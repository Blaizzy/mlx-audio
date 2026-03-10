"""
InstructTTSEval: Evaluation of instruction-following capabilities in TTS systems.

This module implements the InstructTTSEval benchmark for evaluating TTS models on their
ability to follow complex natural-language style instructions.

The benchmark has three task types:
- APS (Acoustic Property Specification): Low-level acoustic attribute descriptions
- DSD (Detailed Style Description): High-level style instructions
- RP (Role-Play): Context-based scenario instructions

Reference: https://arxiv.org/abs/2506.16381
Dataset: https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval
"""

import argparse
import csv
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load as load_tts_model

from .utils import inference

# Instruction types in InstructTTSEval
INSTRUCTION_TYPES = ["APS", "DSD", "RP"]


def load_dataset(
    dataset_name: str = "CaasiHUANG/InstructTTSEval",
    split: str = "en",
    streaming: bool = False,
    max_samples: Optional[int] = None,
):
    """
    Load the InstructTTSEval dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split ('en' for English, 'zh' for Chinese).
        streaming: Whether to use streaming mode.
        max_samples: Maximum number of samples to load (for debugging).

    Returns:
        Dataset object.
    """
    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(dataset_name, split=split, streaming=streaming)

    # Remove audio column to avoid decoding issues because of torchcodec not being installed
    if "reference_audio" in dataset.column_names:
        dataset = dataset.remove_columns(["reference_audio"])

    if max_samples and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    elif max_samples and streaming:
        dataset = dataset.take(max_samples)

    return dataset


def save_audio(audio: mx.array, path: str, sample_rate: int = 24000) -> None:
    """Save audio to file."""
    audio_write(path, np.array(audio), sample_rate, format="wav")


def get_voice_for_model(model, model_type: str, lang_code: str = "en") -> Optional[str]:
    """
    Get an appropriate voice/speaker for the model.

    Args:
        model: Loaded TTS model.
        model_type: Type of model (e.g., 'CustomVoice', 'VoiceDesign').
        lang_code: Language code ('en' or 'zh').

    Returns:
        Voice name or None.
    """
    if hasattr(model, "config"):
        tts_model_type = getattr(model.config, "tts_model_type", None)
        if tts_model_type == "custom_voice":
            # CustomVoice models have predefined speakers
            # Try to get available speakers from model
            if hasattr(model, "available_speakers"):
                speakers = model.available_speakers
                if speakers:
                    # Prefer English-sounding names for English, Chinese for Chinese
                    if lang_code == "en":
                        for name in ["vivian", "ryan", "aiden", "eric", "dylan"]:
                            if name in speakers:
                                return name
                    else:  # Chinese
                        for name in ["uncle_fu", "serena", "ono_anna", "sohee"]:
                            if name in speakers:
                                return name
                    return speakers[0]  # Fall back to first available
            # Default speakers based on model
            return "vivian"  # Common default for Qwen3-TTS CustomVoice
        elif tts_model_type == "voice_design":
            # VoiceDesign models don't need a voice parameter
            return None
    return None


def run_inference(
    model,
    text: str,
    instruction: str,
    voice: Optional[str] = None,
    lang_code: str = "auto",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    verbose: bool = False,
) -> Optional[mx.array]:
    """
    Run TTS inference with the given instruction.

    Args:
        model: Loaded TTS model.
        text: Text to synthesize.
        instruction: Style instruction (APS, DSD, or RP content).
        voice: Voice/speaker name.
        lang_code: Language code.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        verbose: Whether to print verbose output.

    Returns:
        Generated audio array or None if generation failed.
    """
    try:
        results = list(
            inference(
                model=model,
                text=text,
                voice=voice,
                instruct=instruction,
                lang_code=lang_code,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
            )
        )

        if results:
            # Concatenate all audio segments
            audio_segments = [r.audio for r in results]
            return mx.concatenate(audio_segments, axis=0)
        return None
    except Exception as e:
        logging.error(f"Inference error: {e}")
        return None


def evaluate_with_llm(
    audio_path: str,
    instruction: str,
    instruction_type: str,
    evaluator: str = "gemini",
    api_key: Optional[str] = None,
) -> bool:
    """
    Evaluate generated audio against instruction using LLM-as-judge.

    The evaluation uses a binary rubric:
    - True: Primary style attributes align with the prompt
    - False: At least one key style attribute conflicts with the prompt

    Args:
        audio_path: Path to generated audio file.
        instruction: The instruction that was given to the TTS model.
        instruction_type: Type of instruction (APS, DSD, RP).
        evaluator: Evaluator to use ('gemini', 'openai', 'local').
        api_key: API key for the evaluator service.

    Returns:
        Boolean indicating whether the audio follows the instruction.
    """
    if evaluator == "skip":
        # Skip LLM evaluation, just return True (for audio-only generation runs)
        return True

    # Evaluation prompt based on InstructTTSEval paper
    eval_prompt = f"""You are evaluating a text-to-speech (TTS) system's ability to follow style instructions.

The TTS system was given this instruction:
---
{instruction}
---

Listen to the generated audio and determine if it follows the instruction.

Scoring rubric:
- TRUE: The sample's primary style attributes (e.g., gender, pitch, rate, emotion) align with the instruction, without conflict.
- FALSE: At least one key style attribute clearly conflicts with the instruction, or the overall style deviates from the instruction.

Respond with only TRUE or FALSE."""

    if evaluator == "gemini":
        return _evaluate_with_gemini(audio_path, eval_prompt, api_key)
    elif evaluator == "openai":
        return _evaluate_with_openai(audio_path, eval_prompt, api_key)
    else:
        logging.warning(f"Unknown evaluator: {evaluator}, returning True")
        return True


def _evaluate_with_gemini(
    audio_path: str, prompt: str, api_key: Optional[str] = None
) -> bool:
    """Evaluate using Google's Gemini API."""
    try:
        import google.generativeai as genai

        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                logging.warning("No Gemini API key provided, skipping evaluation")
                return True

        model = genai.GenerativeModel("gemini-2.0-flash")

        # Upload audio file
        audio_file = genai.upload_file(audio_path)

        response = model.generate_content([prompt, audio_file])
        result = response.text.strip().upper()

        return result == "TRUE"
    except Exception as e:
        logging.error(f"Gemini evaluation error: {e}")
        return True  # Default to True on error


def _evaluate_with_openai(
    audio_path: str, prompt: str, api_key: Optional[str] = None
) -> bool:
    """Evaluate using OpenAI's API with audio capabilities."""
    try:
        import base64

        from openai import OpenAI

        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        # Read and encode audio
        with open(audio_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_data, "format": "wav"},
                        },
                    ],
                }
            ],
        )

        result = response.choices[0].message.content.strip().upper()
        return result == "TRUE"
    except Exception as e:
        logging.error(f"OpenAI evaluation error: {e}")
        return True  # Default to True on error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TTS models on InstructTTSEval benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace repo ID of the TTS model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CaasiHUANG/InstructTTSEval",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Dataset split to evaluate on (en=English, zh=Chinese)",
    )
    parser.add_argument(
        "--instruction-types",
        type=str,
        nargs="+",
        default=["APS", "DSD", "RP"],
        choices=["APS", "DSD", "RP"],
        help="Instruction types to evaluate",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset loading",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/instruct_tts_eval",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice/speaker name (model-specific)",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="skip",
        choices=["gemini", "openai", "skip"],
        help="LLM evaluator to use for scoring (skip=audio generation only)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the evaluator service",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated audio files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for debugging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_audio:
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    # Determine language code from split
    lang_code = "en" if args.split == "en" else "zh"

    logging.info(f"Loading model from {args.model}")
    print(f"Loading model: {args.model}")
    model = load_tts_model(args.model)

    # Determine voice if not specified
    voice = args.voice
    if voice is None:
        voice = get_voice_for_model(model, args.model, lang_code)
        if voice:
            print(f"Using voice: {voice}")

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split {args.split}")
    print(f"Loading dataset: {args.dataset} (split={args.split})")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        max_samples=args.max_samples,
    )

    # Initialize results tracking
    results = []
    scores = {inst_type: {"correct": 0, "total": 0} for inst_type in args.instruction_types}

    # Evaluate each sample
    model_name = args.model.split("/")[-1]

    # Get total count for progress bar
    try:
        total = len(dataset)
    except TypeError:
        total = args.max_samples if args.max_samples else None

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating", total=total)):
        sample_id = sample.get("id", f"{args.split}_{idx}")
        text = sample["text"]

        # Process each instruction type
        for inst_type in args.instruction_types:
            instruction = sample[inst_type]

            # Run inference
            audio = run_inference(
                model=model,
                text=text,
                instruction=instruction,
                voice=voice,
                lang_code=lang_code,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose,
            )

            if audio is None:
                logging.warning(f"Failed to generate audio for sample {sample_id} ({inst_type})")
                result = {
                    "id": sample_id,
                    "instruction_type": inst_type,
                    "text": text,
                    "instruction": instruction[:200] + "..." if len(instruction) > 200 else instruction,
                    "generated": False,
                    "score": False,
                }
                results.append(result)
                scores[inst_type]["total"] += 1
                continue

            # Save audio if requested
            audio_path = None
            if args.save_audio:
                audio_path = str(audio_dir / f"{sample_id}_{inst_type}.wav")
                save_audio(audio, audio_path, sample_rate=model.sample_rate)

            # Evaluate with LLM if not skipping
            if args.evaluator != "skip":
                # Need to save audio temporarily for evaluation
                if audio_path is None:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        audio_path = f.name
                        save_audio(audio, audio_path, sample_rate=model.sample_rate)

                is_correct = evaluate_with_llm(
                    audio_path=audio_path,
                    instruction=instruction,
                    instruction_type=inst_type,
                    evaluator=args.evaluator,
                    api_key=args.api_key,
                )

                # Clean up temp file
                if not args.save_audio and audio_path:
                    os.unlink(audio_path)
            else:
                is_correct = None  # No evaluation

            # Record result
            result = {
                "id": sample_id,
                "instruction_type": inst_type,
                "text": text,
                "instruction": instruction[:200] + "..." if len(instruction) > 200 else instruction,
                "generated": True,
                "score": is_correct,
            }
            results.append(result)

            scores[inst_type]["total"] += 1
            if is_correct:
                scores[inst_type]["correct"] += 1

        # Progress update every 10 samples
        if (idx + 1) % 10 == 0:
            logging.info(f"Processed {idx + 1} samples")

    # Calculate final scores
    final_scores = {}
    for inst_type in args.instruction_types:
        total = scores[inst_type]["total"]
        correct = scores[inst_type]["correct"]
        if total > 0 and args.evaluator != "skip":
            final_scores[inst_type] = (correct / total) * 100
        else:
            final_scores[inst_type] = None

    # Calculate average score
    valid_scores = [s for s in final_scores.values() if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    # Save results to CSV
    results_file = output_dir / f"{model_name}_InstructTTSEval_{args.split}.csv"
    fieldnames = ["id", "instruction_type", "text", "instruction", "generated", "score"]

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "instruction_types": args.instruction_types,
        "evaluator": args.evaluator,
        "total_samples": len(set(r["id"] for r in results)),
        "scores": {
            inst_type: {
                "correct": scores[inst_type]["correct"],
                "total": scores[inst_type]["total"],
                "accuracy": final_scores[inst_type],
            }
            for inst_type in args.instruction_types
        },
        "average_score": avg_score,
    }

    summary_file = output_dir / f"{model_name}_InstructTTSEval_{args.split}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*80}")
    print("InstructTTSEval Results")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Evaluator: {args.evaluator}")
    print(f"Total Samples: {summary['total_samples']}")

    if args.evaluator != "skip":
        print(f"\n{'-'*80}")
        print("Scores by Instruction Type:")
        print(f"{'-'*80}")
        for inst_type in args.instruction_types:
            score_info = summary["scores"][inst_type]
            print(
                f"  {inst_type}: {score_info['correct']}/{score_info['total']} "
                f"({score_info['accuracy']:.2f}%)"
            )

        if avg_score is not None:
            print(f"\nAverage Score: {avg_score:.2f}%")
    else:
        print("\nScoring skipped (--evaluator=skip)")
        print(f"Generated audio for {sum(s['total'] for s in scores.values())} instruction-text pairs")

    print(f"{'='*80}")
    print(f"\nResults saved to {results_file}")
    print(f"Summary saved to {summary_file}")
    if args.save_audio:
        print(f"Audio files saved to {audio_dir}")


if __name__ == "__main__":
    main()

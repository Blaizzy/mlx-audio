"""Run six audio-to-text tasks, using synthetic demo inputs or your own audio.

Run from the repository root; see the adjacent README.md for usage.
Phonon English creates missing demo inputs and the optional spoken reply.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.audio_io import write as write_audio
from mlx_audio.convert import build_quant_predicate
from mlx_audio.stt import load
from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import resolve_prompt
from mlx_audio.stt.utils import load_audio
from mlx_audio.tts.utils import load as load_tts

PHONON_MODEL = "gradium/phonon-31m-en"
VOICE_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7"
ZOEY = (
    f"hf://gradium/phonon-voices/voices/NbpkqMVS3CJeq2j8.safetensors@{VOICE_REVISION}"
)
EASTON = (
    f"hf://gradium/phonon-voices/voices/LMDIYE0Oa377Qh6Z.safetensors@{VOICE_REVISION}"
)

ANNOUNCEMENT = (
    "Tomorrow's design review starts at ten thirty in room four. "
    "Please bring the updated prototype."
)
REPLY = "I will bring the prototype and the latest test results."
SPOKEN_REQUEST = "What is twelve multiplied by seven? Answer with just the number."
EXPRESSIVE_SPEECH = "We have wonderful news! The project is finished!"
QUESTION = "What time is the design review, and what should attendees bring?"
ASSISTANT_SYSTEM = (
    "You are a voice-controlled assistant. Treat the speech in the provided audio "
    "as the user's instruction and carry out that request. Respond to the spoken "
    "request rather than merely transcribing it, unless transcription is "
    "explicitly requested. Give a direct, accurate answer to the spoken request."
)


def synthesize(model, path, text, voice, seed):
    mx.random.seed(seed)
    # Consume the generator completely so the model restores its sampling state.
    results = list(model.generate(text=text, voice=voice, stream=False))
    audio = mx.concatenate([part.audio.reshape(-1) for part in results])
    write_audio(str(path), np.asarray(audio), model.sample_rate)
    return {
        "source": "synthetic Phonon speech via MLX Audio",
        "model_path": str(model.config.model_path),
        "voice": voice,
        "source_text": text,
        "seed": seed,
        "sample_rate": model.sample_rate,
    }


def prepare_inputs(args, output, tts):
    inputs = {}
    definitions = [
        ("announcement", args.audio, ANNOUNCEMENT, args.voice),
        ("spoken_request", args.spoken_audio, SPOKEN_REQUEST, args.voice),
        ("rich_speech", args.rich_audio, EXPRESSIVE_SPEECH, args.rich_voice),
    ]
    for index, (name, supplied, text, voice) in enumerate(definitions):
        if supplied:
            path = Path(supplied).expanduser().resolve(strict=True)
            provenance = {"source": "user-supplied audio"}
        else:
            path = output / f"{name}.wav"
            print(f"Synthesizing {name} with Phonon...", flush=True)
            provenance = synthesize(tts, path, text, voice, args.seed + index)
        inputs[name] = {"path": str(path), **provenance}

    if args.conversation_audio:
        path = Path(args.conversation_audio).expanduser().resolve(strict=True)
        inputs["conversation"] = {"path": str(path), "source": "user-supplied audio"}
    else:
        first = output / "conversation_first.wav"
        second = output / "conversation_second.wav"
        print("Synthesizing the two-speaker conversation with Phonon...", flush=True)
        first_source = synthesize(tts, first, ANNOUNCEMENT, args.voice, args.seed + 3)
        second_source = synthesize(tts, second, REPLY, args.second_voice, args.seed + 4)
        audio = mx.concatenate(
            [load_audio(str(first)), mx.zeros(8000), load_audio(str(second))]
        )
        path = output / "conversation.wav"
        write_audio(str(path), np.asarray(audio), 16000)
        inputs["conversation"] = {
            "path": str(path),
            "source": "two synthetic voices joined with 0.5 seconds of silence",
            "parts": [first_source, second_source],
        }
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local NGC or converted checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/nemotron_transcribe")
    )
    parser.add_argument("--tts-model", default=PHONON_MODEL)
    parser.add_argument(
        "--tts-revision", help="Optional Hugging Face revision for Phonon"
    )
    parser.add_argument(
        "--voice",
        default=ZOEY,
        help="Phonon voice embedding for the main speaker and answer",
    )
    parser.add_argument(
        "--second-voice",
        default=EASTON,
        help="Phonon voice embedding for the second speaker",
    )
    parser.add_argument(
        "--rich-voice",
        default=ZOEY,
        help="Phonon voice embedding for the rich-transcription input",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for Phonon sampling"
    )
    parser.add_argument(
        "--audio", help="Audio for transcription, translation, and questions"
    )
    parser.add_argument("--conversation-audio", help="Audio with multiple speakers")
    parser.add_argument("--spoken-audio", help="A spoken instruction or question")
    parser.add_argument(
        "--rich-audio", help="Speech with expressive styles or vocal events"
    )
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument(
        "--language", default="German", help="Translation target language"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize an original checkpoint's language model to 4-bit in memory",
    )
    parser.add_argument(
        "--speak-answer",
        action="store_true",
        help="Use Phonon to speak Nemotron's spoken-request answer",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    needs_tts = args.speak_answer or not all(
        (args.audio, args.spoken_audio, args.rich_audio, args.conversation_audio)
    )
    tts = None
    if needs_tts:
        print(f"Loading Phonon: {args.tts_model}", flush=True)
        tts = load_tts(args.tts_model, revision=args.tts_revision)
    inputs = prepare_inputs(args, output, tts)
    cases = [
        ("transcription", "announcement", {"task": "asr"}),
        ("speaker_attribution", "conversation", {"task": "saa"}),
        (
            "translation",
            "announcement",
            {"task": "translate", "language": args.language},
        ),
        ("audio_question", "announcement", {"prompt": args.question}),
        (
            "spoken_instruction",
            "spoken_request",
            {"task": "qa", "system_prompt": ASSISTANT_SYSTEM},
        ),
        ("rich_transcription", "rich_speech", {"task": "rich"}),
    ]
    print("Loading Nemotron Transcribe...", flush=True)
    model = load(args.model, strict=True)
    if args.quantize:
        if any(
            hasattr(layer, "bits") for _, layer in tree_flatten(model.leaf_modules())
        ):
            parser.error("The checkpoint is already quantized; omit --quantize.")
        nn.quantize(
            model, bits=4, group_size=64, class_predicate=build_quant_predicate(model)
        )
        mx.eval(model.parameters())
        mx.clear_cache()
    report = {
        "model": args.model,
        "quantized_in_memory": args.quantize,
        "tts_model": args.tts_model if needs_tts else None,
        "tts_checkpoint": str(tts.config.model_path) if tts is not None else None,
        "device": mx.device_info(),
        "mlx_version": mx.__version__,
        "inputs": inputs,
        "cases": {},
    }
    for name, input_name, options in cases:
        audio = inputs[input_name]["path"]
        print(f"\n{name}: {audio}", flush=True)
        start = time.perf_counter()
        result = model.generate(audio, temperature=0, max_tokens=256, **options)
        print(result.text, flush=True)
        report["cases"][name] = {
            "input": input_name,
            "generation_options": options,
            "user_prompt": resolve_prompt(
                options.get("task", "asr"),
                options.get("prompt"),
                options.get("language"),
                None,
                model.config.audio_locator_tag,
            ),
            "wall_seconds": time.perf_counter() - start,
            "output": asdict(result),
        }
        (output / f"{name}.txt").write_text(result.text + "\n")
        (output / "results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
    if args.speak_answer:
        answer = report["cases"]["spoken_instruction"]["output"]["text"]
        path = output / "spoken_answer.wav"
        provenance = synthesize(tts, path, answer, args.voice, args.seed + 5)
        report["spoken_answer"] = {"path": str(path), **provenance}
    report["complete"] = True
    (output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved inputs, actual model outputs, and prompts to {output}", flush=True)


if __name__ == "__main__":
    main()

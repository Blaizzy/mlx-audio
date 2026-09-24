r"""Speaker-attributed transcription with Nemotron Diarization and mlx-audio ASR.

Usage (from the repository root):
    python examples/nemotron_diarization_asr.py --audio conversation.wav
    python examples/nemotron_diarization_asr.py --audio conversation.wav \
        --language en-US --output transcript.json
    python examples/nemotron_diarization_asr.py --audio conversation.wav \
        --diar-model mlx-community/Nemotron-3-Diarization-8bit \
        --asr-model mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit

By default, Nemotron 3.5 ASR runs a masked feature stream and independent
encoder/decoder caches for each speaker, following NVIDIA's integration approach:
https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md
Both models consume the same 16 kHz mono waveform. Masking does not separate
simultaneous voices; overlap may produce repeated or missing words.

Use --mode timestamps for post-hoc word attribution with Nemotron or Parakeet.
That mode marks overlapping speech ambiguous. RNNT timestamps are emission
times, so timestamp-based attribution near speaker changes is approximate.
"""

import argparse
import json
import math
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path

import numpy as np

DIAR_MODEL = "mlx-community/Nemotron-3-Diarization"
ASR_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16000


def extract_tokens(result):
    """Join BPE continuations into words before assigning speakers.

    Whitespace marks word starts; scripts without spaces retain token granularity.
    Delayed punctuation attaches to the preceding word without extending its time.
    """
    if not hasattr(result, "sentences"):
        raise ValueError(
            "This example needs timestamped AlignedResult output. "
            "Use a Nemotron 3.5 ASR or Parakeet model."
        )
    tokens = []
    whitespace = ""
    for sentence in result.sentences:
        word = None
        for token in sentence.tokens:
            if not token.text.strip():
                whitespace += token.text
                continue
            text = whitespace + token.text
            whitespace = ""
            punctuation = all(
                unicodedata.category(c).startswith("P") for c in text.strip()
            )
            # No-space scripts cannot use BPE whitespace to determine word boundaries.
            no_spaces = any(
                unicodedata.east_asian_width(c) in ("W", "F")
                or "THAI" in unicodedata.name(c, "")
                for c in text.strip()
            )
            if word is not None and punctuation:
                word["text"] += text
            elif word is not None and not text[0].isspace() and not no_spaces:
                word["text"] += text
                word["end"] = max(word["end"], float(token.end))
            else:
                word = {
                    "start": float(token.start),
                    "end": float(token.end),
                    "text": text,
                }
                tokens.append(word)
    return tokens


def assign_speakers(tokens, speaker_probs, frame_duration, threshold=0.5):
    """Label tokens by duration-weighted mean activity over their time intervals.

    Never assign the nearest speaker across silence or beyond the probability
    timeline. If any covered frame has simultaneous speakers, leave the speaker
    unknown and retain the active candidates instead of guessing who said it.
    """
    probs = np.asarray(speaker_probs, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] == 0:
        raise ValueError("speaker_probs must have shape (frames, speakers)")
    if not math.isfinite(frame_duration) or frame_duration <= 0:
        raise ValueError("frame_duration must be positive and finite")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    timeline_end = len(probs) * frame_duration
    attributed = []
    for token in tokens:
        start, end = token["start"], token["end"]
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError(f"Invalid ASR timestamp interval: {start}, {end}")
        item = {
            **token,
            "speaker": None,
            "speaker_score": 0.0,
            "overlap": False,
            "candidates": [],
        }
        # A zero-duration token uses the frame containing its emission time.
        stop = end if end > start else start + frame_duration
        left, right = max(0.0, start), min(timeline_end, stop)
        if right > left:
            first = max(0, math.floor(left / frame_duration))
            last = min(len(probs), math.ceil(right / frame_duration))
            starts = np.arange(first, last) * frame_duration
            weights = np.maximum(
                0, np.minimum(starts + frame_duration, right) - np.maximum(starts, left)
            )
            window = probs[first:last]
            scores = np.sum(window * weights[:, None], axis=0) / weights.sum()
            active = window[weights > 0] > threshold
            candidates = np.flatnonzero(np.any(active, axis=0))
            overlap = bool(np.any(active.sum(axis=1) > 1))
            speaker = int(np.argmax(scores))
            item.update(
                speaker_score=float(scores[speaker]),
                overlap=overlap,
                candidates=[f"speaker_{int(i)}" for i in candidates],
            )
            if not overlap and scores[speaker] > threshold:
                item["speaker"] = f"speaker_{speaker}"
        attributed.append(item)
    return attributed


def merge_tokens(tokens, max_gap=1.0):
    """Combine consecutive tokens with the same attribution into readable turns."""
    segments = []
    for token in tokens:
        key = (
            token["speaker"],
            token["overlap"],
            tuple(token["candidates"]) if token["overlap"] else (),
        )
        if segments:
            previous = segments[-1]
            previous_key = (
                previous["speaker"],
                previous["overlap"],
                tuple(previous["candidates"]) if previous["overlap"] else (),
            )
        if (
            segments
            and key == previous_key
            and token["start"] - previous["end"] <= max_gap
        ):
            previous["text"] += token["text"]
            previous["end"] = max(previous["end"], token["end"])
        else:
            segments.append(
                {
                    k: token[k]
                    for k in (
                        "start",
                        "end",
                        "text",
                        "speaker",
                        "overlap",
                        "candidates",
                    )
                }
            )
    for segment in segments:
        segment["text"] = segment["text"].strip()
    return segments


def transcribe_with_speakers(
    audio,
    diar_model,
    asr_model,
    *,
    language=None,
    threshold=0.5,
    asr_chunk_duration=30.0,
    mode="masked",
):
    """Transcribe using independent speaker streams or timestamp association."""
    if mode == "masked":
        if not callable(getattr(asr_model, "generate_speakers", None)):
            raise ValueError(
                "Masked mode requires Nemotron 3.5 ASR; use --mode timestamps for Parakeet"
            )
        transcripts = asr_model.generate_speakers(
            audio, diar_model, language=language, threshold=threshold
        )
        tokens = []
        segments = []
        for speaker, transcript in transcripts.items():
            words = [
                {**word, "speaker": speaker} for word in extract_tokens(transcript)
            ]
            tokens.extend(words)
            # Group each stream independently, even when its emissions interleave
            # with another speaker. No overlap confidence is implied by a mask.
            turns = merge_tokens(
                [{**word, "overlap": False, "candidates": []} for word in words]
            )
            segments.extend(
                [
                    {key: turn[key] for key in ("start", "end", "text", "speaker")}
                    for turn in turns
                ]
            )
        tokens.sort(key=lambda token: token["start"])
        segments.sort(key=lambda segment: segment["start"])
        return {
            "mode": mode,
            "text": " ".join(segment["text"] for segment in segments),
            "segments": segments,
            "tokens": tokens,
            "speaker_transcripts": {
                speaker: result.text for speaker, result in transcripts.items()
            },
        }
    if mode != "timestamps":
        raise ValueError("mode must be 'masked' or 'timestamps'")
    diarization = diar_model.generate(
        audio, sample_rate=SAMPLE_RATE, threshold=threshold
    )
    # Silence has no speaker stream to transcribe; avoid ASR silence hallucinations.
    if not diarization.segments:
        return {"text": "", "segments": [], "tokens": [], "diarization": []}
    options = {"chunk_duration": asr_chunk_duration}
    if language is not None:
        options["language"] = language
    transcription = asr_model.generate(audio, **options)
    config = diar_model.config
    frame_duration = (
        config.processor_config.hop_length
        / config.processor_config.sampling_rate
        * config.output_subsampling_factor
    )
    tokens = assign_speakers(
        extract_tokens(transcription),
        np.array(diarization.speaker_probs),
        frame_duration,
        threshold,
    )
    return {
        "text": transcription.text,
        "segments": merge_tokens(tokens),
        "tokens": tokens,
        "diarization": [asdict(segment) for segment in diarization.segments],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", "-a", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("masked", "timestamps"),
        default="masked",
        help="Speaker-masked ASR (Nemotron) or timestamp attribution (Nemotron/Parakeet)",
    )
    parser.add_argument(
        "--diar-model",
        default=DIAR_MODEL,
        help="HF repo or local converted Nemotron diarization folder",
    )
    parser.add_argument(
        "--asr-model",
        default=ASR_MODEL,
        help="HF repo or local Nemotron 3.5 ASR/Parakeet folder",
    )
    parser.add_argument(
        "--language", help="ASR language prompt, e.g. en-US or auto for Nemotron"
    )
    parser.add_argument(
        "--diar-preset",
        choices=("offline", "low", "very_low", "ultra_low"),
        default="low",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--asr-chunk-duration",
        type=float,
        default=30.0,
        help="ASR input chunk duration in seconds for --mode timestamps",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional JSON output with attributed tokens and turns",
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if not math.isfinite(args.asr_chunk_duration) or args.asr_chunk_duration <= 2:
        parser.error("--asr-chunk-duration must be finite and greater than 2 seconds")

    from mlx_audio.stt import load as load_asr
    from mlx_audio.stt.utils import load_audio
    from mlx_audio.vad import load as load_diarization

    # Decode, downmix and resample once so both timelines have the same origin.
    audio = load_audio(str(args.audio), sr=SAMPLE_RATE)
    print(f"Loading diarization: {args.diar_model}", file=sys.stderr)
    diar_model = load_diarization(args.diar_model, strict=True)
    diar_model.set_streaming_config(args.diar_preset)
    print(f"Loading ASR: {args.asr_model}", file=sys.stderr)
    asr_model = load_asr(args.asr_model)
    if (
        getattr(getattr(asr_model, "preprocessor_config", None), "sample_rate", None)
        != SAMPLE_RATE
    ):
        parser.error("Use a Nemotron 3.5 ASR or Parakeet model accepting 16 kHz audio")
    print("Diarizing and transcribing...", file=sys.stderr)
    result = transcribe_with_speakers(
        audio,
        diar_model,
        asr_model,
        language=args.language,
        threshold=args.threshold,
        asr_chunk_duration=args.asr_chunk_duration,
        mode=args.mode,
    )
    result["audio"] = str(args.audio)
    result["models"] = {"diarization": args.diar_model, "asr": args.asr_model}
    for segment in result["segments"]:
        label = segment["speaker"] or "unknown"
        if segment.get("overlap"):
            label = "overlap: " + ", ".join(segment["candidates"])
        print(
            f"[{segment['start']:.2f} - {segment['end']:.2f}] {label}: {segment['text']}"
        )
    if not result["segments"]:
        print("No speech was transcribed.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from huggingface_hub import hf_hub_download

from .schema import STTEvalSample

DEFAULT_DATASET_REPO = "zhaochenyang20/seed-tts-eval"
DEFAULT_REVISION = "main"
EN_META_FILE = "en/meta.lst"
EN_TARGET_PREFIX = "en/wavs/"


@dataclass(frozen=True)
class SeedTTSMetadataEntry:
    utt_id: str
    prompt_text: str
    prompt_wav: str
    target_text: str
    target_wav: str


SeedTTSSample = STTEvalSample


def parse_seed_tts_meta_line(line: str) -> Optional[SeedTTSMetadataEntry]:
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split("|")
    if len(parts) == 4:
        utt_id, prompt_text, prompt_wav, target_text = parts
        target_wav = f"{EN_TARGET_PREFIX}{_strip_wav_suffix(utt_id)}.wav"
    elif len(parts) == 5:
        utt_id, prompt_text, prompt_wav, target_text, target_wav = parts
        if not target_wav:
            target_wav = f"{EN_TARGET_PREFIX}{_strip_wav_suffix(utt_id)}.wav"
    else:
        raise ValueError(
            f"Expected 4 or 5 pipe-delimited Seed-TTS fields, got {len(parts)}: {line!r}"
        )

    utt_id = _strip_wav_suffix(utt_id)
    return SeedTTSMetadataEntry(
        utt_id=utt_id,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
        target_text=target_text,
        target_wav=target_wav,
    )


def load_seed_tts_english_references(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    revision: str = DEFAULT_REVISION,
) -> dict[str, SeedTTSMetadataEntry]:
    meta_path = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        revision=revision,
        filename=EN_META_FILE,
    )

    references: dict[str, SeedTTSMetadataEntry] = {}
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        for line in meta_file:
            entry = parse_seed_tts_meta_line(line)
            if entry is None:
                continue
            if not entry.target_wav.startswith(EN_TARGET_PREFIX):
                continue
            references[entry.utt_id] = entry

    if not references:
        raise ValueError(f"No English Seed-TTS references found in {meta_path}")
    return references


def iter_seed_tts_english_samples(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    revision: str = DEFAULT_REVISION,
    audio_cache_dir: str | Path = "audio-cache",
    limit: Optional[int] = None,
    fail_on_missing_audio: bool = True,
) -> Iterator[STTEvalSample]:
    references = load_seed_tts_english_references(
        dataset_repo=dataset_repo,
        revision=revision,
    )
    dataset = _load_streaming_dataset(dataset_repo=dataset_repo, revision=revision)
    cache_dir = Path(audio_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    yielded = 0
    seen_target_ids: set[str] = set()
    for row in dataset:
        audio = row.get("audio") if isinstance(row, dict) else None
        audio_path = _get_streamed_audio_path(audio)
        relpath = _streaming_audio_path_to_relpath(audio_path)
        if relpath is None or not relpath.startswith(EN_TARGET_PREFIX):
            continue

        utt_id = _strip_wav_suffix(Path(relpath).name)
        entry = references.get(utt_id)
        if entry is None:
            if fail_on_missing_audio:
                raise ValueError(f"Streamed audio has no metadata reference: {relpath}")
            continue

        cached_path = materialize_streamed_audio(
            audio=audio,
            audio_path=audio_path,
            cache_dir=cache_dir,
            utt_id=utt_id,
        )
        seen_target_ids.add(utt_id)
        yield STTEvalSample(
            utt_id=utt_id,
            audio_path=cached_path,
            reference_text=entry.target_text,
            source_path=relpath,
            metadata={"dataset": "seed-tts", "locale": "en", "set": "standard"},
        )

        yielded += 1
        if limit is not None and yielded >= limit:
            return

    if fail_on_missing_audio and yielded == 0:
        missing = sorted(set(references) - seen_target_ids)
        detail = f"; first missing: {missing[0]}" if missing else ""
        raise ValueError(f"No English Seed-TTS target wavs were streamed{detail}")


def materialize_streamed_audio(
    audio: object,
    audio_path: str,
    cache_dir: str | Path,
    utt_id: str,
) -> Path:
    cache_path = Path(cache_dir) / f"{_strip_wav_suffix(utt_id)}.wav"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    audio_bytes = _get_streamed_audio_bytes(audio)
    if audio_bytes is None:
        import fsspec

        with fsspec.open(audio_path, "rb") as remote_file:
            audio_bytes = remote_file.read()

    with open(cache_path, "wb") as local_file:
        local_file.write(audio_bytes)

    return cache_path


def _load_streaming_dataset(dataset_repo: str, revision: str):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Seed-TTS streaming eval requires the optional 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(
        dataset_repo,
        split="train",
        streaming=True,
        revision=revision,
    )
    try:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception:
        # Older/newer datasets versions may already expose paths without casting.
        pass
    return dataset


def _get_streamed_audio_path(audio: object) -> str:
    if isinstance(audio, dict):
        path = audio.get("path")
        if path:
            return str(path)

    path = getattr(audio, "path", None)
    if path:
        return str(path)

    raise ValueError("Streamed dataset row does not expose an audio path")


def _get_streamed_audio_bytes(audio: object) -> Optional[bytes]:
    if isinstance(audio, dict):
        audio_bytes = audio.get("bytes")
        if isinstance(audio_bytes, bytes):
            return audio_bytes
    return None


def _streaming_audio_path_to_relpath(audio_path: str) -> Optional[str]:
    normalized = audio_path.replace(os.sep, "/")
    marker = f"/{EN_TARGET_PREFIX}"
    if marker in normalized:
        return EN_TARGET_PREFIX + normalized.split(marker, 1)[1]
    prompt_marker = "/en/prompt-wavs/"
    if prompt_marker in normalized:
        return "en/prompt-wavs/" + normalized.split(prompt_marker, 1)[1]
    if normalized.startswith(EN_TARGET_PREFIX):
        return normalized
    return None


def _strip_wav_suffix(value: str) -> str:
    return value[:-4] if value.endswith(".wav") else value

"""Audit helpers for MOSS-TTS integration contracts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .request import MossNormalizedRequest

MOSS_VARIANT_DIRS = {
    "MOSS-TTS": "MOSS-TTS",
    "MOSS-TTS-Local": "MOSS-TTS-Local-Transformer",
    "MOSS-TTSD": "MOSS-TTSD-v1.0",
    "MOSS-VoiceGenerator": "MOSS-Voice-Generator",
    "MOSS-SoundEffect": "MOSS-SoundEffect",
    "MOSS-TTS-Realtime": "MOSS-TTS-Realtime",
}

DEFAULT_FRAME_RATE_HZ = 12.5
DEFAULT_REALTIME_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class MossVariantInvariant:
    variant_name: str
    model_type: str
    config_path: Path
    sample_rate: int
    sample_rate_is_inferred: bool
    frame_rate_hz: float
    audio_vocab_size: int
    n_vq: int
    vocab_size: int
    special_tokens: Dict[str, int]
    has_local_transformer: bool


@dataclass(frozen=True)
class MossAudioTokenizerAudit:
    source_path: Path
    commit_hash: Optional[str]
    config_model_type: str
    sampling_rate: int
    downsample_rate: int
    frame_rate_hz: float
    num_quantizers: int
    codebook_size: int


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_moss_variant_invariants(
    reference_root: str | Path = "REFERENCE/MOSS-TTS-HF-Repos",
) -> Dict[str, MossVariantInvariant]:
    root = Path(reference_root)
    invariants: Dict[str, MossVariantInvariant] = {}

    for variant_name, relative_dir in MOSS_VARIANT_DIRS.items():
        config_path = root / relative_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config for {variant_name}: {config_path}")

        config = _load_json(config_path)
        sample_rate = config.get("sampling_rate")
        sample_rate_is_inferred = False
        if sample_rate is None:
            sample_rate = DEFAULT_REALTIME_SAMPLE_RATE
            sample_rate_is_inferred = True

        n_vq = config.get("n_vq", config.get("rvq"))
        if n_vq is None:
            raise ValueError(f"Missing n_vq/rvq in config: {config_path}")

        vocab_size = config.get(
            "vocab_size", config.get("language_config", {}).get("vocab_size")
        )
        if vocab_size is None:
            raise ValueError(f"Missing vocab size in config: {config_path}")

        special_token_keys = [
            "audio_start_token_id",
            "audio_end_token_id",
            "audio_user_slot_token_id",
            "audio_assistant_gen_slot_token_id",
            "audio_assistant_delay_slot_token_id",
            "reference_audio_pad",
            "text_pad",
        ]
        special_tokens = {
            key: int(config[key]) for key in special_token_keys if key in config
        }

        invariants[variant_name] = MossVariantInvariant(
            variant_name=variant_name,
            model_type=str(config["model_type"]),
            config_path=config_path,
            sample_rate=int(sample_rate),
            sample_rate_is_inferred=sample_rate_is_inferred,
            frame_rate_hz=DEFAULT_FRAME_RATE_HZ,
            audio_vocab_size=int(config["audio_vocab_size"]),
            n_vq=int(n_vq),
            vocab_size=int(vocab_size),
            special_tokens=special_tokens,
            has_local_transformer="local_num_layers" in config,
        )

    return invariants


def load_moss_audio_tokenizer_audit(
    reference_root: str | Path = "REFERENCE/MOSS-Audio-Tokenizer",
) -> MossAudioTokenizerAudit:
    root = Path(reference_root)
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer config at {config_path}")

    config = _load_json(config_path)
    quantizer_kwargs = config.get("quantizer_kwargs", {})
    sampling_rate = int(config["sampling_rate"])
    downsample_rate = int(config["downsample_rate"])

    commit_hash: Optional[str] = None
    git_dir = root / ".git"
    if git_dir.exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()

    return MossAudioTokenizerAudit(
        source_path=root,
        commit_hash=commit_hash,
        config_model_type=str(config.get("model_type", "")),
        sampling_rate=sampling_rate,
        downsample_rate=downsample_rate,
        frame_rate_hz=sampling_rate / downsample_rate,
        num_quantizers=int(quantizer_kwargs["num_quantizers"]),
        codebook_size=int(quantizer_kwargs["codebook_size"]),
    )


__all__ = [
    "DEFAULT_FRAME_RATE_HZ",
    "DEFAULT_REALTIME_SAMPLE_RATE",
    "MOSS_VARIANT_DIRS",
    "MossAudioTokenizerAudit",
    "MossNormalizedRequest",
    "MossVariantInvariant",
    "load_moss_audio_tokenizer_audit",
    "load_moss_variant_invariants",
]

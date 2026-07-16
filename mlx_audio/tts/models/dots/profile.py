"""Speaker-enrollment profile: cached, target-independent reference-encode artifacts.

A SpeakerProfile holds the four arrays produced once at enrollment (g_cond, the
AudioVAE-encoded prompt latents, and the patch-encoder embeddings) plus the metadata
needed to reconstruct the identical generation schedule. Saved as a ``.dtprofile``
directory: ``cond.safetensors`` (arrays) + ``profile.json`` (metadata). Pure MLX +
stdlib; no torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

SCHEMA_VERSION = 2  # bumped from 1: in-context patch_emb now derived from denormalized
# latents; v1 profiles cached the wrong (normalized-derived) patch_emb.
_ARRAY_FIELDS = ("g_cond", "prompt_patches", "prompt_denorm_latents", "patch_emb")
_META_FIELDS = (
    "prompt_text",
    "prompt_patch_count",
    "speaker_scale",
    "latent_dim",
    "patch_size",
    "hop_size",
    "sample_rate",
    "dtype",
    "compat_hash",
    "schema_version",
)


@dataclass
class SpeakerProfile:
    g_cond: mx.array
    prompt_patches: mx.array
    prompt_denorm_latents: mx.array
    patch_emb: mx.array
    prompt_text: str
    prompt_patch_count: int
    speaker_scale: float
    latent_dim: int
    patch_size: int
    hop_size: int
    sample_rate: int
    dtype: str
    compat_hash: str
    schema_version: int = SCHEMA_VERSION

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {f: getattr(self, f) for f in _ARRAY_FIELDS}
        mx.eval(*arrays.values())
        mx.save_safetensors(str(path / "cond.safetensors"), arrays)
        meta = {f: getattr(self, f) for f in _META_FIELDS}
        (path / "profile.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerProfile":
        path = Path(path)
        cond = path / "cond.safetensors"
        meta_path = path / "profile.json"
        if not cond.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"{path} is not a valid .dtprofile (need cond.safetensors + profile.json)."
            )
        arrays = mx.load(str(cond))
        missing = [f for f in _ARRAY_FIELDS if f not in arrays]
        if missing:
            raise ValueError(f"profile {path} missing arrays: {missing}")
        meta = json.loads(meta_path.read_text())
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"profile schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}"
            )
        return cls(
            **{f: arrays[f] for f in _ARRAY_FIELDS},
            **{f: meta[f] for f in _META_FIELDS},
        )

    def check_compat(self, model_hash: str) -> None:
        if self.compat_hash != model_hash:
            raise ValueError(
                "this speaker profile was enrolled on a different model "
                f"(profile hash {self.compat_hash[:12]}…, current model {model_hash[:12]}…) — "
                "re-enroll the voice on the current model."
            )


def _safetensors_signature(
    path: str | Path, exclude_prefixes: tuple[str, ...] = ()
) -> list:
    """Sorted ``[(name, shape)]`` from a safetensors header — metadata only, no
    tensor reads. ``exclude_prefixes`` drops e.g. the quantized ``llm.`` tensors."""
    path = Path(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n).decode("utf-8"))
    sig = []
    for name, meta in sorted(header.items()):
        if name == "__metadata__":
            continue
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        # NB: dtype is intentionally excluded — non-LLM float components are stored at
        # different on-disk dtypes across variants (bf16 dir: F32; int4/int8 dirs: BF16)
        # but load+cast to the same runtime dtype, yielding identical artifacts. Keying on
        # (name, shape) keeps profiles portable across quant variants while still pinning
        # the model architecture; config.json + latent_stats values pin the rest.
        sig.append((name, tuple(meta["shape"])))
    return sig


def model_compat_hash(model_dir: str | Path) -> str:
    """Hash the artifact-producing components of a converted model dir.

    Pins config dims + latent-stats normalization + the AudioVAE / CAM++ / patch-encoder
    / projection tensor signatures (the bf16 components that produce profile artifacts).
    EXCLUDES the ``quantization`` config block and all ``llm.`` tensors, so a profile is
    portable across int4 / int8 / bf16.
    """
    model_dir = Path(model_dir)
    h = hashlib.sha256()

    cfg = json.loads((model_dir / "config.json").read_text())
    cfg.pop("quantization", None)  # quant-independent
    h.update(json.dumps(cfg, sort_keys=True).encode("utf-8"))
    h.update((model_dir / "latent_stats.npz").read_bytes())

    sigs = {
        "core": _safetensors_signature(
            model_dir / "core.safetensors", exclude_prefixes=("llm.",)
        ),
        "speaker": _safetensors_signature(model_dir / "speaker.safetensors"),
        "vocoder": _safetensors_signature(model_dir / "vocoder.safetensors"),
    }
    h.update(json.dumps(sigs, sort_keys=True, default=list).encode("utf-8"))
    return h.hexdigest()

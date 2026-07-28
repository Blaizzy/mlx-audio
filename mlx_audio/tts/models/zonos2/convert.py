from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

import mlx.core as mx

from .config import Zonos2Config
from .speaker_encoder import DEFAULT_SPEAKER_ENCODER_REPO, convert_speaker_encoder

OFFICIAL_MLX_REPO = "mlx-community/Zyphra-ZONOS2"


def _load_source_dir(hf_path: str, revision: str | None = None) -> Path:
    path = Path(hf_path).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            hf_path,
            revision=revision,
            allow_patterns=["params.json", "model.pth", "model.pt", "README.md"],
        )
    )


def _load_params(source_dir: Path) -> dict:
    params_path = source_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found in {source_dir}")
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    return params.get("model", params)


def _load_torch_state(source_dir: Path):
    import torch

    for name in ("model.pth", "model.pt", "consolidated/consolidated.pth"):
        candidate = source_dir / name
        if candidate.exists():
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            return state.get("model", state)
    raise FileNotFoundError(f"no model.pth/model.pt found in {source_dir}")


def _torch_to_mx(tensor, dtype: str) -> mx.array:
    tensor = tensor.detach().cpu().contiguous()
    array = mx.array(tensor)
    if dtype == "bfloat16" and array.dtype in (mx.float16, mx.float32, mx.bfloat16):
        return array.astype(mx.bfloat16)
    if dtype == "float16" and array.dtype in (mx.float16, mx.float32, mx.bfloat16):
        return array.astype(mx.float16)
    if dtype == "float32" and array.dtype in (mx.float16, mx.float32, mx.bfloat16):
        return array.astype(mx.float32)
    return array


def _remap_parametrized_key(key: str) -> str:
    if ".parametrizations." in key and ".original" in key:
        return key.replace(".parametrizations.", ".").replace(".original", "")
    return key


def _skip_key(key: str) -> bool:
    return ".router.ent_denom" in key or ".router.normalized_entropy" in key


def _router_mlp_key(key: str) -> str:
    return (
        key.replace(".router.router_mlp.0.", ".router.router_mlp.l0.")
        .replace(".router.router_mlp.2.", ".router.router_mlp.l2.")
        .replace(".router.router_mlp.4.", ".router.router_mlp.l4.")
    )


def _expert_prefix(key: str, suffix: str) -> str | None:
    if not key.endswith(suffix):
        return None
    marker = ".feed_forward.experts."
    if marker not in key:
        return None
    return key[: -len(suffix)]


def iter_converted_state_dict(
    state_dict,
    dtype: str = "bfloat16",
) -> Iterator[tuple[str, mx.array]]:
    consumed: set[str] = set()

    for raw_key in list(state_dict.keys()):
        if raw_key in consumed:
            continue
        key = _remap_parametrized_key(raw_key)
        if _skip_key(key):
            consumed.add(raw_key)
            continue

        prefix = _expert_prefix(key, "w13")
        if prefix is not None:
            w13 = state_dict[raw_key]
            gate = w13[:, 0::2, :]
            up = w13[:, 1::2, :]
            yield prefix + "gate_proj.weight", _torch_to_mx(gate, dtype)
            yield prefix + "up_proj.weight", _torch_to_mx(up, dtype)
            consumed.add(raw_key)
            continue

        prefix = _expert_prefix(key, "gate_up_proj")
        if prefix is not None:
            gate_up = state_dict[raw_key]
            half = gate_up.shape[1] // 2
            yield prefix + "gate_proj.weight", _torch_to_mx(gate_up[:, :half, :], dtype)
            yield prefix + "up_proj.weight", _torch_to_mx(gate_up[:, half:, :], dtype)
            consumed.add(raw_key)
            continue

        prefix = _expert_prefix(key, "w1.weight")
        if prefix is not None:
            yield prefix + "gate_proj.weight", _torch_to_mx(state_dict[raw_key], dtype)
            consumed.add(raw_key)
            continue

        prefix = _expert_prefix(key, "w3.weight")
        if prefix is not None:
            yield prefix + "up_proj.weight", _torch_to_mx(state_dict[raw_key], dtype)
            consumed.add(raw_key)
            continue

        prefix = _expert_prefix(key, "w2")
        if prefix is not None:
            yield prefix + "down_proj.weight", _torch_to_mx(state_dict[raw_key], dtype)
            consumed.add(raw_key)
            continue

        prefix = _expert_prefix(key, "w2.weight")
        if prefix is not None:
            yield prefix + "down_proj.weight", _torch_to_mx(state_dict[raw_key], dtype)
            consumed.add(raw_key)
            continue

        prefix = _expert_prefix(key, "down_proj")
        if prefix is not None:
            yield prefix + "down_proj.weight", _torch_to_mx(state_dict[raw_key], dtype)
            consumed.add(raw_key)
            continue

        yield _router_mlp_key(key), _torch_to_mx(state_dict[raw_key], dtype)
        consumed.add(raw_key)


def convert_state_dict(state_dict, dtype: str = "bfloat16") -> dict[str, mx.array]:
    return dict(iter_converted_state_dict(state_dict, dtype=dtype))


def _make_shards(weights: dict[str, mx.array], max_file_size_gb: float):
    max_bytes = int(max_file_size_gb * (1 << 30))
    shards = []
    shard = {}
    shard_size = 0
    for key, value in weights.items():
        if shard and shard_size + value.nbytes > max_bytes:
            shards.append(shard)
            shard = {}
            shard_size = 0
        shard[key] = value
        shard_size += value.nbytes
    if shard:
        shards.append(shard)
    return shards


def save_sharded_safetensors(
    weights: dict[str, mx.array],
    out_dir: Path,
    *,
    max_file_size_gb: float = 4.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = _make_shards(weights, max_file_size_gb)
    shard_count = len(shards)
    file_format = (
        "model-{:05d}-of-{:05d}.safetensors" if shard_count > 1 else "model.safetensors"
    )
    index = {
        "metadata": {
            "total_size": int(sum(v.nbytes for v in weights.values())),
            "total_parameters": int(sum(v.size for v in weights.values())),
        },
        "weight_map": {},
    }
    for idx, shard in enumerate(shards, start=1):
        name = file_format.format(idx, shard_count)
        mx.save_safetensors(str(out_dir / name), shard, metadata={"format": "mlx"})
        for key in shard:
            index["weight_map"][key] = name
    index["weight_map"] = {
        key: index["weight_map"][key] for key in sorted(index["weight_map"])
    }
    with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def save_converted_safetensors(
    state_dict,
    out_dir: Path,
    *,
    dtype: str = "bfloat16",
    max_file_size_gb: float = 4.5,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_file_size_gb * (1 << 30))
    if max_bytes <= 0:
        raise ValueError("max_file_size_gb must be positive")

    temp_files: list[Path] = []
    temp_keys: list[list[str]] = []
    weight_to_file: dict[str, str] = {}
    shard: dict[str, mx.array] = {}
    shard_keys: list[str] = []
    shard_size = 0
    total_size = 0
    total_parameters = 0
    converted_count = 0
    largest: list[tuple[str, tuple[int, ...], int]] = []

    def update_largest(key: str, value: mx.array) -> None:
        largest.append((key, tuple(value.shape), int(value.nbytes)))
        largest.sort(key=lambda item: item[2], reverse=True)
        del largest[10:]

    def flush_shard() -> None:
        nonlocal shard, shard_keys, shard_size
        if not shard:
            return
        temp_path = out_dir / f"model-{len(temp_files) + 1:05d}.tmp.safetensors"
        mx.save_safetensors(str(temp_path), shard, metadata={"format": "mlx"})
        temp_files.append(temp_path)
        temp_keys.append(shard_keys)
        shard = {}
        shard_keys = []
        shard_size = 0

    for key, value in iter_converted_state_dict(state_dict, dtype=dtype):
        converted_count += 1
        value_size = int(value.nbytes)
        if shard and shard_size + value_size > max_bytes:
            flush_shard()
        shard[key] = value
        shard_keys.append(key)
        shard_size += value_size
        total_size += value_size
        total_parameters += int(value.size)
        update_largest(key, value)
    flush_shard()

    shard_count = len(temp_files)
    for idx, (temp_path, keys) in enumerate(
        zip(temp_files, temp_keys, strict=True), start=1
    ):
        if shard_count == 1:
            final_name = "model.safetensors"
        else:
            final_name = f"model-{idx:05d}-of-{shard_count:05d}.safetensors"
        final_path = out_dir / final_name
        temp_path.replace(final_path)
        for key in keys:
            weight_to_file[key] = final_name

    index = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": total_parameters,
        },
        "weight_map": {key: weight_to_file[key] for key in sorted(weight_to_file)},
    }
    with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return {
        "source_keys": len(state_dict),
        "converted_keys": converted_count,
        "total_size": total_size,
        "largest": largest,
    }


def _write_config(
    params: dict,
    out_dir: Path,
    dac_repo: str,
    speaker_encoder_repo: str,
    speaker_encoder_path: str | None,
) -> Zonos2Config:
    cfg = Zonos2Config.from_dict(
        {
            **params,
            "dac_model_id": dac_repo,
            "speaker_encoder_model_id": speaker_encoder_repo,
            "speaker_encoder_path": speaker_encoder_path,
        }
    )
    data = asdict(cfg)
    data.pop("model_path", None)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return cfg


def _write_readme(
    out_dir: Path,
    source: str,
    dtype: str,
    dac_repo: str,
    speaker_encoder_repo: str,
    include_speaker_encoder: bool,
) -> None:
    speaker_line = (
        "- Reference-audio speaker extraction uses the bundled speaker encoder."
        if include_speaker_encoder
        else "- Reference-audio speaker extraction downloads the speaker encoder on first use."
    )
    text = f"""---
license: apache-2.0
base_model:
- {source}
library_name: mlx-audio
tags:
- mlx
- text-to-speech
- voice-cloning
---

# ZONOS2

ZONOS2 is Zyphra's autoregressive text-to-speech model with multi-codebook audio
generation, 44.1 kHz DAC decode, and speaker conditioning from reference audio.

**Original model:** [`{source}`](https://huggingface.co/{source})

## Supported Repositories

| Repository | Format | Notes |
|------------|--------|-------|
| `{OFFICIAL_MLX_REPO}` | {dtype} | Official MLX conversion for `mlx-audio` |

## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("{OFFICIAL_MLX_REPO}", lazy=True)

result = next(model.generate(
    text="Hello, this is ZONOS two running locally with MLX audio.",
    max_tokens=220,
))

audio_write("zonos2.wav", result.audio, result.sample_rate)
```

## Voice Cloning

Pass a short reference clip with `ref_audio`. Clean speech-only clips work best.

```python
result = next(model.generate(
    text="This text will be spoken with the reference speaker.",
    ref_audio="speaker.wav",
    max_tokens=220,
))
```

You can also compute the speaker embedding once and reuse it from the Python
API:

```python
speaker = model.extract_speaker_embedding("speaker.wav")

result = next(model.generate(
    text="This reuses a precomputed speaker embedding.",
    speaker_embedding=speaker,
    max_tokens=220,
))
```

## Batch Generation

Use `batch_generate` from Python for non-streaming batches. A shared
`ref_audio` or `speaker_embedding` applies to every row; use `ref_audios` or
`speaker_embeddings` for per-row speaker conditioning.

```python
texts = [
    "This is the first ZONOS two batch sample.",
    "This is the second sample generated in the same batch.",
]

for result in model.batch_generate(texts, max_tokens=220):
    audio_write(
        f"zonos2_batch_{{result.sequence_idx}}.wav",
        result.audio,
        result.sample_rate,
    )
```

Batch streaming is not implemented yet; use `generate(..., stream=True)` for
single-sequence streaming.

## CLI

```bash
python -m mlx_audio.tts.generate \\
  --model {OFFICIAL_MLX_REPO} \\
  --text "Hello, this is ZONOS two running with MLX audio." \\
  --output_path outputs \\
  --file_prefix zonos2
```

Voice cloning:

```bash
python -m mlx_audio.tts.generate \\
  --model {OFFICIAL_MLX_REPO} \\
  --text "This text will use the voice from the reference clip." \\
  --ref_audio speaker.wav \\
  --output_path outputs \\
  --file_prefix zonos2_clone
```

Streaming playback:

```bash
python -m mlx_audio.tts.generate \\
  --model {OFFICIAL_MLX_REPO} \\
  --text "This streams ZONOS two audio chunks as they are generated." \\
  --stream \\
  --streaming_interval 0.5
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ref_audio` | `None` | Reference audio path or array for voice cloning |
| `ref_audios` | `None` | Per-row reference audio list for batch generation, Python API only |
| `speaker_embedding` | `None` | Precomputed 2048-D speaker embedding, Python API only |
| `speaker_embeddings` | `None` | Per-row 2048-D speaker embeddings for batch generation, Python API only |
| `max_tokens` | 1024 | Maximum number of audio token frames |
| `temperature` | 1.15 | Sampling temperature |
| `top_k` | 106 | Top-k sampling filter |
| `top_p` | 0.0 | Nucleus sampling filter, disabled at 0 |
| `min_p` | 0.18 | Minimum-probability sampling filter |
| `repetition_penalty` | 1.2 | Repetition penalty applied to recent audio tokens |
| `seed` | `None` | Seed for deterministic sampling |
| `stream` | `False` | Yield audio chunks during generation |
| `streaming_interval` | 2.0 | Approximate seconds of audio per streaming chunk |
| `text_normalization` | `True` | English text normalization toggle, Python API only |

## Notes

- Output audio is mono 44.1 kHz.
- DAC dependency: [`{dac_repo}`](https://huggingface.co/{dac_repo})
- Speaker encoder: [`{speaker_encoder_repo}`](https://huggingface.co/{speaker_encoder_repo})
{speaker_line}
- English text normalization handles common written forms; unsupported languages
  fall back to raw UTF-8 byte prompting.
- Streaming decodes completed delayed-codebook frames as they become available;
  final audio is emitted in the last chunk.
- Batch generation currently supports non-streaming Python API usage.

## License

See the [`{source}` model card](https://huggingface.co/{source}) for upstream
license and usage details.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _summarize(keys: Iterable[str], converted: dict[str, mx.array]) -> None:
    print(f"[INFO] Source keys: {len(list(keys))}")
    print(f"[INFO] Converted keys: {len(converted)}")
    total = sum(v.nbytes for v in converted.values())
    print(f"[INFO] Converted tensor bytes: {total:,}")
    largest = sorted(converted.items(), key=lambda kv: kv[1].nbytes, reverse=True)[:10]
    print("[INFO] Largest tensors:")
    for key, value in largest:
        print(f"  {key}: shape={value.shape} bytes={value.nbytes:,}")


def _summarize_stats(stats: dict) -> None:
    print(f"[INFO] Source keys: {stats['source_keys']}")
    print(f"[INFO] Converted keys: {stats['converted_keys']}")
    print(f"[INFO] Converted tensor bytes: {stats['total_size']:,}")
    print("[INFO] Largest tensors:")
    for key, shape, nbytes in stats["largest"]:
        print(f"  {key}: shape={shape} bytes={nbytes:,}")


def convert(
    hf_path: str,
    mlx_path: str,
    *,
    dtype: str = "bfloat16",
    dac_repo: str = "mlx-community/descript-audio-codec-44khz",
    speaker_encoder_repo: str = DEFAULT_SPEAKER_ENCODER_REPO,
    include_speaker_encoder: bool = True,
    revision: str | None = None,
    max_file_size_gb: float = 4.5,
) -> Path:
    source_dir = _load_source_dir(hf_path, revision=revision)
    out_dir = Path(mlx_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = _load_params(source_dir)
    state = _load_torch_state(source_dir)
    speaker_encoder_path = "speaker_encoder" if include_speaker_encoder else None
    _write_config(
        params,
        out_dir,
        dac_repo,
        speaker_encoder_repo,
        speaker_encoder_path,
    )
    stats = save_converted_safetensors(
        state,
        out_dir,
        dtype=dtype,
        max_file_size_gb=max_file_size_gb,
    )
    if include_speaker_encoder:
        convert_speaker_encoder(
            speaker_encoder_repo,
            out_dir / "speaker_encoder",
            dtype=dtype,
        )
    _write_readme(
        out_dir,
        hf_path,
        dtype,
        dac_repo,
        speaker_encoder_repo,
        include_speaker_encoder,
    )
    _summarize_stats(stats)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ZONOS2 to MLX format.")
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--dac-repo", default="mlx-community/descript-audio-codec-44khz"
    )
    parser.add_argument("--speaker-encoder-repo", default=DEFAULT_SPEAKER_ENCODER_REPO)
    parser.add_argument("--skip-speaker-encoder", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-file-size-gb", type=float, default=4.5)
    args = parser.parse_args()
    convert(
        args.hf_path,
        args.mlx_path,
        dtype=args.dtype,
        dac_repo=args.dac_repo,
        speaker_encoder_repo=args.speaker_encoder_repo,
        include_speaker_encoder=not args.skip_speaker_encoder,
        revision=args.revision,
        max_file_size_gb=args.max_file_size_gb,
    )


if __name__ == "__main__":
    main()

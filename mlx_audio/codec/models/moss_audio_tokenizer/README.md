# MOSS Audio Tokenizer (Shared Codec)

Shared codec used by all MOSS-TTS runtimes.

- Package: `mlx_audio/codec/models/moss_audio_tokenizer/`
- Runtime class: `MossAudioTokenizer`
- Typical frame rate: `sampling_rate / downsample_rate` (24 kHz / 1920 => 12.5 Hz for upstream checkpoints)

## Why This Matters For MOSS-TTS

`moss_tts` and `moss_tts_realtime` both depend on this codec for:

- reference-audio encoding into discrete codebook tokens
- generated token decoding back to waveform
- streaming decode paths with cache boundaries

If codec contracts are wrong, generation may "run" but output quality/continuity will degrade.

## Core APIs

- `encode(input_values, ..., num_quantizers=None, chunk_duration=None)`
- `decode(audio_codes, ..., num_quantizers=None, chunk_duration=None)`
- `batch_encode(wav_list, num_quantizers=None)`
- `batch_decode(codes_list, num_quantizers=None)`
- `streaming_decode(audio_codes, chunk_tokens=..., num_quantizers=None)`

## Audio-Code Shape Contracts

`decode(...)` accepts both canonical and transposed layouts:

- 2D: `(NQ, T)` or `(T, NQ)`
- 3D: `(NQ, B, T)` or `(B, T, NQ)`

Internally, decode normalizes to canonical `(NQ, B, T)`.

`batch_decode(...)` expects each list entry to resolve to `batch_size=1` after normalization.

`streaming_decode(...)` currently supports only `batch_size=1`.

## `num_quantizers` Behavior

- If `num_quantizers` is omitted, runtime uses configured quantizer count.
- If provided, it must be `1..configured_nq`.
- Prefix decode is supported (decode with fewer quantizers than checkpoint max).

Ambiguous shape ties are resolved with conservative rules favoring canonical orientation to preserve encode->decode round-trip behavior.

## Chunked Encode/Decode

When `chunk_duration` is used:

- must be positive
- must be `<= causal_transformer_context_duration`
- `chunk_duration * sampling_rate` must be divisible by `downsample_rate`
- streaming chunked paths currently require `batch_size=1`

Chunked decode and `streaming_decode(...)` include explicit `mx.eval(...)`/`mx.clear_cache()` boundaries to keep long runs bounded.

## Checkpoint Sanitization

`sanitize(...)` performs two key operations:

1. Tensor layout transpose when checkpoint 3D layouts differ from current MLX parameter shapes.
2. Weight-norm merge for Conv1d params:
   - combines `parametrizations.weight.original0` (`g`) + `original1` (`v`)
   - produces standard `.weight`

Missing weight-norm pairs fail fast.

## Quantization Guardrail

`model_quant_predicate(...)` disables quantization for embedding modules to protect codebook fidelity.

## Load/Save

- `from_pretrained(path_or_repo, strict=True)` supports local path or HF repo.
- `save_config(path)` writes canonicalized config JSON.

MOSS runtime post-load hooks look for embedded codec folders first (`audio_tokenizer`, `moss_audio_tokenizer`, `codec`) and then fall back to `OpenMOSS-Team/MOSS-Audio-Tokenizer`.

## Validation Anchors

Primary tests:

- `mlx_audio/codec/tests/test_moss_audio_tokenizer.py`
- `mlx_audio/codec/tests/test_moss_audio_tokenizer_config_contracts.py`

These cover layout normalization, prefix decode semantics, tie-resolution edge cases, sanitize behavior, and streaming decode constraints.

# Nemotron 3 Diarization

MLX implementation of [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization): a 31-layer RoPE Transformer with feature stacking, an eight-speaker output head, and learned Arrival-Order Speaker Cache (AOSC) embeddings. The model produces speaker activity probabilities every 10 ms, including overlapping speech.

## Supported Models

| Precision | Hugging Face Repository |
| --- | --- |
| BF16 | [mlx-community/Nemotron-3-Diarization](https://huggingface.co/mlx-community/Nemotron-3-Diarization) |
| 8-bit | [mlx-community/Nemotron-3-Diarization-8bit](https://huggingface.co/mlx-community/Nemotron-3-Diarization-8bit) |

## Python

```python
from mlx_audio.vad import load

model = load("mlx-community/Nemotron-3-Diarization", strict=True)
result = model.generate("meeting.wav", threshold=0.5)
print(result.text)  # RTTM

for segment in result.segments:
    print(segment.start, segment.end, segment.speaker)
```

`generate()` accepts an audio filename, NumPy array, or MLX array. Files are converted to mono and resampled automatically; supply `sample_rate` for arrays. It processes long recordings with bounded speaker-cache/FIFO context, then returns a `DiarizationOutput` with segments, `(frames, 8)` probabilities, speaker count, and elapsed processing time. Empty audio returns empty output. Clips shorter than one 10 ms frame have no predictions.

`threshold`, `min_duration` (seconds), and `merge_gap` (seconds) control segment extraction. The postprocessing uses mlx-audio's thresholding and gap merging, rather than NeMo's complete evaluation pipeline. Speaker labels are generic arrival-order IDs, not personal identities. No speech transcription is included.

## Streaming

```python
model.set_streaming_config("low")
for result in model.generate_stream("meeting.wav"):
    print(result.text)
```

Call `set_streaming_config()` before starting a recording. Available presets use NVIDIA's recommended values, in 80 ms encoder frames:

| Preset | Input buffer | Chunk | Right context | FIFO | Cache update period |
| --- | --- | --- | --- | --- | --- |
| `offline` (default) | 30.4 s | 340 | 40 | 40 | 300 |
| `low` | 1.04 s | 9 | 4 | 264 | 222 |
| `very_low` | 0.64 s | 6 | 2 | 264 | 222 |
| `ultra_low` | 0.32 s | 3 | 1 | 264 | 222 |

All presets retain 264 speaker-cache frames. Input-buffer latency excludes compute and the centered STFT's window lookahead.

For live audio, feed arbitrary **16 kHz mono** PCM chunks and flush at the end:

```python
state = model.init_streaming_state()
for chunk in microphone_chunks:
    result, state = model.feed(chunk, state)
    print(result.text)

result, state = model.feed([], state, final=True)
print(result.text)
```

The frontend retains only the PCM history needed for upcoming windows. Chunk boundaries do not reset preemphasis or STFT context. Output may be empty until enough audio is buffered. `generate_stream()` also accepts an iterable of PCM chunks and flushes it automatically. Streaming timestamps are absolute, but a continuous segment can be split between emitted results. `generate()` merges segments across those boundaries.

The low-level `model(features, lengths)` evaluates a single feature window, returning native-resolution probabilities with padding masked. It does not use the speaker cache. `streaming_step()` accepts precomputed feature windows for integrations that own preprocessing.

## Local conversion

The upstream checkpoint is a `.nemo` archive. PyTorch and PyYAML are required only for conversion; inference is MLX-native and does not require NeMo.

```bash
pip install torch pyyaml
python -m mlx_audio.vad.models.nemotron_diarization.convert \
    --hf-path nvidia/Nemotron-3-Diarization \
    --mlx-path ./output/Nemotron-3-Diarization-bf16 \
    --dtype bfloat16 \
    --model-id mlx-community/Nemotron-3-Diarization
```

Use `--nemo-path /path/to/Nemotron-3-Diarization.nemo` for a local archive. `--model-id your-name/Nemotron-3-Diarization-MLX` sets the future HF model ID in the generated README; it does not upload anything. `--revision` pins the source revision when downloading.

The converter checks every weight name and shape, preserves the checkpoint's mel filters, learned silence vector and auxiliary activity head, transposes the upsampling convolution to MLX layout, and writes `model.safetensors`, `config.json`, and `README.md`. All 363 tensors from the release checkpoint are retained. The auxiliary three-class activity head is preserved for checkpoint fidelity; the public API returns speaker probabilities.

BF16 is the default published distribution. The converter defaults to `float32` for numerical validation when `--dtype` is omitted; `float16` is also supported. The source tensors are bfloat16, so float32 storage does not add weight precision. The converted weights retain the upstream OpenMDW 1.1 license.

For BF16, use `--dtype bfloat16`. Add `--quantize` to produce an 8-bit affine
checkpoint with BF16 activations and unquantized layers:

```bash
python -m mlx_audio.vad.models.nemotron_diarization.convert \
    --nemo-path /path/to/Nemotron-3-Diarization.nemo \
    --mlx-path ./output/Nemotron-3-Diarization-8bit \
    --dtype bfloat16 --quantize
```

Quantization uses groups of 64 weights by default (`--group-size` overrides it).
Eligible linear layers are quantized; the upsampling convolution, normalization
layers, and silence embeddings remain in the chosen floating-point dtype.
The mel frontend buffers remain float32. Both variants use the same `load()` API.

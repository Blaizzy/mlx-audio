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

## Speaker-attributed transcription

The [ASR integration example](../../../../examples/nemotron_diarization_asr.py)
pairs this model with Nemotron 3.5 ASR in mlx-audio. Run it from the repository
root:

```bash
python examples/nemotron_diarization_asr.py \
    --audio conversation.wav \
    --language en-US \
    --output transcript.json
```

For the 8-bit models:

```bash
python examples/nemotron_diarization_asr.py \
    --audio conversation.wav \
    --diar-model mlx-community/Nemotron-3-Diarization-8bit \
    --asr-model mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit \
    --output transcript.json
```

The default `--mode masked` runs a separate feature-masked Nemotron ASR stream
for each active speaker, with independent encoder caches and RNN-T decoder state.
Both models consume the same mono 16 kHz waveform. The script prints speaker
turns and optionally saves tokens, turns, and per-speaker transcripts to JSON.
`--diar-preset low` is the default; use `offline` for the longer diarization window.

Temporal masking cannot separate simultaneous voices: overlapping speech can still
cause duplicate or missing words. Speaker labels are session-local arrival-order IDs,
and timestamps reflect token emission times.

Use `--mode timestamps` for a post-hoc word attribution workflow,
including Parakeet models returning `AlignedResult`. That mode flags overlapping
speech as ambiguous and saves diarization segments as well. Attribution near
speaker changes is approximate.

### Python and live PCM

```python
from mlx_audio.stt import load as load_asr
from mlx_audio.vad import load as load_diarization

asr = load_asr("mlx-community/nemotron-3.5-asr-streaming-0.6b")
diar = load_diarization("mlx-community/Nemotron-3-Diarization", strict=True)
diar.set_streaming_config("low")

for speaker, transcript in asr.generate_speakers(
    "meeting.wav", diar, language="en-US"
).items():
    print(speaker, transcript.text)

# Each delta contains a speaker ID, new tokens, and their absolute timestamps.
# Iterable chunks must be 16 kHz mono PCM; exhaustion flushes the stream.
for delta in asr.stream_generate_speakers(microphone_chunks, diar):
    print(delta.speaker, delta.text)
```

For explicit input control, use
`session = asr.create_speaker_streaming_session(diar)`, call `session.feed(pcm)`,
then `session.feed([], final=True)` once. Each call returns token deltas and may
return an empty list while waiting for activity predictions or ASR lookahead.
`session.reset()` starts a new recording. Calls are synchronous and should run on
one inference worker. This API is separate from the server's plain-text realtime
session protocol.

ASR waits for committed diarization predictions, averages native 10 ms activity
into 80 ms masks, then masks log-mel features before subsampling. ASR streams
are processed sequentially with shared weights and independent state. Inference
buffers remain bounded by the configured windows; collecting a complete result
requires memory proportional to transcript length. The diarization preset stays
fixed for the session and determines additional latency.

Cache gating retains activity for two ASR chunks by default, allowing delayed
tokens after a speaker stops. `cache_gating_buffer_size` changes that history;
`cache_gating=False` processes every speaker stream on every chunk. Inactive
streams otherwise keep their caches frozen and resume on the original audio
clock. The models' sample rates and feature hops must match, and diarization must
use native output resolution (`output_subsampling_factor=1`, the default).

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

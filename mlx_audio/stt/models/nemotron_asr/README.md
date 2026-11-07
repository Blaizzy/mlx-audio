# Nemotron streaming ASR — MLX

MLX port of NVIDIA's
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b):
a 600M-parameter **cache-aware streaming FastConformer-RNNT** with **language-ID prompt
conditioning**, covering 40 language-locales with punctuation and capitalization.

The same implementation loads NVIDIA's native Transformers checkpoint
[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
directly, including its 80, 160, 560, and 1120 ms English streaming modes.

The same module runs NVIDIA's
[`nvidia/nemotron-labs-unified-asr-0.6b`](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b)
(25 European languages, offline and streaming in one checkpoint); see
[Nemotron Unified](#nemotron-unified) below.

## Models

| Model | Repo | Precision | Size |
|-------|------|-----------|------|
| Nemotron 3.5 | [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) | bf16 | ~1.2 GB |
| Nemotron 3.5 | [`mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit) | 8-bit | ~720 MB |
| Nemotron Unified | [`mlx-community/nemotron-labs-unified-asr-0.6b-bf16`](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-bf16) | bf16 | ~1.25 GB |
| Nemotron Unified | [`mlx-community/nemotron-labs-unified-asr-0.6b-8bit`](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-8bit) | 8-bit | ~744 MB |
| Nemotron Unified | [`mlx-community/nemotron-labs-unified-asr-0.6b-4bit`](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-4bit) | 4-bit | ~472 MB |

## Usage

```python
from mlx_audio.stt import load

model = load("mlx-community/nemotron-3.5-asr-streaming-0.6b")

# auto language detection (default)
print(model.generate("speech.wav").text)

# force a language via its prompt key (en-US, es-ES, zh-CN, fr-FR, ...)
print(model.generate("speech.wav", language="en-US").text)
```

The English checkpoint uses the shared streaming contract with
`transcription_delay_ms` set to `80`, `160`, `560`, or `1120`:

```python
model = load("nvidia/nemotron-speech-streaming-en-0.6b")
session = model.create_streaming_session(transcription_delay_ms=160)
for pcm in audio_chunks:
    session.feed(pcm)
    print("".join(session.step()), end="", flush=True)
session.close()
while not session.done:
    print("".join(session.step()), end="", flush=True)
```

CLI:

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/nemotron-3.5-asr-streaming-0.6b --audio speech.wav --format txt
```

## Speaker-masked transcription

Use Nemotron 3 Diarization to run an independent ASR stream for each speaker:

```python
from mlx_audio.vad import load as load_diarization

diar = load_diarization("mlx-community/Nemotron-3-Diarization", strict=True)
diar.set_streaming_config("low")
for speaker, transcript in model.generate_speakers("meeting.wav", diar).items():
    print(speaker, transcript.text)

# Also accepts an iterable of 16 kHz mono PCM chunks, with automatic final flush.
for delta in model.stream_generate_speakers("meeting.wav", diar):
    print(delta.speaker, delta.text)
```

`generate_speakers()` returns a dictionary of speaker IDs to `AlignedResult`.
Streaming deltas contain `speaker`, `tokens`, and `text`; timestamps use the
original recording clock. `create_speaker_streaming_session(diar)` exposes
`feed(pcm)` and `feed([], final=True)` for explicit live input. Between turns,
`close_streams()` ends every speaker stream and returns its last tokens while
diarization keeps running, so speakers keep their IDs into the next turn.

This uses existing weights with feature masking, independent encoder/decoder
caches, and activity-based cache gating. Overlapping voices are not separated.
See the [diarization integration documentation](../../../vad/models/nemotron_diarization/README.md#speaker-attributed-transcription)
for options, latency, and the runnable example.

## Language prompt

Pass `language=<key>` where `<key>` is from the model's `prompt_dictionary`
(`en-US`, `es-ES`, `zh-CN`, `fr-FR`, …, or `auto`). In `auto` mode the model
infers the language; Nemotron 3.5 also emits a leading `<lang>` tag, which is
stripped from the returned text (Unified's vocabulary has no tag pieces).

## Look-ahead / latency

`generate(..., att_context_size=[left, right])` selects one of the trained
look-aheads (`[56,3]`, `[56,0]`, `[56,6]`, `[56,13]`). The default `[56,13]`
gives the best offline accuracy; smaller right contexts trade accuracy for lower latency.

## Architecture

```
audio ─► log-mel (128, no norm) ─► FastConformer encoder (24 layers, d=1024)
                                         │  causal dw-striding subsampling ×8
                                         │  causal depthwise conv, layer-norm
                                         │  rel-pos attn, chunked-limited mask
                                         ▼
                  language one-hot (128) ─┴─► concat ─► prompt_kernel (1152→2048→1024)
                                         ▼
                              RNN-T joint + 2-layer LSTM prediction net ─► greedy decode
```

Run offline, the `chunked_limited` attention mask reproduces the training-time look-ahead,
so a single full-utterance pass matches the streaming model's output.

## Nemotron Unified

`nemotron-labs-unified-asr-0.6b` has a non-causal encoder, so it streams
*buffered*: every step re-encodes a window of left context + chunk + right
context and keeps the chunk's frames, as NeMo's
`speech_to_text_streaming_infer_rnnt.py` does. Latency is chunk + right
context; the left context (5.6 s by default) only costs compute.

```python
model = load("mlx-community/nemotron-labs-unified-asr-0.6b-8bit")

# Offline: one full-context pass (up to 60 s; longer audio streams 10/10/5 s windows).
print(model.generate("speech.wav", language="fr").text)

# Live: latency from 160 to 2080 ms (320 by default).
session = model.create_streaming_session(language="fr", transcription_delay_ms=320)
```

`transcription_delay_ms` is rounded to whole 80 ms frames and clamped to
160-2080 ms. NVIDIA's recommended contexts give the chunk (160/240/320 ms: 80 ms
chunk; 560 ms: 160 ms; 1120 ms: 560 ms; 2080 ms: 1040 ms); a latency between
two rows takes the lower row's chunk and the rest as right context.

Pass the spoken language (`en`, `fr`, `de`, … or `auto`). As in NeMo, a bare
code matches a unique locale key and an unknown or ambiguous language raises.
At low latency, `auto` can emit a few first tokens in the wrong language. `stream_generate`
and `generate(att_context_size=[left, chunk, right])` take explicit contexts in
80 ms encoder frames.

## Converting the original weights (optional)

Most users should just load the pre-converted repos above. If you want to convert the
original `.nemo` yourself (needs `torch` + `pyyaml`):

```bash
python -m mlx_audio.stt.models.nemotron_asr.convert \
    --nemo-path nvidia/nemotron-3.5-asr-streaming-0.6b \
    --mlx-path  ./nemotron-3.5-asr-streaming-0.6b-mlx
```

For Unified, pass `--nemo-path nvidia/nemotron-labs-unified-asr-0.6b`.
`--nemo-path` takes a local `.nemo` file **or** a HuggingFace repo id. Add `--quantize`
(`--q-bits 8`) for a quantized build, or `--dtype float16`.

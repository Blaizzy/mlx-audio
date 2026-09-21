# Qwen3-ASR live streaming

Qwen3-ASR and compatible checkpoints, including
[Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2), share
one live-input adapter. Loading uses the existing `qwen3_asr` architecture;
there is no checkpoint-name detection or separate R2T2 model type.

R2T2 is trained for stable-prefix streaming. Ordinary Qwen3-ASR can use the
same session, but may commit recognition mistakes on incomplete audio. Emitted
text is immutable by the adapter's policy, not a guarantee of recognition
accuracy. Tune `unfixed_token_num`, chunk size, and lookahead for your checkpoint.

## File transcription

```python
from mlx_audio.stt import load

model = load("netease-youdao/Confucius4-R2T2", strict=True)
result = model.generate("speech.wav", language="Russian")
print(result.text)
```

The inherited `generate(..., stream=True)` yields text tokens for an already
available recording. For a microphone or incoming PCM chunks, use a session.

## Live audio

```python
import numpy as np
from mlx_audio.stt import load

model = load("netease-youdao/Confucius4-R2T2")
session = model.create_streaming_session(language="Russian")

# audio_chunks yields mono float32 PCM at 16 kHz, in any chunk size.
for pcm in audio_chunks:
    session.feed(np.asarray(pcm, dtype=np.float32))
    for delta in session.step(max_decode_tokens=4):
        print(delta, end="", flush=True)

session.close()
while not session.done:
    for delta in session.step(max_decode_tokens=4):
        print(delta, end="", flush=True)
```

A live application should schedule `step()` continuously on one model executor;
`feed()` and `close()` may run on a producer thread. An empty delta list means
more audio or decoding may be needed. Keep stepping after `close()` until `done`,
including when input ends exactly on a chunk boundary. `cancel()` discards a
session and must run on the decoder thread.

Only committed text is returned. A small trailing token window is withheld and
regenerated with the next audio update. In an STT–LLM pipeline, these deltas can
feed LLM prefill, while still accounting for the LLM tokenizer's text boundary.
No timestamps are returned by the live session.

| Option | Default | Meaning |
|--------|---------|---------|
| `language` | `None` | Auto-detect, or a supported name such as `Russian` |
| `context` | `""` | Context/hotword hint in the system prompt |
| `chunk_size_sec` | `0.16` | Audio update size, from 0.08 to 2 seconds |
| `lookahead_sec` | `0.16` | Extra audio for the first update |
| `unfixed_token_num` | `1` | Trailing ASR tokens withheld until a later update |
| `temperature` | `0.0` | Live decoding is greedy |
| `max_audio_seconds` | `30.0` | Maximum total audio per session; overflow raises `BufferError` |

`step(max_decode_tokens=4)` bounds token decoding work for that call. It does not
bound encoder or prefill time. The adapter re-encodes the accumulated utterance with
each audio update, so work grows with utterance length. Use VAD/turn boundaries
to close sessions; this adapter does not implement upstream's rolling-window
variant. Finalization re-decodes with the committed prefix and a token cap of at least 512
to flush withheld text and the remaining audio. Streaming and file transcription
may differ, including punctuation. Upstream's published latency figures are not
MLX performance measurements.

## Loading and conversion

Both official Qwen3-ASR and R2T2 checkpoints use `model_type: qwen3_asr`.
The same loader, converter, and configuration work for both. Renaming the model
directory does not affect the implementation selected from its configuration.

```python
model = load("Qwen/Qwen3-ASR-0.6B")
session = model.create_streaming_session(
    language="Russian", unfixed_token_num=5, chunk_size_sec=0.5
)
```

The values above illustrate explicit tuning, not a validated quality preset.
The per-update token allowance grows with `unfixed_token_num` so a larger
withheld window can still make progress before the stream ends. Existing `generate()` and `generate(..., stream=True)` retain their
file-transcription behavior. Forced-alignment models do not expose a live session.

The realtime server discovers the session factory on all Qwen3-ASR checkpoints;
it does not distinguish checkpoints by name. Callers opting into realtime mode
therefore select the same commitment policy regardless of checkpoint.

```bash
python -m mlx_audio.convert \
    --hf-path netease-youdao/Confucius4-R2T2 \
    --mlx-path ./r2t2-4bit \
    --quantize --q-bits 4
```

Conversion preserves the `qwen3_asr` family. The source checkpoint's license
continues to apply; see its model card.

## Sources

- [Upstream streaming implementation](https://github.com/netease-youdao/Confucius4-R2T2/blob/80c22e6140bcb9166fb9906798894fc8b18c8309/r2t2/r2t2_asr.py)
- [Upstream example and adaptive decoding limits](https://github.com/netease-youdao/Confucius4-R2T2/blob/80c22e6140bcb9166fb9906798894fc8b18c8309/example.py)
- Code: Apache-2.0. Weights: NetEase Model Use License Agreement; see the model card.

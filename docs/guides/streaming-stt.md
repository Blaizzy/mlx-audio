# Adding a live-input STT model

`mlx_audio.stt.streaming.StreamingSession` is the shared structural protocol
used by the realtime server. Nemotron and Voxtral Realtime both implement it.
No base class, registration step, or model-name branch in the server is needed.

Implement `create_streaming_session(...) -> StreamingSession` on the model.
Keep model-specific construction and options there; the session presents:

- `input_sample_rate: int`: native input rate.
- `feed(samples: np.ndarray) -> None`: queue mono float32 PCM; no inference.
- `step(*, max_decode_tokens: int = 4) -> list[str]`: advance on one model
  executor and return append-only text deltas, possibly an empty list.
- `close() -> None`: signal end of input, without discarding the pending tail.
- `done: bool`: true once decoding has finished.

`feed` and `close` may run on a producer thread while one consumer calls `step`.
Do not modify submitted arrays while queued, or feed after close/completion.
Keep calling `step` after close until done. An empty result is not completion.
Models may also finish early (for example EOS or a model-specific token cap).
The decode budget is not a time limit: model-specific encoder work can still
make a step expensive. Document input limits and errors in the model guide.
Reset/cancel are optional implementation extensions, not part of this contract.

```python
from mlx_audio.stt.streaming import StreamingSession

session: StreamingSession = model.create_streaming_session(temperature=0.0)
for samples in audio_chunks:
    if session.done:
        break
    session.feed(samples)
    for delta in session.step():
        emit(delta)
session.close()
while not session.done:
    for delta in session.step():
        emit(delta)
```

This example executes sequentially; a live server schedules the producer and
consumer independently. Factory options remain model-specific; the existing
server forwards transcription delay only when the factory declares it.

Add a small model fixture to `mlx_audio/stt/tests/test_streaming_session.py`
to exercise the same server/session contract, plus model-specific decoder
parity tests. Runtime `isinstance` checks only member presence; behavioral tests
are necessary to prove the contract. The shared test uses tiny random weights
and deterministic token selection, not production checkpoints or downloads.

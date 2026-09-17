# Nemotron live input

Nemotron implements the shared [`StreamingSession`](streaming-stt.md) protocol,
also used by Voxtral Realtime. Sessions use greedy
decoding (`temperature=0`), mono float32 PCM at `session.input_sample_rate`, and
independent frontend, encoder and RNNT state.

```python
session = model.create_streaming_session()
session.feed(samples)
print("".join(session.step(max_decode_tokens=8)), end="")
session.close()
while not session.done:
    print("".join(session.step(max_decode_tokens=8)), end="")
```

Call `step` regularly from one decoder thread. Its budget counts joint evaluations,
including blanks and special tokens. Each step ingests at most one native audio
chunk and drains existing encoded frames before ingesting more. Packet boundaries
do not reset frontend or decoder state. Text is emitted as deltas, not cumulative
transcripts. No transcript history is retained by the session.

`feed` copies input and rejects non-finite or non-mono data. Pending audio is
limited to 30 seconds; exceeding that limit raises `BufferError` without dropping
previously queued audio. This is a pending-input limit, not a session duration
limit. Producers must pace input and consumers must keep stepping. `close` is
idempotent and signals input completion, not immediate decoder completion.
`cancel` discards pending work without emitting a successful final; `reset`
starts independent state. Both must run with no concurrent decoder step.

The generic server selects models through `/v1/realtime?model=<model-id>` and
accepts session updates and PCM16 audio append/commit events. Use model-rate PCM
to avoid conflating session behavior with packet-wise server resampling. This
change does not add a Nativ route, authentication policy or new server protocol.

Validation covers tiny-model frontend/encoder behavior, real tiny-decoder parity,
and the actual ASGI WebSocket handler (partial before commit, completion and a
second turn). A local multilingual checkpoint check used five paced runs each of
Portuguese and English synthesized speech, with 20 ms / 16 kHz chunks. Text matched
the existing streaming decoder after trimming outer whitespace. Both paths made
the same Portuguese recognition error; this is parity evidence, not an accuracy
benchmark.

For those ten local session runs, first-partial p50/p95 was 1.912/3.433 seconds and
close-to-final p50/p95 was 0.267/0.676 seconds (maximum 0.921 seconds). Quantiles use
linear interpolation. These small-sample measurements are not WebSocket latency
or Nativ/OpenClaw end-to-end proof. Long-session memory-residency measurements,
broader speech samples and production integration remain outstanding.

# StepAudio2 codec streaming

The StepAudio2 streaming API reuses the existing FP32 flow and HiFT networks.
It returns resident MLX waveform chunks and explicit cache state. The existing
offline `StepAudio2Token2Wav.decode()` remains unchanged. No language model,
speech-token generator, audio device or application framework is required.

## Prepare and decode

```python
from mlx_audio.codec.models.stepaudio2 import StepAudio2Token2Wav
from mlx_audio.codec.models.stepaudio2.streaming import (
    decode_chunk,
    hamming_window,
    prepare_stream,
    reset_stream,
)

# Load/prepare explicitly. These operations may read model/reference files;
# the streaming functions themselves do not download or extract a prompt.
model = StepAudio2Token2Wav.from_pretrained(local_model_path)
prompt = model.prepare_prompt(local_reference_wav)
base = prepare_stream(model.flow, prompt)
state = reset_stream(base)
window = hamming_window()

first_window = [4218] * 3 + first_25_speech_codes
first = decode_chunk(
    model.flow, model.hift, first_window, state, prompt["embedding"], window
)
state = first.state
# first.waveform is FP32 [1,T], still resident in MLX.

continuation = first_window[-3:] + next_25_speech_codes
next_chunk = decode_chunk(
    model.flow, model.hift, continuation, state, prompt["embedding"], window
)
state = next_chunk.state

# Include the three carried lookahead codes in the entire remaining buffer.
final = decode_chunk(
    model.flow, model.hift, remaining_buffer, state, prompt["embedding"], window,
    last_chunk=True,
)
```

`prepare_stream` accepts either the existing prepared prompt dictionary or a
five-array reference tuple: codes `[1,N]`, code lengths `[1]`, speaker embedding
`[1,192]`, aligned mel `[1,2*N,80]`, mel lengths `[1]`. Codes/lengths must be
int32; speaker/mel must be FP32. For an offline prompt dictionary, the aligned
`prompt_feat` shape supplies its streaming length. Earlier prompt preparation
can retain a pre-alignment `prompt_feat_len`, which is not a streaming cache
length; the adapter does not mutate that original dictionary.

The supplied networks must be in evaluation mode with FP32 parameters. The
streaming source itself loads its backend lazily; ordinary MLX Audio codec
namespace imports still import the surrounding device modules. Optional `mx`/`nn`
arguments support injected runtime facades for CPU contract tests; applications
normally omit them.

## Buffered code contract

| Call | Input codes | Codes consumed |
| --- | --- | --- |
| First normal chunk | three silence codes (`4218`) plus 25 generated codes | 25 |
| Continuation | prior last three plus 25 new codes | 25 |
| Final chunk | all remaining buffered codes | all |

The final count is the **remaining buffer length**, not newly generated code
count. Three, 17 and 25 remaining codes are separate final cases. This API does
not implement an upstream speech generator's buffering/flush policy; a native
turn ending can require a non-final flush followed by a final three-code call.

The first non-final waveform includes 160ms leading silence and withholds a
160ms tail. Continuations use the native Hamming overlap and eight mel frames of
HiFT history. A final call emits the retained tail. Output is mono at 24kHz.
Optional `pcm16_bytes()` applies native amplitude clipping and little-endian
PCM16 packaging; it rejects nonfinite values rather than repairing them.

## State, numerics and ownership

`StreamState` contains the four flow caches, three HiFT caches, prompt frame
count, consumed-code count and final status. Treat its mappings/arrays as
read-only. `decode_chunk` evaluates and validates output/state before returning
a new state, so a failed call cannot install partial caches in the input state.
A finalized state rejects further chunks. Reset requires the original unconsumed
prompt base and creates fresh containers with evaluated leaves.

Sequence state is independent, but the network's ESPnet position encoder keeps
a deterministic positional-table memo. Serialize network use; interleaved
independent streams on one owner do not establish concurrent thread safety.
Reset does not reseed MLX or replace persistent flow noise. HiFT retains its
native source-noise generation. For reproducible comparisons, snapshot/restore
the same RNG state externally and use the same evaluated flow noise.

The algorithm preserves ten cosine-scheduled Euler steps, signed native CFG,
current-before-history DiT cache order, and prompt plus last-100-frame cropping.
Default attention is eager. Explicit `attention_mode="sdpa"` uses the MLX GPU
primitive only for FP32 eight-head, 64-wide DiT geometry, with no fallback and
unchanged flow step count. Fused reduction order can differ numerically.

## Tests and evidence

```sh
python -m pytest -q tests/test_stepaudio2_streaming_contract.py
```

The contract suite loads only the streaming source and uses CPU NumPy facades.
It covers lookahead and split/whole synthetic encoder behavior, continuation
and final buffer lengths, cache order/cropping, independent/reset streams,
failure transactions, persistent flow noise and ten-step CFG. It requires no
model download or audio hardware and does not establish trained-network or
voice parity. Real numerical and performance claims require a saved comparison
against the native streaming implementation with matched reference arrays,
codes, persistent noise, weights and RNG; complete native endings must be
measured separately from a single final call.

The streaming algorithm derives from StepAudio2/CosyVoice2 and ESPnet native
cache implementations. Original author/license notices remain in the source;
this derived module retains Apache-2.0, with its license copy packaged beside it.
The weighted networks remain the existing MLX Audio StepAudio2 implementation.
Contributors should retain these component notices instead of describing the
derived streaming implementation as MIT-only.

The caller must bound code-window lengths and total work before passing input to
the networks. Native windows normally contain 28 codes and small final tails;
this low-level API does not choose an application's admission/queue budget.

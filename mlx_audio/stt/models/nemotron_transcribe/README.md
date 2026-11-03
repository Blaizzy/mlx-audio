# Nemotron 3.5 Transcribe

Native MLX inference for NVIDIA's **NVIDIA-Nemotron-3.5-Transcribe-30B-A4B**
consolidated SpeechLM checkpoint. It combines a feature-stacking speech
Transformer, a Sortformer speaker encoder, and the vendored Nemotron-H
Mamba/attention/MoE language model. Runtime inference requires neither NeMo,
PyTorch, nor `mlx-lm`.

## Supported checkpoints

### NVIDIA checkpoints

The official
[NVIDIA Nemotron 3.5 Transcribe collection](https://huggingface.co/collections/nvidia/nemotron-35-transcribe)
contains the following models:

| Format | Model | MLX Audio support |
| --- | --- | --- |
| BF16 | [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) | Supported directly, including word timestamps and MTP speculative decoding. |
| FP8 | [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8) | Supported directly; FP8 weights expand to BF16 during loading. |
| NVFP4 | [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4) | Supported directly; NVFP4 weights stay packed with their global scales. |
| DFlash2 draft | [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2) | Separate draft model; DFlash2 decoding is not currently implemented. |

The BF16, FP8, and NVFP4 checkpoints all support word timestamps and MTP
speculative decoding. ModelOpt tensor layouts and scales are adapted automatically,
without installing `mlx-vlm` or PyTorch. FP8 modules, including the audio encoder
and MTP module in the NVFP4 checkpoint, run in BF16. Activations and KV caches
remain floating point. `speculative=True` uses the main checkpoint's MTP module;
DFlash2 decoding requires a separate implementation.

### MLX exports

Ready-to-load MLX exports are available in the
[nativ-community Nemotron 3.5 Transcribe collection](https://huggingface.co/collections/nativ-community/nemotron-35-transcribe):

| Format | Model |
| --- | --- |
| BF16 | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-bf16](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-bf16) |
| 4-bit | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit) |
| 5-bit | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-5bit](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-5bit) |
| 6-bit | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-6bit](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-6bit) |
| 8-bit | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-8bit](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-8bit) |
| MXFP4 | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-mxfp4](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-mxfp4) |
| NVFP4 | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-nvfp4](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-nvfp4) |
| MXFP8 | [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-mxfp8](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-mxfp8) |

These early-access exports support the transcription, translation, speaker,
and audio-prompt tasks below. Their current weights omit both MTP and CTC
heads, so use one of the NVIDIA BF16, FP8, or NVFP4 checkpoints above for
speculative decoding and word timestamps.

## Load and transcribe

Pass a Hugging Face model ID to download and load the checkpoint automatically:

```python
from mlx_audio.stt import load

model = load("nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit", strict=True)
result = model.generate("audio.wav", max_tokens=1024)
print(result.text)
```

```bash
python -m mlx_audio.stt.generate \
  --model nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit \
  --audio audio.wav --output-path transcript --format txt
```

File inputs are resampled to 16 kHz mono. Array inputs must already be mono
16 kHz PCM. ASR preserves the spoken language by default; thinking is disabled.
Long recordings use NVIDIA's context-extended encoder windows and persistent
speaker cache. The full audio is encoded before text generation starts.

For runnable examples of all six tasks, synthetic audio inputs, recorded
outputs, and an optional separate text-to-speech reply, see the
[six-task demo](../../../../examples/nemotron_transcribe/README.md).

## Other tasks

```python
# Speech translation: language is the target language for this task.
translated = model.generate("audio.wav", task="translate", language="German")

# Speaker-attributed transcription, up to eight speaker IDs.
turns = model.generate("conversation.wav", task="saa")
print(turns.text)      # Native <spk:0>, <spk:1>, ... tags are preserved.
print(turns.segments)  # Dictionaries with speaker and text, without timestamps.

# Preserve non-speech event and speaking-style tags.
rich = model.generate("audio.wav", task="rich")

# Audio questions or contextual transcription use the checkpoint's text prompt.
answer = model.generate("audio.wav", prompt="Summarize the conversation.")
transcript = model.generate("audio.wav", hotwords=["Neywa Labs", "Nemotron"])

# Stream decoded text after encoding the recording.
for part in model.generate("audio.wav", stream=True):
    print(part.text, end="", flush=True)
```

A custom prompt may include exactly one `<|audio|>` placeholder. If omitted, it
is appended. `system_prompt` and `enable_thinking=True` are available for audio
reasoning. `task="qa"` supplies only the audio placeholder; provide an appropriate
system prompt when the spoken audio itself is an instruction.

## Word timestamps and speculative decoding

The NVIDIA BF16, FP8, and NVFP4 checkpoints load directly, with no conversion
step. For example, use NVFP4 to reduce memory use while retaining both features:

```python
model = load("nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4", strict=True)
result = model.generate(
    "conversation.wav", task="saa", word_timestamps=True,
    speculative=True, num_draft_tokens=4,
)
print(result.text)
for word in result.segments:
    print(word["text"], word["start"], word["end"], word["speaker"])
print(result.speculative_stats)
```

Word timestamps use the checkpoint's CTC head and separate alignment tokenizer.
Segments contain one word each, with `start` and `end` in seconds and the
transcript's speaker ID. Chinese and Japanese use character units; Thai word
segmentation is currently unsupported. Alignment requires a transcription task
(`asr`, `transcribe`, `saa`, or `rich`) and a transcript that fits the recording.
Speaker IDs are local to each recording. With `stream=True`, text arrives during
decoding and the final item contains the complete timestamped `segments`.

MTP speculative decoding reuses the trained attention/MoE prediction module and
verifies every proposed token with the language model. It currently supports
greedy decoding (`temperature=0`) and 1–8 draft tokens per verification step.
`speculative_stats` reports proposed tokens, accepted tokens, and verification
steps. Speed depends on draft acceptance and quantization; batched floating-point
evaluation can differ from ordinary greedy decoding near tied logits.

Both features are optional and can be used independently. Earlier converted
checkpoints that omitted MTP or CTC weights still support ordinary transcription;
reconvert from the original checkpoint to enable the missing features. Keep the
`alignment_tokenizer/` directory when copying a converted checkpoint.

Timestamped segments support SRT/VTT export:

```bash
python -m mlx_audio.stt.generate \
  --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 --audio audio.wav \
  --output-path transcript --format srt \
  --gen-kwargs '{"word_timestamps": true, "speculative": true, "num_draft_tokens": 4}'
```

The original NVIDIA BF16 weights require approximately 67.5 GB of storage and
about 68.5 GB of MLX memory in the measured short-audio runs. The NVIDIA FP8
download is approximately 34.7 GB, but expands to BF16 memory use on loading.
The NVIDIA NVFP4 download is approximately 21.6 GB, with about 24.4 GB of loaded
parameters after expanding its FP8 modules; runtime allocations add to this.
The community collection's 4-bit export uses approximately 19.3 GB of weights.
Quantized outputs can differ from BF16.

## Reference and licensing

The port follows [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech),
including its generation-time parallel expert encoder, repeated MTP module,
and `nemotron3p5` prompt format. Timestamp alignment also follows the
[Transformers implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/nemotron3_5_transcribe).
These sources are Apache-2.0. The released checkpoint uses
[OpenMDW-1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/blob/main/LICENSE).
The earlier NGC early-access checkpoint remains supported and has its own
[NVIDIA Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/);
obtain early-access weights through your authorized NGC access.

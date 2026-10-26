# Nemotron 3.5 Transcribe

Native MLX inference for NVIDIA's **Nemotron-3.5-Transcribe-30B-A4B-EA**
consolidated SpeechLM checkpoint. It combines a feature-stacking speech
Transformer, a Sortformer speaker encoder, and the vendored Nemotron-H
Mamba/attention/MoE language model. Runtime inference requires neither NeMo,
PyTorch, nor `mlx-lm`.

## Load and transcribe

Use the local folder downloaded from NVIDIA NGC, containing `config.json`,
`model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`:

```python
from mlx_audio.stt import load

model = load("/path/to/Nemotron-3.5-Transcribe-30B-A4B-EA-0.1.0", strict=True)
result = model.generate("audio.wav", max_tokens=1024)
print(result.text)
```

```bash
python -m mlx_audio.stt.generate \
  --model /path/to/Nemotron-3.5-Transcribe-30B-A4B-EA-0.1.0 \
  --audio audio.wav --output-path transcript --format txt
```

File inputs are resampled to 16 kHz mono. Array inputs must already be mono
16 kHz PCM. ASR preserves the spoken language by default; thinking is disabled.
Long recordings use NVIDIA's context-extended encoder windows and persistent
speaker cache. The full audio is encoded before text generation starts.

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

This port performs ordinary autoregressive decoding; the checkpoint's optional
MTP training/speculation heads are not loaded. It does not produce word
timestamps, so use TXT or JSON output rather than SRT/VTT. Speaker IDs are local
to each recording.

## Convert and quantize

The original BF16 checkpoint can be loaded directly. To save an MLX checkpoint
with a quantized language model:

```bash
python -m mlx_audio.convert \
  --hf-path /path/to/Nemotron-3.5-Transcribe-30B-A4B-EA-0.1.0 \
  --mlx-path /path/to/nemotron-transcribe-4bit \
  --model-domain stt --quantize --q-bits 4 --q-group-size 64
```

The perception encoders and projection remain unquantized. Mamba dynamics and
router correction parameters retain their original float32 precision during
dtype conversion. Quantization changes model outputs; evaluate it on your audio.

## Reference and licensing

The port follows [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech/tree/fc0cbdb795236b88bf6a65ca11e8d763d7598cbe),
including its generation-time parallel expert encoder and `nemotron3p5` prompt
format. NeMo source is Apache-2.0. The early-access checkpoint has its own
[NVIDIA Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/);
obtain weights through your authorized NGC access.

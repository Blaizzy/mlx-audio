# CosyVoice3

A pure MLX implementation of [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) by FunAudioLLM (Alibaba), a state-of-the-art zero-shot voice cloning and text-to-speech model.

## Architecture

CosyVoice3 uses a multi-stage pipeline:

1. **Frontend** - Text tokenization (Qwen2 tokenizer) + speaker embedding extraction (CAMPPlus) + speech tokenization (S3TokenizerV2)
2. **LLM** - Qwen2-based language model that generates speech tokens from text, conditioned on speaker embeddings and reference speech tokens
3. **Flow** - Causal Conditional Flow Matching (CFM) with a DiT (Diffusion Transformer) that converts speech tokens to mel-spectrograms
4. **HiFT** - Causal HiFi-GAN vocoder that synthesizes audio waveforms from mel-spectrograms

## Supported Languages

- Mandarin Chinese
- English
- Cantonese
- Sichuan dialect
- Shanghainese (Wu Chinese)
- Other Chinese dialects

The model handles mixed-language text (e.g., Chinese with English words) natively.

## Usage

### Pre-converted Models (Recommended)

Use pre-converted safetensors models from mlx-community on HuggingFace:

```python
from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import Model

model = Model.from_pretrained("mlx-community/Fun-CosyVoice3-0.5B-2512")
```

### Zero-Shot Voice Cloning

```python
from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import Model

model = Model.from_pretrained("mlx-community/Fun-CosyVoice3-0.5B-2512")

for result in model.inference_zero_shot(
    text="Hello, this is a test of voice cloning.",
    ref_text="Transcript of the reference audio.",
    ref_audio="path/to/reference.wav",
    n_timesteps=10,
    temperature=1.0,
    top_k=25,
):
    audio = result.audio
    sample_rate = result.sample_rate
```

### Instruct Mode (Style/Language Control)

Use `instruct_text` to control speaking style or language while cloning a voice:

```python
# Speed control (fast)
for result in model.inference_instruct(
    text="Receiving a birthday gift sent by a friend filled my heart with happiness.",
    ref_audio="path/to/reference.wav",
    instruct_text="Please speak as fast as possible",
    n_timesteps=10,
):
    audio = result.audio

# Speed control (slow)
for result in model.inference_instruct(
    text="Hello, this is a test of style control.",
    ref_audio="path/to/reference.wav",
    instruct_text="Please speak as slowly as possible",
    n_timesteps=10,
):
    audio = result.audio

# Language control
for result in model.inference_instruct(
    text="Not many, usually only during National Day, those holidays maybe.",
    ref_audio="path/to/reference.wav",
    instruct_text="Please express in Cantonese",
    n_timesteps=10,
):
    audio = result.audio
```

### Cross-Lingual / Fine-Grained Control

Use control tokens like `[breath]` for natural speech patterns:

```python
# Breath control
for result in model.inference_cross_lingual(
    text="[breath]Because their generation[breath]is more accustomed to living in the countryside,[breath]the neighbors are all very friendly.",
    ref_audio="path/to/reference.wav",
    n_timesteps=10,
):
    audio = result.audio

# Cough
for result in model.inference_cross_lingual(
    text="[cough]Sorry about that,[cough]I was saying we should meet tomorrow.",
    ref_audio="path/to/reference.wav",
    n_timesteps=10,
):
    audio = result.audio

# Laughter
for result in model.inference_cross_lingual(
    text="[laughter]That is so funny,[laughter]I cannot believe you said that.",
    ref_audio="path/to/reference.wav",
    n_timesteps=10,
):
    audio = result.audio
```

Supported control tokens: `[breath]`, `[laughter]`, `[cough]`, `[noise]`

### Voice Conversion

Convert the content of `source_audio` to the voice of `ref_audio`. No text input needed — skips the LLM entirely:

```python
for result in model.inference_vc(
    source_audio="path/to/source.wav",
    ref_audio="path/to/target_voice.wav",
    n_timesteps=10,
):
    audio = result.audio
```

### Summary

| Mode | Method | Inputs | Use Case |
|------|--------|--------|----------|
| Zero-Shot | `inference_zero_shot` | `text` + `ref_text` + `ref_audio` | Voice cloning TTS |
| Instruct | `inference_instruct` | `text` + `ref_audio` + `instruct_text` | Style/language control |
| Cross-Lingual | `inference_cross_lingual` | `text` (with control tokens) + `ref_audio` | Fine-grained speech control |
| Voice Conversion | `inference_vc` | `source_audio` + `ref_audio` | Convert audio to target voice |

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `text` | Text to synthesize | (required for TTS modes) |
| `ref_text` | Transcript of the reference audio | (zero-shot only) |
| `ref_audio` | Path to reference audio file (WAV) | (required) |
| `instruct_text` | Style/language instruction | (instruct only) |
| `source_audio` | Path to source audio for conversion | (VC only) |
| `n_timesteps` | Number of flow ODE steps (higher = better quality, slower) | 10 |
| `temperature` | LLM sampling temperature | 1.0 |
| `top_k` | Top-k sampling for LLM token generation | 25 |

### Tips

- Reference audio should be 3-10 seconds of clean speech
- `ref_text` should be an accurate transcript of the reference audio (zero-shot only)
- The model auto-prepends `"You are a helpful assistant.<|endofprompt|>"` to `ref_text` if not already present
- For instruct mode, no transcript is needed — the model handles alignment from audio alone
- For voice conversion, the output duration matches the source audio length

## Converting from PyTorch/ONNX

If you have the original PyTorch model from [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512), you need to convert the weights to safetensors before use.

The conversion script handles all weight formats in one pass:
- `flow.pt`, `hift.pt`, `llm.pt`, `campplus.onnx` -> `model.safetensors`
- `speech_tokenizer_v3.onnx` -> `speech_tokenizer_v3.safetensors`

### Requirements

```bash
pip install torch onnx
```

### Run Conversion

```bash
python -m mlx_audio.tts.models.cosyvoice3.convert --model-dir /path/to/model
```

You can also specify a separate output directory:

```bash
python -m mlx_audio.tts.models.cosyvoice3.convert \
    --model-dir /path/to/original \
    --output-dir /path/to/converted
```

After conversion, your model directory should contain:

```
model-dir/
├── CosyVoice-BlankEN/              # Qwen2 tokenizer
├── model.safetensors               # Flow + HiFT + LLM + CAMPPlus weights
└── speech_tokenizer_v3.safetensors # Speech tokenizer
```

## Model Files

| File | Description | Size |
|------|-------------|------|
| `model.safetensors` | Combined flow, HiFT, LLM, and CAMPPlus weights | ~3.7 GB |
| `speech_tokenizer_v3.safetensors` | S3 speech tokenizer (FSQ) | ~923 MB |
| `CosyVoice-BlankEN/` | Qwen2 tokenizer files | ~5 MB |

## Module Structure

| File | Component |
|------|-----------|
| `cosyvoice3.py` | Main model class with `from_pretrained` and `inference_zero_shot` |
| `llm.py` | Qwen2-based LLM for speech token generation |
| `flow.py` | Causal Conditional Flow Matching with DiT |
| `dit.py` | Diffusion Transformer (estimator for flow) |
| `hift.py` | Causal HiFi-GAN vocoder |
| `frontend.py` | Text tokenization, speaker embedding, speech tokenization |
| `campplus.py` | CAMPPlus speaker embedding extractor |
| `convert.py` | Weight conversion script (PyTorch + ONNX -> safetensors) |

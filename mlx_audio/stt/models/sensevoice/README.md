# SenseVoice

MLX implementation of [SenseVoice Small](https://github.com/FunAudioLLM/SenseVoice) from Alibaba DAMO Academy. Non-autoregressive CTC-based model with speech recognition, language identification, emotion recognition, and audio event detection.

## Features

- 50+ languages (strong on zh, en, ja, ko, yue)
- emotion detection (happy, sad, angry, neutral, etc.)
- audio event detection (speech, bgm, laughter, applause)
- non-autoregressive inference (very fast, ~70ms for 10s audio)
- ~234M parameters

## Usage

```python
from mlx_audio.stt import load

# downloads automatically from huggingface
model = load("mlx-community/SenseVoiceSmall")

result = model.generate("audio.wav", language="auto")
print(result.text)
print(result.language)

# rich info (emotion, event) is in the first segment
seg = result.segments[0]
print(seg["emotion"])  # e.g. "neutral"
print(seg["event"])    # e.g. "Speech"
```

### Language options

`"auto"`, `"zh"`, `"en"`, `"ja"`, `"ko"`, `"yue"`, `"nospeech"`

### Inverse text normalization

```python
result = model.generate("audio.wav", language="auto", use_itn=True)
```

# Whisper

MLX implementation of [Whisper](https://github.com/openai/whisper) by OpenAI. Also natively supports distilled variants like [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3).

## Usage

```python
from mlx_audio.stt import load

# Standard Whisper
model = load("mlx-community/whisper-large-v3-turbo-asr-fp16")

# Distil-Whisper
# model = load("distil-whisper/distil-large-v3")

result = model.generate("audio.wav")
print(result.text)
```

### Timestamps

Segment-level timestamps are enabled by default:

```python
result = model.generate("audio.wav")
for segment in result.segments:
    print(f"[{segment['start']:.2f} -> {segment['end']:.2f}] {segment['text']}")
```

Word-level timestamps use cross-attention alignment heads via DTW:

```python
result = model.generate("audio.wav", word_timestamps=True)
for segment in result.segments:
    for word in segment["words"]:
        print(f"[{word['start']:.2f} -> {word['end']:.2f}] {word['word']} (p={word['probability']:.3f})")
```

To disable timestamps entirely:

```python
result = model.generate("audio.wav", return_timestamps=False)
```

## CLI

Basic transcription with segment timestamps:

```bash
mlx_audio.stt.generate \
  --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
  --audio audio.wav \
  --verbose
```

Word-level timestamps:

```bash
mlx_audio.stt.generate \
  --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
  --audio audio.wav \
  --verbose \
  --gen-kwargs '{"word_timestamps": true}'
```

Export as JSON with word-level timestamps:

```bash
mlx_audio.stt.generate \
  --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
  --audio audio.wav \
  --format json \
  --output-path output.json \
  --gen-kwargs '{"word_timestamps": true}'
```

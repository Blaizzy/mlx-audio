# Chatterbox TTS Quick Reference

Multilingual voice cloning TTS using mlx-audio on Apple Silicon.

> **Note:** mlx-audio is already installed system-wide. Commands work from any folder.


---

## Web UI Usage

The MLX Audio Studio provides a user-friendly interface for Text-to-Speech generation, including full support for Chatterbox voice cloning.

### 1. Starting the Server

Run the following command to start both the API server and the Web UI:

```bash
python -m mlx_audio.server --start-ui
```

- **UI URL:** `http://localhost:3000`
- **API URL:** `http://localhost:8000`

### 2. Using Chatterbox TTS

1. **Select Model:** Choose `Chatterbox` from the **Model** dropdown menu.
2. **Reference Audio:** Click "Choose audio file..." to upload a 5-10 second clip of the voice you want to clone.
   - *Note:* This is required for Chatterbox to work.
3. **Select Language:** Choose the target language from the **Language** dropdown (e.g., English, Chinese, Japanese).
   - *Important:* Ensure this matches the text you are generating to avoid pronunciation issues.
4. **Adjust Settings:**
   - **Emotion Exaggeration:** Controls expressiveness (0.0 - 1.0). Default is 0.5. Higher values make the voice more emotional but can be unstable.
   - **Guidance Weight:** Controls how closely the model follows the text/audio conditioning. Default is 0.5.
5. **Generate:** Enter your text and click **Generate**.
6. **Download:** Click the download icon to save the generated audio as an MP3 file.

### 3. Troubleshooting

- **"Failed to fetch"**: If generation takes too long (Chatterbox is slow), the UI might timeout. We've increased the timeout to 5 minutes for Chatterbox.
- **Pronunciation issues**: If words sound wrong (e.g., "But I" -> "boot yi"), check that the **Language** dropdown is set correctly (e.g., to "English").

---

## CLI Usage

### Basic Chinese TTS

```bash
mlx_audio.tts.generate \
  --model mlx-community/chatterbox-fp16 \
  --text "你好，今天天气真不错！" \
  --lang_code zh \
  --ref_audio /path/to/reference_voice.mp3 \
  --file_prefix output
```

### All CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model path (use `mlx-community/chatterbox-fp16`) | Required |
| `--text` | Text to synthesize | Required |
| `--lang_code` | Language code (see below) | `en` |
| `--ref_audio` | Path to reference audio for voice cloning | Required |
| `--file_prefix` | Output filename prefix | `audio` |
| `--audio_format` | Output format: `wav`, `mp3`, `flac` | `wav` |
| `--exaggeration` | Emotion intensity (0.0-1.0) | `0.5` |
| `--cfg_scale` | Classifier-free guidance (lower = more stable) | `0.5` |
| `--temperature` | Sampling temperature | `0.8` |
| `--max_tokens` | Max tokens to generate (lower = faster) | `1000` |
| `--verbose` | Print detailed output | `false` |
| `--play` | Play audio after generation | `false` |

---

## Python API

```python
import mlx.core as mx
import soundfile as sf
import librosa
from mlx_audio.tts.utils import load_model

# Load multilingual Chatterbox model
model = load_model('mlx-community/chatterbox-fp16')

# Load reference audio (for voice cloning)
audio, sr = librosa.load('/path/to/reference.mp3', sr=24000, mono=True)
conds = model.prepare_conditionals(mx.array(audio), sr, exaggeration=0.5)

# Generate Chinese speech
for result in model.generate(
    text="你好，我是语音合成模型。",
    conds=conds,
    lang_code='zh',
    max_new_tokens=300
):
    sf.write('output.wav', result.audio, result.sample_rate)
```

---

## Language Codes

| Code | Language | Code | Language |
|------|----------|------|----------|
| `zh` | Chinese | `ja` | Japanese |
| `en` | English | `ko` | Korean |
| `es` | Spanish | `ar` | Arabic |
| `fr` | French | `hi` | Hindi |
| `de` | German | `ru` | Russian |
| `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `pl` | Polish |
| `tr` | Turkish | `sv` | Swedish |

Full list: `ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh`

---

## Tips

1. **Reference audio**: Use 3-10 seconds of clean speech
2. **First run**: Downloads 2.6GB model (~5-10 min)
3. **Faster generation**: Use `--max_tokens 200` for short text
4. **exaggeration**: Higher = more emotional expression

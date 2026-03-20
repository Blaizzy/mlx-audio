# Ming Omni TTS (BailingMM) in MLX-Audio

This model implementation powers `model_type: "bailingmm"` (Ming Omni TTS) in MLX-Audio.

## Run with CLI

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --text "This is a quick Ming Omni test." \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix quick_test \
  --verbose
```

## Cookbook Examples (CLI + Python)

### 1) Voice cloning

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "Our vision is to build digital infrastructure for future services and bring many small but meaningful improvements to everyday life." \
  --ref_audio "PATH_TO_AUDIO" \
  --ref_text "This is a sample reference transcript." \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_01_tts \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="Our vision is to build digital infrastructure for future services and bring many small but meaningful improvements to everyday life.",
        ref_audio="PATH_TO_AUDIO",
        ref_text="This is a sample reference transcript.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_01_tts_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 2) Basic style control

Note: only provide `ref_text` when it exactly matches `ref_audio`. A mismatched transcript can collapse audio amplitude.

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "Simply put, this was equivalent to handing over the consumer market to competitors." \
  --ref_audio "PATH_TO_AUDIO" \
  --ref_text "This is a sample reference transcript." \
  --instruct "Speak quickly, with medium pitch and higher volume." \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_02_basic \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="Simply put, this was equivalent to handing over the consumer market to competitors.",
        ref_audio="PATH_TO_AUDIO",
        ref_text="This is a sample reference transcript.",
        instruct="Speak quickly, with medium pitch and higher volume.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_02_basic_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 3) Emotion control

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "I got concert tickets at last! This is amazing. I cannot wait to hear the singer live on stage." \
  --ref_audio "PATH_TO_AUDIO" \
  --ref_text "This is a sample reference transcript." \
  --instruct "Use a happy and excited tone." \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_03_emotion \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="I got concert tickets at last! This is amazing. I cannot wait to hear the singer live on stage.",
        ref_audio="PATH_TO_AUDIO",
        ref_text="This is a sample reference transcript.",
        instruct="Use a happy and excited tone.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_03_emotion_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 4) Accent/dialect instruction

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "I believe both companies and individuals share this responsibility." \
  --ref_audio "PATH_TO_AUDIO" \
  --ref_text "This is a sample reference transcript." \
  --instruct "Speak English with a Cantonese accent." \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_04_dialect \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="I believe both companies and individuals share this responsibility.",
        ref_audio="PATH_TO_AUDIO",
        ref_text="This is a sample reference transcript.",
        instruct="Speak English with a Cantonese accent.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_04_dialect_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 5) Multi-speaker podcast style

The CLI accepts one `--ref_audio`, so merge the two prompt clips first:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.utils import load_audio

audio_a = "/Users/prince_canuma/Documents/mlx-audio-dev/Ming-omni-tts/data/wavs/CTS-CN-F2F-2019-11-11-423-012-A.wav"
audio_b = "/Users/prince_canuma/Documents/mlx-audio-dev/Ming-omni-tts/data/wavs/CTS-CN-F2F-2019-11-11-423-012-B.wav"
out = Path("./audio_io/cookbook_cli_en/ref_podcast_concat.wav")
out.parent.mkdir(parents=True, exist_ok=True)
sample_rate = 44100

a = np.array(load_audio(audio_a, sample_rate=sample_rate), dtype=np.float32).reshape(-1)
b = np.array(load_audio(audio_b, sample_rate=sample_rate), dtype=np.float32).reshape(-1)
audio_write(str(out), np.concatenate([a, b], axis=0), sample_rate, format="wav")
print(out)
PY
```

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text $' speaker_1:Could you summarize it again? I am not even sure whether I watched that movie.\n speaker_2:It is the one that turns into a funny classroom situation.\n speaker_1:Right.\n speaker_2:Yes, it is a comedy film.\n speaker_1:A comedy, got it.\n' \
  --ref_audio "PATH_TO_AUDIO" \
  --ref_text $' speaker_1:We still had monthly assessments, and even written exams for service jobs.\n speaker_2:Exactly, that is strange. Sometimes pay is low and rules are strict just because the brand is famous.\n' \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_05_podcast \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
AUDIO_A = "/Users/prince_canuma/Documents/mlx-audio-dev/Ming-omni-tts/data/wavs/CTS-CN-F2F-2019-11-11-423-012-A.wav"
AUDIO_B = "/Users/prince_canuma/Documents/mlx-audio-dev/Ming-omni-tts/data/wavs/CTS-CN-F2F-2019-11-11-423-012-B.wav"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)
REF_MERGED = OUT / "ref_podcast_concat.wav"

model = load_model(MODEL)
a = np.array(load_audio(AUDIO_A, sample_rate=model.sample_rate), dtype=np.float32).reshape(-1)
b = np.array(load_audio(AUDIO_B, sample_rate=model.sample_rate), dtype=np.float32).reshape(-1)
audio_write(str(REF_MERGED), np.concatenate([a, b], axis=0), model.sample_rate, format="wav")

result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text=(
            " speaker_1:Could you summarize it again? I am not even sure whether I watched that movie.\n"
            " speaker_2:It is the one that turns into a funny classroom situation.\n"
            " speaker_1:Right.\n"
            " speaker_2:Yes, it is a comedy film.\n"
            " speaker_1:A comedy, got it.\n"
        ),
        ref_audio=str(REF_MERGED),
        ref_text=(
            " speaker_1:We still had monthly assessments, and even written exams for service jobs.\n"
            " speaker_2:Exactly, that is strange. Sometimes pay is low and rules are strict just because the brand is famous.\n"
        ),
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_05_podcast_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 6) IP character style (zero speaker embedding)

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "The product name is Ultra Spicy Beef Balls." \
  --instruct "Use a playful mascot brand spokesperson voice." \
  --use_zero_spk_emb \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_06_ip \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="The product name is Ultra Spicy Beef Balls.",
        instruct="Use a playful mascot brand spokesperson voice.",
        use_zero_spk_emb=True,
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_06_ip_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 7) Whisper/ASMR style (zero speaker embedding)

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "I will stay with you here until you slowly drift into the softest and calmest sleep." \
  --instruct "Use a soft ASMR whisper style, very gentle, very low volume, and very slow pace." \
  --use_zero_spk_emb \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_07_style \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="I will stay with you here until you slowly drift into the softest and calmest sleep.",
        instruct="Use a soft ASMR whisper style, very gentle, very low volume, and very slow pace.",
        use_zero_spk_emb=True,
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_07_style_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 8) Text-to-audio events

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate audio events based on given text.\n" \
  --text "Thunder and a gentle rain" \
  --cfg_scale 4.5 --sigma 0.3 --temperature 2.5 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_08_tta \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate audio events based on given text.\n",
        text="Thunder and a gentle rain",
        cfg_scale=4.5,
        sigma=0.3,
        temperature=2.5,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_08_tta_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 9) BGM generation

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate music based on the following description.\n" \
  --text "Genre: electronic dance music. Mood: confident and determined. Instrument: drum kit. Theme: festival. Duration: 30 seconds." \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 400 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_09_bgm \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate music based on the following description.\n",
        text="Genre: electronic dance music. Mood: confident and determined. Instrument: drum kit. Theme: festival. Duration: 30 seconds.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=400,
        lang_code="en",
    )
)

output = OUT / "en_09_bgm_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 10) Speech + BGM instruction

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "The performance decline can largely be attributed to stopping service for several brands." \
  --ref_audio "PATH_TO_AUDIO" \
  --ref_text "The performance decline can largely be attributed to stopping service for several brands." \
  --instruct "Add warm contemporary classical background music with electric guitar and a festive mood at moderate SNR." \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_10_speech_bgm \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="The performance decline can largely be attributed to stopping service for several brands.",
        ref_audio="PATH_TO_AUDIO",
        ref_text="The performance decline can largely be attributed to stopping service for several brands.",
        instruct="Add warm contemporary classical background music with electric guitar and a festive mood at moderate SNR.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_10_speech_bgm_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

### 11) Speech + environment sound instruction

```bash
uv run mlx_audio.tts.generate \
  --model "mlx-community/Ming-omni-tts-16.8B-A3B-bf16" \
  --prompt "Please generate speech based on the following description.\n" \
  --text "The performance decline can largely be attributed to stopping service for several brands." \
  --ref_audio "PATH_TO_AUDIO" \
  --ref_text "The performance decline can largely be attributed to stopping service for several brands." \
  --instruct "Add birds chirping as a soft environmental background at moderate SNR." \
  --cfg_scale 2.0 --sigma 0.25 --temperature 0.0 --max_tokens 200 \
  --lang_code en \
  --output_path "audio_io" \
  --file_prefix en_11_speech_sound \
  --verbose
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

MODEL = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"
OUT = Path("audio_io")
OUT.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL)
result = next(
    model.generate(
        prompt="Please generate speech based on the following description.\n",
        text="The performance decline can largely be attributed to stopping service for several brands.",
        ref_audio="PATH_TO_AUDIO",
        ref_text="The performance decline can largely be attributed to stopping service for several brands.",
        instruct="Add birds chirping as a soft environmental background at moderate SNR.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

output = OUT / "en_11_speech_sound_000.wav"
audio_write(str(output), np.array(result.audio, dtype=np.float32), result.sample_rate, format="wav")
print(output)
```

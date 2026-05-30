---
title: Speech-to-Speech Models
---

# Speech-to-Speech (STS) Models

MLX Audio provides speech-to-speech models for audio source separation, speech enhancement, and multimodal voice interaction.

| Model | Provider | Use Case | Repo |
|-------|----------|----------|------|
| [**SAM-Audio**](#sam-audio) | Meta | Text-guided source separation | [mlx-community/sam-audio-large](https://huggingface.co/mlx-community/sam-audio-large) |
| [**Liquid2.5-Audio**](#liquid25-audio) | LiquidAI | Speech-to-Speech, TTS, ASR | [mlx-community/LFM2.5-Audio-1.5B-4bit](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-4bit) |
| [**Moshi**](#moshi) | Kyutai Labs | Full-duplex voice conversation | [kyutai/moshiko-mlx-q4](https://huggingface.co/kyutai/moshiko-mlx-q4) |
| [**MossFormer2 SE**](#mossformer2-se) | Alibaba | Speech enhancement / noise removal | [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX) |
| [**DeepFilterNet**](#deepfilternet) | -- | Speech enhancement / noise suppression | [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx) |

---

## SAM-Audio

[SAM-Audio](https://github.com/facebookresearch/sam-audio) (Segment Anything Model for Audio) is Meta's foundation model for audio source separation using text prompts. Describe what you want to extract, and SAM-Audio separates it from the mix.

```python
from mlx_audio.sts import load

model = load("mlx-community/sam-audio-large")
```

### Quick Start

=== "Python"

    ```python
    from mlx_audio.sts import SAMAudio, save_audio

    # Load model
    model = SAMAudio.from_pretrained("facebook/sam-audio-large")

    # Separate audio using a text prompt
    result = model.separate(
        audios=["mixed_audio.wav"],
        descriptions=["A person speaking"],
    )

    # Save separated audio
    save_audio(result.target[0], "voice.wav", sample_rate=model.sample_rate)
    save_audio(result.residual[0], "background.wav", sample_rate=model.sample_rate)
    ```

=== "Long audio (chunked)"

    ```python
    from mlx_audio.sts import SAMAudio, save_audio

    model = SAMAudio.from_pretrained("facebook/sam-audio-large")

    result = model.separate_long(
        audios=["long_audio.wav"],
        descriptions=["speech"],
        chunk_seconds=10.0,
        overlap_seconds=3.0,
        ode_opt={"method": "euler", "step_size": 2/64},
    )

    save_audio(result.target[0], "voice.wav", sample_rate=model.sample_rate)
    ```

=== "Streaming"

    ```python
    from mlx_audio.sts import SAMAudio
    from mlx_audio.audio_io import write as audio_write
    import numpy as np

    model = SAMAudio.from_pretrained("facebook/sam-audio-large")

    target_chunks = []
    for result in model.separate_streaming(
        audios=["audio.mp3"],
        descriptions=["speech"],
        chunk_seconds=10.0,
        overlap_seconds=3.0,
        verbose=True,
    ):
        target_chunks.append(np.array(result.target[:, 0]))

    audio_write("target.wav", np.concatenate(target_chunks), 48000)
    ```

### Temporal Anchors

Anchors provide fine-grained control by telling the model which time spans contain (or do not contain) the target sound:

```python
result = model.separate(
    audios=["audio.wav"],
    descriptions=["speech"],
    anchors=[[("+", 1.5, 3.0)]],  # Target speech is at 1.5-3.0s
)

# Multiple anchors: target IS here, but NOT there
result = model.separate(
    audios=["audio.wav"],
    descriptions=["speech"],
    anchors=[[("+", 1.5, 3.0), ("-", 5.0, 7.0)]],
)
```

| Token | Meaning |
|-------|---------|
| `"+"` | Positive -- "the target sound IS here" |
| `"-"` | Negative -- "the target sound is NOT here" |

!!! warning
    Anchors are only supported with `separate()`, not with `separate_long()` or `separate_streaming()`.

### ODE Solver Options

Control the quality vs speed tradeoff:

| Method | Step Size | Steps | Quality | Speed |
|--------|-----------|-------|---------|-------|
| `midpoint` | `2/64` | 32 | Maximum | Slowest |
| `midpoint` | `2/32` | 16 | Best (default) | Slow |
| `midpoint` | `2/16` | 8 | Good | Medium |
| `euler` | `2/64` | 32 | Very Good | Medium |
| `euler` | `2/32` | 16 | Good | Fast |
| `euler` | `2/16` | 8 | Good | Fastest |

---

## Liquid2.5-Audio

[LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) by LiquidAI is a 1.5B parameter multimodal foundation model supporting text-to-speech, speech-to-text, and speech-to-speech in a single model with interleaved text and audio generation.

### Available Models

| Model | Precision | Repo |
|-------|-----------|------|
| **LFM2.5-Audio 4bit** | 4-bit quantized | [mlx-community/LFM2.5-Audio-1.5B-4bit](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-4bit) |
| **LFM2.5-Audio 8bit** | 8-bit quantized | [mlx-community/LFM2.5-Audio-1.5B-8bit](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-8bit) |

```python
from mlx_audio.sts import load

model = load("mlx-community/LFM2.5-Audio-1.5B-4bit")
```

### Text-to-Speech

```python
import mlx.core as mx
from mlx_audio.sts.models.lfm_audio import (
    LFM2AudioModel,
    LFM2AudioProcessor,
    ChatState,
    LFMModality,
)
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN
from mlx_audio.audio_io import write as audio_write

model = LFM2AudioModel.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")
processor = LFM2AudioProcessor.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")

chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Perform TTS. Use a UK male voice.")
chat.end_turn()
chat.new_turn("user")
chat.add_text("Hello, welcome to MLX Audio!")
chat.end_turn()
chat.new_turn("assistant")

audio_codes = []
for token, modality in model.generate_sequential(
    **dict(chat),
    max_new_tokens=2048,
    temperature=0.8,
):
    mx.eval(token)
    if modality == LFMModality.AUDIO_OUT:
        if token[0].item() == AUDIO_EOS_TOKEN:
            break
        audio_codes.append(token)

audio_codes = mx.stack(audio_codes, axis=0)[None, :].transpose(0, 2, 1)
waveform = processor.decode_audio(audio_codes)
audio_write("output.wav", waveform[0].tolist(), model.sample_rate)
```

### Speech-to-Text

```python
import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import read as audio_read
from mlx_audio.sts.models.lfm_audio import (
    LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality,
)

model = LFM2AudioModel.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")
processor = LFM2AudioProcessor.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")

audio, sr = audio_read("input.wav")
audio = mx.array(audio.astype(np.float32))

chat = ChatState(processor)
chat.new_turn("user")
chat.add_audio(audio, sample_rate=sr)
chat.add_text("Transcribe the audio.")
chat.end_turn()
chat.new_turn("assistant")

for token, modality in model.generate_interleaved(**dict(chat), max_new_tokens=512):
    mx.eval(token)
    if modality == LFMModality.TEXT:
        print(processor.decode_text(token[None]), end="", flush=True)
```

### Speech-to-Speech

```python
import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import read as audio_read, write as audio_write
from mlx_audio.sts.models.lfm_audio import (
    LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality,
)

model = LFM2AudioModel.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")
processor = LFM2AudioProcessor.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")

audio, sr = audio_read("input.wav")
audio = mx.array(audio.astype(np.float32))

chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio.")
chat.end_turn()
chat.new_turn("user")
chat.add_audio(audio, sample_rate=sr)
chat.end_turn()
chat.new_turn("assistant")

text_out, audio_out = [], []
for token, modality in model.generate_interleaved(**dict(chat), max_new_tokens=2048):
    mx.eval(token)
    if modality == LFMModality.TEXT:
        text_out.append(token)
        print(processor.decode_text(token[None]), end="", flush=True)
    else:
        audio_out.append(token)

if audio_out:
    audio_codes = mx.stack(audio_out[:-1], axis=1)[None, :]
    waveform = processor.decode_with_detokenizer(audio_codes)
    audio_write("response.wav", waveform[0].tolist(), 24000)
```

### Generation Config

```python
from mlx_audio.sts.models.lfm_audio import GenerationConfig

config = GenerationConfig(
    max_new_tokens=2048,
    temperature=0.9,        # Text sampling temperature
    top_k=50,               # Text top-k sampling
    top_p=1.0,              # Text nucleus sampling
    audio_temperature=0.7,  # Audio sampling temperature
    audio_top_k=30,         # Audio top-k sampling
)
```

---

## Moshi

[Moshi](https://github.com/kyutai-labs/moshi) by Kyutai Labs is a full-duplex speech-to-speech foundation model that can listen and talk at the same time in real-time.

### Features

- **Full-duplex**: handles concurrent input and output audio streams
- **Multi-modal**: simultaneously generates text ("inner monologue") and audio tokens
- **Ultra-low latency**: ~200ms theoretical latency
- **Streaming generation** natively via MLX

### Available Models

| Model | Precision | Memory | Repo |
|-------|-----------|--------|------|
| **moshiko bf16** | Full precision | ~16GB | [kyutai/moshiko-mlx-bf16](https://huggingface.co/kyutai/moshiko-mlx-bf16) |
| **moshiko q8** | 8-bit quantized | -- | [kyutai/moshiko-mlx-q8](https://huggingface.co/kyutai/moshiko-mlx-q8) |
| **moshiko q4** | 4-bit quantized | ~8GB | [kyutai/moshiko-mlx-q4](https://huggingface.co/kyutai/moshiko-mlx-q4) |

```python
from mlx_audio.sts import load

model = load("kyutai/moshiko-mlx-q4")
```

### Usage

```python
from mlx_audio.sts.models.moshi import MoshiSTSModel
import sounddevice as sd
import numpy as np

model = MoshiSTSModel.from_pretrained("kyutai/moshiko-mlx-q4", quantized=4)

stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.float32)
stream.start()

for word, pcm_frame in model.generate(steps=150):
    if word:
        print(word, end="", flush=True)
    if pcm_frame is not None:
        stream.write(np.array(pcm_frame))

stream.stop()
stream.close()
```

---

## MossFormer2 SE

[MossFormer2 SE](https://github.com/modelscope/ClearerVoice-Studio) is a speech enhancement model (~55.3M parameters) from Alibaba's ClearerVoice-Studio, processing audio at 48kHz for high-quality noise removal.

### Features

- 48kHz sample rate for high-quality audio
- Auto-chunking for long audio (>60s) with low RAM usage
- Multiple precision options: fp32, fp16, int8, int6, int4

### Available Models

| Model | Precision | Repo |
|-------|-----------|------|
| **MossFormer2 SE** | fp32 | [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX) |
| **MossFormer2 SE 8bit** | int8 | [starkdmi/MossFormer2_SE_48K_MLX-8bit](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX-8bit) |
| **MossFormer2 SE 4bit** | int4 | [starkdmi/MossFormer2_SE_48K_MLX-4bit](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX-4bit) |

```python
from mlx_audio.sts import load

model = load("starkdmi/MossFormer2_SE_48K_MLX")
```

### Usage

=== "Basic"

    ```python
    from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel, save_audio

    model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")
    enhanced = model.enhance("noisy.wav")
    save_audio(enhanced, "enhanced.wav", 48000)
    ```

=== "Forced chunked mode"

    ```python
    from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel, save_audio

    model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")

    # Force chunked processing for very long audio or limited RAM
    enhanced = model.enhance("long_audio.wav", chunked=True)
    save_audio(enhanced, "enhanced.wav", 48000)
    ```

=== "Full mode (best quality)"

    ```python
    from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel, save_audio

    model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")

    # Force full processing for best quality (if RAM allows)
    enhanced = model.enhance("audio.wav", chunked=False)
    save_audio(enhanced, "enhanced.wav", 48000)
    ```

!!! info "Automatic mode selection"
    By default, `enhance()` automatically selects full mode for audio under 60 seconds and chunked mode for longer audio.

---

## DeepFilterNet

[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) is a speech enhancement model for noise suppression, available in versions 1, 2, and 3. The MLX port supports all three versions with stateful streaming processing.

The standard STS loader defaults to `v3`:

```python
from mlx_audio.sts import load

model = load("mlx-community/DeepFilterNet-mlx")
```

### Usage

=== "Basic"

    ```python
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

    # Load v3 (default)
    model = DeepFilterNetModel.from_pretrained()
    model.enhance_file("noisy.wav", "clean.wav")
    ```

=== "Specific version"

    ```python
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

    # Load v2
    model = DeepFilterNetModel.from_pretrained(version=2)
    model.enhance_file("noisy.wav", "clean.wav")

    # Or specify the subfolder directly
    model = DeepFilterNetModel.from_pretrained(subfolder="v1")
    ```

=== "Streaming / chunked"

    ```python
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

    model = DeepFilterNetModel.from_pretrained()
    streamer = model.create_streamer(pad_end_frames=3, compensate_delay=True)

    out_1 = streamer.process_chunk(chunk_a)
    out_2 = streamer.process_chunk(chunk_b)
    out_tail = streamer.flush()
    ```

!!! tip "Version selection"
    The model version is automatically selected from `config.json`. Version 3 is the default and generally recommended. Use v2 for stateful streaming with per-hop processing.

**Pretrained weights**: [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx)

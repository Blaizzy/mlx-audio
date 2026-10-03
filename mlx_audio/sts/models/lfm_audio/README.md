# LFM2.5-Audio for MLX

MLX implementation of [LiquidAI's LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B), a multimodal foundation model for audio understanding and generation.

## Features

- **Text-to-Speech (TTS)**: Generate natural speech from text
- **Speech-to-Text (ASR)**: Transcribe audio to text
- **Speech-to-Speech (STS)**: Voice conversations with audio input and output
- **Interleaved Generation**: Mixed text and audio responses in a single turn
- **Streaming**: Real-time token-by-token generation for low-latency applications

## Installation

```bash
pip install mlx-audio
```

## Quick Start

### Text-to-Speech (TTS)

```python
import mlx.core as mx
from mlx_audio.sts.models.lfm_audio import (
    LFM2AudioModel,
    LFM2AudioProcessor,
    ChatState,
    LFMModality,
)
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN

# Load model and processor
model = LFM2AudioModel.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")
processor = LFM2AudioProcessor.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")

# Create chat state
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Perform TTS. Use a UK male voice.")
chat.end_turn()
chat.new_turn("user")
chat.add_text("Hello, welcome to MLX Audio!")
chat.end_turn()
chat.new_turn("assistant")

# Generate with interleaved text and audio
audio_codes = []
for token, modality in model.generate_sequential(
    **dict(chat),
    max_new_tokens=2048,
    # Upstream TTS recipe: greedy text, sampled audio
    audio_temperature=0.8,
    audio_top_k=64,
):
    mx.eval(token)
    if modality == LFMModality.AUDIO_OUT:
        if token[0].item() == AUDIO_EOS_TOKEN:
            break
        audio_codes.append(token)

# Decode audio
audio_codes = mx.stack(audio_codes, axis=0)[None, :].transpose(0, 2, 1)
waveform = processor.decode_audio(audio_codes, codec="detokenizer")

# Save audio (24kHz sample rate)
from mlx_audio.audio_io import write as audio_write
audio_write("output.wav", waveform[0].tolist(), model.sample_rate)

```

### Speech-to-Text (ASR)

```python
import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import read as audio_read
from mlx_audio.sts.models.lfm_audio import (
    LFM2AudioModel,
    LFM2AudioProcessor,
    ChatState,
    LFMModality,
)

# Load model and processor
model = LFM2AudioModel.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")
processor = LFM2AudioProcessor.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")

# Load audio (must be 24kHz for audio input)
audio, sr = audio_read("input.wav")
audio = mx.array(audio.astype(np.float32))

# Create chat state with audio input
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Perform ASR.")
chat.end_turn()
chat.new_turn("user")
chat.add_audio(audio, sample_rate=sr)
chat.end_turn()
chat.new_turn("assistant")

# Generate text response (ASR is fully greedy upstream)
text_out = []
for token, modality in model.generate_sequential(**dict(chat), max_new_tokens=512):
    mx.eval(token)
    if modality == LFMModality.TEXT:
        text_out.append(token)
        print(processor.decode_text(token[None]), end="", flush=True)
```

### Speech-to-Speech (STS)

```python
import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import read as audio_read, write as audio_write
from mlx_audio.sts.models.lfm_audio import (
    LFM2AudioModel,
    LFM2AudioProcessor,
    ChatState,
    LFMModality,
)
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN

# Load model and processor
model = LFM2AudioModel.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")
processor = LFM2AudioProcessor.from_pretrained("mlx-community/LFM2.5-Audio-1.5B-4bit")

# Load input audio (24kHz)
audio, sr = audio_read("input.wav")
audio = mx.array(audio.astype(np.float32))

# Create chat state with audio input
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio.")
chat.end_turn()
chat.new_turn("user")
chat.add_audio(audio, sample_rate=sr)
chat.end_turn()
chat.new_turn("assistant")

# Generate response with both text and audio
text_out, audio_out = [], []
for token, modality in model.generate_interleaved(
    **dict(chat), max_new_tokens=2048, audio_temperature=1.0, audio_top_k=4
):
    mx.eval(token)
    if modality == LFMModality.TEXT:
        text_out.append(token)
        print(processor.decode_text(token[None]), end="", flush=True)
    elif token[0].item() != AUDIO_EOS_TOKEN:  # skip end-of-audio frames
        audio_out.append(token)

# Decode audio response
if audio_out:
    audio_codes = mx.stack(audio_out, axis=1)[None, :]  # (1, 8, T)
    waveform = processor.decode_audio(audio_codes, codec="detokenizer")
    audio_write("response.wav", waveform[0].tolist(), 24000)
```

## Interleaved Text and Audio Generation

LFM2.5-Audio uses `generate_interleaved` for mixed text and audio output. The model can respond with text, audio, or both interleaved together.

Each audio token returned by `generate_interleaved` is a complete frame of shape `(8,)` containing all 8 codebook values:

```python
from mlx_audio.sts.models.lfm_audio import LFMModality
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN

text_out, audio_out = [], []
for token, modality in model.generate_interleaved(
    **dict(chat), max_new_tokens=2048, audio_temperature=1.0, audio_top_k=4
):
    mx.eval(token)
    if modality == LFMModality.TEXT:
        text_out.append(token)
        # Stream text output
        print(processor.decode_text(token[None]), end="", flush=True)
    elif token[0].item() != AUDIO_EOS_TOKEN:  # LFMModality.AUDIO_OUT
        audio_out.append(token)  # token shape: (8,)

# Stack audio frames: list of (8,) -> (8, T). An all-EOS frame (2048 in every
# codebook) closes each audio span; it is not a codec token, so it is skipped.
if audio_out:
    audio_codes = mx.stack(audio_out, axis=1)[None, :]  # (1, 8, T)
    waveform = processor.decode_audio(audio_codes, codec="detokenizer")
```

## Audio Decoding Options

LFM2.5-Audio supports two methods for decoding audio codes to waveforms:

### 1. Detokenizer (Recommended for TTS)

The neural detokenizer reconstructs audio using ISTFT from predicted spectrograms:

```python
# Decode using the detokenizer (the default)
audio = processor.decode_audio(codes[None], codec="detokenizer")  # (1, T_audio)
```

### 2. Mimi Codec

The Mimi neural codec provides an alternative decoding path:

```python
# Decode using the Mimi codec
audio = processor.decode_audio(codes, codec="mimi")  # (1, 1, T_audio)
```

## Generation Configuration

```python
from mlx_audio.sts.models.lfm_audio import GenerationConfig

config = GenerationConfig(
    max_new_tokens=2048,    # Maximum tokens to generate
    temperature=None,       # Text sampling temperature (None = greedy)
    top_k=None,             # Text top-k sampling (None = no filtering)
    audio_temperature=1.0,  # Audio sampling temperature (None = greedy)
    audio_top_k=4,          # Audio top-k sampling (None = no filtering)
)
```

Every knob defaults to `None`, i.e. greedy decoding, matching `liquid-audio`. The
upstream recipes are:

| Task | Mode | Text | Audio |
| --- | --- | --- | --- |
| Chat / STS | `generate_interleaved` | greedy | `audio_temperature=1.0`, `audio_top_k=4` |
| TTS | `generate_sequential` | greedy | `audio_temperature=0.8`, `audio_top_k=64` |
| ASR | `generate_sequential` | greedy | n/a |

## Streaming Generation

For real-time audio playback during generation:

```python
from mlx_audio.sts.models.lfm_audio import LFMModality
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN

FRAMES_PER_CHUNK = 10  # Decode every 10 audio frames

audio_buffer = []
for token, modality in model.generate_interleaved(
    **dict(chat), max_new_tokens=2048, audio_temperature=1.0, audio_top_k=4
):
    mx.eval(token)
    if modality == LFMModality.AUDIO_OUT:
        if token[0].item() != AUDIO_EOS_TOKEN:  # skip end-of-audio frames
            audio_buffer.append(token)

        # Decode when we have enough frames
        if len(audio_buffer) >= FRAMES_PER_CHUNK:
            codes = mx.stack(audio_buffer, axis=1)[None, :]  # (1, 8, T)
            chunk = processor.decode_audio(codes, codec="detokenizer")
            # Play chunk with your audio library...
            audio_buffer = []

    elif modality == LFMModality.TEXT:
        # Stream text output
        print(processor.decode_text(token[None]), end="", flush=True)
```

## Model Architecture

LFM2.5-Audio consists of:

- **Audio Encoder**: Conformer-based encoder for processing input audio
- **LFM Backbone**: 1.5B parameter Liquid Foundation Model for multimodal reasoning
- **Audio Decoder**: Depthformer for generating audio codes
- **Detokenizer**: ISTFT-based neural vocoder for waveform reconstruction

## API Reference

### LFM2AudioModel

```python
class LFM2AudioModel:
    @classmethod
    def from_pretrained(cls, model_name: str) -> "LFM2AudioModel":
        """Load pretrained model from HuggingFace Hub."""

    def generate_interleaved(
        self,
        text_tokens: mx.array,
        audio_features: mx.array,
        modalities: mx.array,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        audio_temperature: Optional[float] = None,
        audio_top_k: Optional[int] = None,
    ) -> Generator[Tuple[mx.array, LFMModality], None, None]:
        """Generate interleaved text and audio tokens.

        Yields:
            (token, modality) tuples where:
            - For TEXT: token is scalar, modality is LFMModality.TEXT
            - For AUDIO_OUT: token is (8,) array, modality is LFMModality.AUDIO_OUT
        """
```

### LFM2AudioProcessor

```python
class LFM2AudioProcessor:
    @classmethod
    def from_pretrained(cls, model_name: str) -> "LFM2AudioProcessor":
        """Load pretrained processor from HuggingFace Hub."""

    def preprocess_audio(self, audio: mx.array, sample_rate: int) -> mx.array:
        """Convert audio to mel spectrogram features."""

    def tokenize_audio(self, audio: mx.array, sample_rate: int) -> mx.array:
        """Tokenize audio using Mimi codec."""

    def decode_audio(
        self, codes: mx.array, codec: Optional[str] = "detokenizer"
    ) -> mx.array:
        """Decode audio codes to a waveform via "detokenizer" or "mimi"."""

    def tokenize_text(self, text: str) -> mx.array:
        """Tokenize text."""

    def decode_text(self, tokens: mx.array) -> str:
        """Decode text tokens."""
```

### ChatState

```python
class ChatState:
    def __init__(self, processor: LFM2AudioProcessor):
        """Initialize chat state."""

    def new_turn(self, role: str):
        """Start a new turn (user/assistant/system)."""

    def end_turn(self):
        """End the current turn."""

    def add_text(self, text: str):
        """Add text to current turn."""

    def add_audio(self, audio: mx.array, sample_rate: int):
        """Add audio to current turn."""
```

## License

This implementation follows the license terms of the original LFM2.5-Audio model.
See [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) for details.

## Acknowledgements

- [LiquidAI](https://liquid.ai/) for the LFM2.5-Audio model
- [MLX](https://github.com/ml-explore/mlx) team for the framework

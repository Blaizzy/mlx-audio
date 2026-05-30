# Higgs Audio v2

Llama-3.2-3B-backed TTS with multi-codebook acoustic tokens and delay-pattern streaming, with real-time voice cloning on Apple Silicon. The MLX port targets the 3B open-weights release from Boson AI and reuses the in-tree HiggsAudio acoustic tokenizer (originally added for OmniVoice).

- **Original repo:** [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)
- **Paper / blog:** [Higgs Audio v2](https://boson.ai/blog/higgs-audio-v2)
- **Full MLX port docs:** [`docs/models/tts/higgs_audio.md`](../../../../docs/models/tts/higgs_audio.md)

## Highlights

- Reference-audio voice cloning via ChatML prompt format
- Full `AUDIO_INIT` + delay-pattern ramp-in/out state machine
- Repetition-avoidance sampling (RAS) for stable long-form output
- MLX native 4/6/8-bit quantization with optional per-layer protection
- Conforms to the standard mlx-audio interface (`mlx_audio.tts.generate` CLI works unchanged)

## Usage

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/higgs-audio-v2-3B-mlx-q8 \
  --text "Hello from Higgs Audio on MLX." \
  --ref_audio path/to/reference.wav \
  --ref_text "Transcript of the reference clip."
```

Python API (standard):

```python
from mlx_audio.tts.utils import load
from mlx_audio.audio_io import write as audio_write

model = load("mlx-community/higgs-audio-v2-3B-mlx-q8")

for result in model.generate(
    text="Hello from Higgs Audio on MLX.",
    ref_audio="path/to/reference.wav",
    ref_text="Transcript of the reference clip.",
):
    audio_write("output.wav", result.audio, result.sample_rate)
```

Python API (Higgs-specific surface):

```python
from mlx_audio.tts.models.higgs_audio import HiggsAudioServer

server = HiggsAudioServer(
    model_path="mlx-community/higgs-audio-v2-3B-mlx-q8",
    codec_path="mlx-community/higgs-audio-v2-tokenizer",
)
result = server.generate(
    text="Hello from Higgs Audio on MLX.",
    ref_audio_path="path/to/reference.wav",
    ref_text="Transcript of the reference clip.",
)
```

## Voice Cloning

Best results come from **5-15 seconds of clean reference speech**. Reference audio is encoded through the in-tree HiggsAudioTokenizer and stitched into the assistant turn of a ChatML prompt. `ref_text` is the transcript of the reference clip and is required for stable alignment between the cloned voice and the target text.

Three sample voices ship in [`examples/voice_prompts/`](../../../../examples/voice_prompts/): `en_woman`, `en_man`, `en_man_deep`.

See [`examples/higgs_audio_clone_demo.py`](../../../../examples/higgs_audio_clone_demo.py) for a complete cloning walkthrough.

Without `ref_audio`, generation runs in "smart voice" mode (random voice per sample) — works but is less reliable than voice cloning. For production use, a reference voice is strongly recommended.

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Sampling temperature |
| `top_p` | 0.95 | Nucleus sampling cutoff |
| `top_k` | `None` | Optional top-k cap |
| `max_new_frames` | 1200 | Max acoustic frames to generate (≈ 48s @ 25 fps) |
| `fade_in_ms` | 30.0 | Fade-in on decoded audio |
| `fade_out_ms` | 30.0 | Fade-out on decoded audio |
| `ref_audio` | `None` | Path to reference audio (voice cloning) |
| `ref_text` | `None` | Transcript of the reference clip |

## Available Models

| Model | Parameters | Format | RTF (M5 Max) | Memory |
|-------|-----------|--------|--------------|--------|
| [`bosonai/higgs-audio-v2-generation-3B-base`](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) | 3B | bf16 (authoritative) | 0.60× | 6.8 GB |
| [`mlx-community/higgs-audio-v2-3B-mlx-q8`](https://huggingface.co/mlx-community/higgs-audio-v2-3B-mlx-q8) | 3B | 8-bit | 0.36× | 6.18 GB |
| [`mlx-community/higgs-audio-v2-3B-mlx-q6`](https://huggingface.co/mlx-community/higgs-audio-v2-3B-mlx-q6) | 3B | 6-bit | 0.33× | 4.75 GB |
| [`mlx-community/higgs-audio-v2-tokenizer`](https://huggingface.co/mlx-community/higgs-audio-v2-tokenizer) | — | — | — | acoustic tokenizer (required) |

## Conversion

To quantize or save a pre-converted format:

```bash
python -m mlx_audio.convert \
  --hf-path bosonai/higgs-audio-v2-generation-3B-base \
  --mlx-path ./higgs-audio-v2-3B-mlx-q8 \
  --quantize --q-bits 8
```

## Architecture

- **Llama-3.2-3B backbone** for the text/language stream
- **Multi-codebook acoustic tokens** with `AUDIO_INIT` initialization and delay-pattern ramp-in/out
- **HiggsAudio acoustic tokenizer** (shared with the in-tree OmniVoice entry) at 24kHz
- **ChatML prompt format** for ref-audio conditioning
- **RAS (repetition-avoidance sampling)** for long-form stability

## License

Higgs Audio v2 is released under the [Apache 2.0 License](https://github.com/boson-ai/higgs-audio/blob/main/LICENSE).

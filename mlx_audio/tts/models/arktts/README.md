# Audio8 TTS

Multilingual text-to-speech and zero-shot voice cloning on Apple Silicon, with
44.1 kHz mono audio output.

## Supported Models

| Repository | Precision | Languages |
|---|---|---|
| [nativ-community/Audio8-TTS-Realistic-0.6b-bf16](https://huggingface.co/nativ-community/Audio8-TTS-Realistic-0.6b-bf16) | BF16 | 23 |
| [nativ-community/Audio8-TTS-Realistic-0.6b-4bit](https://huggingface.co/nativ-community/Audio8-TTS-Realistic-0.6b-4bit) | 4-bit | 23 |
| [nativ-community/Audio8-TTS-Realistic-0.6b-8bit](https://huggingface.co/nativ-community/Audio8-TTS-Realistic-0.6b-8bit) | 8-bit | 23 |
| [mlx-community/Audio8-TTS-Preview-0.6b-bf16](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.6b-bf16) | BF16 | 11 |
| [mlx-community/Audio8-TTS-Preview-0.1b-bf16](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.1b-bf16) | BF16 | 8 |

Use any of these repository IDs with the Python and CLI examples below.

The audio codec is included in each checkpoint. No separate codec download or
PyTorch installation is required.

## Generate speech

Python API:

```python
from mlx_audio.audio_io import write
from mlx_audio.tts.utils import load

model = load("nativ-community/Audio8-TTS-Realistic-0.6b-bf16")

for result in model.generate(
    text="Welcome to Audio8 TTS, running on Apple Silicon.",
):
    write("output.wav", result.audio, result.sample_rate)
```

Without `ref_audio` the model synthesizes with its own default voice.

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model nativ-community/Audio8-TTS-Realistic-0.6b-bf16 \
  --text "Welcome to Audio8 TTS, running on Apple Silicon."
```

## Clone a voice

Provide a clean reference recording and its exact transcript:

```python
result = next(model.generate(
    text="This sentence uses the reference speaker.",
    ref_audio="reference.wav",
    ref_text="The exact words spoken in the reference recording.",
))
write("cloned.wav", result.audio, result.sample_rate)
```

For the CLI, add `--ref_audio reference.wav --ref_text "Transcript of the reference clip."`.
Audio arrays passed to the Python API should be mono at 44.1 kHz; file paths are
resampled automatically.

## Languages

Realistic supports 23 languages: Arabic, Cantonese, Chinese, Dutch, English,
Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Malay,
Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, and Vietnamese.

Preview-0.6b supports 11 languages:
Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese, Korean, Polish, Spanish.

Language follows the input text. Quality varies by language.

## Options

| Option | Default | Description |
|---|---|---|
| `temperature` | 0.7 | Controls variation in the generated speech. |
| `top_p` | 0.9 | Sampling probability cutoff. |
| `top_k` | 50 | Maximum number of token candidates at each sampling step. |
| `max_tokens` | 512 | Maximum audio frames; 512 frames allow approximately 23.8 seconds. |
| `do_sample` | `True` | Set `False` for deterministic generation. |
| `seed` | — | Set an integer for reproducible sampling. |

For Realistic, you can use `temperature=0.8`, `top_p=0.95`, and
`max_tokens=1024` as in the [usage guide](../../../../docs/models/tts/audio8.md).

Generation returns one complete waveform per call. Text, reference audio, and
generated audio share a 2,048-position context window. Long input text or reference
recordings leave less room for output; split long passages into shorter utterances.

## License

Realistic-0.6b and Preview-0.6b are released under Apache-2.0. Preview-0.1b uses
the Audio8 Community License v1.0; see its model card for the license terms.

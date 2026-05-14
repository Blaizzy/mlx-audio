# Dramabox

Dramabox is Resemble AI's dialogue-focused text-to-speech model built on an audio-only LTX DiT, a Gemma text encoder, an audio VAE, and a BigVGAN-style vocoder. It generates 48 kHz stereo speech and supports reference-audio voice cloning by conditioning generation on a short voice sample.

**Original model:** [ResembleAI/Dramabox](https://huggingface.co/ResembleAI/Dramabox)

## Usage

Python API:

```python
from mlx_audio.tts.utils import load

model = load("mlx-community/ResembleAI-Dramabox", lazy=True, strict=True)

result = next(model.generate(
    text='A calm narrator says, "The city lights flickered on at dusk."',
))

audio = result.audio  # mlx array, 48kHz stereo
```

## Voice Cloning

Pass a short reference clip with `ref_audio`. Dramabox works best with a clean 3 to 10 second sample:

```python
result = next(model.generate(
    text='A confident announcer says, "Tonight, every secret gets a spotlight."',
    ref_audio="speaker.wav",
    cfg_scale=2.5,
    stg_scale=1.5,
    steps=30,
    seed=42,
))

audio = result.audio
```

## Text Encoder Override

The converted model defaults to `mlx-community/gemma-3-12b-it-8bit` for prompt encoding. Use `text_encoder_model` to use another Gemma checkpoint:

```python
result = next(model.generate(
    text='A woman speaks clearly, "The weather today will be sunny."',
    ref_audio="speaker.wav",
    text_encoder_model="mlx-community/gemma-3-12b-it-8bit",
))
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ref_audio` | `None` | Reference audio path or array for voice cloning |
| `steps` | 30 | Euler denoising steps. Higher is slower and may improve quality |
| `cfg_scale` | 2.5 | Classifier-free guidance strength |
| `stg_scale` | 1.5 | Spatiotemporal guidance strength |
| `stg_block` | 29 | Transformer block used for STG perturbation |
| `rescale_scale` | `"auto"` | CFG rescale schedule used to avoid clipping |
| `gen_duration` / `duration` | auto | Target generation duration in seconds |
| `duration_multiplier` | 1.1 | Multiplier applied to automatic duration estimation |
| `seed` | 42 | Random seed for reproducible sampling |
| `text_encoder_model` | config default | Compatible MLX Gemma text encoder checkpoint |

## CLI

Text-only generation:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/ResembleAI-Dramabox \
  --text 'A calm narrator says, "The city lights flickered on at dusk."' \
  --play
```

Voice cloning:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/ResembleAI-Dramabox \
  --text 'A confident announcer says, "Tonight, every secret gets a spotlight."' \
  --ref_audio speaker.wav \
  --cfg_scale 2.5 \
  --play
```

## Architecture

- **Audio-only LTX DiT:** 48 transformer layers, 32 audio attention heads, split RoPE, AdaLN conditioning
- **Gemma prompt encoder:** collects hidden states across Gemma layers and projects them through the Dramabox audio embeddings connector
- **Audio VAE:** encodes 16 kHz reference mel features and decodes generated latents
- **BigVGAN-style vocoder:** converts decoded mel features to 48 kHz stereo waveform
- **Reference conditioning:** appends frozen reference latent tokens with asymmetric target-to-reference attention

## License

See the [ResembleAI/Dramabox model card](https://huggingface.co/ResembleAI/Dramabox) for upstream license and usage details.

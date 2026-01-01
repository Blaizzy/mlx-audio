# MossFormer2 SE - Speech Enhancement for MLX

MossFormer2 SE is a speech enhancement model based on the MossFormer2 architecture from 
[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio). This MLX port
enables efficient inference on Apple Silicon.

## Features

- 🎯 **48kHz sample rate** for high-quality audio
- ⚡ **Apple Silicon optimized** with Metal acceleration
- 🔄 **Auto-chunking** for long audio (>60s) with low RAM usage
- 📦 **Multiple precision options**: fp32, fp16, int8, int6, int4

## Quick Start

```python
from mlx_audio.sts.models.mossformer2_se import (
    MossFormer2SEModel,
    save_audio,
)

# Load model
model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")

# Enhance audio
enhanced = model.enhance("noisy.wav")

# Save result
save_audio(enhanced, "enhanced.wav", 48000)
```

## Processing Modes

### Automatic (default)
```python
# Automatically selects mode based on duration
# < 60s: Full mode (faster, best quality)
# >= 60s: Chunked mode (lower RAM)
enhanced = model.enhance("audio.wav")
```

### Force Chunked Mode
```python
# For very long audio or limited RAM
enhanced = model.enhance("long_audio.wav", chunked=True)
```

### Force Full Mode
```python
# For best quality (if RAM allows)
enhanced = model.enhance("audio.wav", chunked=False)
```

## Precision Options

Precision is inferred from the repo name suffix:

```python
# Full precision (default)
model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")

# Quantized versions
model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX-4bit")
model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX-8bit")
```

## Model Architecture

| Component | Parameters |
|-----------|------------|
| Total | ~55.3M |
| MaskNet | 24 transformer blocks |
| Features | 180 (60 mel + 60 delta + 60 delta-delta) |
| Output | 961 frequency bins |

## License

- Original model: [Apache License 2.0](https://github.com/modelscope/ClearerVoice-Studio/blob/main/LICENSE)
- MLX port: Apache License 2.0

## Citation

```bibtex
@article{zhao2023mossformer2,
  title={MossFormer2: Combining Transformer and RNN-Free Recurrent Network 
         for Enhanced Time-Domain Monaural Speech Separation},
  author={Zhao, Shengkui and Ma, Yukun and others},
  journal={arXiv preprint arXiv:2312.11825},
  year={2023}
}
```

## Acknowledgments

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) - Speech Lab, Alibaba Group
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Prince Canuma

---
license: apache-2.0
datasets:
- amphion/Emilia-Dataset
language:
- en
base_model:
- Marvis-AI/marvis-tts-250m-v0.1-base
library_name: transformers
tags:
- mlx
- mlx-audio
- transformers
- mlx
---

# Marvis-AI/marvis-tts-250m-v0.1-MLX-4bit
This model was converted to MLX format from [`Marvis-AI/marvis-tts-250m-v0.1`](https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.1) using mlx-audio version **0.2.5**.
Refer to the [original model card](https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.1) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-audio
```

```bash
python -m mlx_audio.tts.generate --model Marvis-AI/marvis-tts-250m-v0.1-MLX-4bit --text "Describe this image."
```

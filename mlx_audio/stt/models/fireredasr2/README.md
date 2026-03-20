# FireRedASR2-AED

MLX implementation of Xiaohongshu's FireRedASR2-AED, a conformer encoder + transformer decoder model for Chinese and English speech recognition.

## Model

| Model | Parameters | Description |
|-------|------------|-------------|
| FireRedASR2-AED | ~1.18B | Conformer-AED, Mandarin + English + Chinese dialects |

Converted weights are available on Hugging Face at [`mlx-community/FireRedASR2-AED-mlx`](https://huggingface.co/mlx-community/FireRedASR2-AED-mlx).

## Python Usage

```python
from mlx_audio.stt import load

model = load("mlx-community/FireRedASR2-AED-mlx")

result = model.generate("audio.wav")
print(result.text)

# with custom beam search parameters
result = model.generate("audio.wav", beam_size=5, softmax_smoothing=1.25, length_penalty=0.6)
```

## Architecture

- 80-dim Kaldi fbank frontend with CMVN normalization
- Conv2d subsampling (4x temporal downsampling)
- 16 layer conformer encoder with relative positional attention (Transformer-XL style)
- 16 layer transformer decoder with cross attention and GELU FFN
- Beam search decoding with GNMT length penalty
- Hybrid tokenizer: Chinese characters + English BPE (SentencePiece, ~8.7k vocab)
- 1280 hidden dim, 20 attention heads, 5120 FFN dim

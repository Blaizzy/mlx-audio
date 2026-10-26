# IndexTTS2 / IndexTTS-2.5

Native MLX support for Bilibili's IndexTTS family. One model package loads both
generations; the converted checkpoint's `version` selects the matching runtime
path.

| Version | Languages | Conditioning | Pace control |
|---|---|---|---|
| IndexTTS2 (`2.0`) | ZH, EN | Conformer speaker conditioning + GPT latent projection | waveform speed adjustment |
| IndexTTS-2.5 (`2.5`) | ZH, EN, JA, ES, AR | CAMPPlus speaker embedding + language embedding | latent `duration_factor` |

Both versions support zero-shot voice cloning from a short reference clip and
the same 22.05 kHz BigVGAN vocoder.

## Convert

Install conversion dependencies first:

```bash
pip install -e ".[convert]"
```

Convert IndexTTS-2.5:

```bash
python -m mlx_audio.tts.models.indextts2.convert_unifiedvoice \
  --version 2.5 --hf-repo IndexTeam/IndexTTS-2.5 --output-dir ./IndexTTS-2.5-MLX
python -m mlx_audio.tts.models.indextts2.convert_s2mel \
  --version 2.5 --hf-repo IndexTeam/IndexTTS-2.5 --output-dir ./IndexTTS-2.5-MLX
python -m mlx_audio.tts.models.indextts2.convert_maskgct_semantic_codec \
  --version 2.5 --hf-repo IndexTeam/IndexTTS-2.5 --output-dir ./IndexTTS-2.5-MLX
python -m mlx_audio.tts.models.indextts2.convert_bigvgan \
  --output-dir ./IndexTTS-2.5-MLX
python -m mlx_audio.tts.models.indextts2.convert_campplus \
  --output-dir ./IndexTTS-2.5-MLX
python -m mlx_audio.tts.models.indextts2.convert_w2vbert \
  --output-dir ./IndexTTS-2.5-MLX
python -m mlx_audio.tts.models.indextts2.convert_w2vbert_stats \
  --stats-pt ./IndexTTS-2.5/checkpoint/wav2vec2bert_stats.pt \
  --output-dir ./IndexTTS-2.5-MLX
python -m mlx_audio.tts.models.indextts2.convert_emotion_matrices \
  --model-dir ./IndexTTS-2.5 --output-dir ./IndexTTS-2.5-MLX
```

For IndexTTS2, use `--version 2.0 --hf-repo IndexTeam/IndexTTS-2`. The
conversion scripts merge their component metadata into one `config.json`;
`version`, `spk_cond_mode`, `tokenizer`, and `use_gpt_latent` are inferred when
the checkpoint is loaded.

## Generate

```python
from mlx_audio.tts import load

model = load("./IndexTTS-2.5-MLX", strict=True)
result = next(
    model.generate(
        "Hello from IndexTTS-2.5 on Apple Silicon.",
        ref_audio="reference.wav",
        language="en",
        duration_factor=1.0,
    )
)
```

`duration_factor > 1.0` produces slower speech and `< 1.0` produces faster
speech (valid range in the upstream model is 0.5–2.0). Emotion control uses
either an official 8-D vector in `[happy, angry, sad, afraid, disgusted,
melancholic, surprised, calm]` order or text analyzed by the bundled Qwen3
emotion model:

```python
# Explicit emotion vector. emo_alpha scales the whole vector.
model.generate(
    "I am so happy to see you!",
    ref_audio="reference.wav",
    language="en",
    emo_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    emo_alpha=0.8,
)

# Text-driven emotion. It uses emo_text when provided, otherwise the main text.
model.generate(
    "I am so happy to see you!",
    ref_audio="reference.wav",
    language="en",
    use_emo_text=True,
    emo_text="非常开心、激动，终于重逢",
    emo_alpha=0.8,
)
```

Emotion control requires `feat1.pt`/`feat2.pt` to be converted into
`emotion_matrices.safetensors`; text emotion additionally requires the
`qwen0.6bemo4-merge` directory from the official 2.5 checkpoint.

Pronunciation overrides use the upstream syntax:

```python
model.generate(
    "行<行|XING2>走，<going|G OW1 . IH0 NG> home.",
    ref_audio="reference.wav",
    language="zh",
)
```

## Validation

The 2.5 path was smoke-tested on Apple Silicon with the existing
`vanch007/mlx-indextts2-2.5-fp16` MLX export (2.0 GB):

- Strict model load: pass
- Text: `Hello from IndexTTS two point five on MLX Audio.`
- Output: 3.51 s at 22.05 kHz, all samples finite, peak 0.94
- `duration_factor=0.5`: 1.393 s
- `duration_factor=1.5`: 4.249 s
- Explicit 8-D emotion vector (`happy=1.0`, `emo_alpha=0.8`): 2.752 s, finite
- Qwen3 text emotion ("非常开心、激动，终于重逢"): 2.786 s, finite
- Unit/regression tests: 20 passed

IndexTTS2 (`2.0`) keeps the original PR's conformer + GPT-latent path, but no
2.0 converted checkpoint was available for an end-to-end rerun in this pass.

## Current limits

- No `mlx-community` converted checkpoint is published yet; convert the official
  PyTorch weights locally.
- Streaming generation is not implemented.
- Emotion control needs the converted `emotion_matrices.safetensors`; text
  emotion additionally needs the Qwen3 emotion checkpoint.
- IndexTTS-2.5 requires `tiktoken`; IndexTTS2 uses the SentencePiece
  `bpe.model`.

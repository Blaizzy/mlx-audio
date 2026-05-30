# Mel-Band-RoFormer

Mel-Band-RoFormer is a transformer-based architecture for audio source separation,
particularly effective for vocal isolation from music. This is an MLX port of the
architecture described in Lu et al., "Mel-Band RoFormer for Music Source Separation"
(2024), with reference implementations in
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) and
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).

## Architecture

- **Input**: Stereo audio at 44.1 kHz (`[B, 2, samples]`)
- **Pipeline**: STFT → CaC interleave → BandSplit → N× DualAxisTransformer → MaskEstimate → complex multiply → iSTFT
- **Output**: Separated stem (e.g., vocals) as stereo audio (`[B, 2, samples]`)

Key design elements:
- **Mel-scale band splitting** into 60 frequency bands (Slaney mel scale, binarized)
- **Dual-axis attention** — alternating time-axis and frequency-axis RoPE transformers
- **Per-head sigmoid gates** on attention outputs
- **Per-band MLP mask estimation** with GLU activation
- **Complex-valued masking** via real/imaginary interleaved representation

## License

Architecture code: MIT (inherited from
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) and
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)).

Pre-trained checkpoints have independent licenses; consult the source repository
of any checkpoint you load.

## Usage

### Loading a model

Each preset matches a specific checkpoint family's architecture. You must
pick the preset that matches the checkpoint you intend to load.

```python
from mlx_audio.sts.models.mel_roformer import MelRoFormer, MelRoFormerConfig

# Pick the config preset matching your checkpoint family
config = MelRoFormerConfig.kim_vocal_2()           # depth=6
# or:
config = MelRoFormerConfig.viperx_vocals()         # depth=12
# or:
config = MelRoFormerConfig.zfturbo_bs_roformer()   # depth=12
# or for non-standard variants:
config = MelRoFormerConfig.custom(depth=8, num_bands=48)

# Load from a local directory (must contain a matching safetensors file)
model = MelRoFormer.from_pretrained("./my_weights_dir", config=config)

# Or from a HuggingFace repo ID (if the repo contains a compatible
# `<name>.config.json` produced by convert.py, the config is loaded from there)
model = MelRoFormer.from_pretrained("your-org/mel-band-roformer-mlx")
```

### Running separation

```python
import mlx.core as mx
from mlx_audio.sts.models.mel_roformer import MelRoFormer, MelRoFormerConfig

model = MelRoFormer.from_pretrained("path/to/weights", config=MelRoFormerConfig.kim_vocal_2())

# Stereo audio at 44.1 kHz — shape [1, 2, samples]
audio = mx.random.normal((1, 2, 44100 * 8))  # 8 seconds of dummy audio

# Separate vocals
vocals = model(audio)
# vocals has shape [1, 2, samples]
```

For audio longer than `chunk_size` (default 8 seconds), split into overlapping
chunks and apply crossfade reconstruction — a convenience wrapper for this is
planned but not yet in-tree.

## Converting PyTorch Checkpoints

If you have a PyTorch `.ckpt` or `.pt` checkpoint from MSS-Training or a
community source, convert it with:

```bash
python -m mlx_audio.sts.models.mel_roformer.convert \
    --input path/to/checkpoint.ckpt \
    --output path/to/output_dir/ \
    --dtype bfloat16   # or float16 / float32
```

The script:
- Accepts `.ckpt`, `.pt`, or `.safetensors` input
- Writes a content-addressed output file (`<basename>.<sha256[:8]>.<dtype>.safetensors`)
  and companion `<basename>.<sha256[:8]>.<dtype>.config.json`
- Skips if the output already exists (idempotent re-runs are free)
- Splits packed `to_qkv` projections into separate `to_q`, `to_k`, `to_v` weights
- Strips training state (optimizer, schedulers, EMA, running stats)

## References

- Lu et al., "Mel-Band RoFormer for Music Source Separation" (2024)
- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) — original PyTorch (MIT)
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — training framework and config files (MIT)

## Credits

Architecture by the paper authors at ByteDance AI Labs.
Reference implementations by lucidrains and ZFTurbo.

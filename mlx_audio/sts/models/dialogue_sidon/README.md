# DialogueSidon

Separate a recording of two speakers into individual audio tracks on Apple
Silicon. DialogueSidon also restores degraded speech and outputs two mono WAV
files at **24 kHz**, with the same duration as the input.

Follow the [MLX Audio installation instructions](../../../../README.md#installation)
before running the examples below.

## Supported models

| Precision | Repository ID | Weight size |
|-----------|---------------|-------------|
| FP32 | [mlx-community/DialogueSidon](https://huggingface.co/mlx-community/DialogueSidon) | 1.78 GB |
| BF16 | [mlx-community/DialogueSidon-bf16](https://huggingface.co/mlx-community/DialogueSidon-bf16) | 889 MB |

Either repository ID works in the examples below. Choose BF16 for a smaller
download and lower weight memory use. You can also load a local model directory.

## Command line

```bash
python -m mlx_audio.sts.generate \
    --model mlx-community/DialogueSidon \
    --audio dialogue.wav --output-path separated.wav \
    --num-steps 30 --seed 0
```

This writes `separated_speaker_1.wav` and `separated_speaker_2.wav`.

## Python

```python
from mlx_audio.sts import load
from mlx_audio.audio_io import write

model = load("mlx-community/DialogueSidon")
result = model.separate("dialogue.wav", num_steps=30, seed=0)
for i, speaker in enumerate(result.speakers, 1):
    write(f"speaker_{i}.wav", speaker, result.sample_rate)
```

`result.speakers` contains two tracks with shape `[2, samples]`, and
`result.sample_rate` is `24000`.

For a NumPy or MLX array, provide its sample rate:

```python
from mlx_audio.audio_io import read

audio, sample_rate = read("dialogue.wav")
result = model.separate(audio, sample_rate=sample_rate, seed=0)
```

Arrays must use `[samples]` or `[samples, channels]` layout. Stereo and other
multichannel inputs are averaged to mono before separation.

## Options and long recordings

| Python option | CLI option | Default | Purpose |
|---------------|------------|---------|---------|
| `num_steps` | `--num-steps` | `30` | Fewer steps run faster but can change output quality. |
| `seed` | `--seed` | Random | Set a seed to repeat a run with the same input, model, and settings. |
| `chunk_seconds` | `--chunk-seconds` | `20.0` | Audio processed at once; smaller chunks reduce memory use. |
| `overlap_seconds` | `--overlap-seconds` | `5.0` | Overlap between chunks, in seconds. Must be less than the chunk duration. |

Long recordings are processed automatically in overlapping chunks. To process
an entire file at once in Python, use `chunk_seconds=None`; this uses more
memory.

The model supports exactly two speakers and processes recorded audio offline.
The tracks are numbered rather than labeled with speaker identities; a speaker
may switch tracks after a long silence. Because the model also restores speech,
adding the tracks together may not reproduce the input exactly.

## License and attribution

Model weights use **CC-BY-NC-4.0**. Original model by Wataru Nakata, Yuki Saito,
Kazuki Yamauchi, Emiru Tsunoo, and Hiroshi Saruwatari (SaruLab).
See the [original model card](https://huggingface.co/sarulab-speech/DialogueSidon).
The upstream [Sidon code](https://github.com/sarulab-speech/Sidon) is MIT licensed.

# MOSS-Music

MOSS-Music support for `mlx-audio` lives in the STT bucket because lyrics ASR is
one of the model's primary tasks. The model is prompt-driven audio-text-to-text,
so it can also answer music captioning, key/tempo, chord, structure, and
general music QA prompts.

## Supported Models

| Repo | Precision | Size |
| ------ | ----------- | ------ |
| [`OpenMOSS-Team/MOSS-Music-8B-Thinking`](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking) | bf16 | ~16.9 GiB |
| [`OpenMOSS-Team/MOSS-Music-8B-Instruct`](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) | bf16 | ~16.9 GiB |
| [`mlx-community/MOSS-Music-8B-Thinking-8bit`](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-8bit) | 8-bit | ~9.5 GiB |
| [`mlx-community/MOSS-Music-8B-Thinking-6bit`](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-6bit) | 6-bit | ~7.6 GiB |
| [`mlx-community/MOSS-Music-8B-Thinking-4bit`](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-4bit) | 4-bit | ~5.6 GiB |

## Python Usage

```python
from mlx_audio.stt import load

model = load("mlx-community/MOSS-Music-8B-Thinking-8bit")

result = model.generate("song.wav")
print(result.text)
```

The default prompt is:

```text
Please give a detailed musical description of this clip.
```

Use `prompt` for transcription or music QA:

```python
result = model.generate(
    "song.wav",
    prompt="Transcribe the lyrics of this song.",
)
print(result.text)
```

## CLI Usage

```bash
mlx_audio.stt.generate \
  --model mlx-community/MOSS-Music-8B-Thinking-8bit \
  --audio song.wav \
  --output-path output \
  --prompt "Please give a detailed musical description of this clip."
```

## Notes

- Time markers are enabled by default to match the upstream serving path. In
  Python, pass `enable_time_marker=False` to `generate()` to disable them for a
  single call.
- `result.text` contains the decoded model response. `result.segments` is
  structured when the model emits timestamp markers such as `[00:12] lyric` or
  `00:12: lyric`; otherwise it falls back to a single full-text segment spanning
  the input audio.
- For timestamped JSON/SRT/VTT output, prompt the model to emit timestamped
  lines and use `--format json`, `--format srt`, or `--format vtt` in the CLI.
- The Thinking checkpoint may emit `<think>...</think>` reasoning. mlx-audio
  strips those blocks by default; pass `strip_thinking=False` in Python to keep
  raw output.

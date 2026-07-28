# Fun-ASR-Nano-2512

MLX implementation of FunAudioLLM's Fun-ASR-Nano-2512, a compact speech recognition model with a SenseVoice-style audio encoder and Qwen3-0.6B text decoder.

## Supported Models

| Model | Parameters | Languages | Description |
|-------|------------|-----------|-------------|
| [mlx-community/Fun-ASR-Nano-2512](https://huggingface.co/mlx-community/Fun-ASR-Nano-2512) | 0.8B | ZH, EN, JA | Transcription-only Fun-ASR-Nano checkpoint |

## Python Usage

```python
from mlx_audio.stt import load

model = load("mlx-community/Fun-ASR-Nano-2512")

result = model.generate(
    "audio.wav",
    language="zh",
    hotwords=["开放时间"],
)
print(result.text)
```

## CLI Usage

```bash
mlx_audio.stt.generate \
  --model mlx-community/Fun-ASR-Nano-2512 \
  --audio audio.wav \
  --output-path transcript \
  --language zh \
  --context "开放时间, 地址"
```

## Options

Language hints use ISO-style codes. Supported hints include `zh`, Chinese dialect codes that map to the Chinese prompt (`yue`, `wuu`, `nan`, `hak`, `gan`, `hsn`, `cjy`), `en`, and `ja`. Pass `None` or `auto` to omit the language constraint.

Hotwords can be supplied as a list of domain terms:

```python
result = model.generate("audio.wav", language="zh", hotwords=["开放时间", "地址"])
```

The shared CLI and OpenAI-compatible server API expose contextual biasing as a
single `context` string. It is an alias for `hotwords`; non-empty values are
mutually exclusive.

Inverse text normalization is enabled by default. Set `itn=False` to ask the model to keep spoken-form text:

```python
result = model.generate("audio.wav", language="zh", itn=False)
```

## Notes

This runtime is transcription-only. The public checkpoint does not include timestamp head weights, so segment and word timestamps are not emitted by this model.

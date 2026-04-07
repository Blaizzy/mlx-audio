# Adding a New Model

## Directory layout

Place the model under the appropriate category:

```
mlx_audio/
├── tts/models/<model_name>/   # Text-to-speech
├── stt/models/<model_name>/   # Speech-to-text
├── sts/models/<model_name>/   # Speech-to-speech / enhancement
└── codec/models/<model_name>/ # Standalone audio codecs / tokenizers
```

Minimum required files:

```
mlx_audio/tts/models/my_model/
├── __init__.py      # must export Model and ModelConfig
├── my_model.py      # Model class + ModelConfig dataclass
└── convert.py       # weight conversion script (PyTorch → safetensors)
```

## Auto-discovery

**No registration needed.** The loader resolves models dynamically:

1. Reads `"model_type"` from `config.json` in the model directory.
2. Imports `mlx_audio.{tts|stt|sts}.models.{model_type}`.
3. Instantiates `module.ModelConfig.from_dict(config)` then `module.Model(config)`.

So `model_type` in `config.json` **must match the directory name exactly**.

## ModelConfig

```python
from dataclasses import dataclass
from mlx_audio.tts.models.base import BaseModelArgs

@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "my_model"  # must match directory name
    sample_rate: int = 24000      # output audio sample rate

    # model-specific fields …

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(
            model_type=config.get("model_type", "my_model"),
            sample_rate=config.get("sample_rate", 24000),
            # …
        )
```

## Model class

```python
import mlx.nn as nn
import mlx.core as mx
from typing import Generator, Optional

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # define layers …

    @property
    def sample_rate(self) -> int:      # required
        return self.config.sample_rate

    @property
    def model_type(self) -> str:       # recommended
        return self.config.model_type

    def sanitize(self, weights: dict) -> dict:
        """Rename / reshape PyTorch keys to match MLX attribute paths.

        Common transforms:
        - Strip "model." prefix
        - Conv1d weights: PyTorch (out, in, k) — MLX loads as (out, k, in),
          so no manual transpose is needed; mlx.load_weights handles it.
        - LayerNorm: .gamma → .weight, .beta → .bias
        - Drop unused keys (position_ids, etc.)
        """
        result = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[4:]
            result[k] = v
        return result

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "en",
        temperature: float = 0.7,
        max_tokens: int = 1200,
        **kwargs,   # absorb unused args from generate_audio()
    ) -> Generator["GenerationResult", None, None]:
        import time
        from mlx_audio.tts.models.base import GenerationResult

        start = time.time()
        audio = self._run_inference(text)   # mx.array [samples]
        elapsed = time.time() - start

        n = int(audio.shape[0])
        dur = n / self.sample_rate
        d = int(dur)

        yield GenerationResult(
            audio=audio,
            samples=n,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=0,
            audio_duration=f"{d//3600:02d}:{(d%3600)//60:02d}:{d%60:02d}.{int((dur%1)*1000):03d}",
            real_time_factor=dur / elapsed if elapsed > 0 else 0.0,
            prompt={"tokens": 0, "tokens-per-sec": 0},
            audio_samples={"samples": n, "samples-per-sec": round(n / elapsed, 2)},
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    @staticmethod
    def post_load_hook(model: "Model", model_path) -> "Model":
        """Optional. Called by the loader after weights are applied.
        Use to load auxiliary tokenizers or preprocessors."""
        return model
```

## `generate()` — kwargs passed by the CLI

`generate_audio()` always passes these kwargs; use `**kwargs` to absorb any
you don't need:

| kwarg | type | description |
|---|---|---|
| `text` | `str` | input text |
| `voice` | `str \| None` | speaker / voice ID |
| `speed` | `float` | playback speed multiplier |
| `lang_code` | `str` | BCP-47 language code |
| `temperature` | `float` | sampling temperature |
| `max_tokens` | `int` | token budget |
| `ref_audio` | `mx.array \| None` | reference waveform for voice cloning |
| `ref_text` | `str \| None` | transcript of reference audio |
| `cfg_scale` | `float \| None` | classifier-free guidance strength |
| `instruct` | `str \| None` | style / emotion instruction |
| `stream` | `bool` | streaming mode |

## Weight conversion (`convert.py`)

```python
# convert.py — minimal pattern
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
import torch

def convert(pt_path: str, output_dir: str, dtype: str = "bfloat16"):
    mlx_dtype = {"float16": np.float16, "bfloat16": np.float32}[dtype]
    weights = torch.load(pt_path, map_location="cpu")
    out = {}
    for k, v in weights.items():
        arr = v.detach().cpu().numpy().astype(mlx_dtype)
        out[k] = arr
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_file(out, f"{output_dir}/model.safetensors")
```

For key renaming and shape fixes, implement `Model.sanitize()` — the loader
calls it automatically after reading the safetensors file.

## Acoustic codecs / tokenizers

If your model needs an audio codec (encode waveform → tokens or decode tokens
→ waveform), add it under `mlx_audio/codec/models/<codec_name>/` and export
from `mlx_audio/codec/__init__.py`. Reference it from your TTS/STT model by
import — do not bundle codec weights inside the TTS model directory.

## Tests

Add tests under `mlx_audio/tts/tests/test_<model_name>.py` (or the equivalent
category). Tests should use random MLX weights — no real checkpoint required:

```python
import unittest
import mlx.core as mx
from mlx_audio.tts.models.my_model import Model, ModelConfig
from mlx_audio.tts.models.base import GenerationResult

class TestMyModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(ModelConfig())

    def test_sample_rate(self):
        self.assertEqual(self.model.sample_rate, 24000)

    def test_generate_yields_result(self):
        results = list(self.model.generate("Hello"))
        self.assertIsInstance(results[0], GenerationResult)
        self.assertGreater(results[0].samples, 0)
```

## Publishing weights to mlx-community

Model weights must be published to the
[mlx-community](https://huggingface.co/mlx-community) HuggingFace organization,
not bundled in this repository. The only exception is when an existing
mlx-community model needs to be updated and the PR is waiting for approval —
in that case a temporary personal fork is acceptable.

### Naming convention

```
mlx-community/<ModelName>[-<Variant>]-<ParameterCount>-<Dtype>
```

| part | description | examples |
|---|---|---|
| `ModelName` | base model name, preserve original casing | `Kokoro`, `Qwen3-TTS`, `Voxtral` |
| `Variant` | optional variant tag | `Base`, `VoiceDesign`, `Realtime` |
| `ParameterCount` | size indicator | `82M`, `0.6B`, `1B`, `4B` |
| `Dtype` | precision / quantization level | `bf16`, `fp16`, `8bit`, `4bit`, `6bit` |

Real examples from mlx-community:

```
mlx-community/Kokoro-82M-bf16
mlx-community/Kokoro-82M-4bit
mlx-community/OuteTTS-1.0-0.6B-fp16
mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
mlx-community/Voxtral-4B-TTS-2603-mlx-bf16
mlx-community/chatterbox-fp16
mlx-community/LongCat-AudioDiT-1B-bf16
mlx-community/whisper-large-v3-turbo-asr-fp16
mlx-community/parakeet-tdt-0.6b-v3
```

**Notes:**
- Prefer `bf16` as the primary upload; add quantized variants (`4bit`, `8bit`) if
  the model is large enough to benefit.
- Include the parameter count when the model family has multiple sizes.
- Do not add an `-mlx` suffix unless the upstream name already contains it.
- Link the mlx-community repo in your PR description so reviewers can verify
  the weights are accessible.

## PR checklist

- [ ] `config.json` has `"model_type"` matching the directory name
- [ ] `__init__.py` exports `Model` and `ModelConfig`
- [ ] `ModelConfig` is a `@dataclass` with `from_dict()` and `sample_rate`
- [ ] `Model.generate()` yields `GenerationResult` and accepts `**kwargs`
- [ ] `Model.sanitize()` covers all key renames / shape fixes
- [ ] `convert.py` produces a loadable `model.safetensors`
- [ ] Tests pass with random weights (no real checkpoint needed)
- [ ] Model listed in `README.md` model table

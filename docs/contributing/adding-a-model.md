# Adding a Model

This guide explains how to add a new TTS or STT model to MLX Audio. The library uses a consistent pattern across all models, so once you understand the structure, adding a new one is straightforward.

## Directory Structure

Every model lives in its own package under the appropriate category directory:

```
mlx_audio/
├── tts/models/
│   ├── base.py              # Shared base classes
│   ├── __init__.py
│   ├── kokoro/              # Example: Kokoro TTS
│   │   ├── __init__.py      # Exports Model, ModelConfig, Pipeline
│   │   ├── kokoro.py        # Model implementation
│   │   └── pipeline.py      # Generation pipeline
│   ├── your_model/          # Your new model
│   │   ├── __init__.py
│   │   ├── your_model.py
│   │   └── ...
├── stt/models/
│   ├── whisper/
│   ├── parakeet/
│   └── your_model/
```

## Step 1: Create the Model Package

Create a new directory under `mlx_audio/tts/models/` (or `stt/models/` for STT):

```
mlx_audio/tts/models/my_model/
├── __init__.py
├── my_model.py
└── README.md          # Optional but recommended
```

## Step 2: Implement the Model

### Base Classes

TTS models use the base classes from `mlx_audio/tts/models/base.py`:

- **`BaseModelArgs`** -- Dataclass for model configuration. Includes a `from_dict()` class method that filters unknown keys automatically.
- **`GenerationResult`** -- Dataclass returned by `generate()`. Contains `audio`, `sample_rate`, `token_count`, timing information, and streaming flags.
- **`BatchGenerationResult`** -- Dataclass for batch generation results.

### Transformer Components

Many speech models wrap a language-model backbone. Import that machinery from
`mlx_audio.lm`, **not** from `mlx-lm` -- mlx-audio vendors it so the package has
no runtime dependency on mlx-lm, and a test fails the build if `mlx_lm` is
imported anywhere outside `sts/voice_pipeline.py`.

| Need | Import from |
|---|---|
| KV caches (`KVCache`, `RotatingKVCache`, `BatchKVCache`, `make_prompt_cache`) | `mlx_audio.lm.models.cache` |
| Attention masks, `scaled_dot_product_attention` | `mlx_audio.lm.models.base` |
| Samplers, logits processors | `mlx_audio.lm.sample_utils` |
| `generate_step`, `stream_generate` | `mlx_audio.lm.generate` |
| Backbones (llama, qwen2, qwen3, gpt2, granite, lfm2, bailing_moe, gemma3) | `mlx_audio.lm.models.<name>` |
| Quantize / save helpers | `mlx_audio.lm.convert` |

If your model needs a backbone that is not vendored yet, copy it from mlx-lm
into `mlx_audio/lm/models/` verbatim and add the provenance header used by the
other files there (upstream path, version, commit).

!!! warning "Size KV caches for prefill *plus* generation"
    When your generation loop writes into a fixed-size KV cache, allocate it
    for the prompt/prefill tokens **and** the generated tokens combined --
    not just `max_tokens`. If an audio (or text) prompt is prefilled first,
    those slots are consumed before decoding starts, and undersized caches
    fail with an index/assertion error mid-generation. A regression test that
    runs one generation step past the prefill length catches this class of
    bug; see `mlx_audio/tts/tests/test_dia.py`.

### Model Configuration

Create a dataclass for your model's config that extends `BaseModelArgs`:

```python
from dataclasses import dataclass
from ..base import BaseModelArgs

@dataclass
class MyModelConfig(BaseModelArgs):
    hidden_size: int = 512
    num_layers: int = 6
    vocab_size: int = 32000
    sample_rate: int = 24000
    model_type: str = "my_model"
```

### Model Class

Your model should be an `mlx.nn.Module` with a `generate()` method that yields `GenerationResult` objects:

```python
import mlx.nn as nn
from ..base import GenerationResult

class MyModel(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.model_type = config.model_type
        # Build layers ...

    def generate(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        lang_code: str = "en",
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ):
        """Generate speech from text.

        Yields:
            GenerationResult for each audio segment.
        """
        # Your generation logic here
        audio = self._synthesize(text)

        yield GenerationResult(
            audio=audio,
            samples=audio.shape[0],
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=0,
            audio_duration="0.0s",
            real_time_factor=0.0,
            prompt={},
            audio_samples={},
            processing_time_seconds=0.0,
            peak_memory_usage=0.0,
        )
```

### `__init__.py`

Export the model class and config so the loader can find them:

```python
from .my_model import MyModel as Model, MyModelConfig as ModelConfig

__all__ = ["Model", "ModelConfig"]
```

!!! important "Naming convention"
    The loader expects `Model` and `ModelConfig` (or `ModelArgs`) to be importable from the package. Use these exact names in your exports.

## Step 3: Register the Model

### Config File

The model type is detected from the `config.json` file in the HuggingFace repo. The `model_type` field must match your directory name (or a key in the remapping dict).

If the `model_type` in `config.json` differs from your directory name, add an entry to the `MODEL_REMAPPING` dict in the appropriate utils file:

=== "TTS"

    ```python
    # mlx_audio/tts/utils.py
    MODEL_REMAPPING = {
        # ... existing entries ...
        "my_model": "my_model",  # config.json model_type -> directory name
    }
    ```

=== "STT"

    ```python
    # mlx_audio/stt/utils.py
    MODEL_REMAPPING = {
        # ... existing entries ...
        "my_model": "my_model",
    }
    ```

### How Model Loading Works

1. `load()` downloads the model from HuggingFace (or uses a local path).
2. It reads `config.json` to determine `model_type`.
3. It looks up the model type in `MODEL_REMAPPING` (or uses the type directly).
4. It dynamically imports the corresponding package from `mlx_audio/tts/models/` (or `stt/models/`).
5. It instantiates `ModelConfig` from the config dict and `Model` from the config.
6. It loads the `.safetensors` weights into the model.

## Step 4: Convert and Test

### Convert Weights

If your model's original weights are in PyTorch format, use the conversion script:

```bash
python -m mlx_audio.convert \
    --hf-path original/my-model \
    --mlx-path ./my-model-bf16 \
    --dtype bfloat16
```

### Prevent Broken Conversions: `model_quant_predicate`

The converter quantizes every `nn.Linear` (and similar) layer whose last
dimension is a multiple of 64. That blanket policy is wrong for some
architectures and produces checkpoints that load fine but generate garbage.
Define a `model_quant_predicate` classmethod on your model when any of the
following apply:

- An `nn.Embedding` doubles as the tied output/logit head (quantizing it
  degrades generation logits).
- A sensitive head or projection drives discrete codebook sampling
  (e.g. residual/RVQ heads).
- Any other path where 4-bit weights are known to be lossy for this
  architecture.

```python
class Model(nn.Module):
    @classmethod
    def model_quant_predicate(cls, path: str, module) -> bool:
        # Keep embeddings and the fast residual path in full precision.
        return (
            not isinstance(module, nn.Embedding)
            and "fast_" not in path
        )
```

The predicate receives `(path, module)` and returns whether quantization is
allowed; it is combined with the converter's base requirements, so returning
`True` never forces quantization of an unsuitable tensor. See
`qwen3_tts`, `fish_qwen3_omni`, or `spark` for examples.

### Publish to Hugging Face

If you plan to share the converted model, prefer publishing it on the
[mlx-community](https://huggingface.co/mlx-community) organization on Hugging Face.
It is the shared home for ready-to-use MLX weights across projects like `mlx-lm`,
`mlx-vlm`, `mlx-swift-examples`, and `mlx-audio`, so publishing there keeps MLX-native
checkpoints discoverable in one place.

If you cannot publish directly to `mlx-community`, use your own namespace first and
link it from the docs. We should still encourage new model contributors to be part of
the `mlx-community` org when possible.

### Test

Prefer offline unit tests that build a tiny randomly-initialized model over
tests that download weights -- they run in seconds, work in CI without
network, and still exercise real generation code paths:

```python
def test_my_model():
    model = MyModel(tiny_config())  # small dims, random init, no downloads
    results = list(model.generate("Hello, world!"))
    assert len(results) > 0
    assert results[0].audio.shape[0] > 0
```

Stub out any component your model downloads at construction time (codec,
tokenizer) with a fake that returns correctly-shaped arrays; see
`mlx_audio/tts/tests/test_dia.py` or `test_qwen3_tts.py` for patterns, and
`mlx_audio/music/tests/test_minimax_music3.py` for a tiny end-to-end config.

If the model supports batch generation, also verify the result contract:
**exactly one result per input item**, even when an item produces no audio
(yield silence rather than skipping it) so streaming consumers never see a
sequence terminate without a final chunk.

Run it:

```bash
pytest mlx_audio/tts/tests/test_my_model.py
```

## Step 5: Add Documentation

1. Add a model page in `docs/models/tts/` (or `stt/`).
2. Add the page to the `nav` section in `mkdocs.yml`.
3. Optionally add a `README.md` inside your model directory for model-specific details.
4. If the model is published on Hugging Face, prefer an `mlx-community/...` repo when available and link that repo from the docs.
5. Make sure your PR includes the docs change. The docs workflow will fail if model files change without a matching docs update.

## Checklist

- [ ] Model package created under `mlx_audio/{tts,stt}/models/`
- [ ] `__init__.py` exports `Model` and `ModelConfig`
- [ ] `generate()` method yields `GenerationResult` objects
- [ ] Model type registered in `MODEL_REMAPPING` (if needed)
- [ ] KV caches sized for prefill + generation (if the model manages caches manually)
- [ ] `model_quant_predicate` defined if blanket quantization would corrupt the model
- [ ] Weights converted to MLX `.safetensors` format and verified after conversion
- [ ] Hugging Face repo chosen and linked in docs (`mlx-community/...` preferred)
- [ ] Offline unit test written and passing
- [ ] Documentation page added
- [ ] PR submitted with a clear description

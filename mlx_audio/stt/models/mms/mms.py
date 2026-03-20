import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import STTOutput
from ..wav2vec.wav2vec import ModelConfig, Wav2Vec2Model


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self._vocab = None
        self._processor = None

    @property
    def sample_rate(self) -> int:
        return 16000

    def __call__(self, input_values: mx.array) -> mx.array:
        outputs = self.wav2vec2(input_values)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits

    def _ctc_decode(self, logits: mx.array) -> List[List[int]]:
        predictions = mx.argmax(logits, axis=-1)
        batch_tokens = []
        for b in range(predictions.shape[0]):
            tokens = []
            prev = -1
            for t in range(predictions.shape[1]):
                token = int(predictions[b, t])
                if token != prev and token != 0:
                    tokens.append(token)
                prev = token
            batch_tokens.append(tokens)
        return batch_tokens

    def _tokens_to_text(self, tokens: List[int]) -> str:
        if self._processor is not None:
            return self._processor.decode(tokens)
        if self._vocab is None:
            return " ".join(str(t) for t in tokens)
        return "".join(self._vocab.get(t, "") for t in tokens).replace("|", " ")

    def generate(
        self,
        audio,
        *,
        verbose: bool = False,
        dtype: mx.Dtype = mx.float32,
        **kwargs,
    ) -> STTOutput:
        kwargs.pop("generation_stream", None)
        kwargs.pop("max_tokens", None)
        kwargs.pop("temperature", None)
        kwargs.pop("language", None)
        kwargs.pop("source_lang", None)
        kwargs.pop("target_lang", None)
        kwargs.pop("stream", None)

        start_time = time.time()

        if isinstance(audio, (str, Path)):
            from mlx_audio.stt.utils import load_audio

            audio = load_audio(str(audio), sr=self.sample_rate, dtype=dtype)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        if audio.ndim == 1:
            audio = audio[None, :]

        if audio.dtype != dtype:
            audio = audio.astype(dtype)

        audio = (audio - mx.mean(audio, axis=-1, keepdims=True)) / (
            mx.std(audio, axis=-1, keepdims=True) + 1e-7
        )

        logits = self(audio)
        mx.eval(logits)

        decoded = self._ctc_decode(logits)
        text = self._tokens_to_text(decoded[0])

        end_time = time.time()
        total_time = end_time - start_time

        if verbose:
            print(f"Text: {text}")

        return STTOutput(
            text=text.strip(),
            segments=[{"text": text.strip(), "start": 0.0, "end": 0.0}],
            total_time=total_time,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        for k, v in weights.items():
            if k.endswith(".conv.weight"):
                v = v.swapaxes(1, 2)
            if k.endswith(".conv.weight_v") or k.endswith(".conv.weight_g"):
                v = v.swapaxes(1, 2)
            if k.endswith(".parametrizations.weight.original0"):
                k = k.replace(".parametrizations.weight.original0", ".weight_g")
                v = v.swapaxes(1, 2)
            if k.endswith(".parametrizations.weight.original1"):
                k = k.replace(".parametrizations.weight.original1", ".weight_v")
                v = v.swapaxes(1, 2)
            if (
                k.startswith("quantizer.")
                or k.startswith("project_")
                or k == "masked_spec_embed"
            ):
                continue

            sanitized[k] = v
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model_path = Path(model_path)

        adapter_path = model_path / "adapter.eng.safetensors"
        if not adapter_path.exists():
            adapters = list(model_path.glob("adapter.*.safetensors"))
            if adapters:
                adapter_path = adapters[0]

        if adapter_path.exists():
            adapter_weights = mx.load(str(adapter_path))
            sanitized = model.sanitize(adapter_weights)
            model.load_weights(list(sanitized.items()), strict=False)

        vocab_path = model_path / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                vocab = json.load(f)
            if isinstance(next(iter(vocab.values())), dict):
                lang_vocab = vocab.get(
                    "eng", vocab.get("en", next(iter(vocab.values())))
                )
                model._vocab = {v: k for k, v in lang_vocab.items()}
            else:
                model._vocab = {v: k for k, v in vocab.items()}

        try:
            from transformers import AutoProcessor

            model._processor = AutoProcessor.from_pretrained(str(model_path))
        except Exception:
            model._processor = None
        return model

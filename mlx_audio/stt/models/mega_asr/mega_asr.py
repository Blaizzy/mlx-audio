from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

from .config import MegaASRConfig
from .convert_lora import LoraModule, load_lora_factors
from .lora import apply_deltas, remove_deltas
from .router import AudioQualityRouter


class Model:
    def __init__(self, config: MegaASRConfig):
        self.config = config

        self._asr = Qwen3ASRModel(config.to_qwen3_config())

        router_cfg = config.router_config or {}
        self._router = AudioQualityRouter(
            d_model=router_cfg.get("d_model", 256),
            nhead=router_cfg.get("nhead", 4),
            dim_feedforward=router_cfg.get("dim_feedforward", 1024),
            num_layers=router_cfg.get("num_layers", 1),
            n_mels=router_cfg.get("n_mels", 80),
            frontend_hidden_dim=router_cfg.get("frontend_hidden_dim", 128),
            classifier_hidden_dim=router_cfg.get("classifier_hidden_dim", 128),
            max_len=router_cfg.get("max_len", 850),
        )

        self._deltas: Dict[str, LoraModule] = {}
        self._lora_active: bool = False

    @property
    def sample_rate(self) -> int:
        return self._asr.sample_rate

    def make_cache(self):
        return self._asr.make_cache()

    def parameters(self):
        return self._asr.parameters()

    def eval(self):
        return self._asr.eval()

    def load_weights(self, weights, strict: bool = False):
        return self._asr.load_weights(weights, strict=strict)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return Qwen3ASRModel.sanitize(weights)

    def model_quant_predicate(self, path: str, module: nn.Module) -> bool:
        return self._asr.model_quant_predicate(path, module)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model_path = Path(model_path)

        model._asr = Qwen3ASRModel.post_load_hook(model._asr, model_path)

        router_path = model_path / model.config.router_weights
        if router_path.exists():
            router_weights = mx.load(str(router_path))
            if not isinstance(router_weights, dict):
                raise ValueError(f"expected a tensor mapping in {router_path!s}")
            model._router = AudioQualityRouter.from_converted(router_weights)
            model._router.eval()

        lora_path = model_path / model.config.lora_weights
        if lora_path.exists():
            model._deltas = load_lora_factors(lora_path)

        return model

    def _set_lora(self, want: bool) -> None:
        if want and not self._lora_active:
            apply_deltas(self._asr, self._deltas)
            self._lora_active = True
        elif not want and self._lora_active:
            remove_deltas(self._asr, self._deltas)
            self._lora_active = False

    def generate(self, audio, **kwargs):
        route = self._router.route(audio)
        self._set_lora(bool(route["use_lora"]))
        return self._asr.generate(audio, **kwargs)

    def stream_transcribe(self, audio, **kwargs):
        route = self._router.route(audio)
        self._set_lora(bool(route["use_lora"]))
        return self._asr.stream_transcribe(audio, **kwargs)

    def __getattr__(self, name: str):
        try:
            asr = self.__dict__["_asr"]
        except KeyError:
            raise AttributeError(name)
        return getattr(asr, name)

    def __call__(self, *args, **kwargs):
        return self._asr(*args, **kwargs)


ModelConfig = MegaASRConfig

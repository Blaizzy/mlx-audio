from dataclasses import dataclass, field

from mlx_audio.base import BaseModelArgs
from mlx_audio.lm.models.bailing_moe_v3 import ModelArgs


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "lingv3_tts"
    llm_config: dict = field(default_factory=dict)
    audio_config: dict = field(default_factory=dict)
    tts_config: dict = field(default_factory=dict)
    flow_config: dict = field(default_factory=dict)
    aggregator_config: dict = field(default_factory=dict)
    model_path: str | None = None

    def __post_init__(self):
        if isinstance(self.llm_config, dict):
            self.llm_config = ModelArgs.from_dict(self.llm_config)
        layout = self.tts_config.get("prompt_format", "legacy_antmmf")
        if layout not in {"legacy_antmmf", "base_v1"}:
            raise ValueError(f"Unsupported Ming Tiny prompt format: {layout}")
        if self.tts_config.get("decode_alignment", "old_shift") != "old_shift":
            raise ValueError("This Ming Tiny port requires old_shift decoding")
        if not self.tts_config.get("with_rml", True):
            raise ValueError("The released Ming Tiny checkpoint requires speaker slots")
        if self.tts_config.get("sampling_contract", "native_fp32") != "native_fp32":
            raise ValueError("This Ming Tiny port requires native_fp32 flow sampling")

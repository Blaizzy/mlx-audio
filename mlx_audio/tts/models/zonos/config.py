from dataclasses import dataclass, field
from typing import Literal

import torch


# https://github.com/state-spaces/mamba/blob//mamba_ssm/utils/generation.py#L18
@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: torch.Tensor | None = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


@dataclass
class BackboneConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 0
    n_layer: int = 16
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5


@dataclass
class PrefixConditionerConfig:
    conditioners: list[dict]
    projection: Literal["none", "linear", "mlp"]


@dataclass
class ZonosConfig:
    backbone: BackboneConfig
    prefix_conditioner: PrefixConditionerConfig
    eos_token_id: int = 1024
    masked_token_id: int = 1025
    pad_vocab_to_multiple_of: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "ZonosConfig":
        d = d.copy()
        backbone_config = BackboneConfig(**d.pop("backbone"))
        prefix_conditioner_config = PrefixConditionerConfig(**d.pop("prefix_conditioner"))
        config = cls(backbone_config, prefix_conditioner_config, **d)
        return config
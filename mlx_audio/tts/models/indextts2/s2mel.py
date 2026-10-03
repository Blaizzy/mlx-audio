from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.nn as nn

from .s2mel_dit import DiTConfig
from .s2mel_flow_matching import CFM
from .s2mel_length_regulator import InterpolateRegulator, InterpolateRegulatorConfig


@dataclass
class S2MelConfig:
    dit: DiTConfig
    length_regulator: InterpolateRegulatorConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "S2MelConfig":
        dit_d = d.get("DiT", {})
        wv_d = d.get("wavenet", {})
        dit = DiTConfig(
            hidden_dim=int(dit_d.get("hidden_dim", 512)),
            num_heads=int(dit_d.get("num_heads", 8)),
            depth=int(dit_d.get("depth", 13)),
            in_channels=int(dit_d.get("in_channels", 80)),
            content_dim=int(dit_d.get("content_dim", 512)),
            style_dim=int(d.get("style_encoder", {}).get("dim", 192)),
            is_causal=bool(dit_d.get("is_causal", False)),
            long_skip_connection=bool(dit_d.get("long_skip_connection", True)),
            uvit_skip_connection=bool(dit_d.get("uvit_skip_connection", True)),
            final_layer_type=str(dit_d.get("final_layer_type", "wavenet")),
            wavenet_hidden_dim=int(wv_d.get("hidden_dim", 512)),
            wavenet_num_layers=int(wv_d.get("num_layers", 8)),
            wavenet_kernel_size=int(wv_d.get("kernel_size", 5)),
            wavenet_dilation_rate=int(wv_d.get("dilation_rate", 1)),
        )

        lr_d = d.get("length_regulator", {})
        lr = InterpolateRegulatorConfig(
            channels=int(lr_d.get("channels", 512)),
            sampling_ratios=tuple(lr_d.get("sampling_ratios", [1, 1, 1, 1])),
            in_channels=int(lr_d.get("in_channels", 1024)),
            out_channels=int(lr_d.get("channels", 512)),
            groups=1,
        )
        return cls(dit=dit, length_regulator=lr)


class S2MelModel(nn.Module):
    def __init__(self, cfg: S2MelConfig):
        super().__init__()
        self.cfg = cfg

        self.cfm = CFM(cfg.dit)
        self.length_regulator = InterpolateRegulator(cfg.length_regulator)
        self.gpt_layer = [
            nn.Linear(1280, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1024),
        ]

    def project_gpt_latent(self, x):
        h = x
        for layer in self.gpt_layer:
            h = layer(h)
        return h

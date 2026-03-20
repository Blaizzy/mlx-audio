from dataclasses import dataclass, field


@dataclass
class EcapaTdnnConfig:
    input_size: int = 60
    channels: int = 1024
    embed_dim: int = 256
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    global_context: bool = False

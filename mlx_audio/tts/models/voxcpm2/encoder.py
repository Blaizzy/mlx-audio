import mlx.core as mx
import mlx.nn as nn

from .config import LMConfig
from .minicpm import MiniCPMModel


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: LMConfig, input_dim: int = 64):
        super().__init__()
        self.config = config

        self.special_token = mx.random.normal((1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)

        self.encoder = MiniCPMModel(config)

    def __call__(self, x):
        B, T, P, D = x.shape

        x = self.in_proj(x)  # (B, T, P, H)

        special_tokens = mx.broadcast_to(
            self.special_token, (B, T, 1, self.config.hidden_size)
        )

        x = mx.concatenate([special_tokens, x], axis=2)  # (B, T, P+1, H)

        x = x.reshape(B * T, P + 1, -1)

        outputs, _ = self.encoder(inputs_embeds=x, is_causal=False)

        cls_output = outputs[:, 0, :]  # (B*T, H)

        return cls_output.reshape(B, T, -1)

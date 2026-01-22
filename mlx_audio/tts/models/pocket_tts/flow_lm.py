from functools import partial
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from .conditioners import LUTConditioner, TokenizedText
from .config import FlowLMConfig
from .mlp import SimpleMLPAdaLN
from .transformer import StreamingTransformer

FlowNet2 = Callable[[mx.array, mx.array, mx.array], mx.array]


def lsd_decode(v_t: FlowNet2, x_0: mx.array, num_steps: int = 1) -> mx.array:
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps
        shape = x_0[..., :1].shape
        s_t = mx.full(shape, s, dtype=x_0.dtype)
        t_t = mx.full(shape, t, dtype=x_0.dtype)
        flow_dir = v_t(s_t, t_t, current)
        current = current + flow_dir / num_steps
    return current


class FlowLMModel(nn.Module):
    def __init__(
        self,
        conditioner: LUTConditioner,
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        dim: int = 128,
        ldim: int = 64,
        stats_ema_decay: float = 0.999,
        text_padding_weight: float = 1.0,
        dtype=None,
    ):
        super().__init__()
        self.conditioner = conditioner
        self.ldim = ldim
        self.stats_ema_decay = stats_ema_decay
        self.dim = dim
        self.text_padding_weight = text_padding_weight
        self.dtype = dtype

        self.flow_net = flow_net
        self.emb_std = mx.ones((ldim,), dtype=dtype or mx.float32)
        self.emb_mean = mx.zeros((ldim,), dtype=dtype or mx.float32)
        self.bos_emb = mx.random.normal((ldim,), dtype=dtype or mx.float32)

        self.input_linear = nn.Linear(self.ldim, dim, bias=False)
        self.transformer = transformer
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.out_eos = nn.Linear(dim, 1)

    def make_cache(self):
        return self.transformer.make_cache()

    def backbone(
        self, input_: mx.array, text_embeddings: mx.array, sequence: mx.array, cache
    ) -> mx.array:
        input_ = mx.concatenate([text_embeddings, input_], axis=1)
        transformer_out = self.transformer(input_, cache)
        if self.out_norm is not None:
            transformer_out = self.out_norm(transformer_out)
        return transformer_out[:, -sequence.shape[1] :]

    def __call__(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        cache,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[mx.array, mx.array]:
        bos = self.bos_emb[None, None, :]
        sequence = mx.where(mx.isnan(sequence), bos, sequence)
        input_ = self.input_linear(sequence)
        transformer_out = self.backbone(input_, text_embeddings, sequence, cache)
        transformer_out = transformer_out.astype(mx.float32)
        if lsd_decode_steps <= 0:
            raise ValueError("lsd_decode_steps must be > 0 for generation.")

        transformer_out = transformer_out[:, -1]
        out_eos = self.out_eos(transformer_out) > eos_threshold

        noise_shape = transformer_out.shape[:-1] + (self.ldim,)
        std = temp**0.5
        noise = mx.random.normal(shape=noise_shape, dtype=transformer_out.dtype) * std
        if noise_clamp is not None:
            noise = mx.clip(noise, -noise_clamp, noise_clamp)
        conditioned_flow = partial(self.flow_net, transformer_out)
        return lsd_decode(conditioned_flow, noise, lsd_decode_steps), out_eos

    def _sample_next_latent(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        cache,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[mx.array, mx.array]:
        return self(
            sequence=sequence,
            text_embeddings=text_embeddings,
            cache=cache,
            lsd_decode_steps=lsd_decode_steps,
            temp=temp,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        )

    @classmethod
    def from_config(cls, config: FlowLMConfig, latent_dim: int) -> "FlowLMModel":
        d_model = config.transformer.d_model
        flow_mlp = SimpleMLPAdaLN.from_pydantic_config(config, latent_dim, d_model)
        conditioner = LUTConditioner(
            n_bins=config.lookup_table.n_bins,
            tokenizer_path=str(config.lookup_table.tokenizer_path),
            dim=config.lookup_table.dim,
            output_dim=d_model,
        )
        transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            dim_feedforward=int(config.transformer.hidden_scale * d_model),
            max_period=float(config.transformer.max_period),
        )
        dtype = getattr(mx, config.dtype) if config.dtype else None
        return cls(
            flow_net=flow_mlp,
            transformer=transformer,
            dim=d_model,
            conditioner=conditioner,
            ldim=latent_dim,
            dtype=dtype,
        )

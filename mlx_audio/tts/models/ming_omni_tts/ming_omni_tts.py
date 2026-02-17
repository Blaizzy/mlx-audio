from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.bailing_moe import Model as BailingMoeModel
from mlx_lm.models.bailing_moe import ModelArgs as BailingMoeModelArgs
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelArgs
from mlx_lm.models.qwen2 import Qwen2Model
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult
from mlx_audio.tts.models.interpolate import interpolate
from mlx_audio.utils import load_audio

from .convert import convert_campplus_onnx_to_safetensors


def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "ming_omni_tts"
    llm_config: Optional[dict] = None
    audio_tokenizer_config: Optional[dict] = None
    ditar_config: Optional[dict] = None
    aggregator_config: Optional[dict] = None
    model_path: Optional[str] = None


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv = mx.arange(0, dim, 2, dtype=mx.float32)
        self.inv_freq = 1.0 / (base ** (inv / dim))

    def _apply(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = mx.stack([out_even, out_odd], axis=-1)
        return out.reshape(x.shape)

    def __call__(
        self, q: mx.array, k: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array]:
        seq_len = q.shape[2]
        pos = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(pos, self.inv_freq.astype(mx.float32))
        cos = mx.cos(freqs)[None, None, :, :]
        sin = mx.sin(freqs)[None, None, :, :]
        return self._apply(q, cos, sin), self._apply(k, cos, sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean((x.astype(mx.float32)) ** 2, axis=-1, keepdims=True)
        y = x * mx.rsqrt(variance + self.eps).astype(x.dtype)
        return y * self.weight.astype(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        # Keep list layout to preserve checkpoint keys (ff.0.0.*, ff.2.*).
        self.ff = [[nn.Linear(dim, inner_dim)], None, nn.Linear(inner_dim, dim_out)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ff[0][0](x)
        x = nn.gelu(x)
        x = self.ff[2](x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        # Keep list layout to preserve checkpoint keys (to_out.0.*).
        self.to_out = [nn.Linear(inner_dim, dim), nn.Dropout(0.0)]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        q = self.to_q(x).reshape(bsz, seqlen, self.heads, self.dim_head).transpose(
            0, 2, 1, 3
        )
        k = self.to_k(x).reshape(bsz, seqlen, self.heads, self.dim_head).transpose(
            0, 2, 1, 3
        )
        v = self.to_v(x).reshape(bsz, seqlen, self.heads, self.dim_head).transpose(
            0, 2, 1, 3
        )

        if rope is not None:
            q, k = rope(q, k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.dim_head)
        )
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seqlen, self.heads * self.dim_head)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        if mask is not None:
            out = mx.where(mask[..., None], out, 0.0)
        return out


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mult=mlp_ratio)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array], rope: RotaryEmbedding
    ) -> mx.array:
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.norm_final(x))


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = scale * x[:, None] * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        # Keep index positions to preserve checkpoint keys (time_mlp.0.*, time_mlp.2.*).
        self.time_mlp = [nn.Linear(freq_embed_dim, dim), None, nn.Linear(dim, dim)]

    def __call__(self, timestep: mx.array) -> mx.array:
        hidden = self.time_embed(timestep).astype(timestep.dtype)
        hidden = self.time_mlp[0](hidden)
        hidden = nn.silu(hidden)
        hidden = self.time_mlp[2](hidden)
        return hidden


class CondEmbedder(nn.Module):
    def __init__(self, input_feature_size: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def __call__(self, llm_cond: mx.array, train: bool = False) -> mx.array:
        # Training-time CFG dropout is not needed for inference.
        return self.cond_embedder(llm_cond)


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_cond_dim: int = 2048,
        cfg_dropout_prob: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size, cfg_dropout_prob)
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = [
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ]
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        c: mx.array,
        latent_history: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        t_embed = self.t_embedder(t)[:, None, :]
        x_now = self.x_embedder(x)
        x_history = self.x_embedder(latent_history)
        x_full = mx.concatenate([x_history, x_now], axis=1)
        c_embed = self.c_embedder(c, train=False)
        y = t_embed + c_embed
        x_full = mx.concatenate([y, x_full], axis=1)

        mask_bool = None
        if mask is not None:
            mask = mask.astype(mx.bool_)
            pad = mx.repeat(mask[:, :1], x_history.shape[1] + c_embed.shape[1], axis=1)
            mask_bool = mx.concatenate([pad, mask], axis=1)

        for block in self.blocks:
            x_full = block(x_full, mask_bool, self.rotary_embed)
        return self.final_layer(x_full)

    def forward_with_cfg(
        self,
        x: mx.array,
        t: mx.array,
        c: mx.array,
        cfg_scale: float,
        latent_history: mx.array,
        patch_size: int,
    ) -> mx.array:
        if cfg_scale != 1.0:
            x = mx.concatenate([x, x], axis=0)
            latent_history = mx.concatenate([latent_history, latent_history], axis=0)
            c = mx.concatenate([c, mx.zeros_like(c)], axis=0)
        if t.ndim == 0:
            t = mx.full((x.shape[0],), t, dtype=x.dtype)
        out = self(x=x, t=t, c=c, latent_history=latent_history)
        return out[:, -patch_size:, :]


class Aggregator(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_input_dim: int = 2048,
        **_: Any,
    ):
        super().__init__()
        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = [
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ]
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.x_embedder(x)
        cls = self.word_embedder(mx.zeros((x.shape[0], 1), dtype=mx.int32))
        x = mx.concatenate([cls, x], axis=1)
        mask_bool = None
        if mask is not None:
            mask = mask.astype(mx.bool_)
            mask_bool = mx.concatenate([mask[:, :1], mask], axis=1)
        for block in self.blocks:
            x = block(x, mask_bool, self.rotary_embed)
        x = self.final_layer(x)
        return x[:, :1, :]


def get_epss_timesteps(n: int, dtype: Any) -> mx.array:
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return mx.linspace(0, 1, n + 1, dtype=dtype)
    return dt * mx.array(t, dtype=dtype)


class Solver:
    def __init__(self, func, y0: mx.array, sigma: float = 0.25, temperature: float = 1.5):
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(self, t: mx.array) -> List[mx.array]:
        y = self.y0
        solution = [y]
        for i in range(1, t.shape[0]):
            t0 = t[i - 1]
            t1 = t[i]
            dt = t1 - t0
            dy = dt * self.func(t0, y)
            y = y + dy
            noise = mx.random.normal(y.shape, dtype=y.dtype)
            shift = (
                self.sigma
                * (self.temperature**0.5)
                * (mx.abs(dt) ** 0.5)
                * noise
            )
            y = y + shift
            solution.append(y)
        return solution


class CFM(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def sample(
        self,
        noise: mx.array,
        c: mx.array,
        latent_history: mx.array,
        steps: int = 10,
        cfg_scale: float = 1.0,
        sway_sampling_coef: float = -1.0,
        use_epss: bool = True,
        patch_size: int = 1,
        sigma: float = 0.25,
        temperature: float = 1.5,
    ) -> Tuple[mx.array, List[mx.array]]:
        def fn(t: mx.array, x: mx.array) -> mx.array:
            if cfg_scale < 1e-5:
                return self.model(
                    x=x,
                    t=mx.full((x.shape[0],), t, dtype=x.dtype),
                    c=c,
                    latent_history=latent_history,
                )
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
                cfg_scale=cfg_scale,
                patch_size=patch_size,
            )
            pred, null_pred = mx.split(pred_cfg, 2, axis=0)
            return pred + (pred - null_pred) * cfg_scale

        y0 = noise.transpose(0, 2, 1)
        if use_epss:
            t = get_epss_timesteps(steps, dtype=noise.dtype)
        else:
            t = mx.linspace(0, 1, steps + 1, dtype=noise.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (mx.cos(math.pi / 2 * t) - 1 + t)

        solver = Solver(fn, y0, sigma=sigma, temperature=temperature)
        trajectory = solver.integrate(t)
        return trajectory[-1], trajectory


class FlowLoss(nn.Module):
    def __init__(self, z_channels: int, llm_cond_dim: int, **kwargs: Any):
        super().__init__()
        self.z_channels = z_channels
        self.cfm = CFM(model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs))

    def sample(
        self,
        z: mx.array,
        latent_history: mx.array,
        cfg: float = 1.0,
        patch_size: int = 1,
        sigma: float = 0.25,
        temperature: float = 0.0,
        steps: int = 10,
    ) -> Tuple[mx.array, List[mx.array]]:
        noise = mx.random.normal((z.shape[0], self.z_channels, patch_size), dtype=z.dtype)
        sampled, trajectory = self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
            steps=steps,
        )
        return sampled, trajectory


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)

    def __call__(self, x: mx.array) -> mx.array:
        import numpy as np

        x_pred = self.out(x).transpose(0, 2, 1)
        mag, phase = mx.split(x_pred, 2, axis=1)
        mag = mx.exp(mag)
        mag = mx.clip(mag, None, 1e2)
        spec = mag * (mx.cos(phase) + 1j * mx.sin(phase))
        # Match Torch Ming implementation: ISTFT with "same" padding via overlap-add.
        spec_np = np.array(spec)
        window = np.hanning(self.n_fft).astype(np.float32)
        batch, _, frames = spec_np.shape
        output_size = (frames - 1) * self.hop_length + self.n_fft
        pad = (self.n_fft - self.hop_length) // 2

        ifft = np.fft.irfft(spec_np, n=self.n_fft, axis=1).astype(np.float32)
        ifft *= window[None, :, None]

        audio = np.zeros((batch, output_size), dtype=np.float32)
        env = np.zeros((output_size,), dtype=np.float32)
        win_sq = window * window

        for t in range(frames):
            start = t * self.hop_length
            end = start + self.n_fft
            audio[:, start:end] += ifft[:, :, t]
            env[start:end] += win_sq

        audio = audio[:, pad:-pad]
        env = env[pad:-pad]
        env = np.clip(env, 1e-11, None)
        audio = audio / env[None, :]
        return mx.array(audio)


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_args: dict,
        input_dim: int = 320,
        hop_size: int = 320,
        latent_dim: int = 64,
        patch_size: int = -1,
    ):
        super().__init__()
        cfg = Qwen2ModelArgs.from_dict(encoder_args)
        self.encoder = Qwen2Model(cfg)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.patch_size = patch_size
        self.fc1 = nn.Linear(input_dim, cfg.hidden_size, bias=False)
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.fc3 = nn.Linear(cfg.hidden_size, latent_dim * 2)
        self.norm = nn.LayerNorm(cfg.hidden_size)
        if patch_size != -1:
            agg_cfg_dict = dict(encoder_args)
            agg_cfg_dict["num_hidden_layers"] = 4
            agg_cfg = Qwen2ModelArgs.from_dict(agg_cfg_dict)
            self.aggregator = Qwen2Model(agg_cfg)
            self.cls_embed = mx.random.normal((1, 1, cfg.hidden_size)).astype(mx.float32)

    def get_frames(self, x: mx.array) -> mx.array:
        # x: (B, T)
        import numpy as np

        x_np = np.array(x)
        num_frames = (x_np.shape[-1] + self.hop_size - 1) // self.hop_size
        expected_len = (num_frames - 1) * self.hop_size + self.input_dim
        pad = expected_len - x_np.shape[-1]
        if pad > 0:
            x_np = np.pad(x_np, ((0, 0), (0, pad)))
        frames = np.lib.stride_tricks.sliding_window_view(x_np, self.input_dim, axis=1)
        frames = frames[:, :: self.hop_size, :]
        return mx.array(frames.astype(np.float32))

    def pad_patch_insert_cls(self, x: mx.array) -> mx.array:
        bsz, t, dim = x.shape
        r = t % self.patch_size
        pad = self.patch_size - r if r else 0
        if pad > 0:
            x = mx.pad(x, ((0, 0), (0, pad), (0, 0)))
        x = x.reshape(-1, self.patch_size, dim)
        cls = mx.broadcast_to(self.cls_embed.astype(x.dtype), (x.shape[0], 1, dim))
        x = mx.concatenate([x, cls], axis=1)
        return x.reshape(bsz, -1, dim)

    def __call__(self, waveform: mx.array) -> mx.array:
        x = self.get_frames(waveform)
        x = self.fc1(x)
        x = self.fc2(x)
        dummy = mx.zeros((x.shape[0], x.shape[1]), dtype=mx.int32)
        x = self.encoder(dummy, input_embeddings=x)
        if self.patch_size != -1:
            x = self.pad_patch_insert_cls(x)
            dummy2 = mx.zeros((x.shape[0], x.shape[1]), dtype=mx.int32)
            x = self.aggregator(dummy2, input_embeddings=x)
            bsz, _, dim = x.shape
            x = x.reshape(-1, self.patch_size + 1, dim)
            x = x[:, -1:, :].reshape(bsz, -1, dim)
        return self.fc3(x)


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_args: dict,
        output_dim: int = 882,
        latent_dim: int = 64,
        patch_size: int = -1,
    ):
        super().__init__()
        cfg = Qwen2ModelArgs.from_dict(decoder_args)
        self.decoder = Qwen2Model(cfg)
        self.fc1 = nn.Linear(latent_dim, cfg.hidden_size)
        self.patch_size = patch_size
        self.head = ISTFTHead(dim=cfg.hidden_size, n_fft=output_dim * 4, hop_length=output_dim)

    def low_level_reconstruct(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        if self.patch_size != -1:
            x = x.transpose(0, 2, 1)
            x = interpolate(
                x,
                scale_factor=float(self.patch_size),
                mode="linear",
                align_corners=False,
            )
            x = x.transpose(0, 2, 1)
        dummy = mx.zeros((x.shape[0], x.shape[1]), dtype=mx.int32)
        x = self.decoder(dummy, input_embeddings=x)
        return self.head(x)


class AudioVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.sample_rate = int(config["sample_rate"])
        self.patch_size = int(config.get("patch_size", -1))
        self.encoder = Encoder(
            encoder_args=config["enc_kwargs"]["backbone"],
            input_dim=int(config["enc_kwargs"]["input_dim"]),
            hop_size=int(config["enc_kwargs"].get("hop_size", 320)),
            latent_dim=int(config["enc_kwargs"]["latent_dim"]),
            patch_size=self.patch_size,
        )
        self.decoder = Decoder(
            decoder_args=config["dec_kwargs"]["backbone"],
            output_dim=int(config["dec_kwargs"]["output_dim"]),
            latent_dim=int(config["dec_kwargs"]["latent_dim"]),
            patch_size=self.patch_size,
        )

    def encode_latent(self, waveform: mx.array, waveform_length: mx.array) -> Tuple[mx.array, mx.array]:
        frame_num = mx.ceil(waveform_length / self.config["enc_kwargs"]["input_dim"]).astype(mx.int32)
        if self.patch_size != -1:
            frame_num = mx.ceil(frame_num / self.patch_size).astype(mx.int32)
        h = self.encoder(waveform)  # (B, T, 2*latent)
        h = h.transpose(0, 2, 1)
        mu, logvar = mx.split(h, 2, axis=1)
        std = mx.exp(0.5 * mx.clip(logvar, -30, 20))
        latent = mu + std * mx.random.normal(mu.shape, dtype=mu.dtype)
        latent = latent.transpose(0, 2, 1)
        return latent, frame_num

    def decode(
        self,
        latent: mx.array,
        past_key_values=None,
        use_cache: bool = False,
        stream_state=(None, None, None),
        last_chunk: bool = False,
    ):
        waveform = self.decoder.low_level_reconstruct(latent)
        return waveform, stream_state, past_key_values


class Model(nn.Module):
    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.model_type = "ming_omni_tts"
        self.tokenizer = None

        if not config.llm_config:
            raise ValueError("Missing llm_config in Ming Omni config")
        if not config.audio_tokenizer_config:
            raise ValueError("Missing audio_tokenizer_config in Ming Omni config")
        if not config.ditar_config:
            raise ValueError("Missing ditar_config in Ming Omni config")
        if not config.aggregator_config:
            raise ValueError("Missing aggregator_config in Ming Omni config")

        llm_cfg = dict(config.llm_config)
        # mlx-lm bailing_moe does not currently support rope_scaling.type == "3D".
        # For this TTS path we do not use image/video position masks, so standard RoPE works.
        if isinstance(llm_cfg.get("rope_scaling"), dict):
            llm_cfg["rope_scaling"] = None
        self.llm_args = BailingMoeModelArgs.from_dict(llm_cfg)
        self.model = BailingMoeModel(self.llm_args)

        self.audio = AudioVAE(config.audio_tokenizer_config)
        self.latent_dim = int(config.audio_tokenizer_config["enc_kwargs"]["latent_dim"])
        self.patch_size = int(config.ditar_config["patch_size"])
        self.history_patch_size = int(config.ditar_config.get("history_patch_size", self.patch_size))

        self.linear_proj_audio = Aggregator(
            in_channels=self.latent_dim,
            llm_input_dim=self.llm_args.hidden_size,
            **config.aggregator_config,
        )
        self.flowloss = FlowLoss(
            z_channels=self.latent_dim,
            llm_cond_dim=self.llm_args.hidden_size,
            **config.ditar_config,
        )
        self.stop_head = nn.Linear(self.llm_args.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.llm_args.hidden_size, bias=True)

    @property
    def sample_rate(self) -> int:
        return self.audio.sample_rate

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # First sanitize the LLM shard with mlx-lm's bailing_moe remapping.
        llm_weights = {
            k[len("model.") :]: v for k, v in weights.items() if k.startswith("model.")
        }
        llm_weights = self.model.sanitize(llm_weights)
        sanitized = {}
        allowed_non_llm_prefixes = (
            "audio.",
            "flowloss.",
            "linear_proj_audio.",
            "spk_head.",
            "stop_head.",
        )
        for k, v in llm_weights.items():
            # These gates are only used with explicit modality masks; not needed for text-only TTS path.
            if ".audio_gate." in k or ".image_gate." in k:
                continue
            sanitized[f"model.{k}"] = v
        for k, v in weights.items():
            if not k.startswith("model."):
                if not k.startswith(allowed_non_llm_prefixes):
                    continue
                if k == "audio.decoder.head.istft.window":
                    continue
                sanitized[k] = v
        return sanitized

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prepare_input_embed(
        self,
        prompt: str,
        text: str,
        instruction: Optional[str] = None,
        prompt_latent: Optional[mx.array] = None,
        prompt_text: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array]:
        prompt_text_token = []
        prompt_latent_token = []
        if prompt_latent is not None and prompt_text is not None:
            bsz = prompt_latent.shape[0]
            prompt_latent = prompt_latent.reshape(-1, self.patch_size, self.latent_dim)
            prompt_latent = self.linear_proj_audio(prompt_latent)
            prompt_latent = prompt_latent.reshape(bsz, -1, prompt_latent.shape[-1])
            prompt_text_token = self._encode(prompt_text)
            patch_token_id = self.tokenizer.convert_tokens_to_ids("<audioPatch>")
            prompt_latent_token = [patch_token_id] * prompt_latent.shape[1]

        prompt2 = self._encode(" Text input:\n")
        if (
            "Genre: " in text
            and "Mood: " in text
            and "Instrument: " in text
            and "Theme: " in text
            and "Duration: " in text
        ):
            prompt2 = []

        instruction_prompt = []
        if instruction is not None:
            instruction_prompt = self._encode(instruction) + self._encode("<|endoftext|>")

        input_part = (
            self._encode("<role>HUMAN</role>")
            + self._encode(prompt)
            + prompt2
            + prompt_text_token
            + self._encode(text)
            + self._encode("<role>ASSISTANT</role>")
            + instruction_prompt
            + self._encode("<audio>")
            + prompt_latent_token
        )

        input_ids = mx.array(input_part, dtype=mx.int32)[None, :]
        inputs_embeds = self.model.model.word_embeddings(input_ids)

        if prompt_latent is not None:
            audio_token_id = self.tokenizer.convert_tokens_to_ids("<audio>")
            audio_positions = mx.where(input_ids[0] == audio_token_id)[0]
            if audio_positions.shape[0] > 0:
                start = int(audio_positions[0]) + 1
                end = start + prompt_latent.shape[1]
                inputs_embeds = inputs_embeds.at[0, start:end, :].set(prompt_latent[0])

        return input_ids, inputs_embeds

    def _forward_llm_hidden(self, inputs_embeds: mx.array, cache: List[KVCache]) -> mx.array:
        h = inputs_embeds
        mask = create_attention_mask(h, cache[0] if cache else None)
        for layer, layer_cache in zip(self.model.model.layers, cache):
            h = layer(h, mask, layer_cache)
        return self.model.model.norm(h)

    def sample(
        self,
        prompt: str,
        text: str,
        instruction: Optional[str] = None,
        prompt_waveform: Optional[mx.array] = None,
        prompt_text: Optional[str] = None,
        max_decode_steps: int = 200,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
        flow_steps: int = 10,
    ):
        prompt_latent = None
        if prompt_waveform is not None and prompt_text is not None:
            if prompt_waveform.ndim == 1:
                prompt_waveform = prompt_waveform[None, :]
            prompt_waveform_length = mx.array([prompt_waveform.shape[1]], dtype=mx.int32)
            prompt_latent, _ = self.audio.encode_latent(prompt_waveform, prompt_waveform_length)

        _, inputs_embeds = self._prepare_input_embed(
            prompt=prompt,
            text=text,
            instruction=instruction,
            prompt_latent=prompt_latent,
            prompt_text=prompt_text,
        )

        cache = [KVCache() for _ in range(self.llm_args.num_hidden_layers)]
        latent_history = None

        for step in range(max_decode_steps):
            hidden = self._forward_llm_hidden(inputs_embeds=inputs_embeds, cache=cache)
            z_diff = hidden[:, -1:, :]
            if step == 0:
                latent_history = mx.zeros(
                    (1, self.history_patch_size, self.latent_dim),
                    dtype=z_diff.dtype,
                )
                if prompt_latent is not None:
                    start = self.history_patch_size - prompt_latent.shape[1]
                    if start < 0:
                        latent_history = prompt_latent[:, -start:, :]
                    else:
                        latent_history = latent_history.at[:, start:, :].set(prompt_latent)

            sampled_token_latent, _ = self.flowloss.sample(
                z_diff,
                latent_history,
                cfg=cfg,
                patch_size=self.patch_size,
                sigma=sigma,
                temperature=temperature,
                steps=flow_steps,
            )
            stop_prob = mx.softmax(self.stop_head(z_diff), axis=-1)[0, 0, 1]
            is_last = bool(stop_prob > 0.5 and step > 3)
            yield sampled_token_latent, is_last
            if is_last:
                break

            inputs_embeds = self.linear_proj_audio(sampled_token_latent)
            latent_history = mx.concatenate(
                [latent_history[:, self.patch_size :, :], sampled_token_latent],
                axis=1,
            )

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "en",
        ref_audio: Optional[Union[str, mx.array]] = None,
        ref_text: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        ddpm_steps: Optional[int] = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Iterable[GenerationResult]:
        del voice, lang_code, stream, streaming_interval

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Load model with load()/from_pretrained first.")

        if isinstance(ref_audio, str):
            ref_audio = load_audio(ref_audio, sample_rate=self.sample_rate)

        start_time = time.perf_counter()
        sampled_tokens_list = []
        prompt = kwargs.get(
            "prompt", "Please generate speech based on the following description.\n"
        )
        cfg = 2.0 if cfg_scale is None else cfg_scale
        flow_steps = 10 if ddpm_steps is None else ddpm_steps
        sigma = float(kwargs.get("sigma", 0.25))
        max_decode_steps = int(kwargs.get("max_decode_steps", max_tokens))

        for sampled_tokens, last_chunk in self.sample(
            prompt=prompt,
            text=text,
            instruction=instruct,
            prompt_waveform=ref_audio,
            prompt_text=ref_text,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            flow_steps=flow_steps,
        ):
            sampled_tokens_list.append(sampled_tokens)
            if last_chunk:
                break

        if not sampled_tokens_list:
            raise RuntimeError("No latent tokens were generated")

        sampled_tokens = mx.concatenate(sampled_tokens_list, axis=1)
        speech, _, _ = self.audio.decode(
            sampled_tokens, past_key_values=None, use_cache=False
        )
        speech = speech[0]

        if speed != 1.0:
            # Speed control is not implemented for Ming Omni yet; keep native timing.
            if verbose:
                print("[WARN] speed parameter is currently ignored for Ming Omni TTS.")

        mx.eval(speech)
        elapsed = max(1e-6, time.perf_counter() - start_time)
        samples = int(speech.shape[0])
        token_count = len(self._encode(text))
        duration_seconds = samples / self.sample_rate
        audio_sps = samples / elapsed
        token_tps = token_count / elapsed

        yield GenerationResult(
            audio=speech,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=format_duration(duration_seconds),
            real_time_factor=elapsed / max(duration_seconds, 1e-6),
            prompt={"tokens": token_count, "tokens-per-sec": token_tps},
            audio_samples={"samples": samples, "samples-per-sec": audio_sps},
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=False,
            is_final_chunk=True,
        )

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        # Convert campplus ONNX -> safetensors once to satisfy safetensors-only workflows.
        onnx_path = model_path / "campplus.onnx"
        safetensors_path = model_path / "campplus.safetensors"
        if onnx_path.exists() and not safetensors_path.exists():
            try:
                convert_campplus_onnx_to_safetensors(
                    onnx_path=onnx_path,
                    output_path=safetensors_path,
                    verify_allclose=True,
                )
            except Exception as e:
                print(f"[WARN] campplus ONNX conversion failed: {e}")

        model.tokenizer = cls._load_compatible_tokenizer(
            model_path=model_path,
            llm_vocab_size=int(model.llm_args.vocab_size),
        )
        return model

    @staticmethod
    def _is_tokenizer_compatible(tokenizer, llm_vocab_size: int) -> bool:
        required_tokens = ["<audio>", "<audioPatch>", "<|endoftext|>"]
        for token in required_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0 or token_id >= llm_vocab_size:
                return False
        probe = (
            tokenizer.encode("<role>HUMAN</role>", add_special_tokens=False)
            + tokenizer.encode("<role>ASSISTANT</role>", add_special_tokens=False)
            + tokenizer.encode("<audio>", add_special_tokens=False)
        )
        return (not probe) or (max(probe) < llm_vocab_size and min(probe) >= 0)

    @classmethod
    def _load_compatible_tokenizer(
        cls, model_path: Path, llm_vocab_size: int
    ):
        candidates = []

        # 1) Prefer colocated tokenizer files if present.
        candidates.append(model_path)

        # 2) Optional user override.
        import os

        override = os.environ.get("MING_OMNI_TOKENIZER_PATH")
        if override:
            candidates.append(Path(override).expanduser())

        # 3) Common local layout: .../mlx-audio-dev/{mlx-audio,Ming-omni-tts}
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            candidate = parent / "Ming-omni-tts"
            candidates.append(candidate)

        seen = set()
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
            except Exception:
                continue
            key = candidate.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if not (candidate / "tokenizer.json").exists():
                continue
            try:
                tok = PreTrainedTokenizerFast.from_pretrained(candidate.as_posix())
                if cls._is_tokenizer_compatible(tok, llm_vocab_size):
                    return tok
            except Exception:
                continue

        # 4) Last-resort remote tokenizer load.
        for repo, trust_remote_code in [
            (str(model_path), True),
            ("inclusionAI/Ming-omni-tts-0.5B", True),
        ]:
            try:
                tok = AutoTokenizer.from_pretrained(
                    repo,
                    trust_remote_code=trust_remote_code,
                )
                if cls._is_tokenizer_compatible(tok, llm_vocab_size):
                    return tok
            except Exception:
                continue

        raise RuntimeError(
            "Could not find a tokenizer compatible with Ming-omni-tts-16.8B-A3B. "
            "Set MING_OMNI_TOKENIZER_PATH to a directory containing tokenizer.json."
        )

    @classmethod
    def from_pretrained(cls, path_or_hf_repo: str) -> "Model":
        local = Path(path_or_hf_repo)
        if not local.exists():
            local = Path(snapshot_download(repo_id=path_or_hf_repo))
        import json

        with open(local / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["model_path"] = str(local)
        if cfg.get("model_type") == "bailingmm":
            cfg["model_type"] = "ming_omni_tts"
        model = cls(cfg)
        weights = {}
        for wf in local.glob("*.safetensors"):
            if wf.name.startswith("campplus"):
                continue
            weights.update(mx.load(wf.as_posix()))
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        model.eval()
        return cls.post_load_hook(model, local)

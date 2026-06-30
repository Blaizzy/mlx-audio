"""Stable Audio 3 — text-to-audio diffusion model for MLX.

Generates stereo 44.1kHz audio (sound effects up to 30s, music up to 30s)
from text prompts using a DiT backbone, T5Gemma text encoder, and SAME decoder.

Supports two official variants:
  - stabilityai/stable-audio-3-small-sfx  (sound effects)
  - stabilityai/stable-audio-3-small-music (music)
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelArgs, GenerationResult
from .dit import DiffusionTransformer, DiTConfig
from .same import Pretransform, SAMEConfig

CONV_SUFFIXES = {
    "preprocess_conv.weight",
    "postprocess_conv.weight",
    "mapping.weight",
    "out_mapping.weight",
}


@dataclass
class ModelConfig(BaseModelArgs):
    dit: dict = field(default_factory=dict)
    same: dict = field(default_factory=dict)
    sample_rate: int = 44100
    audio_channels: int = 2
    model_type: str = "stable_audio_3"
    model_path: str = ""

    def get_dit_config(self) -> DiTConfig:
        return DiTConfig(**self.dit) if self.dit else DiTConfig()

    def get_same_config(self) -> SAMEConfig:
        return SAMEConfig(**self.same) if self.same else SAMEConfig()

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        model = d.get("model", d)
        diff_cfg = model.get("diffusion", {}).get("config", {})
        ae_cfg = model.get("pretransform", {}).get("config", {})
        enc_cfg = ae_cfg.get("encoder", {}).get("config", {})
        dec_cfg = ae_cfg.get("decoder", {}).get("config", {})
        patch_cfg = ae_cfg.get("pretransform", {}).get("config", {})

        dit = dict(
            io_channels=diff_cfg.get("io_channels", 256),
            embed_dim=diff_cfg.get("embed_dim", 1024),
            depth=diff_cfg.get("depth", 20),
            num_heads=diff_cfg.get("num_heads", 16),
            cond_token_dim=diff_cfg.get("cond_token_dim", 768),
            global_cond_dim=diff_cfg.get("global_cond_dim", 768),
            local_add_cond_dim=diff_cfg.get("local_add_cond_dim", 257),
            num_memory_tokens=diff_cfg.get("num_memory_tokens", 64),
            ff_mult=diff_cfg.get("ff_kwargs", {}).get("mult", 4.0),
        )

        channels = enc_cfg.get("channels", 128)
        c_mults = enc_cfg.get("c_mults", [6])
        strides = enc_cfg.get("strides", [16])
        patch_size = patch_cfg.get("patch_size", 256)
        audio_channels = patch_cfg.get("channels", 2)
        downsampling_ratio = patch_size
        for s in strides:
            downsampling_ratio *= s

        same = dict(
            latent_dim=ae_cfg.get("latent_dim", 256),
            patch_size=patch_size,
            audio_channels=audio_channels,
            encoder_channels=channels,
            encoder_c_mults=c_mults,
            encoder_strides=strides,
            encoder_depths=enc_cfg.get("transformer_depths", [6]),
            dim_heads=enc_cfg.get("dim_heads", 64),
            downsampling_ratio=downsampling_ratio,
            ff_mult=dec_cfg.get("ff_mult", 3),
            chunk_size=dec_cfg.get("chunk_size", 32),
            chunk_midpoint_shift=dec_cfg.get("chunk_midpoint_shift", True),
            conv_mapping=dec_cfg.get("conv_mapping", True),
            differential=dec_cfg.get("differential", True),
            decoder_out_channels=dec_cfg.get(
                "out_channels", patch_size * audio_channels
            ),
        )

        return cls(
            dit=dit,
            same=same,
            sample_rate=d.get("sample_rate", 44100),
            audio_channels=d.get("audio_channels", 2),
            model_type=d.get("model_type", "stable_audio_3"),
            model_path=d.get("model_path", ""),
        )


class _DiffusionWrapper(nn.Module):
    """Matches PyTorch weight prefix: model.model.*"""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.model = DiffusionTransformer(config)


class _PromptConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.padding_embedding = mx.zeros((768,))


class _DurationEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = [None, nn.Linear(256, 768, bias=True)]


class _DurationConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = _DurationEmbedder()


class _ConditionerSet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = _PromptConditioner()
        self.seconds_total = _DurationConditioner()


class _Conditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioners = _ConditionerSet()


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dit_config = config.get_dit_config()
        same_config = config.get_same_config()

        self.model = _DiffusionWrapper(dit_config)
        self.conditioner = _Conditioner()
        self.pretransform = Pretransform(same_config)

        self._t5_tokenizer = None
        self._t5_model = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def sanitize(self, weights: dict) -> dict:
        result = {}
        skip_keys = set()

        for key in sorted(weights.keys()):
            if key in skip_keys:
                continue

            # Merge weight normalization pairs (weight_g + weight_v -> weight)
            if key.endswith(".weight_g"):
                base = key[: -len(".weight_g")]
                v_key = base + ".weight_v"
                if v_key in weights:
                    g = np.array(weights[key])
                    v = np.array(weights[v_key])
                    norm = np.sqrt(np.sum(v**2, axis=(1, 2), keepdims=True))
                    norm = np.maximum(norm, 1e-12)
                    merged = (g * v / norm).astype(np.float32)
                    weight_key = base + ".weight"
                    if (
                        any(weight_key.endswith(s) for s in CONV_SUFFIXES)
                        and len(merged.shape) == 3
                        and merged.shape[-1] < merged.shape[-2]
                    ):
                        merged = np.transpose(merged, (0, 2, 1))
                    result[weight_key] = mx.array(merged)
                    skip_keys.add(key)
                    skip_keys.add(v_key)
                    continue

            # Filter training-only zero params (but keep running_std)
            if "noise_scaling_factor" in key:
                arr = np.array(weights[key])
                if arr.size <= 1 or np.all(arr == 0):
                    continue

            value = weights[key]
            arr = np.array(value) if not isinstance(value, np.ndarray) else value

            # Transpose Conv1d weights: PyTorch [out, in, K] -> MLX [out, K, in]
            # Skip if already in MLX format (shape[-1] >= shape[-2])
            if (
                any(key.endswith(s) for s in CONV_SUFFIXES)
                and len(arr.shape) == 3
                and arr.shape[-1] < arr.shape[-2]
            ):
                arr = np.transpose(arr, (0, 2, 1))

            result[key] = mx.array(arr)

        return result

    def _load_t5gemma(self, model_path: str):
        import torch
        from transformers import AutoConfig, AutoTokenizer, T5GemmaEncoderModel

        t5_path = Path(model_path) / "t5gemma-b-b-ul2"
        if not t5_path.exists():
            raise FileNotFoundError(
                f"T5Gemma tokenizer not found at {t5_path}. "
                "Ensure the model was downloaded with the full HF repo."
            )
        self._t5_tokenizer = AutoTokenizer.from_pretrained(str(t5_path))
        config = AutoConfig.from_pretrained(str(t5_path))
        config.is_encoder_decoder = False
        self._t5_model = T5GemmaEncoderModel.from_pretrained(
            str(t5_path), config=config, torch_dtype=torch.float32
        )
        self._t5_model.eval()

    def _encode_text(
        self, prompt: str, max_length: int = 256
    ) -> tuple[mx.array, mx.array]:
        import torch

        inputs = self._t5_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = self._t5_model(**inputs)
            embeddings = outputs.last_hidden_state

        text_emb = mx.array(embeddings.numpy())
        attn_mask = mx.array(inputs["attention_mask"].numpy())
        return text_emb, attn_mask

    def _encode_duration(self, duration: float) -> mx.array:
        t = mx.array([duration / 384.0])
        half = 128
        freqs = mx.exp(mx.linspace(0.0, 1.0, half) * math.log(10000.0))
        t_expanded = mx.expand_dims(t, -1)
        fourier = mx.concatenate(
            [mx.sin(t_expanded * freqs), mx.cos(t_expanded * freqs)], axis=-1
        )
        emb = self.conditioner.conditioners.seconds_total.embedder.embedding[1]
        return emb(fourier)

    def _build_schedule(
        self,
        steps: int,
        seq_len: int,
        sigma_max: float = 1.0,
        min_length: int = 256,
        max_length: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> mx.array:
        t = mx.linspace(sigma_max, 0.0, steps + 1)
        seq_len = max(min(seq_len, max_length), min_length)
        mu = -(
            base_shift
            + (max_shift - base_shift)
            * (seq_len - min_length)
            / (max_length - min_length)
        )
        t_shifted = 1.0 - math.exp(mu) / (
            math.exp(mu) + (1.0 / (1.0 - t + 1e-10) - 1.0)
        )
        t_shifted = mx.where(t <= 0, mx.zeros_like(t_shifted), t_shifted)
        t_shifted = mx.where(t >= 1, mx.ones_like(t_shifted), t_shifted)
        return mx.concatenate([mx.array([sigma_max]), t_shifted[1:]])

    def _sample_euler(
        self,
        noise: mx.array,
        sigmas: mx.array,
        cond_tokens: mx.array,
        global_cond: mx.array,
        cfg_scale: float,
        negative_cond_tokens: Optional[mx.array],
        negative_global_cond: Optional[mx.array],
        verbose: bool,
    ) -> mx.array:
        x = noise
        num_steps = sigmas.shape[0] - 1
        dit = self.model.model

        for i in range(num_steps):
            t_curr = sigmas[i]
            t_next = sigmas[i + 1]
            dt = t_next - t_curr
            t_batch = mx.broadcast_to(mx.array([t_curr.item()]), (x.shape[0],))

            if cfg_scale != 1.0 and negative_cond_tokens is not None:
                x_double = mx.concatenate([x, x], axis=0)
                t_double = mx.concatenate([t_batch, t_batch], axis=0)
                cond_double = mx.concatenate(
                    [cond_tokens, negative_cond_tokens], axis=0
                )
                global_double = mx.concatenate(
                    [global_cond, negative_global_cond], axis=0
                )
                v_both = dit(x_double, t_double, cond_double, global_double)
                v_cond, v_uncond = mx.split(v_both, 2, axis=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = dit(x, t_batch, cond_tokens, global_cond)

            x = x + dt * v
            mx.async_eval(x)
            if verbose:
                print(
                    f"  Step {i + 1}/{num_steps} "
                    f"(t={t_curr.item():.3f} -> {t_next.item():.3f})"
                )
        return x

    def _sample_pingpong(
        self,
        noise: mx.array,
        sigmas: mx.array,
        cond_tokens: mx.array,
        global_cond: mx.array,
        cfg_scale: float,
        negative_cond_tokens: Optional[mx.array],
        negative_global_cond: Optional[mx.array],
        verbose: bool,
    ) -> mx.array:
        x = noise
        num_steps = sigmas.shape[0] - 1
        dit = self.model.model

        for i in range(num_steps):
            t_curr = sigmas[i]
            t_next = sigmas[i + 1]
            t_batch = mx.broadcast_to(mx.array([t_curr.item()]), (x.shape[0],))

            if cfg_scale != 1.0 and negative_cond_tokens is not None:
                x_double = mx.concatenate([x, x], axis=0)
                t_double = mx.concatenate([t_batch, t_batch], axis=0)
                cond_double = mx.concatenate(
                    [cond_tokens, negative_cond_tokens], axis=0
                )
                global_double = mx.concatenate(
                    [global_cond, negative_global_cond], axis=0
                )
                v_both = dit(x_double, t_double, cond_double, global_double)
                v_cond, v_uncond = mx.split(v_both, 2, axis=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = dit(x, t_batch, cond_tokens, global_cond)

            t_curr_expanded = mx.expand_dims(mx.expand_dims(t_curr, 0), 0)
            denoised = x - t_curr_expanded * v

            t_next_expanded = mx.expand_dims(mx.expand_dims(t_next, 0), 0)
            x = (1 - t_next_expanded) * denoised + t_next_expanded * mx.random.normal(
                x.shape
            )
            mx.async_eval(x)
            if verbose:
                print(
                    f"  Step {i + 1}/{num_steps} "
                    f"(t={t_curr.item():.3f} -> {t_next.item():.3f})"
                )
        return x

    def generate(
        self,
        text: str,
        model_path: Optional[str] = None,
        duration: Optional[float] = None,
        ddpm_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        sampler: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        if duration is None:
            duration = 10.0
        if ddpm_steps is None:
            ddpm_steps = 8
        if cfg_scale is None:
            cfg_scale = 1.0
        if sampler is None:
            sampler = "pingpong"
        if seed is None:
            seed = 42
        start_time = time.time()

        if self._t5_tokenizer is None:
            path = model_path or self.config.model_path
            if not path:
                raise ValueError(
                    "model_path is required on first call to load T5Gemma encoder"
                )
            if verbose:
                print("Loading T5Gemma text encoder...")
            self._load_t5gemma(path)

        if verbose:
            print(f"Encoding prompt: '{text}'")
        text_emb, text_mask = self._encode_text(text)

        padding_emb = self.conditioner.conditioners.prompt.padding_embedding
        inv_mask = 1.0 - mx.expand_dims(text_mask, -1)
        text_emb = text_emb * mx.expand_dims(text_mask, -1) + padding_emb * inv_mask

        if verbose:
            print(f"Encoding duration: {duration}s")
        duration_emb = self._encode_duration(duration)

        duration_emb_expanded = mx.expand_dims(duration_emb, 1)
        cond_tokens = mx.concatenate([text_emb, duration_emb_expanded], axis=1)
        global_cond = duration_emb

        negative_cond_tokens = None
        negative_global_cond = None
        if cfg_scale != 1.0:
            neg_text_emb = mx.broadcast_to(padding_emb, text_emb.shape)
            negative_cond_tokens = mx.concatenate(
                [neg_text_emb, mx.expand_dims(duration_emb, 1)], axis=1
            )
            negative_global_cond = duration_emb

        dit_config = self.config.get_dit_config()
        same_config = self.config.get_same_config()
        audio_samples = int(duration * self.config.sample_rate)
        latent_len = audio_samples // same_config.downsampling_ratio
        padding_tokens = int(
            6.0 * self.config.sample_rate / same_config.downsampling_ratio
        )
        latent_len += padding_tokens

        if verbose:
            print(f"Latent shape: [1, {dit_config.io_channels}, {latent_len}]")

        mx.random.seed(seed)
        noise = mx.random.normal((1, dit_config.io_channels, latent_len))
        sigmas = self._build_schedule(ddpm_steps, seq_len=latent_len)

        if verbose:
            print(f"Sampling with {sampler} ({ddpm_steps} steps)...")

        sample_fn = (
            self._sample_pingpong if sampler == "pingpong" else self._sample_euler
        )
        latents = sample_fn(
            noise,
            sigmas,
            cond_tokens,
            global_cond,
            cfg_scale,
            negative_cond_tokens,
            negative_global_cond,
            verbose,
        )

        if verbose:
            print("Decoding latents to audio...")
        audio = self.pretransform.decode(latents)
        mx.eval(audio)

        target_samples = int(duration * self.config.sample_rate)
        audio = audio[:, :, :target_samples]

        elapsed = time.time() - start_time
        audio_duration_seconds = target_samples / self.config.sample_rate
        rtf = elapsed / audio_duration_seconds if audio_duration_seconds > 0 else 0

        duration_hours = int(audio_duration_seconds // 3600)
        duration_mins = int(audio_duration_seconds % 3600 // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = (
            f"{duration_hours:02d}:{duration_mins:02d}:"
            f"{duration_secs:02d}.{duration_ms:03d}"
        )

        # audio shape is [1, channels, samples]; flatten to [samples, channels]
        audio_out = audio[0].transpose(1, 0)

        yield GenerationResult(
            audio=audio_out,
            samples=target_samples,
            sample_rate=self.config.sample_rate,
            segment_idx=0,
            token_count=latent_len * ddpm_steps,
            audio_duration=duration_str,
            real_time_factor=round(rtf, 2),
            prompt={
                "tokens": latent_len,
                "tokens-per-sec": (
                    round(latent_len / elapsed, 2) if elapsed > 0 else 0
                ),
            },
            audio_samples={
                "samples": target_samples,
                "samples-per-sec": (
                    round(target_samples / elapsed, 2) if elapsed > 0 else 0
                ),
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

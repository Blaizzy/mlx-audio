from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.codec.models.dacvae import DACVAE
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import load_audio as load_audio_any

from .config import ModelConfig
from .model import IrodoriDiT
from .sampling import sample_euler_cfg
from .text import encode_text, normalize_text


def _find_silence_point(
    latent: mx.array,
    window_size: int = 20,
    std_threshold: float = 0.05,
) -> int:
    """Detect trailing silence in generated latent (same heuristic as Echo TTS)."""
    # latent: (T, D)
    padded = mx.concatenate(
        [latent, mx.zeros((window_size, latent.shape[-1]), dtype=latent.dtype)], axis=0
    )
    for i in range(int(padded.shape[0] - window_size)):
        window = padded[i : i + window_size]
        if float(window.std()) < std_threshold and abs(float(window.mean())) < 0.1:
            return i
    return int(latent.shape[0])


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = IrodoriDiT(config.dit)
        self.dacvae: DACVAE | None = None
        self._tokenizer = None
        self._caption_tokenizer = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def model_type(self) -> str:
        return self.config.model_type

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Weight loading hooks
    # ------------------------------------------------------------------

    def sanitize(self, weights: dict) -> dict:
        """
        Remap Irodori PyTorch weight keys to mlx-audio MLX conventions:
          cond_module.0.weight  → cond_module.layers.0.weight
          <key>                 → model.<key>
        """
        out = {}
        for k, v in weights.items():
            # PyTorch Sequential uses integer keys; MLX nn.Sequential uses "layers.N"
            if k.startswith("cond_module."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    k = ".".join(["cond_module", "layers", parts[1], *parts[2:]])
            # Nest under self.model
            out_key = f"model.{k}" if not k.startswith("model.") else k
            out[out_key] = v
        return out

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Load DACVAE codec after model weights are loaded.

        Looks for a local dacvae/ subdirectory first (produced by convert.py),
        then falls back to downloading model.config.dacvae_repo from HuggingFace.
        """
        import json as _json

        from mlx_audio.codec.models.dacvae import DACVAEConfig

        local_dacvae = Path(model_path) / "dacvae"
        try:
            if local_dacvae.is_dir():
                with open(local_dacvae / "config.json") as f:
                    cfg = DACVAEConfig(**_json.load(f))
                dac = DACVAE(cfg)
                dac.load_weights(str(local_dacvae / "model.safetensors"))
                import mlx.core as _mx

                _mx.eval(dac.parameters())
                model.dacvae = dac
            else:
                model.dacvae = DACVAE.from_pretrained(model.config.dacvae_repo)
        except Exception as e:
            import warnings

            warnings.warn(
                f"Could not load DACVAE: {e}\n"
                "Set model.dacvae manually before calling generate()."
            )
            model.dacvae = None
        return model

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.dit.text_tokenizer_repo
            )
        return self._tokenizer

    def _get_caption_tokenizer(self):
        if self._caption_tokenizer is None:
            from transformers import AutoTokenizer

            repo = self.config.dit.caption_tokenizer_repo_resolved
            # Reuse text tokenizer if repo is the same
            if repo == self.config.dit.text_tokenizer_repo:
                self._caption_tokenizer = self._get_tokenizer()
            else:
                self._caption_tokenizer = AutoTokenizer.from_pretrained(repo)
        return self._caption_tokenizer

    def _prepare_text(
        self, text: str, max_length: Optional[int] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Normalise and tokenise text, returning (input_ids, mask) as MLX arrays.
        Matches Irodori's PretrainedTextTokenizer: BOS prepended manually,
        right-padded to max_length.
        """
        if max_length is None:
            max_length = self.config.max_text_length

        text = normalize_text(text)
        return encode_text(
            text,
            tokenizer=self._get_tokenizer(),
            max_length=max_length,
            add_bos=self.config.dit.text_add_bos,
        )

    def _prepare_caption(
        self, caption: str, max_length: Optional[int] = None
    ) -> tuple[mx.array, mx.array]:
        if max_length is None:
            max_length = self.config.max_caption_length
        return encode_text(
            caption,
            tokenizer=self._get_caption_tokenizer(),
            max_length=max_length,
            add_bos=self.config.dit.caption_add_bos_resolved,
        )

    # ------------------------------------------------------------------
    # Reference audio encoding
    # ------------------------------------------------------------------

    def _encode_ref_audio(self, audio: mx.array) -> tuple[mx.array, mx.array]:
        """
        Encode reference waveform with DACVAE.
        audio: (1, samples) at config.sample_rate
        Returns (latent, mask): latent (1, T, 128), mask (1, T) bool
        """
        assert self.dacvae is not None, "DACVAE not loaded"

        max_samples = (
            self.config.max_speaker_latent_length * self.config.audio_downsample_factor
        )
        audio = audio[:, :max_samples]

        # DACVAE encode expects (B, L, 1)
        audio_in = audio[:, :, None]  # (1, L, 1)
        latent = self.dacvae.encode(audio_in)  # (1, 128, T) channels-first
        latent = mx.transpose(latent, (0, 2, 1))  # (1, T, 128) sequence-first

        actual_t = int(audio.shape[1]) // self.config.audio_downsample_factor
        actual_t = min(actual_t, latent.shape[1])
        latent = latent[:, :actual_t]
        mask = mx.ones((1, actual_t), dtype=mx.bool_)

        # Align to speaker_patch_size
        p = self.config.dit.speaker_patch_size
        if p > 1 and actual_t % p != 0:
            trim = (actual_t // p) * p
            latent = latent[:, :trim]
            mask = mask[:, :trim]

        return latent, mask

    # ------------------------------------------------------------------
    # Latent generation (sampling)
    # ------------------------------------------------------------------

    def generate_latents(
        self,
        text: str,
        ref_latent: Optional[mx.array] = None,
        ref_mask: Optional[mx.array] = None,
        caption: Optional[str] = None,
        rng_seed: int = 0,
        **sampling_kwargs,
    ) -> mx.array:
        text_input_ids, text_mask = self._prepare_text(text)

        caption_input_ids: Optional[mx.array] = None
        caption_mask: Optional[mx.array] = None

        if self.config.dit.use_caption_condition:
            cap = caption or ""
            caption_input_ids, caption_mask = self._prepare_caption(cap)
        else:
            if ref_latent is None:
                ref_latent = mx.zeros((1, 1, self.config.dit.latent_dim))
            if ref_mask is None:
                ref_mask = mx.zeros((1, ref_latent.shape[1]), dtype=mx.bool_)

        sampler_cfg = dict(self.config.sampler.__dict__)
        for k, v in sampling_kwargs.items():
            if k in sampler_cfg:
                sampler_cfg[k] = v

        return sample_euler_cfg(
            model=self.model,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            rng_seed=rng_seed,
            latent_dim=self.config.dit.patched_latent_dim,
            **sampler_cfg,
        )

    # ------------------------------------------------------------------
    # Main generate interface
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | mx.array | None = None,
        caption: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        # instruct is an alias for caption (mlx-audio convention)
        caption = caption or kwargs.pop("instruct", None)
        if stream:
            raise NotImplementedError("Irodori-TTS streaming is not yet implemented.")

        if self.dacvae is None:
            raise ValueError(
                "Irodori-TTS requires DACVAE to be loaded. "
                "Use mlx_audio.tts.load(...) or set model.dacvae manually."
            )

        start_time = time.perf_counter()
        text_input_ids, _ = self._prepare_text(text)
        token_count = int(text_input_ids.shape[1])

        # Encode reference audio if provided
        ref_latent = None
        ref_mask = None
        if ref_audio is not None:
            audio = (
                load_audio_any(ref_audio, sample_rate=self.sample_rate)
                if isinstance(ref_audio, str)
                else ref_audio
            )
            if audio.ndim == 1:
                audio = audio[None, :]
            elif audio.ndim == 2 and audio.shape[0] > 1:
                audio = mx.mean(audio, axis=0, keepdims=True)
            ref_latent, ref_mask = self._encode_ref_audio(audio)

        # Run diffusion sampler
        latent_out = self.generate_latents(
            text=text,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption=caption,
            rng_seed=int(kwargs.get("rng_seed", 0)),
            **{k: v for k, v in kwargs.items() if k != "rng_seed"},
        )

        # Decode latent → waveform
        # latent_out: (1, T, latent_dim)
        latent_for_decode = mx.transpose(latent_out, (0, 2, 1))  # (1, latent_dim, T)
        # Use chunked decoding to avoid large ConvTranspose1d intermediates on
        # 16 GB unified-memory systems (stride-8 block creates a ~17 GB tensor
        # at T=750 in a single pass; 50-frame chunks keep it under ~1.2 GB).
        audio_out = self.dacvae.decode(latent_for_decode, chunk_size=50)  # (1, L, 1)
        audio_out = audio_out[:, :, 0]  # (1, L)
        mx.eval(audio_out)

        # Trim trailing silence
        silence_t = _find_silence_point(latent_out[0])
        trim_samples = silence_t * self.config.audio_downsample_factor
        audio_out = audio_out[:, :trim_samples]

        audio = audio_out[0]  # (L,)
        samples = int(audio.shape[0])
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        audio_duration_seconds = (
            samples / self.sample_rate if self.sample_rate > 0 else 0.0
        )

        h = int(audio_duration_seconds // 3600)
        m = int((audio_duration_seconds % 3600) // 60)
        s = int(audio_duration_seconds % 60)
        ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=audio_duration_seconds / elapsed,
            prompt={"tokens": token_count, "tokens-per-sec": token_count / elapsed},
            audio_samples={"samples": samples, "samples-per-sec": samples / elapsed},
            processing_time_seconds=elapsed,
            peak_memory_usage=float(mx.get_peak_memory() / 1e9),
        )

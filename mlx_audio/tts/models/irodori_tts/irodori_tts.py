from __future__ import annotations

import math
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
from .duration import build_duration_features
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
        self._model_path: Path | None = None

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

        model._model_path = Path(model_path)

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

    def _load_tokenizer(self, repo: str):
        """Prefer a tokenizer bundled with the converted weights (v4 ships one)
        so inference does not depend on the upstream repo staying available."""
        from transformers import AutoTokenizer

        if self._model_path is not None:
            bundled = self._model_path / "tokenizer"
            if (bundled / "tokenizer_config.json").is_file():
                return AutoTokenizer.from_pretrained(str(bundled))
        return AutoTokenizer.from_pretrained(
            repo, revision=self.config.dit.text_encoder_revision
        )

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer(self.config.dit.text_tokenizer_repo)
        return self._tokenizer

    def _get_caption_tokenizer(self):
        if self._caption_tokenizer is None:
            repo = self.config.dit.caption_tokenizer_repo_resolved
            # Reuse text tokenizer if repo is the same
            if repo == self.config.dit.text_tokenizer_repo:
                self._caption_tokenizer = self._get_tokenizer()
            else:
                self._caption_tokenizer = self._load_tokenizer(repo)
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

    def _load_ref_waveform(self, ref: str | mx.array) -> mx.array:
        """Load a reference clip as mono (1, samples) at the model sample rate."""
        audio = (
            load_audio_any(ref, sample_rate=self.sample_rate)
            if isinstance(ref, str)
            else ref
        )
        if audio.ndim == 1:
            audio = audio[None, :]
        elif audio.ndim == 2 and audio.shape[0] > 1:
            audio = mx.mean(audio, axis=0, keepdims=True)
        return audio

    def _max_ref_latent_steps(self, max_ref_seconds: Optional[float] = None) -> int:
        """Latent-step budget for the reference, from ref_max_seconds and the
        hard max_speaker_latent_length cap."""
        seconds = (
            self.config.ref_max_seconds
            if max_ref_seconds is None
            else float(max_ref_seconds)
        )
        limit = self.config.max_speaker_latent_length
        if seconds > 0:
            steps = int(
                seconds * self.config.sample_rate / self.config.audio_downsample_factor
            )
            limit = min(limit, max(1, steps))
        return limit

    def _encode_ref_audio(
        self,
        audio: mx.array,
        max_ref_seconds: Optional[float] = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Encode a single reference waveform with DACVAE.
        audio: (1, samples) at config.sample_rate
        Returns (latent, mask): latent (1, T, latent_dim), mask (1, T) bool
        """
        assert self.dacvae is not None, "DACVAE not loaded"

        max_steps = self._max_ref_latent_steps(max_ref_seconds)
        audio = audio[:, : max_steps * self.config.audio_downsample_factor]

        # DACVAE encode expects (B, L, 1)
        audio_in = audio[:, :, None]  # (1, L, 1)
        latent = self.dacvae.encode(audio_in)  # (1, latent_dim, T) channels-first
        latent = mx.transpose(latent, (0, 2, 1))  # (1, T, latent_dim) sequence-first

        actual_t = int(audio.shape[1]) // self.config.audio_downsample_factor
        actual_t = min(actual_t, latent.shape[1])
        latent = latent[:, :actual_t]

        return latent, mx.ones((1, actual_t), dtype=mx.bool_)

    def _encode_ref_audios(
        self,
        audios: list[mx.array],
        max_ref_seconds: Optional[float] = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Encode reference clips independently and concatenate them in input order,
        then trim to the reference budget. v4 was trained on multiple shorter
        clips from the same speaker rather than one long recording.
        """
        max_steps = self._max_ref_latent_steps(max_ref_seconds)

        pieces: list[mx.array] = []
        total = 0
        for audio in audios:
            latent, _ = self._encode_ref_audio(audio, max_ref_seconds=max_ref_seconds)
            if latent.shape[1] == 0:
                continue
            pieces.append(latent)
            total += int(latent.shape[1])
            if total >= max_steps:
                break
        if not pieces:
            raise ValueError("Reference audio produced an empty latent.")

        latent = mx.concatenate(pieces, axis=1)[:, :max_steps]
        steps = int(latent.shape[1])

        # Align to speaker_patch_size so no partial patch is dropped silently
        p = self.config.dit.speaker_patch_size
        if p > 1 and steps % p != 0:
            steps = (steps // p) * p
            latent = latent[:, :steps]
        if steps == 0:
            raise ValueError(
                f"Reference audio is too short: at least {p} latent frames "
                f"({p * self.config.audio_downsample_factor / self.config.sample_rate:.2f}s) "
                "are required."
            )

        return latent, mx.ones((1, steps), dtype=mx.bool_)

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
        seconds: Optional[float] = None,
        duration_scale: float = 1.0,
        min_seconds: float = 0.5,
        max_seconds: float = 30.0,
        **sampling_kwargs,
    ) -> tuple[mx.array, int]:
        """
        Generate latents from text and optional reference audio.
        Returns (latent, latent_steps) where latent_steps is the number of frames.
        """
        text_input_ids, text_mask = self._prepare_text(text)

        caption_input_ids: Optional[mx.array] = None
        caption_mask: Optional[mx.array] = None

        if self.config.dit.use_caption_condition:
            cap = caption or ""
            caption_input_ids, caption_mask = self._prepare_caption(cap)
            if not cap.strip():
                # An empty caption still tokenizes to a BOS token. Leaving it
                # unmasked would present it as a real caption to the DiT and to
                # the duration predictor, so mask it out entirely.
                caption_mask = mx.zeros_like(caption_mask)

        if self.config.dit.use_speaker_condition_resolved:
            if ref_latent is None:
                ref_latent = mx.zeros(
                    # At least one full speaker patch, otherwise patching
                    # yields an empty sequence (v4 uses speaker_patch_size=4).
                    (
                        1,
                        self.config.dit.speaker_patch_size,
                        self.config.dit.latent_dim,
                    )
                )
            if ref_mask is None:
                ref_mask = mx.zeros((1, ref_latent.shape[1]), dtype=mx.bool_)
        elif not self.config.dit.use_caption_condition:
            # legacy speaker-only (use_speaker_condition=None, use_caption_condition=False)
            if ref_latent is None:
                ref_latent = mx.zeros(
                    # At least one full speaker patch, otherwise patching
                    # yields an empty sequence (v4 uses speaker_patch_size=4).
                    (
                        1,
                        self.config.dit.speaker_patch_size,
                        self.config.dit.latent_dim,
                    )
                )
            if ref_mask is None:
                ref_mask = mx.zeros((1, ref_latent.shape[1]), dtype=mx.bool_)

        # Determine sequence length
        if seconds is not None:
            # Manual duration
            clamped_seconds = min(max_seconds, max(min_seconds, float(seconds)))
            target_samples = int(clamped_seconds * self.config.sample_rate)
            latent_steps = math.ceil(
                target_samples / self.config.audio_downsample_factor
            )
        elif self.config.dit.use_duration_predictor:
            # Predict duration using the integrated predictor
            text_for_features = normalize_text(text)
            token_count = int(text_mask.sum())
            has_speaker = bool(ref_mask is not None and mx.any(ref_mask))

            duration_features = build_duration_features(
                [text_for_features],
                token_counts=[token_count],
                max_text_len=self.config.max_text_length,
                has_speaker=[has_speaker],
            )

            # Encode conditions for duration prediction
            (
                text_state,
                text_mask_full,
                speaker_state,
                speaker_mask,
                caption_state_dur,
                caption_mask_dur,
            ) = self.model.encode_conditions_full(
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                caption_input_ids=caption_input_ids,
                caption_mask=caption_mask,
            )

            has_speaker_arr = mx.array([has_speaker], dtype=mx.bool_)
            has_caption = bool(
                caption_input_ids is not None
                and caption_mask is not None
                and mx.any(caption_mask)
            )
            has_caption_arr = mx.array([has_caption], dtype=mx.bool_)
            pred_log_frames = self.model.predict_duration_log_frames(
                text_state=text_state,
                text_mask=text_mask_full,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                duration_features=duration_features,
                has_speaker=has_speaker_arr,
                caption_state=caption_state_dur,
                caption_mask=caption_mask_dur,
                has_caption=has_caption_arr,
            )
            pred_frames = float(mx.expm1(pred_log_frames[0]).item())
            scaled_frames = pred_frames * duration_scale
            min_frames = max(
                1,
                math.ceil(
                    min_seconds
                    * self.config.sample_rate
                    / self.config.audio_downsample_factor
                ),
            )
            max_frames = max(
                1,
                math.floor(
                    max_seconds
                    * self.config.sample_rate
                    / self.config.audio_downsample_factor
                ),
            )
            latent_steps = int(round(scaled_frames))
            latent_steps = max(min_frames, min(max_frames, latent_steps))
        else:
            # Fallback to 30 seconds
            latent_steps = self.config.sampler.sequence_length

        patched_steps = math.ceil(latent_steps / self.config.dit.latent_patch_size)

        sampler_cfg = dict(self.config.sampler.__dict__)
        # Remove sequence_length since we compute it dynamically
        sampler_cfg.pop("sequence_length", None)
        for k, v in sampling_kwargs.items():
            if k in sampler_cfg:
                sampler_cfg[k] = v

        latent_out = sample_euler_cfg(
            model=self.model,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            rng_seed=rng_seed,
            latent_dim=self.config.dit.patched_latent_dim,
            sequence_length=patched_steps,
            **sampler_cfg,
        )

        return latent_out, latent_steps

    # ------------------------------------------------------------------
    # Main generate interface
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | mx.array | list[str | mx.array] | None = None,
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

        # Encode reference audio if provided. A list/tuple encodes each clip
        # separately and concatenates them (v4 multi-clip cloning).
        ref_latent = None
        ref_mask = None
        if ref_audio is not None:
            refs = ref_audio if isinstance(ref_audio, (list, tuple)) else [ref_audio]
            audios = [self._load_ref_waveform(r) for r in refs]
            ref_latent, ref_mask = self._encode_ref_audios(
                audios, max_ref_seconds=kwargs.get("max_ref_seconds")
            )

        # Run diffusion sampler
        latent_out, latent_steps = self.generate_latents(
            text=text,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption=caption,
            rng_seed=int(kwargs.get("rng_seed", 0)),
            seconds=kwargs.get("seconds"),
            duration_scale=float(kwargs.get("duration_scale", 1.0)),
            min_seconds=float(
                kwargs.get("min_seconds", self.config.sampler.min_seconds)
            ),
            max_seconds=float(
                kwargs.get("max_seconds", self.config.sampler.max_seconds)
            ),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in (
                    "rng_seed",
                    "seconds",
                    "duration_scale",
                    "min_seconds",
                    "max_seconds",
                    "max_ref_seconds",
                )
            },
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

        # Clamp to the predicted/manual duration, then trim trailing silence.
        # A silence point of 0 means the heuristic flagged the very first
        # window, which must not collapse the output to nothing.
        target_samples = latent_steps * self.config.audio_downsample_factor
        trim_samples = target_samples
        silence_samples = (
            _find_silence_point(latent_out[0]) * self.config.audio_downsample_factor
        )
        if silence_samples > 0:
            trim_samples = min(trim_samples, silence_samples)
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

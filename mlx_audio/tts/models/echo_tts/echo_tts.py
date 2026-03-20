from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.fish_s1_dac import DAC as FishS1DAC
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import load_audio as load_audio_any

from .audio import (
    PCAState,
    ae_decode,
    crop_audio_to_flattening_point,
    get_speaker_latent_and_mask,
    load_pca_state,
)
from .config import ModelConfig
from .model import EchoDiT
from .sampling import (
    sample_blockwise_euler_cfg_independent_guidances,
    sample_euler_cfg_independent_guidances,
)
from .text import get_text_input_ids_and_mask


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dit_kwargs = dict(config.dit.__dict__)
        dit_kwargs["enable_blockwise_modules"] = not config.delete_blockwise_modules
        self.model = EchoDiT(**dit_kwargs)
        self.fish_ae: FishS1DAC | None = None
        self.pca_state: PCAState | None = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def model_type(self) -> str:
        return self.config.model_type

    @property
    def latent_size(self) -> int:
        return self.config.dit.latent_size

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def sanitize(self, weights):
        skip = {"pca_components", "pca_mean", "latent_scale"}

        def _is_blockwise_key(key: str) -> bool:
            base = key[6:] if key.startswith("model.") else key
            return (
                base.startswith("latent_encoder.")
                or base.startswith("latent_norm.")
                or ".wk_latent." in base
                or ".wv_latent." in base
            )

        out = {}
        for k, v in weights.items():
            if k in skip:
                continue
            if self.config.delete_blockwise_modules and _is_blockwise_key(k):
                continue
            if k.startswith("cond_module.") and len(k.split(".")) > 1:
                parts = k.split(".")
                if parts[1].isdigit():
                    k = ".".join(["cond_module", "layers", parts[1], *parts[2:]])
            out[f"model.{k}" if not k.startswith("model.") else k] = v
        return out

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        pca_path = model_path / model.config.pca_filename
        if pca_path.exists():
            model.pca_state = load_pca_state(str(pca_path))

        try:
            model.fish_ae = FishS1DAC.from_pretrained(model.config.fish_codec_repo)
        except Exception:
            model.fish_ae = None

        return model

    def _prepare_text(self, text: str, max_length: Optional[int] = None):
        if max_length is None:
            max_length = self.config.max_text_length
        return get_text_input_ids_and_mask(
            [text],
            max_length=max_length,
            normalize=self.config.normalize_text,
            return_normalized_text=True,
            pad_to_max=False,
        )

    def generate_latents(
        self,
        text: str,
        speaker_latent: Optional[mx.array] = None,
        speaker_mask: Optional[mx.array] = None,
        rng_seed: int = 0,
        block_sizes: Optional[list[int]] = None,
        **sampling_kwargs,
    ) -> mx.array:
        text_input_ids, text_mask, _ = self._prepare_text(text)
        if speaker_latent is None:
            speaker_latent = mx.zeros(
                (1, self.config.dit.speaker_patch_size, self.latent_size)
            )
        if speaker_mask is None:
            speaker_mask = mx.zeros((1, speaker_latent.shape[1]), dtype=mx.bool_)

        default_sampling = dict(self.config.sampler.__dict__)
        # Ignore generic TTS kwargs (e.g. `speed`) that are not Echo sampler args.
        for k, v in sampling_kwargs.items():
            if k in default_sampling:
                default_sampling[k] = v

        if block_sizes is None:
            return sample_euler_cfg_independent_guidances(
                model=self.model,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                rng_seed=rng_seed,
                **default_sampling,
            )

        if self.config.delete_blockwise_modules:
            raise ValueError(
                "Blockwise generation requires latent-prefix modules. "
                "Set `delete_blockwise_modules=False` in the model config."
            )

        blockwise_sampling = dict(default_sampling)
        blockwise_sampling.pop("sequence_length", None)
        return sample_blockwise_euler_cfg_independent_guidances(
            model=self.model,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            rng_seed=rng_seed,
            block_sizes=block_sizes,
            **blockwise_sampling,
        )

    def generate(
        self,
        text: str,
        voice: str | None = None,
        ref_audio: str | mx.array | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        if stream:
            raise NotImplementedError("Echo-TTS streaming is not implemented yet.")

        if self.fish_ae is None or self.pca_state is None:
            raise ValueError(
                "Echo-TTS requires Fish S1 DAC and PCA state loaded. "
                "Load via `mlx_audio.tts.load(...)` or set `model.fish_ae` and `model.pca_state`."
            )

        start_time = time.perf_counter()
        text_input_ids, _, _ = self._prepare_text(text)
        token_count = int(text_input_ids.shape[1])

        speaker_latent = None
        speaker_mask = None
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

            speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                self.fish_ae,
                self.pca_state,
                audio,
                max_speaker_latent_length=self.config.max_speaker_latent_length,
                divis_by_patch_size=self.config.dit.speaker_patch_size,
            )

        latent_out = self.generate_latents(
            text=text,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            rng_seed=int(kwargs.get("rng_seed", 0)),
            block_sizes=kwargs.get("block_sizes"),
            **{k: v for k, v in kwargs.items() if k not in {"rng_seed", "block_sizes"}},
        )

        audio_out = ae_decode(self.fish_ae, self.pca_state, latent_out)
        audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])
        audio = audio_out[0, 0]

        samples = int(audio.shape[0])
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        audio_duration_seconds = (
            samples / self.sample_rate if self.sample_rate > 0 else 0.0
        )
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=audio_duration_seconds / elapsed if elapsed > 0 else 0.0,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": token_count / elapsed if elapsed > 0 else 0.0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples / elapsed if elapsed > 0 else 0.0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=float(mx.get_peak_memory() / 1e9),
        )

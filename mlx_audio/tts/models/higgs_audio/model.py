"""Higgs Audio v2 — framework-conforming entry point.

Defines `Model` and `ModelConfig` in the shape expected by
`mlx_audio.tts.utils.load()` and the top-level
`python -m mlx_audio.tts.generate` CLI.

`Model` subclasses `HiggsAudioModel` so the safetensors checkpoint loads
directly against it (no key remapping). Tokenizer + codec are attached
via `post_load_hook`; `generate(text, voice=..., ref_audio=..., ref_text=...)`
matches the framework's standard signature and yields a
`mlx_audio.tts.models.base.GenerationResult`.

The richer, kwarg-style API (`HiggsAudioServer.from_pretrained(...).generate(target_text=...)`)
in `serve.py` is preserved as an additional Python entrypoint.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import GenerationResult
from .config import HiggsAudioConfig
from .higgs_audio import HiggsAudioModel
from .serve import build_prompt, iter_overlap_add_pcm

# Higgs v2 codec emits one frame per ~40 ms (24 kHz × 8× upsample × 240
# patch). Used to convert the framework-level ``streaming_interval``
# (seconds) into ``emit_every_frames`` for the overlap-add helper.
_HIGGS_CODEC_FRAME_S = 0.04

# Framework loader reads `module.ModelConfig.from_dict(config_json)`; aliasing
# HiggsAudioConfig keeps a single source of truth.
ModelConfig = HiggsAudioConfig


_DEFAULT_CODEC_REPO = "mlx-community/higgs-audio-v2-tokenizer"


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class Model(HiggsAudioModel):
    """Framework-conforming wrapper around `HiggsAudioModel`.

    Inherits the full model structure so the checkpoint loads without any
    key remapping. Adds a tokenizer + codec (populated via `post_load_hook`)
    and a `generate()` method matching the standard TTS interface.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self._config = config
        self._tokenizer = None
        self._codec = None
        self._sample_rate = 24000

    # --- framework hooks ------------------------------------------------

    def model_quant_predicate(self, name: str, module: nn.Module) -> bool:
        """Quantize everything except the audio head + codebook embeddings.

        These two components are most sensitive to quantization noise: q4/q6
        pushed the audio head to collapse to stream-EOS or drift a semitone.
        Keeping them at bf16 preserves voice character; the Llama backbone
        and text head compress cleanly.
        """
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        protected = ("audio_codebook_embeddings", "audio_decoder_proj.audio_lm_head")
        return not any(p in name for p in protected)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path) -> "Model":
        """Attach the HF text tokenizer (from model_path) and the Higgs codec
        (from the default mlx-community repo, overridable post-hoc via
        `model.codec = ...`)."""
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from ....codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        model._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        codec_dir = snapshot_download(repo_id=_DEFAULT_CODEC_REPO)
        model._codec = HiggsAudioTokenizer.from_pretrained(codec_dir)
        return model

    # --- accessors ------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self._tokenizer = value

    @property
    def codec(self):
        return self._codec

    @codec.setter
    def codec(self, value) -> None:
        self._codec = value

    # --- generation -----------------------------------------------------

    def generate(  # type: ignore[override]
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio=None,
        ref_text: Optional[str] = None,
        max_new_frames: int = 1200,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        ras_win_len: Optional[int] = 7,
        ras_max_repeat: int = 2,
        sampling_warmup_frames: int = 0,
        fade_in_ms: float = 30.0,
        fade_out_ms: float = 15.0,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        overlap_ms: float = 40.0,
        **kwargs,  # absorb framework kwargs we don't use (speed, lang_code, ...)
    ) -> Iterator[GenerationResult]:
        """Synthesize ``text`` and yield ``GenerationResult``(s).

        When ``stream=False`` (default): yields a single ``GenerationResult``
        containing the full utterance — identical to the pre-streaming path.

        When ``stream=True``: yields intermediate chunks via overlap-add
        mid-generation streaming, with ``is_streaming_chunk=True`` on every
        yield and ``is_final_chunk=True`` on the last. ``streaming_interval``
        (seconds) controls chunk granularity — a Higgs codec frame is ~40 ms,
        so the framework default of 2.0 s ≈ 50 frames per chunk. Lower
        interval means lower TTFB at the cost of more re-decode overhead;
        ~0.6 s is a reasonable conversational-voice setting.

        Voice cloning: pass ``ref_audio`` (mono 24 kHz ``mx.array`` or
        ``numpy.ndarray``, loaded by the top-level CLI from ``--ref_audio``)
        together with ``ref_text`` (its transcript). Without ``ref_audio``,
        runs smart-voice mode — works but less reliable; recommended to
        always supply a reference for production use.
        """
        if self._tokenizer is None or self._codec is None:
            raise RuntimeError(
                "Model not fully loaded — tokenizer/codec missing. Load via "
                "mlx_audio.tts.utils.load(...) so post_load_hook runs."
            )

        ref_audio_24k = None
        if ref_audio is not None:
            ref_audio_24k = np.asarray(ref_audio, dtype=np.float32).reshape(-1)

        full_embeds, audio_out_mask, _info = build_prompt(
            text,
            ref_text=ref_text,
            ref_audio_24k=ref_audio_24k,
            config=self._config,
            tokenizer=self._tokenizer,
            codec=self._codec,
            embed_tokens=self.embed_tokens,
            audio_codebook_embeddings=self.audio_codebook_embeddings,
        )

        if stream:
            yield from self._generate_streaming(
                full_embeds=full_embeds,
                audio_out_mask=audio_out_mask,
                max_new_frames=max_new_frames,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ras_win_len=ras_win_len,
                ras_max_repeat=ras_max_repeat,
                sampling_warmup_frames=sampling_warmup_frames,
                streaming_interval=streaming_interval,
                overlap_ms=overlap_ms,
                fade_in_ms=fade_in_ms,
                fade_out_ms=fade_out_ms,
            )
        else:
            yield from self._generate_oneshot(
                full_embeds=full_embeds,
                audio_out_mask=audio_out_mask,
                max_new_frames=max_new_frames,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ras_win_len=ras_win_len,
                ras_max_repeat=ras_max_repeat,
                sampling_warmup_frames=sampling_warmup_frames,
                fade_in_ms=fade_in_ms,
                fade_out_ms=fade_out_ms,
            )

    # --- generation backends -------------------------------------------

    def _generate_oneshot(
        self,
        *,
        full_embeds: mx.array,
        audio_out_mask: mx.array,
        max_new_frames: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        ras_win_len: Optional[int],
        ras_max_repeat: int,
        sampling_warmup_frames: int,
        fade_in_ms: float,
        fade_out_ms: float,
    ) -> Iterator[GenerationResult]:
        """Non-streaming path — generate all frames, decode once, yield one result."""
        start = time.perf_counter()

        # Call the inherited HiggsAudioModel.generate (delay-pattern state machine).
        aligned, gen_info = HiggsAudioModel.generate(
            self,
            inputs_embeds=full_embeds,
            audio_out_mask=audio_out_mask,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=ras_win_len,
            ras_max_repeat=ras_max_repeat,
            sampling_warmup_frames=sampling_warmup_frames,
        )
        mx.eval(aligned)

        pcm = self._codec.decode(aligned.T)
        mx.eval(pcm)
        pcm_np = np.array(pcm).astype(np.float32).reshape(-1)

        sr = self._sample_rate
        n_in = int(fade_in_ms * sr / 1000.0)
        n_out = int(fade_out_ms * sr / 1000.0)
        if n_in > 0 and pcm_np.size > n_in:
            pcm_np[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        if n_out > 0 and pcm_np.size > n_out:
            pcm_np[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)

        audio_mx = mx.array(pcm_np)
        samples = audio_mx.shape[0]
        elapsed = time.perf_counter() - start
        duration_s = samples / sr
        rtf = elapsed / duration_s if duration_s > 0 else 0.0
        tok = gen_info["num_frames_aligned"]

        yield GenerationResult(
            audio=audio_mx,
            samples=samples,
            sample_rate=sr,
            segment_idx=0,
            token_count=tok,
            audio_duration=_format_duration(duration_s),
            real_time_factor=round(rtf, 3),
            prompt={
                "tokens": tok,
                "tokens-per-sec": round(tok / elapsed, 2) if elapsed > 0 else 0.0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0.0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _generate_streaming(
        self,
        *,
        full_embeds: mx.array,
        audio_out_mask: mx.array,
        max_new_frames: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        ras_win_len: Optional[int],
        ras_max_repeat: int,
        sampling_warmup_frames: int,
        streaming_interval: float,
        overlap_ms: float,
        fade_in_ms: float,
        fade_out_ms: float,
    ) -> Iterator[GenerationResult]:
        """Streaming path — overlap-add mid-generation, one ``GenerationResult`` per chunk."""
        sr = self._sample_rate
        emit_every_frames = max(1, int(streaming_interval / _HIGGS_CODEC_FRAME_S))

        chunks_iter = iter_overlap_add_pcm(
            model=self,
            codec=self._codec,
            config=self._config,
            full_embeds=full_embeds,
            audio_out_mask=audio_out_mask,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=ras_win_len,
            ras_max_repeat=ras_max_repeat,
            sampling_warmup_frames=sampling_warmup_frames,
            emit_every_frames=emit_every_frames,
            overlap_ms=overlap_ms,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            sample_rate=sr,
        )

        chunk_idx = 0
        prev_frames_total = 0
        chunk_start = time.perf_counter()
        for pcm_chunk, meta in chunks_iter:
            elapsed = time.perf_counter() - chunk_start
            chunk_samples = int(pcm_chunk.size)
            chunk_duration = chunk_samples / sr if sr > 0 else 0.0
            frames_total = int(meta.get("frames_total", 0))
            chunk_tokens = max(0, frames_total - prev_frames_total)
            is_final = bool(meta.get("is_final"))

            yield GenerationResult(
                audio=mx.array(pcm_chunk),
                samples=chunk_samples,
                sample_rate=sr,
                segment_idx=chunk_idx,
                token_count=chunk_tokens,
                audio_duration=_format_duration(chunk_duration),
                real_time_factor=(
                    round(chunk_duration / elapsed, 3) if elapsed > 0 else 0.0
                ),
                prompt={
                    "tokens": chunk_tokens,
                    "tokens-per-sec": (
                        round(chunk_tokens / elapsed, 2) if elapsed > 0 else 0.0
                    ),
                },
                audio_samples={
                    "samples": chunk_samples,
                    "samples-per-sec": (
                        round(chunk_samples / elapsed, 2) if elapsed > 0 else 0.0
                    ),
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=True,
                is_final_chunk=is_final,
            )

            chunk_idx += 1
            prev_frames_total = frames_total
            chunk_start = time.perf_counter()

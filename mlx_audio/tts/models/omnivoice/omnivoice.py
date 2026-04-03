import math
import time
from pathlib import Path
from typing import Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import GenerationResult
from .backbone import BackboneConfig, OmniVoiceBackbone
from .config import OmniVoiceConfig


class Model(nn.Module):
    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config

        llm_cfg = config.llm_config or {}
        self.backbone = OmniVoiceBackbone(
            BackboneConfig(
                **{
                    k: v
                    for k, v in llm_cfg.items()
                    if k in BackboneConfig.__dataclass_fields__
                }
            )
        )

        hidden = self.backbone.embed_tokens.weight.shape[-1]
        C = config.num_audio_codebook
        V = config.audio_vocab_size  # 1025 (includes mask token)

        # 8 independent embedding tables for 8 codebooks
        self.audio_embeddings: List[nn.Embedding] = [
            nn.Embedding(V, hidden) for _ in range(C)
        ]
        # 8 independent prediction heads
        self.audio_heads: List[nn.Linear] = [
            nn.Linear(hidden, V, bias=False) for _ in range(C)
        ]

    def _embed(
        self,
        input_ids: mx.array,  # [B, S] text token ids
        audio_tokens: mx.array,  # [B, T, 8] audio codebook tokens (may include MASK_ID)
    ) -> mx.array:  # [B, S+T, hidden]
        text_embeds = self.backbone.embed_tokens(input_ids)  # [B, S, H]
        # Sum embeddings across 8 codebooks (not concat)
        audio_embeds = sum(
            self.audio_embeddings[i](audio_tokens[:, :, i])
            for i in range(self.config.num_audio_codebook)
        )  # [B, T, H]
        return mx.concatenate([text_embeds, audio_embeds], axis=1)  # [B, S+T, H]

    def build_cond_embeds(
        self,
        input_ids: mx.array,  # [1, S]
        ref_tokens: Optional[mx.array] = None,  # [1, T_ref, 8]
    ) -> mx.array:  # [1, S + T_ref, D]
        """Build conditioning embedding: text + optional reference audio tokens."""
        text_embeds = self.backbone.embed_tokens(input_ids)  # [1, S, D]
        if ref_tokens is None:
            return text_embeds
        ref_embeds = sum(
            self.audio_embeddings[i](ref_tokens[:, :, i])
            for i in range(self.config.num_audio_codebook)
        )  # [1, T_ref, D]
        return mx.concatenate([text_embeds, ref_embeds], axis=1)

    def __call__(
        self,
        inputs_embeds: mx.array,  # [B, prefix_len+T, D]
        prefix_len: int,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:  # [B, T, 8, V]
        hidden = self.backbone(inputs_embeds, attention_mask)  # [B, prefix_len+T, H]
        audio_hidden = hidden[:, prefix_len:, :]  # [B, T, H]
        logits = mx.stack(
            [
                self.audio_heads[i](audio_hidden)
                for i in range(self.config.num_audio_codebook)
            ],
            axis=2,
        )  # [B, T, 8, V]
        return logits

    def sanitize(self, weights: dict) -> dict:
        """Remap k2-fsa/OmniVoice PyTorch keys to mlx-audio naming convention.

        Key transforms:
        - ``llm.*``               → ``backbone.*``
        - ``audio_embeddings.weight [8*V, H]`` → 8× ``audio_embeddings.N.weight [V, H]``
        - ``audio_heads.weight [8*V, H]``      → 8× ``audio_heads.N.weight [V, H]``
        - ``codebook_layer_offsets``           → dropped (not needed)
        """
        C = self.config.num_audio_codebook  # 8
        V = self.config.audio_vocab_size  # 1025
        result = {}
        for k, v in weights.items():
            if k == "codebook_layer_offsets":
                continue
            elif k == "audio_embeddings.weight":
                for i in range(C):
                    result[f"audio_embeddings.{i}.weight"] = v[i * V : (i + 1) * V]
            elif k == "audio_heads.weight":
                for i in range(C):
                    result[f"audio_heads.{i}.weight"] = v[i * V : (i + 1) * V]
            elif k.startswith("llm."):
                result["backbone." + k[4:]] = v
            else:
                result[k] = v
        return result

    def _encode_text(
        self,
        text: str,
        language: str = "None",
        instruct: str = "None",
        text_tokenizer=None,
    ) -> mx.array:
        """Wrap text with OmniVoice special tokens and encode to ids."""
        wrapped = (
            f"<|denoise|>"
            f"<|lang_start|>{language}<|lang_end|>"
            f"<|instruct_start|>{instruct}<|instruct_end|>"
            f"<|text_start|>{text}<|text_end|>"
        )
        ids = text_tokenizer.encode(wrapped, add_special_tokens=True)
        return mx.array(ids, dtype=mx.int32)

    def generate(
        self,
        text: Optional[str] = None,
        duration_s: float = 5.0,
        language: str = "None",
        lang_code: str = "None",  # alias used by generate_audio()
        instruct: str = "None",
        ref_audio=None,  # str | Path | mx.array (pre-loaded at sample_rate)
        ref_audio_max_duration_s: float = 10.0,
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
        position_temperature: float = 5.0,
        layer_penalty_factor: float = 5.0,
        t_shift: float = 0.1,
        tokenizer=None,
        text_tokenizer=None,
        # low-level override: pre-encoded ref tokens
        ref_tokens: Optional[mx.array] = None,
        # low-level override: pre-encoded text ids
        input_ids: Optional[mx.array] = None,
        **kwargs,  # absorb: voice, speed, cfg_scale, temperature, max_tokens, etc.
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        Args:
            text: Input text to synthesize.
            duration_s: Maximum output duration in seconds (default 5.0).
            language: BCP-47 language tag e.g. "en", "ru", "zh" (default "None" = auto).
            instruct: Style instruction tag (default "None").
            ref_audio: Path to reference audio file for voice cloning (WAV, any SR).
                       Requires ``tokenizer`` to be set.
            ref_audio_max_duration_s: Clip reference audio to this many seconds (default 10).
            num_steps: Iterative unmasking steps (default 32).
            guidance_scale: CFG strength (default 2.0; 0 = no CFG).
            class_temperature: Token sampling temperature (0 = greedy, default 0.0).
            position_temperature: Gumbel noise on confidence scores (default 5.0).
            layer_penalty_factor: Layer-ordering penalty (default 5.0).
            t_shift: Time-step warp factor (default 0.1).
            tokenizer: HiggsAudioTokenizer instance. Required to decode audio or
                       encode ref_audio for voice cloning.
            text_tokenizer: Hugging Face tokenizer for text encoding (AutoTokenizer).
                            Required unless ``input_ids`` is provided directly.
            ref_tokens: Pre-encoded reference tokens [T_ref, 8]. When set, ``ref_audio``
                        is ignored.
            input_ids: Pre-encoded text token ids [S]. When set, ``text`` and
                       ``text_tokenizer`` are ignored.

        Yields:
            GenerationResult with audio, metrics, and token count.
        """
        from .generation import iterative_unmask
        from .utils import create_voice_clone_prompt

        # lang_code is the name used by generate_audio(); language takes precedence
        if language == "None" and lang_code != "None":
            language = lang_code

        # Fall back to tokenizers set by post_load_hook
        if text_tokenizer is None:
            text_tokenizer = getattr(self, "text_tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self, "audio_tokenizer", None)

        # --- text encoding ---
        if input_ids is None:
            if text_tokenizer is None:
                raise ValueError(
                    "text_tokenizer is required when input_ids is not provided. "
                    "Pass an AutoTokenizer or use input_ids directly."
                )
            input_ids = self._encode_text(text, language, instruct, text_tokenizer)

        # --- voice cloning ---
        if ref_tokens is None and ref_audio is not None:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer (HiggsAudioTokenizer) is required for voice cloning via ref_audio."
                )
            if isinstance(ref_audio, (str, Path)):
                ref_tokens = create_voice_clone_prompt(
                    str(ref_audio),
                    tokenizer=tokenizer,
                    max_duration_s=ref_audio_max_duration_s,
                )
            else:
                # Already loaded as 1D mx.array at self.config.sample_rate by generate_audio()
                wav = ref_audio
                if not isinstance(wav, mx.array):
                    wav = mx.array(wav)
                if wav.ndim == 1:
                    wav = wav[None, :, None]  # [1, T, 1]
                ref_tokens = tokenizer.encode(wav)[0]  # [T_ref, 8]

        # HiggsAudio hop = 8*5*4*2*3 = 960 samples/token at 24kHz → 25 tokens/sec
        T = math.ceil(duration_s * self.config.sample_rate / 960)
        input_ids_b = input_ids[None]  # [1, S]
        ref_b = ref_tokens[None] if ref_tokens is not None else None

        cond_embeds = self.build_cond_embeds(input_ids_b, ref_b)
        # Unconditional: empty prefix — backbone sees only audio mask tokens
        uncond_embeds = mx.zeros((1, 0, cond_embeds.shape[-1]), dtype=cond_embeds.dtype)

        start_time = time.time()
        tokens = iterative_unmask(
            self,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            T=T,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
            position_temperature=position_temperature,
            layer_penalty_factor=layer_penalty_factor,
            t_shift=t_shift,
        )
        elapsed = time.time() - start_time

        if tokenizer is not None:
            audio = tokenizer.decode(tokens)  # [T*960] 1D for 2D tokens input
        else:
            audio = mx.zeros((T * 960,), dtype=mx.float32)

        n_samples = T * 960
        audio_duration_s = n_samples / self.config.sample_rate
        rtf = audio_duration_s / elapsed if elapsed > 0 else 0.0
        d = int(audio_duration_s)
        duration_str = f"{d // 3600:02d}:{(d % 3600) // 60:02d}:{d % 60:02d}.{int((audio_duration_s % 1) * 1000):03d}"

        yield GenerationResult(
            audio=audio,
            samples=n_samples,
            sample_rate=self.config.sample_rate,
            segment_idx=0,
            token_count=T,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": T,
                "tokens-per-sec": round(T / elapsed, 2) if elapsed > 0 else 0,
            },
            audio_samples={
                "samples": n_samples,
                "samples-per-sec": round(n_samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    @property
    def model_type(self) -> str:
        return self.config.model_type

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        """Load text tokenizer and HiggsAudio tokenizer after weight loading."""
        import warnings

        try:
            from transformers import AutoTokenizer

            model.text_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception as e:
            warnings.warn(f"Could not load text tokenizer: {e}")
            model.text_tokenizer = None

        try:
            from mlx_audio.codec.models.higgs_audio.higgs_audio import (
                HiggsAudioTokenizer,
            )

            model.audio_tokenizer = HiggsAudioTokenizer.from_pretrained(str(model_path))
        except Exception as e:
            warnings.warn(f"Could not load audio tokenizer: {e}")
            model.audio_tokenizer = None

        return model

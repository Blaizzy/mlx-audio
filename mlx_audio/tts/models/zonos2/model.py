from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import BatchKVCache, make_prompt_cache
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import BatchGenerationResult, GenerationResult
from .config import Zonos2Config
from .generation import (
    TTSSamplingParams,
    Zonos2GenerationState,
    format_duration,
    sample_frame,
)
from .prompt import TTSPromptBuilder, TTSPromptConfig, shear_up
from .textnorm import TTSTextNormalizer

ModelConfig = Zonos2Config


def _rms_norm(x: mx.array, weight: Optional[mx.array], eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


class Zonos2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        if affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight if "weight" in self else None, self.eps)


class Zonos2FusedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        if affine:
            self.weight = mx.ones((dim,))

    def __call__(
        self, x: mx.array, residual: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        if residual is None:
            return _rms_norm(x, self.weight if "weight" in self else None, self.eps), x
        residual = residual + x
        return (
            _rms_norm(residual, self.weight if "weight" in self else None, self.eps),
            residual,
        )


class ChunkedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, divisor: int):
        super().__init__()
        if out_features % divisor:
            raise ValueError("out_features must be divisible by divisor")
        scale = in_features**-0.5
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.divisor = int(divisor)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(divisor, out_features // divisor, in_features),
        )

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight.reshape(self.out_features, self.in_features)
        return x @ weight.T


class MultiEmbedding(nn.Module):
    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.embedders = [
            nn.Embedding(config.audio_vocab_size, config.dim)
            for _ in range(config.n_codebooks)
        ]
        if config.text_vocab is None:
            raise ValueError("ZONOS2 requires text_vocab")
        self.embedders.append(nn.Embedding(config.text_vocab + 1, config.dim))

    def __call__(self, input_ids: mx.array) -> mx.array:
        if input_ids.shape[-1] != len(self.embedders):
            raise ValueError(
                f"expected frame width {len(self.embedders)}, got {input_ids.shape[-1]}"
            )
        out = self.embedders[0](input_ids[..., 0])
        for idx in range(1, len(self.embedders)):
            out = out + self.embedders[idx](input_ids[..., idx])
        return out


class Zonos2Attention(nn.Module):
    def __init__(self, config: Zonos2Config, layer_idx: int):
        super().__init__()
        del layer_idx
        self.hidden_size = config.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.wq = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.wkv = ChunkedLinear(
            config.dim, 2 * self.num_kv_heads * self.head_dim, divisor=2
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, config.dim, bias=False)
        self.temp = mx.ones((1, self.num_heads, 1), dtype=mx.float32)
        self.gater = nn.Linear(config.dim, self.num_heads, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            # Reference FlashInfer rotary uses is_neox=False, i.e. interleaved
            # consecutive-pair rotation. In MLX this is traditional=True.
            traditional=True,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        gate = mx.sigmoid(self.gater(x))
        q = self.wq(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        kv = self.wkv(x)
        kv_dim = self.num_kv_heads * self.head_dim
        k, v = mx.split(kv, [kv_dim], axis=-1)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = _rms_norm(q, None, 1e-6) * mx.abs(self.temp).astype(q.dtype)
        k = _rms_norm(k, None, 1e-6)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache, self.scale, mask)
        out = out.transpose(0, 2, 1, 3).reshape(
            bsz, seqlen, self.num_heads, self.head_dim
        )
        out = out * gate[..., None]
        out = out.reshape(bsz, seqlen, self.num_heads * self.head_dim)
        return self.wo(out)


class DenseFeedForward(nn.Module):
    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.w_in = ChunkedLinear(config.dim, 2 * self.intermediate_size, divisor=2)
        self.w_out = nn.Linear(self.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h_gate = self.w_in(x)
        h, gate = mx.split(h_gate, [self.intermediate_size], axis=-1)
        return self.w_out(h * nn.silu(gate))


class RouterMLP(nn.Module):
    def __init__(self, router_dim: int, num_experts: int):
        super().__init__()
        self.l0 = nn.Linear(router_dim, router_dim, bias=True)
        self.l2 = nn.Linear(router_dim, router_dim, bias=True)
        self.l4 = nn.Linear(router_dim, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.l0(x))
        x = nn.gelu(self.l2(x))
        return self.l4(x)


class Zonos2Router(nn.Module):
    def __init__(self, config: Zonos2Config, layer_idx: int):
        super().__init__()
        self.router_dim = config.moe_router_dim
        self.num_experts = config.moe_n_experts
        self.top_k = config.num_experts_per_tok(layer_idx)
        self.use_legacy_balancing = config.moe_balancing_strategy != "quantile"
        self.use_eda = layer_idx != config.moe_start_from_layer
        self.down_proj = nn.Linear(config.dim, self.router_dim, bias=True)
        self.router_mlp = RouterMLP(self.router_dim, self.num_experts)
        self.rmsnorm_eda = Zonos2RMSNorm(self.router_dim, config.norm_eps, affine=True)
        if self.use_eda:
            self.router_states_scale = mx.ones((self.router_dim,))
        self.balancing_biases = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(
        self, x: mx.array, router_states: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        hidden = self.down_proj(x)
        if self.use_eda and router_states is not None:
            hidden = hidden + router_states * self.router_states_scale
        next_router_states = hidden
        hidden = self.rmsnorm_eda(hidden)
        expert_prob = mx.softmax(self.router_mlp(hidden).astype(mx.float32), axis=-1)
        bias = self.balancing_biases.astype(mx.float32)
        routing_scores = (
            expert_prob + bias if self.use_legacy_balancing else expert_prob - bias
        )

        if self.top_k == 1:
            topk_ids = mx.argmax(routing_scores, axis=-1, keepdims=True)
        else:
            topk_ids = mx.argpartition(routing_scores, kth=-self.top_k, axis=-1)[
                ..., -self.top_k :
            ]
            selected = mx.take_along_axis(routing_scores, topk_ids, axis=-1)
            order = mx.argsort(-selected, axis=-1)
            topk_ids = mx.take_along_axis(topk_ids, order, axis=-1)

        topk_weights = mx.take_along_axis(expert_prob, topk_ids, axis=-1)
        return topk_weights, mx.stop_gradient(topk_ids), next_router_states


class Zonos2MoEFeedForward(nn.Module):
    def __init__(self, config: Zonos2Config, layer_idx: int):
        super().__init__()
        self.router = Zonos2Router(config, layer_idx)
        self.experts = SwitchGLU(
            config.dim,
            config.intermediate_size,
            config.moe_n_experts,
            bias=False,
        )
        self.norm_topk_prob = bool(config.norm_topk_prob)

    def __call__(
        self, x: mx.array, router_states: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        topk_weights, topk_ids, next_router_states = self.router(x, router_states)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(axis=-1, keepdims=True) + 1e-8
            )
        expert_outputs = self.experts(x, topk_ids)
        out = (expert_outputs.astype(mx.float32) * topk_weights[..., None]).sum(axis=-2)
        return out.astype(x.dtype), next_router_states


class Zonos2Block(nn.Module):
    def __init__(self, config: Zonos2Config, layer_idx: int):
        super().__init__()
        self.attention = Zonos2Attention(config, layer_idx)
        self.attention_norm = Zonos2FusedRMSNorm(
            config.dim, config.norm_eps, affine=True
        )
        self.ffn_norm = Zonos2FusedRMSNorm(config.dim, config.norm_eps, affine=True)
        self.is_moe = config.is_moe_layer(layer_idx)
        if self.is_moe:
            self.feed_forward = Zonos2MoEFeedForward(config, layer_idx)
        else:
            self.feed_forward = DenseFeedForward(config)

    def __call__(
        self,
        x: mx.array,
        residual: Optional[mx.array],
        router_states: Optional[mx.array],
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> tuple[mx.array, mx.array, Optional[mx.array]]:
        x, residual = self.attention_norm(x, residual)
        x = self.attention(x, mask=mask, cache=cache)
        x, residual = self.ffn_norm(x, residual)
        if self.is_moe:
            x, router_states = self.feed_forward(x, router_states)
        else:
            x = self.feed_forward(x)
            router_states = None
        return x, residual, router_states


class Model(nn.Module):
    preserve_ref_audio_path = True

    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.multi_embedder = MultiEmbedding(config)
        self.emb_norm = Zonos2FusedRMSNorm(config.dim, config.norm_eps, affine=False)
        self.speaker_lda_projection = (
            nn.Linear(
                config.speaker_embedding_dim, int(config.speaker_lda_dim), bias=True
            )
            if config.speaker_enabled and config.speaker_lda_dim
            else None
        )
        speaker_dim = (
            int(config.speaker_lda_dim)
            if self.speaker_lda_projection is not None
            else config.speaker_embedding_dim
        )
        self.speaker_projection = (
            nn.Linear(speaker_dim, config.dim, bias=True)
            if config.speaker_enabled
            else None
        )
        self.layers = [Zonos2Block(config, i) for i in range(config.n_layers)]
        self.out_norm = Zonos2FusedRMSNorm(config.dim, config.norm_eps, affine=True)
        self.multi_output = nn.Linear(
            config.dim,
            config.n_codebooks * config.audio_vocab_size,
            bias=False,
        )
        self._prompt_builder = self._make_prompt_builder()
        self._text_normalizer: Optional[TTSTextNormalizer] = None
        self._dac = None
        self._speaker_extractor = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model.config.model_path = str(model_path)
        return model

    def _make_prompt_builder(self) -> TTSPromptBuilder:
        cfg = self.config
        return TTSPromptBuilder(
            TTSPromptConfig(
                n_codebooks=cfg.n_codebooks,
                audio_pad_id=cfg.audio_pad_id,
                text_vocab=int(cfg.text_vocab),
                speaking_rate_num_buckets=cfg.speaking_rate_num_buckets,
                quality_bucket_counts=cfg.quality_bucket_counts,
                speaker_background_num_buckets=cfg.speaker_background_num_buckets,
                accurate_mode_num_buckets=cfg.accurate_mode_num_buckets,
                prepend_silence=True,
            )
        )

    def _normalize_text(
        self, text: str, *, language: str, text_normalization: bool
    ) -> str:
        if not text_normalization:
            return text
        if self._text_normalizer is None:
            self._text_normalizer = TTSTextNormalizer()
        return self._text_normalizer.normalize(text, language)

    def _resolve_quality_buckets(
        self, quality_buckets: dict[str, int | None] | list[int | None] | str | None
    ) -> list[int | None] | None:
        counts = self.config.quality_bucket_counts
        if not counts:
            return None
        if quality_buckets is None:
            quality_buckets = {"trailing_silence_s": 3}
        if isinstance(quality_buckets, str):
            value = quality_buckets.strip()
            if value.startswith(("{", "[")):
                quality_buckets = json.loads(value)
            else:
                quality_buckets = [
                    None if item.strip().lower() in {"", "none", "null"} else int(item)
                    for item in value.split(",")
                ]
        if isinstance(quality_buckets, dict):
            return [
                quality_buckets.get(feature) for feature in self.config.quality_features
            ]
        result = list(quality_buckets)[: len(counts)]
        result.extend([None] * (len(counts) - len(result)))
        return result

    def _load_speaker_embedding(self, speaker_embedding: Any) -> Optional[mx.array]:
        if speaker_embedding is None:
            return None
        if isinstance(speaker_embedding, (str, Path)):
            data = mx.load(str(speaker_embedding))
            if isinstance(data, dict):
                if not data:
                    raise ValueError("speaker embedding archive is empty")
                data = next(iter(data.values()))
            speaker_embedding = data
        arr = mx.array(speaker_embedding, dtype=mx.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"speaker_embedding must be 1-D, got {arr.shape}")
        if arr.shape[0] != self.config.speaker_embedding_dim:
            raise ValueError(
                f"speaker_embedding must have dim {self.config.speaker_embedding_dim}, "
                f"got {arr.shape[0]}"
            )
        return arr[None, :]

    def _load_speaker_extractor(self):
        if self._speaker_extractor is None:
            from .speaker_encoder import (
                Zonos2SpeakerEmbeddingExtractor,
                resolve_speaker_encoder_path,
            )

            encoder_path = resolve_speaker_encoder_path(
                self.config.model_path,
                speaker_encoder_path=self.config.speaker_encoder_path,
                speaker_encoder_model_id=self.config.speaker_encoder_model_id,
            )
            self._speaker_extractor = Zonos2SpeakerEmbeddingExtractor(
                encoder_path,
                sample_rate=self.config.speaker_encoder_sample_rate,
            )
        return self._speaker_extractor

    def extract_speaker_embedding(
        self,
        ref_audio: Any,
        *,
        sample_rate: int | None = None,
    ) -> mx.array:
        if isinstance(ref_audio, (list, tuple)) and not (
            len(ref_audio) == 2 and isinstance(ref_audio[1], int)
        ):
            if len(ref_audio) != 1:
                raise ValueError("ZONOS2 speaker cloning expects one reference audio")
            ref_audio = ref_audio[0]
        extractor = self._load_speaker_extractor()
        embedding = extractor.encode(ref_audio, sample_rate=sample_rate)
        if embedding.shape[-1] != self.config.speaker_embedding_dim:
            raise ValueError(
                f"speaker encoder produced dim {embedding.shape[-1]}, "
                f"expected {self.config.speaker_embedding_dim}"
            )
        return embedding

    def _resolve_speaker_embedding(
        self,
        *,
        speaker_embedding: Any,
        ref_audio: Any,
        ref_audio_sample_rate: int | None,
    ) -> Optional[mx.array]:
        if speaker_embedding is not None and ref_audio is not None:
            raise ValueError("provide either speaker_embedding or ref_audio, not both")
        if speaker_embedding is not None:
            return self._load_speaker_embedding(speaker_embedding)
        if ref_audio is not None:
            return self.extract_speaker_embedding(
                ref_audio,
                sample_rate=ref_audio_sample_rate,
            )
        return None

    def _inject_speaker(
        self,
        x: mx.array,
        speaker_embedding: Optional[mx.array],
        speaker_token_position: Any,
    ) -> mx.array:
        if (
            speaker_embedding is None
            or speaker_token_position is None
            or self.speaker_projection is None
        ):
            return x
        emb = speaker_embedding
        if self.speaker_lda_projection is not None:
            emb = self.speaker_lda_projection(
                emb.astype(self.speaker_lda_projection.weight.dtype)
            )
        projected = self.speaker_projection(
            emb.astype(self.speaker_projection.weight.dtype)
        ).astype(x.dtype)
        if x.shape[0] != projected.shape[0]:
            if projected.shape[0] == 1:
                projected = mx.broadcast_to(projected, (x.shape[0], projected.shape[1]))
            else:
                raise ValueError("speaker batch size does not match input batch size")
        if isinstance(speaker_token_position, int):
            pos = int(speaker_token_position)
            if pos < 0 or pos >= x.shape[1]:
                return x
            return mx.concatenate(
                [x[:, :pos, :], projected[:, None, :], x[:, pos + 1 :, :]],
                axis=1,
            )
        positions = mx.array(speaker_token_position, dtype=mx.int32)
        if positions.ndim == 0:
            positions = mx.broadcast_to(positions[None], (x.shape[0],))
        if positions.shape[0] != x.shape[0]:
            raise ValueError("speaker token positions must match input batch size")
        valid = (positions >= 0) & (positions < x.shape[1])
        mask = (mx.arange(x.shape[1])[None, :] == positions[:, None]) & valid[:, None]
        return mx.where(mask[..., None], projected[:, None, :], x)

    def _forward_hidden(
        self,
        input_ids: mx.array,
        cache=None,
        speaker_embedding: Optional[mx.array] = None,
        speaker_token_position: Any = None,
    ) -> mx.array:
        h = self.multi_embedder(input_ids)
        h = self._inject_speaker(h, speaker_embedding, speaker_token_position)
        h, _ = self.emb_norm(h, None)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        residual = None
        router_states = None
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            h, residual, router_states = layer(
                h, residual, router_states, mask, layer_cache
            )
        h, _ = self.out_norm(h, residual)
        return h

    def compute_logits(self, hidden: mx.array) -> mx.array:
        logits = self.multi_output(hidden)
        logits = logits.reshape(
            *logits.shape[:-1],
            self.config.n_codebooks,
            self.config.audio_vocab_size,
        )
        if self.config.loss_softcap > 0:
            cap = float(self.config.loss_softcap)
            logits = cap * mx.tanh(logits / cap)
        return logits

    def __call__(
        self,
        input_ids: mx.array,
        cache=None,
        speaker_embedding: Optional[mx.array] = None,
        speaker_token_position: Any = None,
    ) -> mx.array:
        return self.compute_logits(
            self._forward_hidden(
                input_ids,
                cache=cache,
                speaker_embedding=speaker_embedding,
                speaker_token_position=speaker_token_position,
            )
        )

    def _load_dac(self):
        if self._dac is None:
            from mlx_audio.codec.models.descript import DAC

            self._dac = DAC.from_pretrained(self.config.dac_model_id)
            self._dac.eval()
        return self._dac

    def _decode_audio(
        self,
        delayed_rows: list[list[int]],
        eos_frame: Optional[int],
        frame_limit: Optional[int] = None,
    ) -> mx.array:
        if not delayed_rows:
            return mx.zeros((0,), dtype=mx.float32)
        raw = mx.array(delayed_rows, dtype=mx.int32)
        codes = shear_up(raw, self.config.audio_pad_id)
        if eos_frame is not None:
            limit = max(0, int(eos_frame))
        elif frame_limit is not None:
            limit = max(0, min(int(frame_limit), codes.shape[0]))
        else:
            limit = None
        if limit is not None:
            codes = codes[:limit]
        if codes.size == 0:
            return mx.zeros((0,), dtype=mx.float32)
        codes = mx.clip(codes, 0, self.config.codebook_size - 1)
        dac = self._load_dac()
        # DAC expects [B, codebooks, frames].
        z = dac.quantizer.from_codes(codes.T[None, :, :].astype(mx.int32))[0]
        audio = dac.decode(z).astype(mx.float32).reshape(-1)
        audio = audio[: int(codes.shape[0]) * 512]
        mx.eval(audio)
        return audio

    def _make_generation_result(
        self,
        audio: mx.array,
        *,
        token_count: int,
        prompt_tokens: int,
        elapsed: float,
        segment_idx: int = 0,
        is_streaming_chunk: bool = False,
        is_final_chunk: bool = False,
    ) -> GenerationResult:
        samples = int(audio.shape[0])
        duration_s = samples / self.sample_rate if self.sample_rate else 0.0
        elapsed = max(float(elapsed), 1e-9)
        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=segment_idx,
            token_count=int(token_count),
            audio_duration=format_duration(duration_s),
            real_time_factor=round(elapsed / duration_s, 3) if duration_s else 0.0,
            prompt={
                "tokens": int(prompt_tokens),
                "completion_tokens": int(token_count),
                "tokens-per-sec": (
                    round((int(prompt_tokens) + int(token_count)) / elapsed, 2)
                    if elapsed > 0
                    else 0.0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0.0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

    def _make_batch_generation_result(
        self,
        audio: mx.array,
        *,
        sequence_idx: int,
        token_count: int,
        elapsed: float,
        is_streaming_chunk: bool = False,
        is_final_chunk: bool = False,
    ) -> BatchGenerationResult:
        samples = int(audio.shape[0])
        duration_s = samples / self.sample_rate if self.sample_rate else 0.0
        return BatchGenerationResult(
            audio=audio,
            sequence_idx=int(sequence_idx),
            samples=samples,
            sample_rate=self.sample_rate,
            token_count=int(token_count),
            audio_duration=format_duration(duration_s),
            processing_time_seconds=max(float(elapsed), 1e-9),
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

    def _build_prompt_rows(
        self,
        text: str,
        *,
        speaking_rate_bucket: int | None,
        quality_buckets: dict[str, int | None] | list[int | None] | None,
        speaker_conditioned: bool,
        clean_speaker_background: bool,
        accurate_mode: bool,
    ) -> tuple[list[list[int]], Optional[int]]:
        rows = self._prompt_builder.build_list(
            text,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=self._resolve_quality_buckets(quality_buckets),
        )
        speaker_token_position = None
        if speaker_conditioned:
            prefix = self._prompt_builder.speaker_marker_prefix(
                clean_speaker_background=clean_speaker_background,
                accurate_mode=accurate_mode,
            )
            rows = prefix + rows
            speaker_token_position = 0
        return rows, speaker_token_position

    def _build_prompt(
        self,
        text: str,
        *,
        speaking_rate_bucket: int | None,
        quality_buckets: dict[str, int | None] | list[int | None] | None,
        speaker_embedding: Optional[mx.array],
        clean_speaker_background: bool,
        accurate_mode: bool,
    ) -> tuple[mx.array, Optional[int]]:
        rows, speaker_token_position = self._build_prompt_rows(
            text,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=quality_buckets,
            speaker_conditioned=speaker_embedding is not None,
            clean_speaker_background=clean_speaker_background,
            accurate_mode=accurate_mode,
        )
        return mx.array(rows, dtype=mx.int32)[None, :, :], speaker_token_position

    def _batch_prompt(
        self, prompt_rows: Sequence[list[list[int]]]
    ) -> tuple[mx.array, list[int]]:
        max_len = max(len(rows) for rows in prompt_rows)
        pad_row = [self.config.audio_pad_id] * self.config.n_codebooks + [
            int(self.config.text_vocab)
        ]
        left_padding = [max_len - len(rows) for rows in prompt_rows]
        padded = [
            [pad_row.copy() for _ in range(pad_count)] + rows
            for pad_count, rows in zip(left_padding, prompt_rows, strict=True)
        ]
        return mx.array(padded, dtype=mx.int32), left_padding

    def _stack_speaker_embeddings(
        self, speaker_embeddings: Any, batch_size: int
    ) -> mx.array:
        if isinstance(speaker_embeddings, (str, Path)):
            raise ValueError(
                "speaker_embeddings must be a per-sequence list or a [B, D] array; "
                "use speaker_embedding for a shared path"
            )
        if isinstance(speaker_embeddings, (list, tuple)):
            if len(speaker_embeddings) != batch_size:
                raise ValueError(
                    f"speaker_embeddings length ({len(speaker_embeddings)}) must "
                    f"match texts length ({batch_size})"
                )
            embeddings = []
            for embedding in speaker_embeddings:
                if embedding is None:
                    raise ValueError(
                        "ZONOS2 batch generation does not support mixed speaker-conditioned "
                        "and unconditioned rows"
                    )
                loaded = self._load_speaker_embedding(embedding)
                if loaded is None:
                    raise ValueError("speaker_embeddings entries must not be None")
                embeddings.append(loaded)
            return mx.concatenate(embeddings, axis=0)

        arr = mx.array(speaker_embeddings, dtype=mx.float32)
        if arr.ndim != 2:
            raise ValueError(f"speaker_embeddings must be 2-D, got {arr.shape}")
        if arr.shape != (batch_size, self.config.speaker_embedding_dim):
            raise ValueError(
                "speaker_embeddings must have shape "
                f"({batch_size}, {self.config.speaker_embedding_dim}), got {arr.shape}"
            )
        return arr

    def _resolve_batch_speaker_embeddings(
        self,
        *,
        batch_size: int,
        speaker_embedding: Any,
        speaker_embeddings: Any,
        ref_audio: Any,
        ref_audios: Optional[Sequence[Any]],
        ref_audio_sample_rate: int | None,
        ref_audio_sample_rates: Optional[Sequence[int | None]],
    ) -> Optional[mx.array]:
        shared_inputs = sum(item is not None for item in (speaker_embedding, ref_audio))
        per_sequence_inputs = sum(
            item is not None for item in (speaker_embeddings, ref_audios)
        )
        if shared_inputs + per_sequence_inputs > 1:
            raise ValueError(
                "provide only one of speaker_embedding, speaker_embeddings, "
                "ref_audio, or ref_audios"
            )
        if speaker_embedding is not None:
            embedding = self._load_speaker_embedding(speaker_embedding)
            if embedding is None:
                return None
            return mx.broadcast_to(embedding, (batch_size, embedding.shape[-1]))
        if speaker_embeddings is not None:
            return self._stack_speaker_embeddings(speaker_embeddings, batch_size)
        if ref_audio is not None:
            embedding = self.extract_speaker_embedding(
                ref_audio,
                sample_rate=ref_audio_sample_rate,
            )
            return mx.broadcast_to(embedding, (batch_size, embedding.shape[-1]))
        if ref_audios is not None:
            if len(ref_audios) != batch_size:
                raise ValueError(
                    f"ref_audios length ({len(ref_audios)}) must match texts length "
                    f"({batch_size})"
                )
            if (
                ref_audio_sample_rates is not None
                and len(ref_audio_sample_rates) != batch_size
            ):
                raise ValueError(
                    "ref_audio_sample_rates length must match texts length "
                    f"({batch_size})"
                )
            embeddings = []
            for idx, item in enumerate(ref_audios):
                if item is None:
                    raise ValueError(
                        "ZONOS2 batch generation does not support mixed speaker-conditioned "
                        "and unconditioned rows"
                    )
                sample_rate = (
                    ref_audio_sample_rates[idx]
                    if ref_audio_sample_rates is not None
                    else ref_audio_sample_rate
                )
                embeddings.append(
                    self.extract_speaker_embedding(item, sample_rate=sample_rate)
                )
            return mx.concatenate(embeddings, axis=0)
        return None

    def _sample_batch_frames(
        self,
        last_logits: mx.array,
        states: Sequence[Zonos2GenerationState],
        params: TTSSamplingParams,
        finished: Sequence[bool],
        *,
        step: int,
        seed: Optional[int],
    ) -> list[list[int]]:
        mx.eval(last_logits)
        text_vocab = int(self.config.text_vocab)
        inactive_frame = [self.config.eoa_id] * self.config.n_codebooks + [text_vocab]
        frames = []
        for idx, state in enumerate(states):
            if finished[idx]:
                frames.append(inactive_frame)
                continue
            sample_key = (
                mx.random.key(int(seed) + step * len(states) + idx)
                if seed is not None
                else None
            )
            frames.append(sample_frame(last_logits[idx], state, params, key=sample_key))
        return frames

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "en_us",
        ref_audio=None,
        ref_text=None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.15,
        top_p: float = 0.0,
        top_k: int = 106,
        min_p: float = 0.18,
        repetition_window: int = 50,
        repetition_penalty: float = 1.2,
        repetition_codebooks: int = 8,
        seed: Optional[int] = None,
        ignore_eos: bool = False,
        speaking_rate_bucket: Optional[int] = None,
        quality_buckets: dict[str, int | None] | list[int | None] | None = None,
        speaker_embedding: Any = None,
        clean_speaker_background: bool = False,
        accurate_mode: bool = True,
        text_normalization: bool = True,
        stream: bool = False,
        streaming_interval: float = 2.0,
        verbose: bool = False,
        **kwargs,
    ) -> Iterator[GenerationResult]:
        ref_audio_sample_rate = kwargs.pop("ref_audio_sample_rate", None)
        del voice, speed, ref_text, verbose, kwargs
        limit = max_new_tokens if max_new_tokens is not None else max_tokens
        if limit is None:
            limit = 1024

        start = time.perf_counter()
        normalized_text = self._normalize_text(
            text, language=lang_code, text_normalization=text_normalization
        )
        speaker_emb = self._resolve_speaker_embedding(
            speaker_embedding=speaker_embedding,
            ref_audio=ref_audio,
            ref_audio_sample_rate=ref_audio_sample_rate,
        )
        prompt, speaker_pos = self._build_prompt(
            normalized_text,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=quality_buckets,
            speaker_embedding=speaker_emb,
            clean_speaker_background=clean_speaker_background,
            accurate_mode=accurate_mode,
        )
        mx.eval(prompt)

        cache = make_prompt_cache(self)
        logits = self(
            prompt,
            cache=cache,
            speaker_embedding=speaker_emb,
            speaker_token_position=speaker_pos,
        )
        last_logits = logits[0, -1]
        params = TTSSamplingParams(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            min_p=float(min_p),
            max_tokens=int(limit),
            ignore_eos=bool(ignore_eos),
            repetition_window=int(repetition_window),
            repetition_penalty=float(repetition_penalty),
            repetition_codebooks=int(repetition_codebooks),
            seed=seed,
        )
        state = Zonos2GenerationState(
            n_codebooks=self.config.n_codebooks,
            eoa_id=self.config.eoa_id,
            text_vocab=int(self.config.text_vocab),
        )
        prompt_tokens = int(prompt.shape[1])
        samples_per_frame = 512
        frames_per_chunk = max(
            1,
            int(float(streaming_interval) * self.sample_rate / samples_per_frame),
        )
        decode_delay = max(0, self.config.n_codebooks - 1)
        emitted_samples = 0
        chunk_token_start = 0
        chunk_start = time.perf_counter()

        for step in range(int(limit)):
            sample_key = mx.random.key(int(seed) + step) if seed is not None else None
            frame = sample_frame(last_logits, state, params, key=sample_key)
            state.append(frame, ignore_eos=params.ignore_eos)
            if state.finished:
                break

            if stream:
                complete_frames = max(0, len(state.generated) - decode_delay)
                ready_frames = complete_frames - (emitted_samples // samples_per_frame)
                if ready_frames >= frames_per_chunk:
                    audio_prefix = self._decode_audio(
                        state.generated,
                        eos_frame=None,
                        frame_limit=complete_frames,
                    )
                    if int(audio_prefix.shape[0]) > emitted_samples:
                        audio_chunk = audio_prefix[emitted_samples:]
                        mx.eval(audio_chunk)
                        elapsed = time.perf_counter() - chunk_start
                        yield self._make_generation_result(
                            audio_chunk,
                            token_count=len(state.generated) - chunk_token_start,
                            prompt_tokens=prompt_tokens,
                            elapsed=elapsed,
                            segment_idx=0,
                            is_streaming_chunk=True,
                            is_final_chunk=False,
                        )
                        emitted_samples = int(audio_prefix.shape[0])
                        chunk_token_start = len(state.generated)
                        chunk_start = time.perf_counter()
                        mx.clear_cache()

            next_ids = mx.array(frame, dtype=mx.int32)[None, None, :]
            logits = self(next_ids, cache=cache)
            last_logits = logits[0, -1]

        audio = self._decode_audio(state.generated, state.eos_frame)
        if stream:
            if int(audio.shape[0]) > emitted_samples:
                audio = audio[emitted_samples:]
            else:
                audio = mx.zeros((0,), dtype=mx.float32)
            mx.async_eval(audio)
            yield self._make_generation_result(
                audio,
                token_count=len(state.generated) - chunk_token_start,
                prompt_tokens=prompt_tokens,
                elapsed=time.perf_counter() - chunk_start,
                segment_idx=0,
                is_streaming_chunk=True,
                is_final_chunk=True,
            )
            return

        mx.async_eval(audio)
        yield self._make_generation_result(
            audio,
            token_count=len(state.generated),
            prompt_tokens=prompt_tokens,
            elapsed=time.perf_counter() - start,
            segment_idx=0,
            is_streaming_chunk=False,
            is_final_chunk=False,
        )

    def batch_generate(
        self,
        texts: list[str],
        voices: Optional[list[Optional[str]]] = None,
        speed: float = 1.0,
        lang_code: str = "en_us",
        ref_audio: Any = None,
        ref_audios: Optional[list[Any]] = None,
        ref_text: Any = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.15,
        top_p: float = 0.0,
        top_k: int = 106,
        min_p: float = 0.18,
        repetition_window: int = 50,
        repetition_penalty: float = 1.2,
        repetition_codebooks: int = 8,
        seed: Optional[int] = None,
        ignore_eos: bool = False,
        speaking_rate_bucket: Optional[int] = None,
        quality_buckets: dict[str, int | None] | list[int | None] | None = None,
        speaker_embedding: Any = None,
        speaker_embeddings: Any = None,
        clean_speaker_background: bool = False,
        accurate_mode: bool = True,
        text_normalization: bool = True,
        stream: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Iterator[BatchGenerationResult]:
        if isinstance(texts, str):
            raise TypeError("texts must be a list of strings")
        if stream:
            raise NotImplementedError("ZONOS2 batch streaming is not implemented")
        batch_size = len(texts)
        if batch_size == 0:
            return
        if voices is not None and len(voices) != batch_size:
            raise ValueError(
                f"voices length ({len(voices)}) must match texts length ({batch_size})"
            )
        if voices is not None and any(voice is not None for voice in voices):
            raise ValueError("ZONOS2 batch_generate does not support voices")

        ref_audio_sample_rate = kwargs.pop("ref_audio_sample_rate", None)
        ref_audio_sample_rates = kwargs.pop("ref_audio_sample_rates", None)
        del speed, ref_text, verbose, kwargs

        limit = max_new_tokens if max_new_tokens is not None else max_tokens
        if limit is None:
            limit = 1024

        start = time.perf_counter()
        normalized_texts = [
            self._normalize_text(
                text, language=lang_code, text_normalization=text_normalization
            )
            for text in texts
        ]
        speaker_emb = self._resolve_batch_speaker_embeddings(
            batch_size=batch_size,
            speaker_embedding=speaker_embedding,
            speaker_embeddings=speaker_embeddings,
            ref_audio=ref_audio,
            ref_audios=ref_audios,
            ref_audio_sample_rate=ref_audio_sample_rate,
            ref_audio_sample_rates=ref_audio_sample_rates,
        )
        prompt_rows = []
        prompt_speaker_positions: list[Optional[int]] = []
        for text in normalized_texts:
            rows, speaker_pos = self._build_prompt_rows(
                text,
                speaking_rate_bucket=speaking_rate_bucket,
                quality_buckets=quality_buckets,
                speaker_conditioned=speaker_emb is not None,
                clean_speaker_background=clean_speaker_background,
                accurate_mode=accurate_mode,
            )
            prompt_rows.append(rows)
            prompt_speaker_positions.append(speaker_pos)
        prompt, left_padding = self._batch_prompt(prompt_rows)
        speaker_token_positions = None
        if speaker_emb is not None:
            speaker_token_positions = [
                left_padding[idx] + int(prompt_speaker_positions[idx] or 0)
                for idx in range(batch_size)
            ]
        mx.eval(prompt)
        if speaker_emb is not None:
            mx.eval(speaker_emb)

        cache = [BatchKVCache(left_padding) for _ in self.layers]
        logits = self(
            prompt,
            cache=cache,
            speaker_embedding=speaker_emb,
            speaker_token_position=speaker_token_positions,
        )
        last_logits = logits[:, -1]
        params = TTSSamplingParams(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            min_p=float(min_p),
            max_tokens=int(limit),
            ignore_eos=bool(ignore_eos),
            repetition_window=int(repetition_window),
            repetition_penalty=float(repetition_penalty),
            repetition_codebooks=int(repetition_codebooks),
            seed=seed,
        )
        states = [
            Zonos2GenerationState(
                n_codebooks=self.config.n_codebooks,
                eoa_id=self.config.eoa_id,
                text_vocab=int(self.config.text_vocab),
            )
            for _ in range(batch_size)
        ]
        finished = [False] * batch_size

        for step in range(int(limit)):
            frames = self._sample_batch_frames(
                last_logits,
                states,
                params,
                finished,
                step=step,
                seed=seed,
            )
            for idx, frame in enumerate(frames):
                if finished[idx]:
                    continue
                states[idx].append(frame, ignore_eos=params.ignore_eos)
                finished[idx] = states[idx].finished
            if all(finished):
                break

            next_ids = mx.array(frames, dtype=mx.int32)[:, None, :]
            logits = self(next_ids, cache=cache)
            last_logits = logits[:, -1]

        elapsed = time.perf_counter() - start
        for idx, state in enumerate(states):
            audio = self._decode_audio(state.generated, state.eos_frame)
            mx.async_eval(audio)
            yield self._make_batch_generation_result(
                audio,
                sequence_idx=idx,
                token_count=len(state.generated),
                elapsed=elapsed,
                is_streaming_chunk=False,
                is_final_chunk=False,
            )

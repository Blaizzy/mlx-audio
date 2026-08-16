"""Hierarchical autoregressive generation for MiniMax Music 3."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_audio.lm.models.cache import make_prompt_cache
from mlx_audio.lm.models.qwen3 import Model as Qwen3Model

from .config import AR_CFG_SCALE, AR_CFG_TOP_K, AR_SAMPLING_TOP_K, ModelConfig
from .depth import RVQDepthDecoder
from .fusion import fuse_frame_hiddens
from .sampling import sample_top_k


@dataclass
class ARFrame:
    semantic_code: mx.array
    residual_codes: mx.array
    frame_hidden: mx.array
    last_hidden: mx.array
    cache: list
    ended: bool


def qwen3_hidden(
    language_model: Qwen3Model,
    input_embeddings: mx.array,
    cache=None,
) -> tuple[mx.array, list]:
    if cache is None:
        cache = make_prompt_cache(language_model)
    hidden = language_model.model(
        mx.zeros(input_embeddings.shape[:2], dtype=mx.int32),
        cache,
        input_embeddings=input_embeddings,
    )
    return hidden, cache


def lm_logits(language_model: Qwen3Model, hidden: mx.array) -> mx.array:
    if language_model.args.tie_word_embeddings:
        return language_model.model.embed_tokens.as_linear(hidden)
    return language_model.lm_head(hidden)


def _embed_audio_frame(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: ModelConfig,
    frame_codes: mx.array,
) -> mx.array:
    embeddings = language_model.model.embed_tokens(
        frame_codes[:, :1] + config.audio_code_offset
    )
    offsets = mx.arange(config.residual_codebooks) * config.audio_vocab_size
    residual = depth.audio_embeddings(frame_codes[:, 1:] + offsets[None, :]).sum(
        axis=1, keepdims=True
    )
    return (embeddings + residual.astype(embeddings.dtype)) * (
        config.num_codebooks**-0.5
    )


def generate_depth_codes(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: ModelConfig,
    last_hidden: mx.array,
    semantic_code: mx.array,
    rng_key: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    sequence = [depth.projection(last_hidden)[:, None, :]]
    code_embedding = language_model.model.embed_tokens(
        semantic_code + config.audio_code_offset
    )
    sequence.append(depth.projection(code_embedding)[:, None, :])
    codes = [semantic_code]
    hidden_parts = []
    key = rng_key
    for index in range(1, config.num_codebooks):
        hidden = depth(mx.concatenate(sequence, axis=1))[:, -1]
        hidden_parts.append(hidden[:1])
        logits = depth.audio_heads[index - 1](hidden)
        conditional = logits[:1].astype(mx.float32)
        unconditional = logits[1:2].astype(mx.float32)
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        sampled, key = sample_top_k(guided, key, AR_SAMPLING_TOP_K)
        code = mx.concatenate([sampled, sampled], axis=0)
        codes.append(code)
        if index < config.num_codebooks - 1:
            embedding = depth.audio_embeddings(
                code + (index - 1) * config.audio_vocab_size
            )
            sequence.append(depth.projection(embedding)[:, None, :])
    return mx.stack(codes, axis=1), mx.concatenate(hidden_parts, axis=-1), key


def ar_one_frame(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: ModelConfig,
    last_hidden: mx.array,
    cache: list,
    rng_key: mx.array,
    emit_frame: bool = True,
) -> ARFrame:
    logits = lm_logits(language_model, last_hidden).astype(mx.float32)
    token_ids = mx.arange(logits.shape[-1])
    allowed = mx.logical_or(
        mx.logical_and(
            token_ids >= config.audio_code_offset,
            token_ids < config.audio_code_offset + config.semantic_vocab_size,
        ),
        token_ids == config.audio_end_token_id,
    )
    logits = mx.where(allowed[None, :], logits, -1e9)
    conditional, unconditional = logits[:1], logits[1:2]
    guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
    k = min(AR_CFG_TOP_K, conditional.shape[-1])
    threshold = mx.min(mx.topk(conditional, k, axis=-1), axis=-1, keepdims=True)
    guided = mx.where(conditional < threshold, -1e9, guided)
    guided = mx.where(allowed[None, :], guided, -1e9)
    sampled, key = sample_top_k(guided, rng_key, AR_SAMPLING_TOP_K)

    if int(sampled.item()) == config.audio_end_token_id:
        return ARFrame(
            semantic_code=sampled,
            residual_codes=mx.zeros((2, config.residual_codebooks), dtype=mx.int32),
            frame_hidden=mx.zeros((1, config.num_codebooks * config.hidden_size)),
            last_hidden=last_hidden,
            cache=cache,
            ended=True,
        )

    semantic_code = (
        mx.concatenate([sampled, sampled], axis=0) - config.audio_code_offset
    )
    frame_codes, depth_hidden, key = generate_depth_codes(
        language_model,
        depth,
        config,
        last_hidden,
        semantic_code,
        key,
    )
    frame_hidden = (
        fuse_frame_hiddens(last_hidden[:1], depth_hidden)
        if emit_frame
        else mx.zeros((1, config.num_codebooks * config.hidden_size))
    )
    feedback = _embed_audio_frame(language_model, depth, config, frame_codes)
    hidden, cache = qwen3_hidden(language_model, feedback, cache)
    return ARFrame(
        semantic_code=semantic_code,
        residual_codes=frame_codes[:, 1:],
        frame_hidden=frame_hidden,
        last_hidden=hidden[:, -1],
        cache=cache,
        ended=False,
    )


def generate_frame_hiddens(
    language_model: Qwen3Model,
    depth: RVQDepthDecoder,
    config: ModelConfig,
    text_ids: mx.array,
    max_frames: int,
    seed: int = 0,
) -> mx.array:
    mx.random.seed(seed)
    key = mx.random.key(seed)
    embeddings = language_model.model.embed_tokens(text_ids)
    hidden, cache = qwen3_hidden(language_model, embeddings)
    last_hidden = hidden[:, -1]
    frames = []
    for frame_index in range(max_frames + 1):
        key, subkey = mx.random.split(key)
        result = ar_one_frame(
            language_model,
            depth,
            config,
            last_hidden,
            cache,
            subkey,
            emit_frame=frame_index > 0,
        )
        last_hidden, cache = result.last_hidden, result.cache
        if result.ended:
            break
        if frame_index > 0:
            frames.append(result.frame_hidden)
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError("MiniMax Music 3 generated zero audio frames")
    return mx.stack(frames, axis=1)

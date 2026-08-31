"""MLX port of the Audio8 TTS preview models (model_type: arktts).

Mirrors the reference modeling_arktts.py structure: a DualAR transformer — a
slow AR stack predicting one semantic token per audio frame, and a fast AR
stack predicting the frame's residual codec codebooks — plus the bundled
44.1 kHz codec (codec.py). Sampling reproduces the reference exactly:
semantic-range logit filtering, the legacy top-k/top-p order (filter before
temperature), exponential-race sampling, and RAS repetition rescue.

Two checkpoints share this ``model_type`` and are told apart by ``slow_backbone``:

``arktts`` (default, Audio8-TTS-Preview-0.6b)
    Pure-attention slow stack, full-vocabulary semantic logits tied to the input
    embedding, EOS at ``eos_token_id``.

``falcon_h1`` (Audio8-TTS-Preview-0.1b)
    Falcon-H1 hybrid slow stack — every layer carries Mamba-2 + attention + MLP —
    consumed from the vendored ``mlx_audio.lm.models.falcon_h1``. Adds a dedicated
    ``semantic_output`` head emitting COMPACT logits of width
    ``codebook_size + 1``: index ``i`` means semantic token
    ``semantic_begin_id + i`` and index ``codebook_size`` means EOS. The fast AR,
    the prompt layout, and the codec are identical across both.

Both share the codec byte-for-byte (same upstream ``codec.pth`` checksum), so a
conversion of either can reuse the other's converted codec tensors.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.base import create_attention_mask, create_ssm_mask
from mlx_audio.lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_audio.lm.models.falcon_h1 import FalconH1DecoderLayer
from mlx_audio.lm.models.falcon_h1 import ModelArgs as FalconH1Args
from mlx_audio.lm.models.falcon_h1 import compute_mup_vector

from ..base import BaseModelArgs, GenerationResult
from .codec import ArkttsCodec


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "arktts"
    vocab_size: int = 155776
    dim: int = 896
    n_layer: int = 24
    n_head: int = 14
    n_local_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    max_seq_len: int = 2048
    rope_base: float = 1_000_000
    norm_eps: float = 1e-6
    attention_qkv_bias: bool = True
    attention_qk_norm: bool = False
    attention_o_bias: bool = False
    tie_word_embeddings: bool = True
    codebook_size: int = 4096
    num_codebooks: int = 10
    semantic_begin_id: int = 151678
    semantic_end_id: int = 155773
    n_fast_layer: int = 4
    fast_dim: int = 896
    fast_n_head: int = 14
    fast_n_local_heads: int = 2
    fast_head_dim: int = 64
    fast_intermediate_size: int = 4864
    fast_attention_qkv_bias: bool = False
    fast_attention_qk_norm: bool = False
    fast_attention_o_bias: bool = False
    norm_fastlayer_input: bool = True
    codec_filename: str = "codec.pth"
    codec_sample_rate: int = 44100
    codec_frame_size: int = 2048
    codec_post_n_layer: int = 8
    codec_post_n_head: int = 16
    codec_post_n_local_heads: int = 8
    codec_post_intermediate_size: int = 1216
    ras_window_size: int = 10
    ras_temperature: float = 1.0
    ras_top_p: float = 0.9
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    sample_rate: int = 44100

    # -- slow-backbone selection --------------------------------------------
    # "arktts" = the 0.6b pure-attention stack (fields above). "falcon_h1" = the
    # 0.1b hybrid stack (fields below). Defaults keep 0.6b checkpoints, which
    # carry none of these keys, loading exactly as before.
    slow_backbone: str = "arktts"

    # Falcon-H1 slow backbone. Names mirror the reference config verbatim so
    # _falcon_args() is a rename-free pass-through; do not "tidy" them.
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 1.0
    key_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    lm_head_multiplier: float = 1.0
    expansion_factor: float = 2.0
    mlp_bias: bool = False
    mlp_multipliers: tuple = (1.0, 1.0)
    projectors_bias: bool = False
    ssm_in_multiplier: float = 1.0
    ssm_multipliers: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    ssm_out_multiplier: float = 1.0
    mamba_chunk_size: int = 128
    mamba_conv_bias: bool = True
    mamba_d_conv: int = 4
    mamba_d_head: int = 32
    mamba_d_ssm: int = 768
    mamba_d_state: int = 64
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_n_heads: int = 24
    mamba_norm_before_gate: bool = False
    mamba_proj_bias: bool = False
    mamba_rms_norm: bool = False
    mamba_use_mlp: bool = True
    initializer_range: float = 0.02

    @property
    def uses_falcon_slow(self) -> bool:
        return self.slow_backbone == "falcon_h1"


def _precompute_rope(length: int, head_dim: int, base: float) -> mx.array:
    frequencies = 1.0 / (
        base
        ** (mx.arange(0, head_dim, 2).astype(mx.float32)[: head_dim // 2] / head_dim)
    )
    phases = mx.arange(length).astype(mx.float32)[:, None] * frequencies[None, :]
    # reference stores the (real, imag) table in bf16; keep the same rounding
    return mx.stack((mx.cos(phases), mx.sin(phases)), axis=-1).astype(mx.bfloat16)


def _apply_rope(x: mx.array, rope: mx.array) -> mx.array:
    # x: (B, T, H, D); rope: (T, D/2, 2) or (B, T, D/2, 2)
    shaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    rope = rope.astype(mx.float32)
    if rope.ndim == 3:
        rope = rope[None, :, None]
    elif rope.ndim == 4:
        rope = rope[:, :, None]
    else:
        raise ValueError(f"Unexpected RoPE shape: {tuple(rope.shape)}")
    output = mx.stack(
        (
            shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
            shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1],
        ),
        axis=-1,
    )
    return output.flatten(3).astype(x.dtype)


def _falcon_args(config: ModelConfig) -> FalconH1Args:
    """Mirror of ArkttsModel._build_falcon_config in the reference remote code."""
    return FalconH1Args(
        model_type="falcon_h1",
        vocab_size=config.vocab_size,
        hidden_size=config.dim,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.n_local_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.norm_eps,
        rope_theta=config.rope_base,
        max_position_embeddings=config.max_seq_len,
        attention_bias=config.attention_bias,
        attention_in_multiplier=config.attention_in_multiplier,
        attention_out_multiplier=config.attention_out_multiplier,
        key_multiplier=config.key_multiplier,
        embedding_multiplier=config.embedding_multiplier,
        lm_head_multiplier=config.lm_head_multiplier,
        mlp_bias=config.mlp_bias,
        mlp_multipliers=list(config.mlp_multipliers),
        mamba_chunk_size=config.mamba_chunk_size,
        mamba_conv_bias=config.mamba_conv_bias,
        mamba_d_conv=config.mamba_d_conv,
        mamba_d_head=config.mamba_d_head,
        mamba_d_ssm=config.mamba_d_ssm,
        mamba_d_state=config.mamba_d_state,
        mamba_expand=config.mamba_expand,
        mamba_n_groups=config.mamba_n_groups,
        mamba_n_heads=config.mamba_n_heads,
        mamba_norm_before_gate=config.mamba_norm_before_gate,
        mamba_proj_bias=config.mamba_proj_bias,
        mamba_rms_norm=config.mamba_rms_norm,
        mamba_use_mlp=config.mamba_use_mlp,
        projectors_bias=config.projectors_bias,
        ssm_in_multiplier=config.ssm_in_multiplier,
        ssm_multipliers=list(config.ssm_multipliers),
        ssm_out_multiplier=config.ssm_out_multiplier,
        tie_word_embeddings=config.tie_word_embeddings,
    )


class FalconSlowStack(nn.Module):
    """The 0.1b slow backbone: Falcon-H1 decoder layers driven by an INJECTED embedding.

    ``mlx_audio.lm.models.falcon_h1.FalconH1Model`` embeds its own token ids and offers no
    ``inputs_embeds`` seam, which arktts needs — its slow input is

        (text_embed + sum_of_ten_codebook_embeds) * embedding_multiplier

    So this holds the same submodules under the same names (weights map straight
    across) and reimplements only the ten-line forward, entering at the embedding.

    THE MULTIPLIER IS APPLIED HERE, to the composite, and must NOT be folded into
    ``embed_tokens.weight`` the way ``FalconH1Model.sanitize`` does — folding scales
    the text half and silently leaves the codebook half at 1.0. Measured against the
    PyTorch oracle that is a 77% relative error on the embedding and 38% on the final
    hidden state, while still producing plausible audio of the right length. See
    ``Model.sanitize`` below, which deliberately omits the embedding fold.
    """

    #: muP multipliers that the upstream FalconH1 folds into weights during `sanitize`.
    #: This port does not fold them (see `Model.sanitize`), so it only supports checkpoints
    #: that carry them at unity — which every published arktts checkpoint does.
    _MUST_BE_UNITY = (
        "attention_in_multiplier",
        "attention_out_multiplier",
        "key_multiplier",
        "ssm_in_multiplier",
        "ssm_out_multiplier",
    )
    _MUST_BE_UNITY_SEQUENCES = ("mlp_multipliers", "ssm_multipliers")

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Fail loudly rather than load silently-unscaled weights. `embedding_multiplier` is
        # deliberately absent from this check: it is applied at runtime to the COMPOSITE
        # embedding (see this class's docstring), and `lm_head_multiplier` is unused because
        # arktts brings its own semantic head.
        offenders = [
            name for name in self._MUST_BE_UNITY if float(getattr(config, name)) != 1.0
        ]
        offenders += [
            name
            for name in self._MUST_BE_UNITY_SEQUENCES
            if set(float(v) for v in getattr(config, name)) != {1.0}
        ]
        if offenders:
            raise ValueError(
                "arktts's falcon_h1 backbone supports unity muP multipliers only, but this "
                f"config sets {', '.join(offenders)} away from 1.0. Folding them belongs in "
                "sanitize(), which is contractually self-contained here and cannot read the "
                "config — so this checkpoint needs that seam reworked rather than silently "
                "loading unscaled weights."
            )
        args = _falcon_args(config)
        self.args = args
        self.embedding_multiplier = config.embedding_multiplier
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [
            FalconH1DecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.final_layernorm = nn.RMSNorm(config.dim, eps=args.rms_norm_eps)
        self._mup_vector = compute_mup_vector(args)
        mx.eval(self._mup_vector)

    def make_cache(self):
        return [
            CacheList(ArraysCache(size=2), KVCache())
            for _ in range(self.args.num_hidden_layers)
        ]

    def __call__(self, hidden: mx.array, cache=None) -> mx.array:
        """`hidden` is the UNSCALED composite embedding; the multiplier is applied here."""
        hidden = hidden * self.embedding_multiplier
        cache = cache if cache is not None else [(None, None)] * len(self.layers)
        mamba_mask = create_ssm_mask(hidden, cache[0][0])
        attn_mask = create_attention_mask(hidden, cache[0][1])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(
                hidden, cache=layer_cache, attn_mask=attn_mask, mamba_mask=mamba_mask
            )
        return self.final_layernorm(hidden)


class ArkttsRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        normalized = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return normalized.astype(x.dtype) * self.weight


class ArkttsKVCache:
    def __init__(
        self, batch_size: int, max_length: int, heads: int, head_dim: int, dtype
    ):
        shape = (batch_size, heads, max_length, head_dim)
        self.keys = mx.zeros(shape, dtype=dtype)
        self.values = mx.zeros(shape, dtype=dtype)

    def update(self, start: int, keys: mx.array, values: mx.array):
        length = keys.shape[2]
        self.keys[:, :, start : start + length] = keys
        self.values[:, :, start : start + length] = values
        return self.keys, self.values


class ArkttsAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        total = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total, bias=qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=output_bias)
        self.n_head = int(n_head)
        self.n_local_heads = int(n_local_heads)
        self.head_dim = int(head_dim)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = ArkttsRMSNorm(head_dim, norm_eps)
            self.k_norm = ArkttsRMSNorm(head_dim, norm_eps)
        self.kv_cache: Optional[ArkttsKVCache] = None

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        attention_mask: Optional[mx.array],
        cache_start: Optional[int] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        qkv = self.wqkv(x)
        query, key, value = mx.split(qkv, [query_size, query_size + kv_size], axis=-1)
        query = query.reshape(batch, length, self.n_head, self.head_dim)
        key = key.reshape(batch, length, self.n_local_heads, self.head_dim)
        value = value.reshape(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = _apply_rope(query, rope).transpose(0, 2, 1, 3)
        key = _apply_rope(key, rope).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        if self.kv_cache is not None:
            if cache_start is None:
                raise ValueError("cache_start is required when KV cache is enabled")
            key, value = self.kv_cache.update(cache_start, key, value)
        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, query_size)
        return self.wo(output)


class ArkttsFeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class ArkttsTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.attention = ArkttsAttention(
            dim,
            n_head,
            n_local_heads,
            head_dim,
            qkv_bias,
            output_bias,
            qk_norm,
            norm_eps,
        )
        self.feed_forward = ArkttsFeedForward(dim, intermediate_size)
        self.ffn_norm = ArkttsRMSNorm(dim, norm_eps)
        self.attention_norm = ArkttsRMSNorm(dim, norm_eps)

    def __call__(self, x, rope, attention_mask, cache_start=None):
        hidden = x + self.attention(
            self.attention_norm(x), rope, attention_mask, cache_start
        )
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class ArkttsModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        if config.uses_falcon_slow:
            # 0.1b: hybrid Mamba+attention slow stack with its own text embedding,
            # plus a dedicated compact semantic head (codebook_size + 1 wide).
            self.slow = FalconSlowStack(config)
            self.semantic_output = nn.Linear(
                config.dim, config.codebook_size + 1, bias=False
            )
        else:
            # 0.6b: pure-attention slow stack, semantic logits tied to the embedding.
            self.embeddings = nn.Embedding(config.vocab_size, config.dim)
            self.layers = [
                ArkttsTransformerBlock(
                    config.dim,
                    config.intermediate_size,
                    config.n_head,
                    config.n_local_heads,
                    config.head_dim,
                    config.attention_qkv_bias,
                    config.attention_o_bias,
                    config.attention_qk_norm,
                    config.norm_eps,
                )
                for _ in range(config.n_layer)
            ]
            self.norm = ArkttsRMSNorm(config.dim, config.norm_eps)
        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim)
            if config.fast_dim != config.dim
            else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = [
            ArkttsTransformerBlock(
                config.fast_dim,
                config.fast_intermediate_size,
                config.fast_n_head,
                config.fast_n_local_heads,
                config.fast_head_dim,
                config.fast_attention_qkv_bias,
                config.fast_attention_o_bias,
                config.fast_attention_qk_norm,
                config.norm_eps,
            )
            for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = ArkttsRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        # The Falcon slow stack builds its own rope internally, so the slow table is
        # only needed by the pure-attention path. The fast table is shared.
        self._freqs_cis = (
            None
            if config.uses_falcon_slow
            else _precompute_rope(config.max_seq_len, config.head_dim, config.rope_base)
        )
        self._fast_freqs_cis = _precompute_rope(
            config.num_codebooks, config.fast_head_dim, config.rope_base
        )
        # Materialize the rope tables now. They are plain attributes rather than module
        # parameters, so nothing else forces them; left lazy, their graph would first be
        # evaluated inside whichever stream/thread happens to run the first forward, which
        # crashes when the model is moved across streams.
        mx.eval(
            (self._fast_freqs_cis,)
            if self._freqs_cis is None
            else (self._freqs_cis, self._fast_freqs_cis)
        )
        self._slow_cache = None

    # -- embedding -----------------------------------------------------------
    @property
    def text_embeddings(self) -> nn.Embedding:
        """The text-token embedding table, wherever this variant keeps it."""
        return (
            self.slow.embed_tokens if self.config.uses_falcon_slow else self.embeddings
        )

    def _embed(self, input_ids: mx.array) -> mx.array:
        config = self.config
        codebook_embeds = []
        for index in range(config.num_codebooks):
            codebook_embeds.append(
                self.codebook_embeddings(
                    input_ids[:, index + 1] + index * config.codebook_size
                )
            )
        codebook_sum = mx.stack(codebook_embeds, axis=1).sum(axis=1)
        semantic = mx.logical_and(
            input_ids[:, 0] >= config.semantic_begin_id,
            input_ids[:, 0] <= config.semantic_end_id,
        )
        codebook_sum = mx.where(semantic[..., None], codebook_sum, 0.0)
        # Returned UNSCALED. On the falcon path FalconSlowStack applies
        # embedding_multiplier to this composite; see its docstring.
        return self.text_embeddings(input_ids[:, 0]) + codebook_sum

    @staticmethod
    def _causal_mask(
        attention_mask: mx.array, query_positions: mx.array, key_length: int
    ) -> mx.array:
        if attention_mask.shape[1] < key_length:
            attention_mask = mx.pad(
                attention_mask, ((0, 0), (0, key_length - attention_mask.shape[1]))
            )
        key_positions = mx.arange(key_length)
        causal = key_positions[None, :] <= query_positions[:, None]
        return mx.logical_and(
            causal[None, None],
            attention_mask[:, None, None, :key_length].astype(mx.bool_),
        )

    # -- prefill (no-cache) forward, for parity ------------------------------
    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None):
        config = self.config
        if input_ids.ndim != 3 or input_ids.shape[1] != config.num_codebooks + 1:
            raise ValueError(
                f"input_ids must have shape [B, {config.num_codebooks + 1}, T]"
            )
        batch, _, length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch, length), dtype=mx.int64)
        hidden = self._embed(input_ids)
        if config.uses_falcon_slow:
            normalized = self.slow(hidden)
            return self.semantic_output(normalized), normalized
        position_ids = mx.maximum(
            mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1, 0
        )
        rope = self._freqs_cis[position_ids]
        mask = self._causal_mask(attention_mask, mx.arange(length), length)
        for layer in self.layers:
            hidden = layer(hidden, rope, mask)
        normalized = self.norm(hidden)
        logits = normalized @ self.embeddings.weight.T
        return logits, hidden

    # -- cached decode steps -------------------------------------------------
    def _setup_generation_caches(self, batch_size: int, dtype):
        config = self.config
        if config.uses_falcon_slow:
            # Hybrid cache: per layer an ArraysCache(2) for the Mamba conv+ssm state
            # and a KVCache for the attention half. Unlike the pure-attention path's
            # static full-buffer cache, this one tracks its own offset, so the slow
            # step feeds only the new column and passes no explicit mask.
            self._slow_cache = self.slow.make_cache()
        else:
            for layer in self.layers:
                layer.attention.kv_cache = ArkttsKVCache(
                    batch_size,
                    config.max_seq_len,
                    config.n_local_heads,
                    config.head_dim,
                    dtype,
                )
        for layer in self.fast_layers:
            layer.attention.kv_cache = ArkttsKVCache(
                batch_size,
                config.num_codebooks,
                config.fast_n_local_heads,
                config.fast_head_dim,
                dtype,
            )

    def _clear_generation_caches(self):
        self._slow_cache = None
        slow_layers = [] if self.config.uses_falcon_slow else list(self.layers)
        for layer in slow_layers + list(self.fast_layers):
            layer.attention.kv_cache = None

    def _slow_step(
        self,
        input_ids: mx.array,
        cache_start: int,
        cache_positions: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array,
    ):
        config = self.config
        hidden = self._embed(input_ids)
        if config.uses_falcon_slow:
            # The hybrid cache carries its own offset and mask construction, so the
            # explicit position/mask arguments are unused here. The reference returns
            # the normalized hidden for BOTH the logits and the fast-AR input
            # (norm_fastlayer_input is not consulted on this path).
            normalized = self.slow(hidden, cache=self._slow_cache)[:, -1:]
            return self.semantic_output(normalized)[:, -1], normalized
        rope = self._freqs_cis[position_ids]
        mask = self._causal_mask(attention_mask, cache_positions, config.max_seq_len)
        for layer in self.layers:
            hidden = layer(hidden, rope, mask, cache_start)
        hidden = hidden[:, -1:]
        normalized = self.norm(hidden)
        logits = (normalized @ self.embeddings.weight.T)[:, -1]
        fast_hidden = normalized if config.norm_fastlayer_input else hidden
        return logits, fast_hidden

    def _fast_step(self, hidden: mx.array, position: int) -> mx.array:
        config = self.config
        rope = self._fast_freqs_cis[mx.array([position])]
        key_mask = mx.ones((hidden.shape[0], config.num_codebooks), dtype=mx.bool_)
        mask = self._causal_mask(key_mask, mx.array([position]), config.num_codebooks)
        for layer in self.fast_layers:
            hidden = layer(hidden, rope, mask, position)
        return self.fast_output(self.fast_norm(hidden))[:, -1]

    # -- sampling (mirrors the reference exactly) ----------------------------
    @staticmethod
    def _legacy_top_k_top_p(scores: mx.array, top_k: int, top_p: float) -> mx.array:
        sorted_indices = mx.argsort(-scores, axis=-1)
        sorted_scores = mx.take_along_axis(scores, sorted_indices, axis=-1)
        cumulative = mx.cumsum(mx.softmax(sorted_scores, axis=-1), axis=-1)
        positions = mx.arange(scores.shape[-1])
        remove_sorted = mx.logical_or(cumulative > top_p, positions[None, :] >= top_k)
        remove_sorted[:, 0] = False
        remove = mx.zeros_like(remove_sorted)
        remove = mx.put_along_axis(remove, sorted_indices, remove_sorted, axis=-1)
        return mx.where(remove, mx.array(-mx.inf, dtype=scores.dtype), scores)

    def _semantic_filter(self, scores: mx.array) -> mx.array:
        config = self.config
        if config.uses_falcon_slow:
            # Compact head: the reference builds this filter with
            # (begin, end, eos) = (0, codebook_size - 1, codebook_size), which spans
            # all codebook_size + 1 columns. Provably a no-op, so skip the full-width
            # scatter rather than pay for it every frame.
            return scores
        filtered = mx.full(scores.shape, -mx.inf, dtype=scores.dtype)
        filtered[:, config.semantic_begin_id : config.semantic_end_id + 1] = scores[
            :, config.semantic_begin_id : config.semantic_end_id + 1
        ]
        filtered[:, config.eos_token_id] = scores[:, config.eos_token_id]
        return filtered

    @staticmethod
    def _sample(scores: mx.array, rng_key) -> mx.array:
        probabilities = mx.softmax(scores, axis=-1)
        random = mx.random.uniform(shape=probabilities.shape, key=rng_key)
        noise = -mx.log(random)
        return mx.argmax(probabilities / noise, axis=-1).astype(mx.int64)

    def _processed_scores(self, scores, top_k, top_p, temperature, semantic: bool):
        if semantic:
            scores = self._semantic_filter(scores)
        scores = self._legacy_top_k_top_p(scores, top_k, top_p)
        return scores / max(temperature, 1e-5)

    def _sample_semantic(
        self, logits, top_k, top_p, temperature, previous, do_sample, rng_keys
    ):
        config = self.config
        regular_scores = self._processed_scores(logits, top_k, top_p, temperature, True)
        if not do_sample:
            return mx.argmax(regular_scores, axis=-1).astype(mx.int64)
        normal = self._sample(regular_scores, rng_keys[0])
        high_scores = self._processed_scores(
            logits, top_k, config.ras_top_p, config.ras_temperature, True
        )
        high = self._sample(high_scores, rng_keys[1])
        if previous is None:
            return normal
        repeated = mx.any(previous == normal[:, None], axis=1)
        if config.uses_falcon_slow:
            # Compact space: semantic ids are 0..codebook_size-1, EOS is codebook_size.
            semantic = normal < config.codebook_size
        else:
            semantic = mx.logical_and(
                normal >= config.semantic_begin_id, normal <= config.semantic_end_id
            )
        return mx.where(mx.logical_and(repeated, semantic), high, normal)

    def _generate_codebooks(
        self, slow_hidden, semantic, top_k, top_p, temperature, do_sample, rng_key
    ):
        config = self.config
        hidden = self.fast_project_in(slow_hidden)
        self._fast_step(hidden, 0)
        # Codebook 0 IS the semantic token, expressed in codebook space. The compact
        # head already emits that space; the full-vocab head needs the offset removed.
        raw = (
            semantic if config.uses_falcon_slow else semantic - config.semantic_begin_id
        )
        current = mx.clip(raw, 0, config.codebook_size - 1)
        codebooks = [current]
        hidden = self.fast_embeddings(current)[:, None]
        for position in range(1, config.num_codebooks):
            scores = self._fast_step(hidden, position)
            scores = self._processed_scores(scores, top_k, top_p, temperature, False)
            if do_sample:
                rng_key, sub = mx.random.split(rng_key)
                current = self._sample(scores, sub)
            else:
                current = mx.argmax(scores, axis=-1).astype(mx.int64)
            codebooks.append(current)
            hidden = self.fast_embeddings(current)[:, None]
        return mx.stack(codebooks, axis=1)

    # -- prompt building -----------------------------------------------------
    def _prepare_prompt(
        self,
        prefix_input_ids: np.ndarray,
        suffix_input_ids: np.ndarray,
        reference_codes: Optional[np.ndarray] = None,
        reference_code_lengths: Optional[np.ndarray] = None,
    ):
        """Single-row (B=1 per row) numpy prompt assembly, mirroring the reference."""
        config = self.config
        rows = []
        batch_size = len(prefix_input_ids)
        for batch_index in range(batch_size):
            prefix = np.asarray(prefix_input_ids[batch_index], dtype=np.int64)
            suffix = np.asarray(suffix_input_ids[batch_index], dtype=np.int64)
            if reference_codes is None:
                semantic_row = np.concatenate((prefix, suffix))
                values = np.zeros(
                    (config.num_codebooks + 1, semantic_row.size), dtype=np.int64
                )
                values[0] = semantic_row
            else:
                length = int(reference_code_lengths[batch_index])
                codes = np.asarray(
                    reference_codes[batch_index][:, :length], dtype=np.int64
                )
                semantic_codes = codes[0] + config.semantic_begin_id
                semantic_row = np.concatenate((prefix, semantic_codes, suffix))
                values = np.zeros(
                    (config.num_codebooks + 1, semantic_row.size), dtype=np.int64
                )
                values[0] = semantic_row
                values[1:, prefix.size : prefix.size + length] = codes
            rows.append(values)
        max_length = max(row.shape[1] for row in rows)
        prompt = np.zeros(
            (batch_size, config.num_codebooks + 1, max_length), dtype=np.int64
        )
        prompt[:, 0] = config.pad_token_id
        prompt_mask = np.zeros((batch_size, max_length), dtype=np.int64)
        for batch_index, row in enumerate(rows):
            start = max_length - row.shape[1]
            prompt[batch_index, :, start:] = row
            prompt_mask[batch_index, start:] = 1
        return mx.array(prompt), mx.array(prompt_mask)

    # -- generation ----------------------------------------------------------
    def generate_codes(
        self,
        prefix_input_ids,
        suffix_input_ids,
        reference_codes=None,
        reference_code_lengths=None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> mx.array:
        """Returns generated codec codes (B, num_codebooks, T). Batch size 1."""
        config = self.config
        prompt, prompt_mask = self._prepare_prompt(
            prefix_input_ids, suffix_input_ids, reference_codes, reference_code_lengths
        )
        batch_size, _, prompt_width = prompt.shape
        if prompt_width >= config.max_seq_len:
            raise ValueError(
                f"Prompt length {prompt_width} must be smaller than {config.max_seq_len}"
            )
        max_new_tokens = min(max_new_tokens, config.max_seq_len - prompt_width)
        dtype = self.text_embeddings.weight.dtype
        # In the compact semantic space EOS is the last column of the head, not the
        # tokenizer's eos_token_id.
        eos_index = (
            config.codebook_size if config.uses_falcon_slow else config.eos_token_id
        )
        self._setup_generation_caches(batch_size, dtype)
        rng_key = mx.random.key(
            seed if seed is not None else int(time.time_ns() % (2**63))
        )

        position_ids = mx.maximum(mx.cumsum(prompt_mask, axis=-1) - 1, 0)
        logits, slow_hidden = self._slow_step(
            prompt, 0, mx.arange(prompt_width), position_ids, prompt_mask
        )
        prompt_lengths = prompt_mask.sum(axis=-1)
        previous = None
        finished = mx.zeros((batch_size,), dtype=mx.bool_)
        code_lengths = mx.zeros((batch_size,), dtype=mx.int64)
        generated_frames = []
        prompt_mask_np = prompt_mask

        for step in range(max_new_tokens):
            active_before = mx.logical_not(finished)
            keys = mx.random.split(rng_key, 4)
            rng_key = keys[3]
            semantic = self._sample_semantic(
                logits,
                top_k,
                top_p,
                temperature,
                previous,
                do_sample,
                (keys[0], keys[1]),
            )
            codebooks = self._generate_codebooks(
                slow_hidden, semantic, top_k, top_p, temperature, do_sample, keys[2]
            )
            emitted = mx.logical_and(active_before, semantic != eos_index)
            frame = mx.where(emitted[:, None], codebooks, -1)
            generated_frames.append(frame)
            code_lengths = code_lengths + emitted.astype(mx.int64)

            if previous is None:
                previous = mx.zeros(
                    (batch_size, config.ras_window_size), dtype=mx.int64
                )
            else:
                previous = mx.concatenate((previous[:, 1:], semantic[:, None]), axis=1)
            finished = mx.logical_or(finished, semantic == eos_index)
            mx.eval(finished, frame, code_lengths)
            if bool(mx.all(finished)):
                break

            # The slow stack is always fed FULL-vocabulary ids, even when the head
            # emits the compact space, because the prompt rows are full-vocabulary.
            if config.uses_falcon_slow:
                semantic_in = mx.where(
                    semantic == eos_index,
                    mx.full(semantic.shape, config.eos_token_id, dtype=semantic.dtype),
                    semantic + config.semantic_begin_id,
                )
            else:
                semantic_in = semantic
            next_column = mx.concatenate((semantic_in[:, None], codebooks), axis=1)[
                ..., None
            ]
            new_valid = active_before.astype(mx.int64)[:, None]
            prompt_mask_np = mx.concatenate((prompt_mask_np, new_valid), axis=1)
            physical_position = prompt_width + step
            token_position = (prompt_lengths + step)[:, None]
            logits, slow_hidden = self._slow_step(
                next_column,
                physical_position,
                mx.array([physical_position]),
                token_position,
                prompt_mask_np,
            )

        self._clear_generation_caches()
        if generated_frames:
            codes = mx.stack(generated_frames, axis=2)
            max_valid = int(mx.max(code_lengths).item()) if code_lengths.size else 0
            codes = codes[:, :, :max_valid]
        else:
            codes = mx.zeros((batch_size, config.num_codebooks, 0), dtype=mx.int64)
        return codes


class Model(nn.Module):
    """mlx-audio entry point wrapping ArkttsModel + ArkttsCodec + the prompt builder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = ArkttsModel(config)
        self.codec = ArkttsCodec(config)
        self._tokenizer = None
        self.model_path: Optional[Path] = None

    @property
    def sample_rate(self) -> int:
        return self.config.codec_sample_rate

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model.model_path = Path(model_path)
        return model

    # tokenizer / prompt -----------------------------------------------------
    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            # the reference processor pins fix_mistral_regex=False; match it
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), use_fast=True, fix_mistral_regex=False
            )
        return self._tokenizer

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text).strip().split())

    def _format_reference_text(self, text: str) -> str:
        import re

        cleaned = self._clean_text(text)
        if re.search(r"<\|speaker:\d+\|>", cleaned):
            return cleaned
        return f"<|speaker:0|>{cleaned}"

    def _prompt_segments(
        self, text: str, reference_text: Optional[str], has_reference: bool
    ):
        tokenizer = self._load_tokenizer()

        def encode_parts(parts):
            out = []
            for part in parts:
                out.extend(tokenizer.encode(part, add_special_tokens=False))
            return np.asarray(out, dtype=np.int64)

        target = self._clean_text(text)
        if not target:
            raise ValueError("text must not be empty")
        if not has_reference:
            full = encode_parts(
                [
                    "<|im_start|>system\n",
                    "convert the provided text to speech",
                    "<|im_end|>\n",
                    "<|im_start|>user\n",
                    target,
                    "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>",
                ]
            )
            return full, np.asarray([], dtype=np.int64)
        if not reference_text:
            raise ValueError(
                "reference_text is required when a reference voice is provided"
            )
        prefix = encode_parts(
            [
                "<|im_start|>system\n",
                "convert the provided text to speech reference to the following:\n\nText:\n",
                self._format_reference_text(reference_text),
                "\n\nSpeech:\n",
            ]
        )
        suffix = encode_parts(
            [
                "<|im_end|>\n",
                "<|im_start|>user\n",
                target,
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        )
        return prefix, suffix

    # audio ------------------------------------------------------------------
    def _load_reference_audio(self, ref_audio) -> mx.array:
        import soundfile as sf

        from mlx_audio.utils import resample_audio

        if isinstance(ref_audio, (str, Path)):
            array, source_rate = sf.read(
                str(ref_audio), dtype="float32", always_2d=True
            )
            array = array.mean(axis=1)
        else:
            array = np.asarray(ref_audio, dtype=np.float32)
            source_rate = self.config.codec_sample_rate
        if int(source_rate) != self.config.codec_sample_rate:
            array = np.asarray(
                resample_audio(array, int(source_rate), self.config.codec_sample_rate),
                dtype=np.float32,
            )
        return mx.array(array)

    def encode_reference(self, ref_audio) -> tuple[mx.array, mx.array]:
        audio = self._load_reference_audio(ref_audio)
        codes, code_lengths = self.codec.encode(
            audio[None], mx.array([audio.shape[0]], dtype=mx.int64)
        )
        return codes, code_lengths

    # generation -------------------------------------------------------------
    def generate(
        self,
        text: str,
        ref_audio=None,
        ref_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 512,
        do_sample: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        start = time.time()
        has_reference = ref_audio is not None
        prefix, suffix = self._prompt_segments(text, ref_text, has_reference)
        reference_codes = reference_code_lengths = None
        if has_reference:
            codes, lengths = self.encode_reference(ref_audio)
            mx.eval(codes, lengths)
            reference_codes = [np.asarray(codes[0])]
            reference_code_lengths = np.asarray(lengths)

        codes = self.model.generate_codes(
            [prefix],
            [suffix],
            reference_codes,
            reference_code_lengths,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            seed=seed,
        )
        mx.eval(codes)
        token_count = codes.shape[-1]
        if token_count == 0:
            raise RuntimeError("Model generated no audio frames")
        waveform = self.codec.decode(codes)[0]
        mx.eval(waveform)
        elapsed = time.time() - start
        samples = waveform.shape[0]
        duration = samples / self.sample_rate
        yield GenerationResult(
            audio=waveform,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=time.strftime("%H:%M:%S", time.gmtime(duration)),
            real_time_factor=elapsed / duration if duration else 0.0,
            prompt={"text": text, "ref_text": ref_text},
            audio_samples={"samples-per-sec": samples / elapsed if elapsed else 0.0},
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    # weights ----------------------------------------------------------------
    def sanitize(self, weights: dict) -> dict:
        """Remap reference checkpoints to this module tree.

        Handles: LM keys (prefix under model.), codec keys (prefix under codec.),
        weight-norm folding (both parametrizations and legacy g/v), conv layout
        transposes, Snake alpha reshape, and dropped rope buffers.

        Idempotent: weights already carrying the model./codec. prefixes (i.e. a
        pre-converted mlx repo) pass through untouched.
        """
        if all(key.startswith(("model.", "codec.")) for key in weights):
            return weights
        out = {}
        pending: dict[str, dict] = {}

        def fold_weight_norm(g: mx.array, v: mx.array) -> mx.array:
            # PyTorch weight_norm dim=0: per-output-channel norm over (I, K)
            norm = mx.sqrt(
                mx.sum(mx.square(v.reshape(v.shape[0], -1)), axis=1, keepdims=True)
            ).reshape(v.shape[0], *([1] * (v.ndim - 1)))
            return g * v / norm

        for key, value in weights.items():
            if key.endswith(("freqs_cis", "causal_mask")):
                continue
            if key.startswith("generator."):
                key = key[len("generator.") :]
            if not key.startswith(("model.", "codec.")):
                # raw reference LM checkpoint keys arrive unprefixed;
                # raw codec.pth keys arrive as encoder./decoder./quantizer.
                if key.split(".")[0] in ("encoder", "decoder", "quantizer"):
                    key = "codec." + key
                else:
                    key = "model." + key
            # collect weight-norm pairs for folding
            if ".parametrizations.weight.original0" in key:
                base = key.replace(".parametrizations.weight.original0", ".weight")
                pending.setdefault(base, {})["g"] = value
                continue
            if ".parametrizations.weight.original1" in key:
                base = key.replace(".parametrizations.weight.original1", ".weight")
                pending.setdefault(base, {})["v"] = value
                continue
            if key.endswith(".weight_g"):
                base = key[: -len(".weight_g")] + ".weight"
                pending.setdefault(base, {})["g"] = value
                continue
            if key.endswith(".weight_v"):
                base = key[: -len(".weight_v")] + ".weight"
                pending.setdefault(base, {})["v"] = value
                continue
            out[key] = value

        for base, pair in pending.items():
            out[base] = fold_weight_norm(pair["g"], pair["v"])

        # SANITIZE IS SELF-CONTAINED, by contract: the test suite calls it on a
        # `Model.__new__(Model)` with no __init__, so it must not read instance state.
        # The falcon variant is therefore detected from the KEYS, and the only work its
        # weights need here is a layout transpose, which is decidable from shapes.
        #
        # The muP multipliers the upstream FalconH1 sanitize folds are NOT folded here.
        # Every published arktts checkpoint carries them at unity, and `FalconSlowStack`
        # ASSERTS that at construction — a loud failure on a future checkpoint that
        # changes them, rather than dead code here that has to read a config it cannot see.
        remapped = {}
        for key, value in out.items():
            if key.startswith("model.slow."):
                # torch Conv1d (C, 1, K) -> mlx (C, K, 1). `shape[1] == 1` identifies the
                # torch layout (the kernel width is 4, never 1), so this cannot double-apply.
                if (
                    key.endswith("conv1d.weight")
                    and value.ndim == 3
                    and value.shape[1] == 1
                ):
                    value = value.transpose(0, 2, 1)
                remapped[key] = value
                continue
            if key.startswith("codec."):
                if key.endswith("alpha"):
                    # Snake (1, C, 1) -> (1, 1, C)
                    value = value.transpose(0, 2, 1)
                elif key.endswith(".conv.weight") and value.ndim == 3:
                    if isinstance_transpose(key):
                        # ConvTranspose1d: torch (I, O, K) -> mlx (O, K, I)
                        value = value.transpose(1, 2, 0)
                    else:
                        # Conv1d: torch (O, I, K) -> mlx (O, K, I)
                        value = value.transpose(0, 2, 1)
                elif (
                    ".in_proj.weight" in key or ".out_proj.weight" in key
                ) and value.ndim == 3:
                    # VQ 1x1 convs: torch (O, I, 1) -> mlx (O, 1, I)
                    value = value.transpose(0, 2, 1)
            remapped[key] = value
        return remapped


def isinstance_transpose(key: str) -> bool:
    """Conv keys that belong to ArkttsCausalConvTranspose1d modules:
    decoder blocks 1..4 hold theirs at block.1.conv; the quantizer's upsample
    stages hold theirs at upsample.<i>.0.conv (the .1 ConvNeXt dwconv is a
    regular grouped conv)."""
    import re

    return (
        re.search(r"decoder\.model\.[1-4]\.block\.1\.conv\.weight$", key) is not None
        or re.search(r"quantizer\.upsample\.\d+\.0\.conv\.weight$", key) is not None
    )

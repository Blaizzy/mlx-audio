from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

import mlx_audio.lm.models.bailing_moe as bailing_moe_impl
import mlx_audio.lm.models.qwen2 as qwen2_impl
from mlx_audio.lm.models.bailing_moe import Model as BailingMoeModel
from mlx_audio.lm.models.bailing_moe import ModelArgs as BailingMoeModelArgs
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen2 import ModelArgs as Qwen2ModelArgs
from mlx_audio.lm.models.qwen2 import Qwen2Model
from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult
from mlx_audio.utils import load_audio


def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# Each fragment shares 512 cache positions between its prompt and generated patches.
MING3_MAX_CACHE_LEN = 512
# Fragment size in "semantic units": one per Chinese character or English word.
MING3_MAX_FRAGMENT_LENGTH = 24

# Ming-3.0 adds a "Target language:" line for the languages beyond zh/en.
MING3_TARGET_LANGUAGES = {
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
}

_CJK_RE = re.compile(r"[一-鿿]")


def _semantic_length(text: str) -> int:
    return len(re.findall(r"[一-龥]", text)) + len(re.findall(r"[a-zA-Z]+", text))


def _add_period(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text
    period = "。" if _CJK_RE.search(text) else "."
    if text[-1] in ",，、;；":
        return text[:-1] + period
    if text[-1].isalnum():
        return text + period
    return text


def _split_fragments(text: str, max_length: int) -> List[str]:
    """Split text into sentence-aligned fragments of at most ``max_length``.

    Pack whole sentences, splitting long sentences at commas and long clauses
    at words or Chinese characters.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    text = " ".join(text.split())
    if _semantic_length(text) <= max_length:
        return [text]

    # Pieces keep their trailing whitespace, so joining them restores the
    # text exactly. ASCII punctuation only ends a piece before whitespace,
    # which keeps "3.14", "1,000", "U.S." and "..." intact.
    sentence_re = re.compile(r".*?(?:[。！？]|[.!?](?=\s)|$)\s*")
    clause_re = re.compile(r".*?(?:[，；、]|[,;](?=\s)|$)\s*")
    # Keep tags and non-Chinese words intact, including numbers and abbreviations.
    word_re = re.compile(r"\[[^\]]*\]|[一-龥]|[^\s一-龥]+|\s+")

    def units(pattern: re.Pattern, value: str) -> List[str]:
        return [m.group(0) for m in pattern.finditer(value) if m.group(0)]

    def pack(pieces: List[str]) -> List[str]:
        out, cur = [], ""
        for piece in pieces:
            if cur and _semantic_length(cur + piece) > max_length:
                out.append(cur)
                cur = piece
            else:
                cur += piece
        if cur:
            out.append(cur)
        return out

    pieces = []
    for sentence in units(sentence_re, text):
        if _semantic_length(sentence) <= max_length:
            pieces.append(sentence)
        else:
            for clause in units(clause_re, sentence):
                if _semantic_length(clause) > max_length:
                    pieces.extend(pack(word_re.findall(clause)))
                else:
                    pieces.append(clause)
    return [f.strip() for f in pack(pieces) if f.strip()]


_MING3_CAPTION_FIELDS = {
    "zh": ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感", "IP"),
    "en": (
        "id",
        "speaker",
        "dialect",
        "style",
        "speed",
        "pitch",
        "volume",
        "emotion",
        "ip",
    ),
}


def ming3_instruction(
    *,
    dialect: Optional[str] = None,
    style: Optional[str] = None,
    speed: Optional[str] = None,
    pitch: Optional[str] = None,
    volume: Optional[str] = None,
    emotion: Optional[str] = None,
    ip: Optional[str] = None,
    schema: Optional[str] = None,
) -> str:
    """Build a Ming-3.0 control caption to pass as ``instruct``.

    The JSON layout and key order are what the checkpoint was trained on.
    Values follow the Ming-3.0 cookbook, e.g. ``emotion="高兴"``,
    ``speed="慢速"``, ``dialect="cantonese"`` or ``ip="US_James"``. Dialect
    captions use the English schema, everything else the Chinese one.
    """
    if schema is None:
        schema = "en" if dialect is not None else "zh"
    fields = _MING3_CAPTION_FIELDS[schema]
    values = (1, "speaker_1", dialect, style, speed, pitch, volume, emotion)
    item = dict(zip(fields[:-1], values))
    item["BGM"] = {
        key: None for key in ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")
    }
    item[fields[-1]] = ip
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


# Built-in voice names: "US_James"-style English voices and Chinese names
# such as "灵小柔" or "灵信远_高兴".
_MING3_VOICE_RE = re.compile(
    r"^(?:(?:US|UK)_[A-Z][a-z]+|[\u4e00-\u9fff][\u4e00-\u9fff_]*)$"
)


def _is_style_instruction(instruct: Optional[str]) -> bool:
    """True for freeform text and voice-design (style) captions."""
    if instruct is None:
        return False
    try:
        caption = json.loads(instruct)
        item = caption["audio_sequence"][0]
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return True
    return item.get("风格", item.get("style")) is not None


def _trim_trailing_silence(
    speech: mx.array,
    sample_rate: int,
    threshold: float = 1e-3,
    tail_seconds: float = 0.3,
) -> mx.array:
    """Cut trailing near-silence, keeping a fixed tail."""
    frame = int(sample_rate * 0.1)
    tail = int(sample_rate * tail_seconds)
    if speech.shape[-1] < frame:
        return speech
    num_frames = speech.shape[-1] // frame
    # Discard any incomplete analysis frame.
    speech = speech[: num_frames * frame]
    scores = np.abs(np.array(speech, dtype=np.float32))
    scores = scores.reshape(num_frames, frame).mean(axis=-1)
    loud = np.nonzero(scores > threshold)[0]
    if loud.size == 0:
        return speech[:tail]
    return speech[: int(loud[-1]) * frame + frame + tail]


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "ming_omni_tts"
    text_config: Optional[dict] = None
    audio_tokenizer_config: Optional[dict] = None
    ditar_config: Optional[dict] = None
    aggregator_config: Optional[dict] = None
    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(
            model_type=config.get("model_type", "ming_omni_tts"),
            text_config=config.get("llm_config"),
            audio_tokenizer_config=config.get("audio_tokenizer_config"),
            ditar_config=config.get("ditar_config"),
            aggregator_config=config.get("aggregator_config"),
        )


def _apply_rope(x: mx.array, base: float, dims: int, offset: int) -> mx.array:
    dims = min(int(dims), x.shape[-1])
    if dims <= 0:
        return x

    x_rope = x[..., :dims]
    x_pass = x[..., dims:]
    seqlen = x.shape[2]
    # Compute RoPE angles in fp32 for stable long-context phase precision.
    pos = mx.arange(offset, offset + seqlen, dtype=mx.float32)
    inv = 1.0 / (float(base) ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    freqs = mx.outer(pos, inv)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb).astype(x.dtype)[None, None, :, :]
    sin = mx.sin(emb).astype(x.dtype)[None, None, :, :]

    half = dims // 2
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    x_rope = (x_rope * cos) + (rotated * sin)
    if dims < x.shape[-1]:
        return mx.concatenate([x_rope, x_pass], axis=-1)
    return x_rope


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    unsqueeze_dim: int = 1,
) -> Tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _apply_multimodal_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    mrope_section: Optional[List[int]] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[mx.array, mx.array]:
    mrope_section = list(mrope_section) if mrope_section is not None else [16, 24, 24]
    split_sizes = [int(s) * 2 for s in mrope_section]
    if sum(split_sizes) != cos.shape[-1]:
        # Fallback to first stream if section metadata doesn't match this head dim.
        return _apply_rotary_pos_emb(q, k, cos[0], sin[0], unsqueeze_dim=unsqueeze_dim)

    split_points = []
    running = 0
    for size in split_sizes[:-1]:
        running += size
        split_points.append(running)

    cos_chunks = mx.split(cos, split_points, axis=-1)
    sin_chunks = mx.split(sin, split_points, axis=-1)
    cos_m = mx.concatenate(
        [chunk[i % 3] for i, chunk in enumerate(cos_chunks)], axis=-1
    )
    sin_m = mx.concatenate(
        [chunk[i % 3] for i, chunk in enumerate(sin_chunks)], axis=-1
    )
    return _apply_rotary_pos_emb(q, k, cos_m, sin_m, unsqueeze_dim=unsqueeze_dim)


class MingBailingMoe3DRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.dim = dim
        self.base = base
        self._inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        mx.eval(self._inv_freq)

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        inv_freq = mx.broadcast_to(
            self._inv_freq[None, None, :, None],
            (3, position_ids.shape[1], self._inv_freq.shape[0], 1),
        )
        pos = mx.expand_dims(position_ids.astype(mx.float32), axis=2)
        freqs = mx.swapaxes(inv_freq @ pos, 2, 3)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def _prepare_sdpa_mask(
    mask: Optional[Union[str, mx.array]],
    q_len: int,
    kv_len: int,
    cache_offset: int,
    out_dtype: mx.Dtype,
) -> Optional[mx.array]:
    if mask is None:
        return None

    neg_inf = mx.array(-3.4028235e38, dtype=mx.float32).astype(out_dtype)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"Unsupported attention mask spec: {mask}")
        q_pos = mx.arange(cache_offset, cache_offset + q_len, dtype=mx.int32)[:, None]
        k_pos = mx.arange(kv_len, dtype=mx.int32)[None, :]
        causal = q_pos >= k_pos
        return mx.where(causal, mx.array(0.0, dtype=out_dtype), neg_inf)[
            None, None, :, :
        ]

    add_mask = mask
    if add_mask.dtype == mx.bool_:
        add_mask = mx.where(add_mask, mx.array(0.0, dtype=out_dtype), neg_inf)
    else:
        add_mask = add_mask.astype(out_dtype)

    while add_mask.ndim < 4:
        add_mask = mx.expand_dims(add_mask, axis=0)

    if add_mask.shape[-1] != kv_len:
        add_mask = add_mask[..., -kv_len:]
    if add_mask.shape[-2] != q_len:
        add_mask = add_mask[..., -q_len:, :]
    return add_mask


@mx.compile
def _group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    score_function,
):
    in_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)

    orig_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if n_group is not None and n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_top2 = mx.topk(scores, 2, axis=-1)[0]
        group_scores = group_top2.sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0, scores.dtype), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    selected_scores = mx.take_along_axis(scores, inds, axis=-1)
    order = mx.argsort(-selected_scores, axis=-1)
    inds = mx.take_along_axis(inds, order, axis=-1)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator
    scores = scores * routed_scaling_factor
    return inds, scores.astype(in_type)


class MingBailingMoeGate(bailing_moe_impl.BailingMoeGate):
    def __call__(self, x):
        logits = self.gate_proj(x.astype(mx.float32))
        return _group_expert_select(
            logits,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class MingBailingMoeMLP(bailing_moe_impl.BailingMoeMLP):
    def __call__(self, x) -> mx.array:
        x32 = x.astype(mx.float32)
        gate = self.gate_proj(x32)
        up = self.up_proj(x32)
        out = self.down_proj((gate * mx.sigmoid(gate)) * up)
        return out.astype(x.dtype)


class MingBailingMoeSparseMoeBlock(bailing_moe_impl.BailingMoeSparseMoeBlock):
    def __init__(self, args: BailingMoeModelArgs):
        super().__init__(args)
        self.gate = MingBailingMoeGate(args)
        shared_dim = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        )
        if args.num_shared_experts > 0 and args.moe_router_enable_shared_expert:
            self.shared_experts = MingBailingMoeMLP(
                args=args,
                intermediate_size=shared_dim * args.num_shared_experts,
            )

    def __call__(self, x):
        in_dtype = x.dtype
        x32 = x.astype(mx.float32)
        topk_idx, topk_weight = self.gate(x32)
        out = self.switch_mlp(x32, topk_idx)
        out = bailing_moe_impl.aggregate_expert_outputs(
            out.astype(mx.float32), topk_weight.astype(mx.float32)
        )
        if self.shared_experts is not None:
            out = out + self.shared_experts(x32).astype(mx.float32)
        return out.astype(in_dtype)


class MingBailingMoeAttention(nn.Module):
    def __init__(self, args: BailingMoeModelArgs):
        super().__init__()
        self.use_qk_norm = args.use_qk_norm
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        if (rope_dim := args.rotary_dim) is None:
            rope_dim = int(self.head_dim * args.partial_rotary_factor)
        self.rope_dims = int(rope_dim)
        self.rope_base = float(args.rope_theta)

        rope_scaling = args.rope_scaling if args.rope_scaling is not None else {}
        rope_type = str(rope_scaling.get("type", "")).lower() if rope_scaling else ""
        self.use_mrope = rope_type == "3d"
        self.mrope_section = (
            list(rope_scaling.get("mrope_section", [16, 24, 24]))
            if self.use_mrope
            else None
        )
        self.mrope = (
            MingBailingMoe3DRotaryEmbedding(self.rope_dims, self.rope_base)
            if self.use_mrope
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        qkv = self.query_key_value(x)

        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)
        queries = q.reshape(
            bsz, seqlen, self.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = k.reshape(
            bsz, seqlen, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = v.reshape(
            bsz, seqlen, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        cache_offset = cache.offset if cache is not None else 0
        if self.use_mrope:
            if position_ids is None:
                pos = mx.arange(cache_offset, cache_offset + seqlen, dtype=mx.int32)[
                    None, :
                ]
                pos = mx.broadcast_to(pos, (bsz, seqlen))
                position_ids = mx.stack([pos, pos, pos], axis=0)
            cos, sin = self.mrope(values, position_ids=position_ids)
            queries, keys = _apply_multimodal_rotary_pos_emb(
                queries,
                keys,
                cos,
                sin,
                mrope_section=self.mrope_section,
            )
        else:
            queries = _apply_rope(queries, self.rope_base, self.rope_dims, cache_offset)
            keys = _apply_rope(keys, self.rope_base, self.rope_dims, cache_offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        attn_mask = _prepare_sdpa_mask(
            mask=mask,
            q_len=seqlen,
            kv_len=keys.shape[-2],
            cache_offset=cache_offset,
            out_dtype=queries.dtype,
        )

        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=attn_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return self.dense(output)


class MingBailingMoeDecoderLayer(bailing_moe_impl.BailingMoeDecoderLayer):
    def __init__(self, args: BailingMoeModelArgs, layer_idx: int):
        nn.Module.__init__(self)
        self.attention = MingBailingMoeAttention(args)
        self.mlp = (
            MingBailingMoeSparseMoeBlock(args)
            if args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            else MingBailingMoeMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.attention(
            self.input_layernorm(x), mask, cache, position_ids=position_ids
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class MingBailingMoeBackbone(bailing_moe_impl.BailingMoeModel):
    def __init__(self, args: BailingMoeModelArgs):
        nn.Module.__init__(self)
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MingBailingMoeDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ):
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_ids=position_ids)

        return self.norm(h)


class MingBailingMoeModel(BailingMoeModel):
    def __init__(self, args: BailingMoeModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = MingBailingMoeBackbone(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        position_ids: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, position_ids=position_ids)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out


class MingQwen2Attention(qwen2_impl.Attention):
    def __init__(self, args: Qwen2ModelArgs, sliding_window: Optional[int]):
        super().__init__(args)
        self.sliding_window = (
            int(sliding_window)
            if sliding_window is not None and int(sliding_window) > 0
            else None
        )

    @staticmethod
    def _build_sliding_additive_mask(
        mask: Optional[Union[str, mx.array]],
        q_len: int,
        kv_len: int,
        cache_offset: int,
        sliding_window: int,
        dtype: Optional[mx.Dtype] = None,
    ) -> mx.array:
        dtype = dtype if dtype is not None else mx.float32
        neg_inf = mx.array(-3.4028235e38, dtype=mx.float32).astype(dtype)
        q_pos = mx.arange(cache_offset, cache_offset + q_len, dtype=mx.int32)[:, None]
        k_pos = mx.arange(kv_len, dtype=mx.int32)[None, :]
        sw_allowed = k_pos > (q_pos - sliding_window)
        sw_add = mx.where(sw_allowed, mx.array(0.0, dtype=dtype), neg_inf)[
            None, None, :, :
        ]

        if mask is None:
            return sw_add

        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unsupported attention mask spec: {mask}")
            causal_allowed = q_pos >= k_pos
            causal_add = mx.where(causal_allowed, mx.array(0.0, dtype=dtype), neg_inf)[
                None, None, :, :
            ]
            return mx.minimum(causal_add, sw_add)

        add_mask = mask
        if add_mask.dtype == mx.bool_:
            add_mask = mx.where(add_mask, mx.array(0.0, dtype=dtype), neg_inf)
        else:
            add_mask = add_mask.astype(dtype)
        while add_mask.ndim < 4:
            add_mask = mx.expand_dims(add_mask, axis=0)
        return mx.minimum(add_mask, sw_add)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.sliding_window is None:
            return super().__call__(x, mask, cache)

        bsz, seqlen, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(bsz, seqlen, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cache_offset = cache.offset if cache is not None else 0
        if cache is not None:
            queries = self.rope(queries, offset=cache_offset)
            keys = self.rope(keys, offset=cache_offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        attn_mask = self._build_sliding_additive_mask(
            mask=mask,
            q_len=seqlen,
            kv_len=keys.shape[2],
            cache_offset=cache_offset,
            sliding_window=self.sliding_window,
            dtype=queries.dtype,
        )
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=attn_mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class MingQwen2Model(Qwen2Model):
    def __init__(self, args: Qwen2ModelArgs, cfg_dict: Dict[str, Any]):
        super().__init__(args)
        use_sliding = bool(cfg_dict.get("use_sliding_window", False))
        sliding_window = int(cfg_dict.get("sliding_window", 0)) if use_sliding else 0
        max_window_layers = int(cfg_dict.get("max_window_layers", len(self.layers)))
        layer_types = cfg_dict.get("layer_types")

        per_layer_sliding: List[Optional[int]] = []
        for idx in range(len(self.layers)):
            layer_sliding: Optional[int] = None
            if sliding_window > 0:
                if isinstance(layer_types, list) and idx < len(layer_types):
                    if layer_types[idx] == "sliding_attention":
                        layer_sliding = sliding_window
                elif idx >= max_window_layers:
                    layer_sliding = sliding_window
            per_layer_sliding.append(layer_sliding)

        for idx, sliding_window in enumerate(per_layer_sliding):
            self.layers[idx].self_attn = MingQwen2Attention(args, sliding_window)

    @property
    def word_embeddings(self):
        # Shared callers use this alias for both dense and MoE backbones.
        return self.embed_tokens


class MingQwen2ForCausalLM(nn.Module):
    def __init__(self, args: Qwen2ModelArgs, cfg_dict: Dict[str, Any]):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MingQwen2Model(args, cfg_dict)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }


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
        return out.reshape(x.shape).astype(x.dtype)

    def __call__(
        self, q: mx.array, k: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array]:
        seq_len = q.shape[2]
        # Preserve attention precision through phase generation and rotation.
        pos = mx.arange(offset, offset + seq_len, dtype=self.inv_freq.dtype)
        freqs = mx.outer(pos, self.inv_freq)
        cos = mx.cos(freqs)[None, None, :, :]
        sin = mx.sin(freqs)[None, None, :, :]
        return self._apply(q, cos, sin), self._apply(k, cos, sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        self.ff = [[nn.Linear(dim, inner_dim)], None, nn.Linear(inner_dim, dim_out)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ff[0][0](x)
        x = nn.gelu_approx(x)
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
        q = (
            self.to_q(x)
            .reshape(bsz, seqlen, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.to_k(x)
            .reshape(bsz, seqlen, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.to_v(x)
            .reshape(bsz, seqlen, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
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
        # Ming guidance is pred + (pred - null_pred) * cfg_scale, so even a
        # scale of 1 needs both branches (and 0 selects the conditional one).
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
    def __init__(
        self, func, y0: mx.array, sigma: float = 0.25, temperature: float = 1.5
    ):
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(
        self, t: mx.array, sde_noise: Optional[mx.array] = None
    ) -> List[mx.array]:
        y = self.y0
        solution = [y]
        for i in range(1, t.shape[0]):
            t0 = t[i - 1]
            t1 = t[i]
            dt = t1 - t0
            # Keep each Euler update in the state dtype.
            f0 = self.func(t0, y)
            if f0.dtype != y.dtype:
                f0 = f0.astype(y.dtype)
            dy = dt.astype(y.dtype) * f0
            y = y + dy
            noise = (
                mx.random.normal(y.shape, dtype=y.dtype)
                if sde_noise is None
                else sde_noise[i - 1]
            )
            shift = self.sigma * (self.temperature**0.5) * (mx.abs(dt) ** 0.5) * noise
            y = y + shift
            solution.append(y)
        return solution


class CFM(nn.Module):
    def __init__(self, model: nn.Module, ming3_sampling: bool = False):
        super().__init__()
        self.model = model
        self.ming3_sampling = ming3_sampling

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
        sde_noise: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        solver_dtype = noise.dtype if self.ming3_sampling else mx.float32

        def fn(t: mx.array, x: mx.array) -> mx.array:
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
                cfg_scale=cfg_scale,
                patch_size=patch_size,
            )
            pred, null_pred = mx.split(pred_cfg, 2, axis=0)
            return (pred + (pred - null_pred) * cfg_scale).astype(solver_dtype)

        y0 = noise if self.ming3_sampling else noise.transpose(0, 2, 1)
        y0 = y0.astype(solver_dtype)
        if use_epss:
            t = get_epss_timesteps(steps, dtype=solver_dtype)
        else:
            t = mx.linspace(0, 1, steps + 1, dtype=solver_dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (mx.cos(math.pi / 2 * t) - 1 + t)

        solver = Solver(fn, y0, sigma=sigma, temperature=temperature)
        trajectory = solver.integrate(t, sde_noise=sde_noise)
        return trajectory[-1], trajectory


class FlowLoss(nn.Module):
    def __init__(
        self,
        z_channels: int,
        llm_cond_dim: int,
        ming3_sampling: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.cfm = CFM(
            model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs),
            ming3_sampling=ming3_sampling,
        )

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
        if self.cfm.ming3_sampling:
            # Draw all noise up front, including at zero temperature.
            noise = mx.random.normal(
                (z.shape[0], patch_size, self.z_channels), dtype=z.dtype
            )
            sde_noise = mx.random.normal((steps, *noise.shape), dtype=z.dtype)
        else:
            noise = mx.random.normal(
                (z.shape[0], self.z_channels, patch_size), dtype=mx.float32
            )
            sde_noise = None
        sampled, trajectory = self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
            steps=steps,
            sde_noise=sde_noise,
        )
        return sampled, trajectory


class _ISTFTWindowState(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()
        n = n_fft
        # Periodic Hann window.
        self.window = mx.array(
            0.5 - 0.5 * mx.cos(2.0 * math.pi * mx.arange(n, dtype=mx.float32) / n),
            dtype=mx.float32,
        )


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = _ISTFTWindowState(n_fft)

    def _buffer_process(
        self,
        x,
        buffer,
        pad: int,
        last_chunk: bool = False,
        streaming: bool = False,
    ):
        import numpy as np

        buffer_len = self.n_fft - self.hop_length
        if streaming:
            if buffer is None:
                x = x[:, pad:]
            if buffer is not None:
                x[:, :buffer_len] += buffer
            buffer = x[:, -buffer_len:]
            if not last_chunk:
                x = x[:, :-buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]
            buffer = None

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        return x, buffer

    def __call__(
        self,
        x: mx.array,
        audio_buffer=None,
        window_buffer=None,
        streaming: bool = False,
        last_chunk: bool = False,
    ):
        import numpy as np

        x_pred = self.out(x).transpose(0, 2, 1)
        mag, phase = mx.split(x_pred, 2, axis=1)
        mag = mx.exp(mag)
        mag = mx.clip(mag, None, 1e2)
        spec = mag * (mx.cos(phase) + 1j * mx.sin(phase))

        spec_np = np.array(spec)
        window = np.array(self.istft.window.astype(mx.float32))
        batch, _, frames = spec_np.shape
        output_size = (frames - 1) * self.hop_length + self.n_fft
        pad = (self.n_fft - self.hop_length) // 2

        ifft = np.fft.irfft(spec_np, n=self.n_fft, axis=1).astype(np.float32)
        ifft *= window[None, :, None]

        audio = np.zeros((batch, output_size), dtype=np.float32)
        env = np.zeros((batch, output_size), dtype=np.float32)
        win_sq = window * window

        for t in range(frames):
            start = t * self.hop_length
            end = start + self.n_fft
            audio[:, start:end] += ifft[:, :, t]
            env[:, start:end] += win_sq

        audio, audio_buffer = self._buffer_process(
            audio,
            audio_buffer,
            pad=pad,
            last_chunk=last_chunk,
            streaming=streaming,
        )
        env, window_buffer = self._buffer_process(
            env,
            window_buffer,
            pad=pad,
            last_chunk=last_chunk,
            streaming=streaming,
        )

        env = np.clip(env, 1e-11, None)
        audio = audio / env
        audio_mx = mx.array(audio)
        if streaming:
            return audio_mx, x_pred, audio_buffer, window_buffer
        return audio_mx


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
        self.encoder = MingQwen2Model(cfg, encoder_args)
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
            self.aggregator = MingQwen2Model(agg_cfg, agg_cfg_dict)
            self.cls_embed = mx.random.normal((1, 1, cfg.hidden_size)).astype(
                mx.float32
            )

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
        # Frame extraction returns float32; project in the encoder's dtype.
        weight = getattr(self.fc1, "scales", self.fc1.weight)
        x = x.astype(weight.dtype)
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
        self.decoder = MingQwen2Model(cfg, decoder_args)
        self.fc1 = nn.Linear(latent_dim, cfg.hidden_size)
        self.patch_size = patch_size
        self.head = ISTFTHead(
            dim=cfg.hidden_size, n_fft=output_dim * 4, hop_length=output_dim
        )

    def _upsample(self, x: mx.array) -> mx.array:
        x_bc = x.transpose(0, 2, 1)
        y_bc = self._linear_upsample_1d(x_bc)
        return y_bc.transpose(0, 2, 1)

    def _linear_upsample_1d(self, x_bc: mx.array) -> mx.array:
        # Keep interpolation on-device; NumPy cannot consume MLX bfloat16 buffers.
        x = x_bc.astype(mx.float32)
        in_t = x.shape[-1]
        scale = int(self.patch_size)
        out_t = in_t * scale
        idx = (mx.arange(out_t, dtype=mx.float32) + 0.5) / scale - 0.5
        left = mx.floor(idx).astype(mx.int32)
        right = left + 1
        w_right = idx - left.astype(mx.float32)
        left_v = mx.take(x, mx.clip(left, 0, in_t - 1), axis=-1)
        right_v = mx.take(x, mx.clip(right, 0, in_t - 1), axis=-1)
        return (
            left_v * (1.0 - w_right)[None, None, :] + right_v * w_right[None, None, :]
        )

    def _streaming_linear_upsample(self, x: mx.array, state, is_last: bool = False):
        if state is None:
            state = {
                "prev_chunk": None,
                "history_last": None,
                "is_first": True,
            }

        if x is None and not is_last:
            return None, state

        if state["is_first"] and is_last:
            out = self._upsample(x)
            return out, None

        output_chunks = []

        if state["is_first"]:
            state["prev_chunk"] = x
            state["is_first"] = False
            if not is_last:
                return None, state

        if state["prev_chunk"] is not None:
            p = state["prev_chunk"].transpose(0, 2, 1)
            if x is None:
                lookahead = p[:, :, -1:]
            else:
                lookahead = x[:, :1, :].transpose(0, 2, 1)

            if state["history_last"] is None:
                inp = mx.concatenate([p, lookahead], axis=2)
                up = self._linear_upsample_1d(inp)
                out_prev = up[:, :, : p.shape[2] * self.patch_size]
            else:
                inp = mx.concatenate([state["history_last"], p, lookahead], axis=2)
                up = self._linear_upsample_1d(inp)
                start = self.patch_size
                end = start + p.shape[2] * self.patch_size
                out_prev = up[:, :, start:end]

            output_chunks.append(out_prev.transpose(0, 2, 1))
            state["history_last"] = p[:, :, -1:]
            state["prev_chunk"] = x

        if is_last:
            p = state["prev_chunk"].transpose(0, 2, 1)
            history_last = state["history_last"]
            if history_last is None:
                history_last = p[:, :, :1]
            inp = mx.concatenate([history_last, p], axis=2)
            up = self._linear_upsample_1d(inp)
            out_last = up[:, :, self.patch_size :]
            output_chunks.append(out_last.transpose(0, 2, 1))
            state = None

        final_out = mx.concatenate(output_chunks, axis=1) if output_chunks else None
        return final_out, state

    def low_level_reconstruct(
        self,
        x: mx.array,
        past_key_values=None,
        use_cache: bool = False,
        stream_state=(None, None, None),
        last_chunk: bool = False,
    ):
        upsample_state, audio_buffer, window_buffer = stream_state
        bsz = int(x.shape[0])
        x = self.fc1(x)
        if self.patch_size != -1:
            if use_cache:
                x, upsample_state = self._streaming_linear_upsample(
                    x, state=upsample_state, is_last=last_chunk
                )
                if x is None:
                    empty = mx.zeros((bsz, 0), dtype=mx.float32)
                    return (
                        empty,
                        (upsample_state, audio_buffer, window_buffer),
                        past_key_values,
                    )
            else:
                x = self._upsample(x)

        dummy = mx.zeros((x.shape[0], x.shape[1]), dtype=mx.int32)
        if use_cache:
            if past_key_values is None:
                past_key_values = [KVCache() for _ in range(len(self.decoder.layers))]
            x = self.decoder(dummy, cache=past_key_values, input_embeddings=x)
            waveform, _, audio_buffer, window_buffer = self.head(
                x,
                audio_buffer=audio_buffer,
                window_buffer=window_buffer,
                streaming=True,
                last_chunk=last_chunk,
            )
            stream_state = (upsample_state, audio_buffer, window_buffer)
            return waveform, stream_state, past_key_values

        x = self.decoder(dummy, input_embeddings=x)
        waveform = self.head(x)
        return waveform, (None, None, None), None


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

    def encode_latent(
        self, waveform: mx.array, waveform_length: mx.array
    ) -> Tuple[mx.array, mx.array]:
        frame_num = mx.ceil(
            waveform_length / self.config["enc_kwargs"]["input_dim"]
        ).astype(mx.int32)
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
        return self.decoder.low_level_reconstruct(
            latent,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=stream_state,
            last_chunk=last_chunk,
        )


class Model(nn.Module):
    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.model_type = "ming_omni_tts"
        self.tokenizer = None

        if not config.text_config:
            raise ValueError("Missing text_config in Ming Omni config")
        if not config.audio_tokenizer_config:
            raise ValueError("Missing audio_tokenizer_config in Ming Omni config")
        if not config.ditar_config:
            raise ValueError("Missing ditar_config in Ming Omni config")
        if not config.aggregator_config:
            raise ValueError("Missing aggregator_config in Ming Omni config")

        llm_cfg = dict(config.text_config)
        # Dense backbones use Qwen chat turns; MoE backbones use role tags.
        self.chat_layout = not self._is_moe_llm_config(llm_cfg)
        if self._is_moe_llm_config(llm_cfg):
            self.llm_args = BailingMoeModelArgs.from_dict(llm_cfg)
            self.model = MingBailingMoeModel(self.llm_args)
        else:
            self.llm_args = Qwen2ModelArgs.from_dict(llm_cfg)
            self.model = MingQwen2ForCausalLM(self.llm_args, llm_cfg)

        self.audio = AudioVAE(config.audio_tokenizer_config)
        self.latent_dim = int(config.audio_tokenizer_config["enc_kwargs"]["latent_dim"])
        self.patch_size = int(config.ditar_config["patch_size"])
        self.history_patch_size = int(
            config.ditar_config.get("history_patch_size", self.patch_size)
        )
        # One-latent patches distinguish Ming-3.0 from Ming Omni dense models.
        self.ming3 = (
            self.chat_layout and self.patch_size == 1 and self.history_patch_size == 1
        )
        self.generate_config = {"temperature": 0.0}

        self.linear_proj_audio = Aggregator(
            in_channels=self.latent_dim,
            llm_input_dim=self.llm_args.hidden_size,
            **config.aggregator_config,
        )
        self.flowloss = FlowLoss(
            z_channels=self.latent_dim,
            llm_cond_dim=self.llm_args.hidden_size,
            ming3_sampling=self.ming3,
            **config.ditar_config,
        )
        self.stop_head = nn.Linear(self.llm_args.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.llm_args.hidden_size, bias=True)

    @property
    def sample_rate(self) -> int:
        return self.audio.sample_rate

    @property
    def layers(self):
        return self.model.model.layers

    @staticmethod
    def _is_moe_llm_config(llm_cfg: Dict[str, Any]) -> bool:
        moe_keys = (
            "moe_intermediate_size",
            "num_experts",
            "num_shared_experts",
            "norm_topk_prob",
            "num_experts_per_tok",
            "first_k_dense_replace",
        )
        return all(llm_cfg.get(k) is not None for k in moe_keys)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Apply the backbone's weight remapping to the LLM shard.
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

            if ".audio_gate." in k or ".image_gate." in k:
                continue
            sanitized[f"model.{k}"] = v
        for k, v in weights.items():
            if not k.startswith("model."):
                if not k.startswith(allowed_non_llm_prefixes):
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
        spk_emb: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, mx.array]:
        if self.chat_layout:
            spk_start, spk_pad, spk_end = (
                "<|vision_start|>",
                "<|vision_pad|>",
                "<|vision_end|>\n",
            )
        else:
            spk_start, spk_pad, spk_end = "<spk>", "<audioPatch>", "</spk>\n"
        spk_emb = spk_emb or []
        spk_prompt = []
        for i in range(len(spk_emb)):
            spk_prompt += (
                self._encode(f"  speaker_{i + 1}:")
                + self._encode(spk_start)
                + self._encode(spk_pad)
                + self._encode(spk_end)
            )

        prompt_text_token = []
        prompt_latent_token = []
        if prompt_latent is not None and prompt_text is not None:
            if self.ming3:
                embedding = self.model.model.word_embeddings
                weight = getattr(embedding, "scales", embedding.weight)
                prompt_latent = prompt_latent.astype(weight.dtype)
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
            instruction_eos = "<|im_end|>" if self.ming3 else "<|endoftext|>"
            instruction_prompt = self._encode(instruction) + self._encode(
                instruction_eos
            )

        if self.chat_layout:
            user_start = self._encode(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ) + self._encode("<|im_start|>user\n")
            turn_end = self._encode("<|im_end|>\n") + self._encode(
                "<|im_start|>assistant\n"
            )
        else:
            user_start = self._encode("<role>HUMAN</role>")
            turn_end = self._encode("<role>ASSISTANT</role>")

        input_part = (
            user_start
            + self._encode(prompt)
            + spk_prompt
            + prompt2
            + prompt_text_token
            + self._encode(text)
            + turn_end
            + instruction_prompt
            + self._encode("<audio>")
            + prompt_latent_token
        )

        input_ids = mx.array(input_part, dtype=mx.int32)[None, :]
        inputs_embeds = self.model.model.word_embeddings(input_ids)

        if spk_emb:
            spk_start_id = self.tokenizer.convert_tokens_to_ids(spk_start)
            starts = np.where(np.array(input_ids[0]) == spk_start_id)[0]
            pieces, prev = [], 0
            for pos, emb in zip(starts, spk_emb):
                pos = int(pos) + 1
                pieces += [
                    inputs_embeds[:, prev:pos, :],
                    emb.reshape(1, 1, -1).astype(inputs_embeds.dtype),
                ]
                prev = pos + 1
            pieces.append(inputs_embeds[:, prev:, :])
            inputs_embeds = mx.concatenate(pieces, axis=1)

        if prompt_latent is not None:
            audio_token_id = self.tokenizer.convert_tokens_to_ids("<audio>")
            audio_positions = np.where(np.array(input_ids[0]) == audio_token_id)[0]
            if audio_positions.size > 0:
                start = int(audio_positions[0]) + 1
                end = start + prompt_latent.shape[1]
                prefix = inputs_embeds[:, :start, :]
                suffix = inputs_embeds[:, end:, :]
                inputs_embeds = mx.concatenate([prefix, prompt_latent, suffix], axis=1)

        return input_ids, inputs_embeds

    def _pad_prompt_waveform(self, waveform: mx.array) -> mx.array:
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        # Align to the tokenizer's 12.5 Hz frames.
        pad_align = int((1 / 12.5) * self.patch_size * self.sample_rate)
        if pad_align <= 0:
            return waveform
        cur_len = int(waveform.shape[-1])
        new_len = ((cur_len + pad_align - 1) // pad_align) * pad_align
        if new_len == cur_len:
            return waveform
        pad = mx.zeros((waveform.shape[0], new_len - cur_len), dtype=waveform.dtype)
        if self.ming3:
            # Ming-3.0 pads at the front: the prompt's tail seeds the latent
            # history, so trailing silence is what the model would continue.
            return mx.concatenate([pad, waveform], axis=1)
        return mx.concatenate([waveform, pad], axis=1)

    def _encode_prompt_latent(self, prompt_waveform: mx.array) -> mx.array:
        prompt_waveform = self._pad_prompt_waveform(prompt_waveform)
        prompt_waveform_length = mx.array([prompt_waveform.shape[1]], dtype=mx.int32)
        prompt_latent, _ = self.audio.encode_latent(
            prompt_waveform, prompt_waveform_length
        )
        return prompt_latent

    def _speaker_embeddings(
        self, spk_emb: Optional[Any], use_zero_spk_emb: bool
    ) -> Optional[List[mx.array]]:
        """Project 192-dim speaker embeddings into the LLM's speaker slots."""
        if use_zero_spk_emb:
            # Zero the projected speaker slot.
            return [mx.zeros((self.llm_args.hidden_size,))]
        if spk_emb is None:
            return None
        if not isinstance(spk_emb, (list, tuple)):
            spk_emb = [spk_emb]
        projected = []
        for emb in spk_emb:
            emb = mx.array(emb, dtype=mx.float32).reshape(1, -1)
            if emb.shape[-1] != 192:
                raise ValueError(f"spk_emb must be 192-dim, got {emb.shape[-1]} values")
            projected.append(self.spk_head(emb)[0])
        return projected

    def _forward_llm_hidden(
        self, inputs_embeds: mx.array, cache: List[KVCache]
    ) -> mx.array:
        h = inputs_embeds
        mask = create_attention_mask(h, cache[0] if cache else None)
        first_layer = self.model.model.layers[0] if self.model.model.layers else None
        first_attention = getattr(first_layer, "attention", None)
        use_mrope = bool(
            first_attention is not None and getattr(first_attention, "use_mrope", False)
        )
        position_ids = None
        if use_mrope:
            offset = cache[0].offset if cache else 0
            seq_len = h.shape[1]
            pos = mx.arange(offset, offset + seq_len, dtype=mx.int32)[None, :]
            pos = mx.broadcast_to(pos, (h.shape[0], seq_len))
            position_ids = mx.stack([pos, pos, pos], axis=0)

        is_moe_backbone = bool(
            self.model.model.layers and hasattr(self.model.model.layers[0], "attention")
        )
        for layer, layer_cache in zip(self.model.model.layers, cache):
            if is_moe_backbone:
                h = layer(h, mask, layer_cache, position_ids=position_ids)
            else:
                h = layer(h, mask, layer_cache)
        return self.model.model.norm(h)

    def sample(
        self,
        prompt: str,
        text: str,
        instruction: Optional[str] = None,
        prompt_waveform: Optional[mx.array] = None,
        prompt_text: Optional[str] = None,
        max_decode_steps: Optional[int] = None,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
        flow_steps: int = 10,
        prompt_latent: Optional[mx.array] = None,
        spk_emb: Optional[List[mx.array]] = None,
    ):
        if prompt_text is None:
            prompt_latent = None
        elif prompt_latent is None and prompt_waveform is not None:
            prompt_latent = self._encode_prompt_latent(prompt_waveform)

        _, inputs_embeds = self._prepare_input_embed(
            prompt=prompt,
            text=text,
            instruction=instruction,
            prompt_latent=prompt_latent,
            prompt_text=prompt_text,
            spk_emb=spk_emb,
        )

        cache = [KVCache() for _ in range(self.llm_args.num_hidden_layers)]
        latent_history = None
        if max_decode_steps is None:
            max_decode_steps = 1000 if self.ming3 else 200

        if self.ming3:
            budget = (MING3_MAX_CACHE_LEN - inputs_embeds.shape[1]) // self.patch_size
            if budget <= 0:
                raise ValueError(
                    f"Ming-3.0 prompt has {inputs_embeds.shape[1]} positions and "
                    f"exhausts the {MING3_MAX_CACHE_LEN}-position cache. Shorten "
                    "the text, instruction, or reference audio."
                )
            max_decode_steps = min(max_decode_steps, budget)

        for step in range(max_decode_steps):
            hidden = self._forward_llm_hidden(inputs_embeds=inputs_embeds, cache=cache)
            z_diff = hidden[:, -1:, :]
            if step == 0:
                latent_history = mx.zeros(
                    (1, self.history_patch_size, self.latent_dim),
                    dtype=z_diff.dtype,
                )
                if prompt_latent is not None:
                    prompt_latent = prompt_latent.astype(z_diff.dtype)
                    start = self.history_patch_size - prompt_latent.shape[1]
                    if start < 0:
                        latent_history = prompt_latent[:, -start:, :]
                    else:
                        latent_history = mx.concatenate(
                            [latent_history[:, :start, :], prompt_latent],
                            axis=1,
                        )

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
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Iterable[GenerationResult]:
        del streaming_interval

        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is not initialized. Load model with load()/from_pretrained first."
            )

        # Built-in voices are an IP caption over an empty speaker slot. Only
        # Ming voice names are mapped, so a default voice that a client sends
        # for every model (e.g. "alloy") is ignored rather than becoming a
        # caption.
        if (
            self.ming3
            and voice
            and ref_audio is None
            and _MING3_VOICE_RE.match(str(voice))
        ):
            if instruct is not None:
                raise ValueError("Pass either voice or instruct, not both")
            instruct = ming3_instruction(ip=voice)
            kwargs["use_zero_spk_emb"] = True

        if isinstance(ref_audio, str):
            ref_audio = load_audio(ref_audio, sample_rate=self.sample_rate)

        start_time = time.perf_counter()
        prompt = kwargs.get(
            "prompt", "Please generate speech based on the following description.\n"
        )
        language = MING3_TARGET_LANGUAGES.get(str(lang_code or "").strip().lower())
        if self.ming3 and language and "prompt" not in kwargs:
            prompt += f"Target language: {language}.\n"
        cfg = 2.0 if cfg_scale is None else cfg_scale
        flow_steps = 10 if ddpm_steps is None else ddpm_steps
        sigma = float(kwargs.get("sigma", 0.25))
        if max_tokens is None:
            # Ming-3.0 is bounded per fragment by its 512-position cache.
            max_tokens = 1000 if self.ming3 else 200
        max_decode_steps = int(kwargs.get("max_decode_steps", max_tokens))
        spk_emb = self._speaker_embeddings(
            kwargs.get("spk_emb"), bool(kwargs.get("use_zero_spk_emb", False))
        )

        # An acoustic prompt is only usable together with its transcript.
        prompt_latent = None
        if ref_audio is not None and ref_text is not None:
            prompt_latent = self._encode_prompt_latent(ref_audio)

        if self.ming3 and kwargs.get("raw_text", False):
            fragments = [text]
        elif self.ming3 and _is_style_instruction(instruct):
            # Apply style captions to the whole utterance.
            fragments = [_add_period(" ".join(text.split()))]
        elif self.ming3:
            max_length = int(kwargs.get("max_length", MING3_MAX_FRAGMENT_LENGTH))
            fragments = _split_fragments(_add_period(text), max_length)
        else:
            fragments = [text]

        if speed != 1.0:
            if verbose:
                print("[WARN] speed parameter is currently ignored for Ming Omni TTS.")

        def result(audio, text, started, segment_idx, is_streaming_chunk, is_final):
            mx.eval(audio)
            elapsed = max(1e-6, time.perf_counter() - started)
            samples = int(audio.shape[0])
            token_count = len(self._encode(text))
            duration_seconds = samples / self.sample_rate
            return GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=token_count,
                audio_duration=format_duration(duration_seconds),
                real_time_factor=elapsed / max(duration_seconds, 1e-6),
                prompt={"tokens": token_count, "tokens-per-sec": token_count / elapsed},
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": samples / elapsed,
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=is_streaming_chunk,
                is_final_chunk=is_final,
            )

        speech_parts = []
        chunk_start = start_time
        for idx, fragment in enumerate(fragments):
            sampler = self.sample(
                prompt=prompt,
                text=fragment,
                instruction=instruct,
                prompt_text=ref_text,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                flow_steps=flow_steps,
                prompt_latent=prompt_latent,
                spk_emb=spk_emb,
            )
            if self.ming3:
                # Decode all latents in one pass per fragment.
                latents = [tokens for tokens, _ in sampler]
                if not latents:
                    raise RuntimeError(
                        "No audio latents were generated for a text fragment"
                    )
                part, _, _ = self.audio.decode(
                    mx.concatenate(latents, axis=1), use_cache=False, last_chunk=True
                )
                part = part[0]
                if kwargs.get("trim_silence", True):
                    part = _trim_trailing_silence(part, self.sample_rate)
                if stream:
                    # Fragments decode independently, so each can play once ready.
                    is_final = idx == len(fragments) - 1
                    yield result(part, fragment, chunk_start, idx, True, is_final)
                    chunk_start = time.perf_counter()
                else:
                    speech_parts.append(part)
                continue

            speech_chunks = []
            stream_state = (None, None, None)
            past_key_values = None
            for sampled_tokens, last_chunk in sampler:
                speech_tmp, stream_state, past_key_values = self.audio.decode(
                    sampled_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    stream_state=stream_state,
                    last_chunk=last_chunk,
                )
                speech_chunks.append(speech_tmp)
                if last_chunk:
                    break
            if speech_chunks:
                speech_parts.append(mx.concatenate(speech_chunks, axis=1)[0])

        if self.ming3 and stream:
            return
        if not speech_parts:
            raise RuntimeError("No audio chunks were generated")
        speech = (
            speech_parts[0]
            if len(speech_parts) == 1
            else mx.concatenate(speech_parts, axis=0)
        )
        yield result(speech, text, start_time, 0, False, True)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from transformers import AutoTokenizer

        # Convert campplus ONNX -> safetensors
        onnx_path = model_path / "campplus.onnx"
        safetensors_path = model_path / "campplus.safetensors"
        if onnx_path.exists() and not safetensors_path.exists():
            try:
                from .convert import convert_campplus_onnx_to_safetensors

                convert_campplus_onnx_to_safetensors(
                    onnx_path=onnx_path,
                    output_path=safetensors_path,
                    verify_allclose=True,
                )
            except Exception as e:
                print(f"[WARN] campplus ONNX conversion failed: {e}")
                raise e

        model.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        return model

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

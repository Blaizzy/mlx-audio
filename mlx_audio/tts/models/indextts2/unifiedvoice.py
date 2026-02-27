from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler

from mlx_audio.tts.models.indextts.gpt2 import GPT2Model
from mlx_lm.models.gpt2 import ModelArgs as GPT2Args

from .unifiedvoice_conformer import ConformerEncoder, ConformerEncoderConfig


def _pad_left(x: mx.array, pad: int, value: float = 0.0) -> mx.array:
    if pad <= 0:
        return x
    return mx.pad(x, ((0, 0), (pad, 0)), mode="constant", constant_values=value)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, model_dim: int):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)

    def __call__(self, x: mx.array) -> mx.array:
        sl = x.shape[1]
        return self.emb(mx.arange(sl))

    def get_fixed_embedding(self, ind: int) -> mx.array:
        return self.emb(mx.array([ind], dtype=mx.int32))[None, :, :]


class GEGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(x, 2, axis=-1)
        return nn.gelu(gate) * x


class RMSNormGamma(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        x_f = x.astype(mx.float32)
        denom = mx.rsqrt(mx.mean(x_f * x_f, axis=-1, keepdims=True) + self.eps)
        return (x_f * denom * self.gamma).astype(x.dtype)


class PerceiverAttend(nn.Module):
    def __init__(self, *, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.dropout = dropout
        self.causal = causal

    def __call__(self, q: mx.array, k: mx.array, v: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # q,k,v: (B, H, L, D)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        return out


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_context: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 8,
        cross_attn_include_queries: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        dim_context = dim if dim_context is None else dim_context

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.attend = PerceiverAttend()

    def __call__(self, x: mx.array, context: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # x: (B, N, D), context: (B, M, Dctx), mask: (B, M)
        if self.cross_attn_include_queries:
            context = mx.concatenate([x, context], axis=-2)
            if mask is not None:
                qmask = mx.ones((mask.shape[0], x.shape[1]), dtype=mask.dtype)
                mask = mx.concatenate([qmask, mask], axis=-1)

        q = self.to_q(x)
        k, v = mx.split(self.to_kv(context), 2, axis=-1)

        B, N, _ = q.shape
        H = self.heads

        q = q.reshape(B, N, H, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, context.shape[1], H, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, context.shape[1], H, self.dim_head).transpose(0, 2, 1, 3)

        attn_mask = None
        if mask is not None:
            attn_mask = (1.0 - mask.astype(mx.float32))[:, None, None, :] * (-1e9)

        out = self.attend(q, k, v, mask=attn_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, H * self.dim_head)
        return self.to_out(out)


def _perceiver_ff(dim: int, mult: int = 4) -> nn.Module:
    dim_inner = int(dim * mult * 2 / 3)
    return [nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Linear(dim_inner, dim)]


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int = 2,
        dim_context: Optional[int] = None,
        num_latents: int = 32,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        dim_context = dim if dim_context is None else dim_context

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        self.latents = mx.random.normal((num_latents, dim)).astype(mx.float32) * 0.02

        self.layers = []
        for _ in range(depth):
            self.layers.append(
                [
                    PerceiverAttention(
                        dim=dim,
                        dim_context=dim,
                        dim_head=dim_head,
                        heads=heads,
                        cross_attn_include_queries=True,
                    ),
                    _perceiver_ff(dim, mult=ff_mult),
                ]
            )
        self.norm = RMSNormGamma(dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B = x.shape[0]
        x = self.proj_context(x)
        latents = mx.broadcast_to(self.latents, (B, *self.latents.shape))
        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            h = latents
            for layer in ff:
                h = layer(h)
            latents = h + latents
        return self.norm(latents)


@dataclass
class UnifiedVoiceConfig:
    model_dim: int
    heads: int
    layers: int
    max_mel_tokens: int
    max_text_tokens: int
    number_text_tokens: int
    number_mel_codes: int
    start_mel_token: int
    stop_mel_token: int
    start_text_token: int
    stop_text_token: int
    condition_type: str
    condition_module: Dict[str, Any]
    emo_condition_module: Dict[str, Any]
    condition_num_latent: int = 32
    max_conditioning_inputs: int = 1
    mel_length_compression: int = 1024

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedVoiceConfig":
        return cls(
            model_dim=int(d["model_dim"]),
            heads=int(d["heads"]),
            layers=int(d["layers"]),
            max_mel_tokens=int(d["max_mel_tokens"]),
            max_text_tokens=int(d["max_text_tokens"]),
            number_text_tokens=int(d["number_text_tokens"]),
            number_mel_codes=int(d["number_mel_codes"]),
            start_mel_token=int(d["start_mel_token"]),
            stop_mel_token=int(d["stop_mel_token"]),
            start_text_token=int(d["start_text_token"]),
            stop_text_token=int(d["stop_text_token"]),
            condition_type=str(d["condition_type"]),
            condition_module=dict(d.get("condition_module", {})),
            emo_condition_module=dict(d.get("emo_condition_module", {})),
            condition_num_latent=int(d.get("condition_num_latent", 32)),
            max_conditioning_inputs=int(d.get("max_conditioning_inputs", 1)),
            mel_length_compression=int(d.get("mel_length_compression", 1024)),
        )


class UnifiedVoice(nn.Module):
    def __init__(self, cfg: UnifiedVoiceConfig, *, bpe_model: str):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = spm.SentencePieceProcessor(model_file=bpe_model)

        self.text_embedding = nn.Embedding(cfg.number_text_tokens + 1, cfg.model_dim)
        self.mel_embedding = nn.Embedding(cfg.number_mel_codes, cfg.model_dim)
        self.mel_pos_embedding = LearnedPositionEmbeddings(
            cfg.max_mel_tokens + 2 + cfg.max_conditioning_inputs, cfg.model_dim
        )
        self.text_pos_embedding = LearnedPositionEmbeddings(cfg.max_text_tokens + 2, cfg.model_dim)

        self.text_head = nn.Linear(cfg.model_dim, cfg.number_text_tokens + 1)
        self.mel_head = nn.Linear(cfg.model_dim, cfg.number_mel_codes)

        # Conditioning encoders (port of index-tts ConformerEncoder)
        self.conditioning_encoder = ConformerEncoder(
            ConformerEncoderConfig(
                input_size=1024,
                output_size=int(cfg.condition_module["output_size"]),
                linear_units=int(cfg.condition_module["linear_units"]),
                attention_heads=int(cfg.condition_module["attention_heads"]),
                num_blocks=int(cfg.condition_module["num_blocks"]),
                input_layer=str(cfg.condition_module["input_layer"]),
            )
        )
        self.perceiver_encoder = PerceiverResampler(
            cfg.model_dim,
            dim_context=int(cfg.condition_module["output_size"]),
            heads=int(cfg.condition_module["attention_heads"]),
            ff_mult=int(cfg.condition_module.get("perceiver_mult", 2)),
            num_latents=cfg.condition_num_latent,
        )

        self.emo_conditioning_encoder = ConformerEncoder(
            ConformerEncoderConfig(
                input_size=1024,
                output_size=int(cfg.emo_condition_module["output_size"]),
                linear_units=int(cfg.emo_condition_module["linear_units"]),
                attention_heads=int(cfg.emo_condition_module["attention_heads"]),
                num_blocks=int(cfg.emo_condition_module["num_blocks"]),
                input_layer=str(cfg.emo_condition_module["input_layer"]),
            )
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=int(cfg.emo_condition_module["output_size"]),
            heads=int(cfg.emo_condition_module["attention_heads"]),
            ff_mult=int(cfg.emo_condition_module.get("perceiver_mult", 2)),
            num_latents=1,
        )

        self.emo_layer = nn.Linear(cfg.model_dim, cfg.model_dim)
        self.emovec_layer = nn.Linear(1024, cfg.model_dim)

        self.speed_emb = nn.Embedding(2, cfg.model_dim)
        self.final_norm = nn.LayerNorm(cfg.model_dim)

        self.gpt = GPT2Model(
            GPT2Args(
                "gpt2",
                1,
                cfg.model_dim,
                cfg.heads,
                cfg.layers,
                1,
                1e-5,
                1,
            )
        )
        # Patch GPT2 token/pos embeddings; we feed embeddings directly.
        self.gpt.wte = nn.Identity()  # type: ignore
        self.gpt.wpe = nn.Identity()  # type: ignore

    @staticmethod
    def _pretokenize_text(text: str) -> str:
        # Match official IndexTTS tokenizer pre-tokenizer behavior:
        # split around CJK chars and uppercase non-CJK spans.
        cjk_range = (
            r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
        )
        parts = re.split(cjk_range, text.strip())
        return " ".join(p.strip().upper() for p in parts if p.strip())

    def encode_text(self, text: str) -> mx.array:
        text = self._pretokenize_text(text)
        ids = self.tokenizer.encode(text)
        return mx.array(ids, dtype=mx.int32)[None, :]

    def get_conditioning(self, x: mx.array, x_lens: mx.array) -> mx.array:
        hs, mask = self.conditioning_encoder(x, x_lens)
        # Build perceiver mask: allow attending to context tokens only
        return self.perceiver_encoder(hs, mask=mask[:, 0, :])

    def get_emo_conditioning(self, x: mx.array, x_lens: mx.array) -> mx.array:
        hs, mask = self.emo_conditioning_encoder(x, x_lens)
        lat = self.emo_perceiver_encoder(hs, mask=mask[:, 0, :])
        return lat[:, 0, :]

    def get_emovec(self, emo_cond: mx.array, emo_lens: mx.array) -> mx.array:
        v = self.get_emo_conditioning(emo_cond, emo_lens)
        v = self.emovec_layer(v)
        return self.emo_layer(v)

    def merge_emovec(
        self,
        spk_cond: mx.array,
        emo_cond: mx.array,
        spk_lens: mx.array,
        emo_lens: mx.array,
        *,
        alpha: float = 1.0,
    ) -> mx.array:
        emo_vec = self.get_emovec(emo_cond, emo_lens)
        base_vec = self.get_emovec(spk_cond, spk_lens)
        return base_vec + float(alpha) * (emo_vec - base_vec)

    def forward_latent(
        self,
        speech_conditioning_latent: mx.array,
        text_tokens: mx.array,
        codes: mx.array,
        emo_vec: mx.array,
    ) -> mx.array:
        # Build embeddings for forward pass similar to official.
        # speech_conditioning_latent: (B, 32, D)
        B = text_tokens.shape[0]
        use_speed = mx.zeros((B,), dtype=mx.int32)

        duration_emb = self.speed_emb(use_speed)
        duration_emb_half = self.speed_emb(mx.ones_like(use_speed))
        conds = mx.concatenate(
            [speech_conditioning_latent + emo_vec[:, None, :], duration_emb_half[:, None, :], duration_emb[:, None, :]],
            axis=1,
        )

        # Text: add stop token
        text_tokens = mx.concatenate(
            [text_tokens, mx.full((B, 1), self.cfg.stop_text_token, dtype=mx.int32)],
            axis=1,
        )
        text_inp = mx.concatenate(
            [mx.full((B, 1), self.cfg.start_text_token, dtype=mx.int32), text_tokens],
            axis=1,
        )
        text_emb = self.text_embedding(text_inp) + self.text_pos_embedding(text_inp)

        # Codes: add stop token
        codes = mx.concatenate(
            [codes, mx.full((B, 1), self.cfg.stop_mel_token, dtype=mx.int32)],
            axis=1,
        )
        mel_inp = mx.concatenate(
            [mx.full((B, 1), self.cfg.start_mel_token, dtype=mx.int32), codes],
            axis=1,
        )
        mel_emb = self.mel_embedding(mel_inp) + self.mel_pos_embedding(mel_inp)

        emb = mx.concatenate([conds, text_emb, mel_emb], axis=1)
        mask = create_attention_mask(emb, cache=None)
        hs = self.gpt(emb, mask=mask, cache=None)
        hs = self.final_norm(hs[:, conds.shape[1] :, :])

        # Return mel latent portion (strip the two extra tokens)
        mel_lat = hs[:, text_emb.shape[1] : text_emb.shape[1] + mel_emb.shape[1], :]
        return mel_lat[:, :-2, :]

    def inference_speech(
        self,
        speech_condition: mx.array,
        text_tokens: mx.array,
        emo_condition: Optional[mx.array] = None,
        *,
        alpha: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        max_generate_length: int = 1500,
        repetition_penalty: float = 10.0,
    ) -> Tuple[mx.array, mx.array]:
        # speech_condition: (B, T, 1024)
        if emo_condition is None:
            emo_condition = speech_condition

        B = speech_condition.shape[0]
        spk_lens = mx.array([speech_condition.shape[1]] * B, dtype=mx.int32)
        emo_lens = mx.array([emo_condition.shape[1]] * B, dtype=mx.int32)

        speech_conditioning_latent = self.get_conditioning(speech_condition, spk_lens)
        emo_vec = self.merge_emovec(speech_condition, emo_condition, spk_lens, emo_lens, alpha=alpha)

        use_speed = mx.zeros((B,), dtype=mx.int32)
        duration_emb = self.speed_emb(use_speed)
        duration_emb_half = self.speed_emb(mx.ones_like(use_speed))
        conds = mx.concatenate(
            [speech_conditioning_latent + emo_vec[:, None, :], duration_emb_half[:, None, :], duration_emb[:, None, :]],
            axis=1,
        )

        # Build text emb with start/stop
        text_tokens = mx.concatenate(
            [mx.full((B, 1), self.cfg.start_text_token, dtype=mx.int32), text_tokens],
            axis=1,
        )
        text_tokens = mx.concatenate(
            [text_tokens, mx.full((B, 1), self.cfg.stop_text_token, dtype=mx.int32)],
            axis=1,
        )
        text_emb = self.text_embedding(text_tokens) + self.text_pos_embedding(text_tokens)

        # Start mel
        cur = mx.full((B, 1), self.cfg.start_mel_token, dtype=mx.int32)
        mel_emb0 = self.mel_embedding(cur) + self.mel_pos_embedding(cur)

        # First forward with full prefix
        emb0 = mx.concatenate([conds, text_emb, mel_emb0], axis=1)
        mask0 = create_attention_mask(emb0, cache=None)
        cache = [KVCache() for _ in range(self.cfg.layers)]
        hs = self.gpt(emb0, mask=mask0, cache=cache)
        hs_last = self.final_norm(hs[:, -1:, :])
        logits = self.mel_head(hs_last)[:, 0, :]

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

        def apply_repetition_penalty(logits_: mx.array, generated: list[mx.array]) -> mx.array:
            if repetition_penalty is None or repetition_penalty <= 1.0 or len(generated) == 0:
                return logits_
            # HF-style repetition penalty for batch size 1 (IndexTTS2 inference path).
            # For this model we run B=1 in practice.
            row = logits_[0].tolist()
            seen = {int(t.item()) for t in generated}
            p = float(repetition_penalty)
            for tok in seen:
                v = float(row[tok])
                row[tok] = v * p if v < 0 else v / p
            return mx.array([row], dtype=logits_.dtype)

        tokens = []
        mel_pos = 1
        for step in range(max_generate_length):
            logits_step = apply_repetition_penalty(logits, tokens)
            next_tok = sampler(logits_step)
            tokens.append(next_tok)
            if int(next_tok.item()) == self.cfg.stop_mel_token:
                break

            cur = next_tok.reshape(B, 1).astype(mx.int32)
            pos = self.mel_pos_embedding.get_fixed_embedding(mel_pos)
            if pos.shape[0] != B:
                pos = mx.broadcast_to(pos, (B, pos.shape[1], pos.shape[2]))
            emb = self.mel_embedding(cur) + pos
            mask = create_attention_mask(emb, cache=cache)
            hs = self.gpt(emb, mask=mask, cache=cache)
            hs_last = self.final_norm(hs[:, -1:, :])
            logits = self.mel_head(hs_last)[:, 0, :]
            mel_pos += 1

        codes = mx.stack(tokens, axis=1)
        return codes, speech_conditioning_latent

"""Cache-aware FastConformer encoder for Nemotron 3.5 ASR (offline path).

Differences from Parakeet's Conformer (all required for parity):
  - causal dw_striding subsampling (CausalConv2D: pad (k-1, s-1) both dims)
  - causal depthwise conv + LayerNorm (not symmetric + BatchNorm)
  - no bias on FFN / attention / conv1d
  - chunked_limited attention mask (block-causal, block size att_ctx[1]+1)
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.nemotron_asr.attention import (
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)


@dataclass
class EncoderArgs:
    feat_in: int = 128
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion_factor: int = 4
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    conv_kernel_size: int = 9
    pos_emb_max_len: int = 5000
    att_context_size: tuple = (56, 3)


def _silu(x):
    return x * mx.sigmoid(x)


class CausalDwStridingSubsampling(nn.Module):
    """8x dw_striding subsampling with causal padding (matches NeMo CausalConv2D)."""

    def __init__(self, args: EncoderArgs):
        super().__init__()
        c = args.subsampling_conv_channels
        self.k = 3
        self.stride = 2
        self.left = self.k - 1  # 2
        self.right = self.stride - 1  # 1
        self.n_stages = 3  # 8x = 2^3
        # indices 0,2,5 are stride-2 (causal); 3,6 pointwise k1; 1,4,7 ReLU
        self.conv = [
            nn.Conv2d(1, c, 3, stride=2, padding=0, bias=True),  # 0
            nn.ReLU(),  # 1
            nn.Conv2d(c, c, 3, stride=2, padding=0, groups=c, bias=True),  # 2 dw
            nn.Conv2d(c, c, 1, stride=1, padding=0, bias=True),  # 3 pw
            nn.ReLU(),  # 4
            nn.Conv2d(c, c, 3, stride=2, padding=0, groups=c, bias=True),  # 5 dw
            nn.Conv2d(c, c, 1, stride=1, padding=0, bias=True),  # 6 pw
            nn.ReLU(),  # 7
        ]
        self._stride_idx = {0, 2, 5}
        # final freq dim after 3 causal stride-2 stages on feat_in
        f = args.feat_in
        for _ in range(self.n_stages):
            f = (f + self.left + self.right - self.k) // self.stride + 1
        self.out = nn.Linear(c * f, args.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, F) -> NHWC (B, T, F, 1)
        x = mx.expand_dims(x, axis=-1)
        for i, layer in enumerate(self.conv):
            if i in self._stride_idx:
                # causal pad on time (H) and freq (W): (left, right) each
                x = mx.pad(
                    x,
                    [(0, 0), (self.left, self.right), (self.left, self.right), (0, 0)],
                )
            x = layer(x)
        # x: (B, T', F', C) -> flatten C-major: (B, T', C, F') -> (B, T', C*F')
        B, T, F, C = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(B, T, C * F)
        return self.out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        return self.linear2(_silu(self.linear1(x)))


class Convolution(nn.Module):
    """Causal depthwise conv module with LayerNorm (conv_norm_type=layer_norm)."""

    def __init__(self, args: EncoderArgs):
        super().__init__()
        d = args.d_model
        k = args.conv_kernel_size
        self.left = k - 1  # causal: pad left k-1, right 0
        self.pointwise_conv1 = nn.Conv1d(d, d * 2, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(d, d, k, padding=0, groups=d, bias=False)
        self.batch_norm = nn.LayerNorm(d)  # legacy attr name; actually LayerNorm
        self.pointwise_conv2 = nn.Conv1d(d, d, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        x = self.pointwise_conv1(x)  # (B, T, 2C)
        x = nn.glu(x, axis=-1)  # (B, T, C)
        x = mx.pad(x, [(0, 0), (self.left, 0), (0, 0)])  # causal time pad
        x = self.depthwise_conv(x)  # (B, T, C)
        x = self.batch_norm(x)  # LayerNorm over C
        x = _silu(x)
        return self.pointwise_conv2(x)


class ConformerBlock(nn.Module):
    def __init__(self, args: EncoderArgs):
        super().__init__()
        d = args.d_model
        ff = d * args.ff_expansion_factor
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = FeedForward(d, ff)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = RelPositionMultiHeadAttention(args.n_heads, d)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = Convolution(args)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = FeedForward(d, ff)
        self.norm_out = nn.LayerNorm(d)

    def __call__(self, x, pos_emb, mask=None):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb=pos_emb, mask=mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


def chunked_limited_mask(T: int, att_context_size, dtype) -> mx.array:
    """Block-causal additive mask (0 / -inf), shape (1,1,T,T).

    chunk_size = right+1; frame i (chunk ci=i//cs) attends to j (cj) iff
    cj <= ci and ci - cj <= left//cs.
    """
    left, right = int(att_context_size[0]), int(att_context_size[1])
    cs = right + 1
    left_chunks = (left // cs) if left >= 0 else 10**9
    idx = mx.arange(T)
    ci = (idx // cs).reshape(T, 1)
    cj = (idx // cs).reshape(1, T)
    diff = ci - cj
    allowed = (diff >= 0) & (diff <= left_chunks)
    neg = mx.array(-1e9, dtype=dtype)
    mask = mx.where(allowed, mx.array(0.0, dtype=dtype), neg)
    return mask.reshape(1, 1, T, T)


class Conformer(nn.Module):
    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.args = args
        self.pre_encode = CausalDwStridingSubsampling(args)
        self.pos_enc = RelPositionalEncoding(args.d_model, args.pos_emb_max_len)
        self.layers = [ConformerBlock(args) for _ in range(args.n_layers)]

    def __call__(self, mel: mx.array) -> mx.array:
        # mel: (B, T, F) -> (B, T', d_model)
        x = self.pre_encode(mel)
        pos_emb = self.pos_enc(x)
        mask = chunked_limited_mask(x.shape[1], self.args.att_context_size, x.dtype)
        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb, mask=mask)
        return x

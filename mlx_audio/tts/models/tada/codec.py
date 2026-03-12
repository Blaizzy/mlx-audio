import math
from typing import List, Literal, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# ============================================================================
# Shared components
# ============================================================================


class Snake1d(nn.Module):
    """Snake activation: x + sin^2(alpha * x) / alpha."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        alpha = self.alpha
        return x + (1.0 / alpha) * mx.power(mx.sin(alpha * x), 2)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = [
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        y = x
        for layer in self.block:
            y = layer(y)
        # Handle potential size mismatch from dilation
        if x.shape[1] != y.shape[1]:
            pad = (x.shape[1] - y.shape[1]) // 2
            if pad > 0:
                x = x[:, pad:-pad, :]
        return x + y


# ============================================================================
# Encoder components
# ============================================================================


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = [
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class WavEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = None,
        d_latent: int = 1024,
    ):
        super().__init__()
        if strides is None:
            strides = [6, 5, 4, 4]

        self.block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block.append(EncoderBlock(d_model, stride=stride))
        self.block.append(Snake1d(d_model))
        self.block.append(nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, 1) in MLX channels-last
        for layer in self.block:
            x = layer(x)
        return x


# ============================================================================
# Decoder components
# ============================================================================


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        pad = math.ceil(stride / 2)
        out_pad = stride + 2 * pad - 2 * stride
        self.block = [
            Snake1d(dim),
            nn.ConvTranspose1d(
                dim,
                dim // 2,
                kernel_size=2 * stride,
                stride=stride,
                padding=pad,
                output_padding=out_pad,
            ),
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class DACDecoder(nn.Module):
    def __init__(self, d_model: int = 96, strides: list = None, d_latent: int = 1024):
        super().__init__()
        if strides is None:
            strides = [4, 4, 5, 6]

        mult = 2 ** len(strides)
        self.model = [nn.Conv1d(d_latent, d_model * mult, kernel_size=7, padding=3)]
        for i, stride in enumerate(strides):
            self.model.append(DecoderBlock(d_model * mult, stride=stride))
            mult //= 2
        self.model.append(Snake1d(d_model))
        self.model.append(nn.Conv1d(d_model, 1, kernel_size=7, padding=3))
        # Tanh is applied in forward (no weights)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D) in MLX channels-last
        for layer in self.model:
            x = layer(x)
        x = mx.tanh(x)
        return x


# ============================================================================
# Local attention for codec encoder/decoder
# ============================================================================


class LocalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # RoPE config (computed on-the-fly to avoid parameter registration)
        self._rope_inv_freq = 1.0 / (
            10000.0
            ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        )
        self.freeze(keys=["_rope_inv_freq"])

    def _apply_rope(self, x: mx.array, seq_len: int) -> mx.array:
        B, H, L, D = x.shape
        positions = mx.arange(L, dtype=mx.float32)
        freqs = mx.outer(positions, self._rope_inv_freq)
        cos = mx.cos(freqs)[:L]  # (L, D//2)
        sin = mx.sin(freqs)[:L]  # (L, D//2)

        x_reshaped = x.reshape(B, H, L, D // 2, 2)
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]

        cos_b = cos[None, None, :, :]  # (1, 1, L, D//2)
        sin_b = sin[None, None, :, :]

        x_rotated_0 = x0 * cos_b - x1 * sin_b
        x_rotated_1 = x0 * sin_b + x1 * cos_b

        x_rotated = mx.stack([x_rotated_0, x_rotated_1], axis=-1)
        return x_rotated.reshape(B, H, L, D)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self._apply_rope(q, L)
        k = self._apply_rope(k, L)

        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            if mask.ndim == 2:
                attn_mask = mx.where(
                    mask[None, None, :, :],
                    mx.array(float("-inf")),
                    mx.array(0.0),
                )
            elif mask.ndim == 3:
                attn_mask = mx.where(
                    mask[:, None, :, :],
                    mx.array(float("-inf")),
                    mx.array(0.0),
                )
            else:
                attn_mask = mx.array(0.0)
            attn_scores = attn_scores + attn_mask

        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, D)

        output = self.out_proj(attn_output)
        output = self.layer_norm(x + output)
        return output


class LocalAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.self_attn = LocalSelfAttention(d_model, num_heads, max_seq_len)
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.self_attn(x, mask=mask)
        x = self.norm(x + self.ffn_out(nn.gelu(self.ffn_in(x))))
        return x


class LocalAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 8192,
        d_input: int = None,
    ):
        super().__init__()
        self.layers = [
            LocalAttentionEncoderLayer(d_model, num_heads, d_ff, max_seq_len)
            for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(d_model)

        if d_input is not None and d_input != d_model:
            self.input_proj = nn.Linear(d_input, d_model)
        else:
            self.input_proj = None

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        if self.input_proj is not None:
            x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)


# ============================================================================
# Segment attention mask
# ============================================================================


def create_segment_attention_mask(
    text_token_mask: mx.array, version: str = "v2"
) -> mx.array:
    """Create segment-based attention mask.

    Args:
        text_token_mask: (batch, seq_len) binary mask where 1 = text token boundary
        version: "v1" or "v2"

    Returns:
        mask: (batch, seq_len, seq_len) bool where True = masked (cannot attend)
    """
    if version == "v1":
        block_ids = mx.cumsum(text_token_mask, axis=1) - text_token_mask
        block_ids_i = mx.expand_dims(block_ids, 2)
        block_ids_j = mx.expand_dims(block_ids, 1)
        same_block = block_ids_j == block_ids_i
        block_ids_j_ex = mx.where(
            text_token_mask.astype(mx.bool_),
            mx.array(-10),
            block_ids_j[:, 0, :],
        )
        block_ids_j_ex = mx.expand_dims(block_ids_j_ex, 1)
        next_block = block_ids_j_ex == (block_ids_i + 1)
        can_attend = same_block | next_block
        return ~can_attend
    elif version == "v2":
        block_ids = mx.cumsum(text_token_mask, axis=1) - text_token_mask
        block_ids_i = mx.expand_dims(block_ids, 2)
        block_ids_j = mx.expand_dims(block_ids, 1)
        same_block = block_ids_j == block_ids_i
        prev_block = block_ids_j == (block_ids_i - 1)
        can_attend = same_block | prev_block
        return ~can_attend
    else:
        raise ValueError(f"Unknown version: {version}")


# ============================================================================
# Codec Decoder (full pipeline)
# ============================================================================


class CodecDecoder(nn.Module):
    """Codec decoder: local attention + DAC waveform decoder."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        embed_dim: int = 512,
        d_model: int = 96,
        strides: list = None,
        num_attn_layers: int = 6,
        num_attn_heads: int = 8,
        attn_dim_feedforward: int = 4096,
        block_attention: str = "v2",
    ):
        super().__init__()
        if strides is None:
            strides = [4, 4, 5, 6]

        self.decoder_proj = nn.Linear(embed_dim, hidden_dim)
        self.local_attention_decoder = LocalAttentionEncoder(
            d_model=hidden_dim,
            num_layers=num_attn_layers,
            num_heads=num_attn_heads,
            d_ff=attn_dim_feedforward,
        )
        self.wav_decoder = DACDecoder(
            d_model=d_model, strides=strides, d_latent=hidden_dim
        )
        self.block_attention = block_attention

    def generate(self, encoded_expanded: mx.array, token_masks: mx.array) -> mx.array:
        x = self.decoder_proj(encoded_expanded)

        attn_mask = create_segment_attention_mask(
            token_masks, version=self.block_attention
        )
        x = self.local_attention_decoder(x, mask=attn_mask)
        wav = self.wav_decoder(x)
        return wav


# ============================================================================
# Codec Encoder (full pipeline)
# ============================================================================


class CodecEncoder(nn.Module):
    """Codec encoder: waveform encoder + local attention + aligner."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        embed_dim: int = 512,
        strides: list = None,
        num_attn_layers: int = 6,
        num_attn_heads: int = 8,
        attn_dim_feedforward: int = 4096,
        block_attention: str = "v2",
        std: float = 0.5,
        acoustic_mean: float = 0.0,
        acoustic_std: float = 1.5,
    ):
        super().__init__()
        if strides is None:
            strides = [6, 5, 4, 4]

        self.wav_encoder = WavEncoder(d_model=64, strides=strides, d_latent=hidden_dim)
        self.local_attention_encoder = LocalAttentionEncoder(
            d_model=hidden_dim,
            num_layers=num_attn_layers,
            num_heads=num_attn_heads,
            d_ff=attn_dim_feedforward,
        )

        if hidden_dim != embed_dim:
            self.hidden_linear = nn.Linear(hidden_dim, embed_dim)
        else:
            self.hidden_linear = None

        self.pos_emb = nn.Embedding(2, hidden_dim)
        self.block_attention = block_attention
        self.std = std
        self.acoustic_mean = acoustic_mean
        self.acoustic_std = acoustic_std

    def get_encoder_outputs(
        self, audio: mx.array, token_masks: mx.array
    ) -> Tuple[mx.array, mx.array]:
        # audio: (B, T) raw waveform at 24kHz
        # Pad audio and reshape: (B, T) -> (B, T, 1)
        padded = mx.pad(audio[:, :, None], [(0, 0), (0, 960), (0, 0)])
        enc_out = self.wav_encoder(padded)  # (B, T', D)

        seq_len = enc_out.shape[1]
        if token_masks.shape[1] < seq_len:
            pad_len = seq_len - token_masks.shape[1]
            token_masks = mx.pad(token_masks, [(0, 0), (0, pad_len)])
        elif token_masks.shape[1] > seq_len:
            token_masks = token_masks[:, :seq_len]

        enc_out = enc_out + self.pos_emb(token_masks.astype(mx.int32))

        attn_mask = create_segment_attention_mask(
            token_masks, version=self.block_attention
        )
        enc_out = self.local_attention_encoder(enc_out, mask=attn_mask)

        if self.hidden_linear is not None:
            enc_out = self.hidden_linear(enc_out)

        return enc_out, token_masks

    def forward(
        self,
        audio: mx.array,
        token_positions: mx.array,
        token_masks: mx.array,
        sample: bool = True,
    ) -> mx.array:
        """Encode audio with pre-computed alignment.

        Returns:
            token_values: (B, num_tokens, embed_dim) acoustic features per text token
        """
        enc_out, token_masks = self.get_encoder_outputs(audio, token_masks)

        # Gather token values at aligned positions
        encoded_expanded = mx.where(
            mx.expand_dims(token_masks, -1) == 0,
            mx.zeros_like(enc_out),
            enc_out,
        )

        if self.std > 0.0 and sample:
            noise = mx.random.normal(encoded_expanded.shape) * self.std
            encoded_expanded = mx.where(
                mx.expand_dims(token_masks, -1) == 0,
                encoded_expanded,
                encoded_expanded + noise,
            )

        # Gather at token positions
        positions = mx.clip(token_positions - 1, 0, encoded_expanded.shape[1] - 1)
        B = encoded_expanded.shape[0]
        token_values = mx.zeros((B, positions.shape[1], encoded_expanded.shape[2]))
        for b in range(B):
            for i in range(positions.shape[1]):
                pos = int(positions[b, i].item())
                token_values[b, i] = encoded_expanded[b, pos]

        token_values = (token_values - self.acoustic_mean) / self.acoustic_std
        return token_values

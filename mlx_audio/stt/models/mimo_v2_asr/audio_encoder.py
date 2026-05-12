"""
Audio Encoder for MiMo-Audio-Tokenizer (MLX port).

Ported from ``src/mimo_audio_tokenizer/modeling_audio_tokenizer.py:AudioEncoder``.

Architecture
------------
Mel (128, T) → Conv1d(128→1280, k=3, pad=1) → GELU
             → Conv1d(1280→1280, k=3, stride=2, pad=1) → GELU   [2× down-sampling]
             → 32 × TransformerLayer (d=1280, 20 heads, FFN=5120, RoPE)
             → skip connection from layer 3
             → LayerNorm
             → AvgPool (2×, optional) → LayerNorm
             → RVQ encode → codes [n_q=20, T_tokens]

Key design decisions:
  - Custom ``Attention`` class with HF-compatible key names (q_proj, k_proj, v_proj)
    so weight loading requires no key remapping for attention weights.
  - RoPE is applied to Q and K inside the Attention module, matching the reference.
  - k_proj has bias=False (matching HF); a zero bias is injected during weight loading.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .quantization import ResidualVectorQuantizer


# ── Rotary Position Embedding helpers ───────────────────────────────

def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary position embedding (HF-compatible)."""
    return (x * cos) + (rotate_half(x) * sin)


# ── Config ──────────────────────────────────────────────────────────

@dataclass
class AudioEncoderConfig:
    """Configuration matching MiMoAudioTokenizerConfig encoder fields."""

    n_mels: int = 128
    d_model: int = 1280
    kernel_size: int = 3
    stride_size: int = 2  # conv2 stride
    scale_embedding: bool = False
    activation_function: str = "gelu"
    encoder_layers: int = 32
    encoder_skip_layer_id: Optional[int] = 3  # 1-indexed; None → no skip
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_causal: bool = False
    avg_pooler: int = 2
    num_quantizers: int = 20
    codebook_size: Union[int, List[int]] = 1024
    threshold_ema_dead_code: int = 2
    rope_theta: float = 10000.0
    rope_type: str = "default"
    ln_type: str = "LayerNorm"
    sampling_rate: int = 24000
    hop_length: int = 240
    max_audio_seconds: int = 1800

    @property
    def max_source_positions(self) -> int:
        return self.max_audio_seconds * self.sampling_rate // self.hop_length // self.stride_size

    @classmethod
    def from_dict(cls, d: dict) -> "AudioEncoderConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ── Activation helper ───────────────────────────────────────────────

def _act_fn(name: str):
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


# ── Layer norm factory ──────────────────────────────────────────────

def _layer_norm(ln_type: str, dim: int, eps: float = 1e-5):
    if ln_type == "LayerNorm":
        return nn.LayerNorm(dim, eps=eps)
    if ln_type == "RMSNorm":
        return nn.RMSNorm(dim, eps=eps)
    raise ValueError(f"Unknown layer norm type: {ln_type}")


# ── Rotary Embedding ────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    RoPE with HF-compatible parameter names.

    Computes cos/sin tables for a given sequence length.
    Uses the "default" rope_type (theta-based, no scaling).
    """

    def __init__(self, base: float, dim: int, max_seq_len: int):
        super().__init__()
        self.base = base
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies: 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, x: mx.array, position_ids: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """
        Compute cos/sin for the given input.

        Parameters
        ----------
        x : mx.array, shape (seq_len, d_model) — used only for dtype inference
        position_ids : mx.array, optional — if None, uses arange(seq_len)

        Returns
        -------
        cos, sin : each shape (seq_len, dim)
        """
        seq_len = x.shape[0]
        if position_ids is None:
            position_ids = mx.arange(seq_len, dtype=mx.float32)

        # (dim/2,) @ (seq_len,) → (seq_len, dim/2)
        freqs = mx.matmul(
            self.inv_freq[:, None],  # (dim/2, 1)
            position_ids[None, :],   # (1, seq_len)
        ).T  # (seq_len, dim/2)

        # Duplicate for full rotary dim: cat(freqs, freqs)
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (seq_len, dim)

        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)
        return cos, sin


# ── Attention (HF-compatible key names) ─────────────────────────────

class Attention(nn.Module):
    """
    Multi-head attention with HF-compatible parameter names.

    Matches the reference ``Attention`` module in:
      src/mimo_audio_tokenizer/modeling_audio_tokenizer.py

    Key names: q_proj, k_proj, v_proj, out_proj
    k_proj has bias=False (matching HF).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        # Match HF key names exactly
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        rope_cos: Optional[mx.array] = None,
        rope_sin: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Parameters
        ----------
        hidden_states : mx.array, shape (seq_len, embed_dim)
        rope_cos, rope_sin : mx.array, shape (seq_len, head_dim) — RoPE embeddings
        mask : mx.array, optional — attention mask

        Returns
        -------
        mx.array, shape (seq_len, embed_dim)
        """
        seq_len = hidden_states.shape[0]

        query = self.q_proj(hidden_states)  # (seq_len, embed_dim)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (1, seq_len, n_heads, head_dim)
        query = query.reshape(1, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(1, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(1, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            # Add head dim for broadcasting: (seq_len, 1, head_dim)
            cos = rope_cos[:, None, :]
            sin = rope_sin[:, None, :]
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        # Transpose to (1, n_heads, seq_len, head_dim) for SDPA
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value,
            scale=self.scale,
            mask=mask,
        )

        # (1, n_heads, seq_len, head_dim) → (1, seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(1, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output[0]  # remove batch dim


# ── Transformer Layer ───────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """Pre-norm Transformer layer with RoPE and full (non-causal) attention."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, ln_type: str):
        super().__init__()
        self.self_attn = Attention(embed_dim=d_model, num_heads=n_heads, causal=False)
        self.self_attn_layer_norm = _layer_norm(ln_type, d_model)

        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = _layer_norm(ln_type, d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        rope_cos: Optional[mx.array] = None,
        rope_sin: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, rope_cos=rope_cos, rope_sin=rope_sin, mask=mask
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ── Audio Encoder ───────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """MiMo Audio Tokenizer encoder (Conv → Transformer → RVQ)."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # Position embedding
        head_dim = config.d_model // config.encoder_attention_heads
        max_seq_len = config.max_source_positions
        self.position_embedding = RotaryEmbedding(
            base=config.rope_theta,
            dim=head_dim,
            max_seq_len=max_seq_len,
        )

        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.skip_layer_idx = config.encoder_skip_layer_id

        # Convolutional front-end
        self.conv1 = nn.Conv1d(config.n_mels, config.d_model, kernel_size=config.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            config.d_model, config.d_model,
            kernel_size=config.kernel_size, stride=config.stride_size, padding=1,
        )

        # Transformer layers
        self.layers = [
            TransformerLayer(
                config.d_model,
                config.encoder_attention_heads,
                config.encoder_ffn_dim,
                config.ln_type,
            )
            for _ in range(config.encoder_layers)
        ]
        self.layer_norm = _layer_norm(config.ln_type, config.d_model)

        # Optional down-sampling via Conv1d (matching HF's Sequential[Conv1d, GELU])
        if config.avg_pooler > 1:
            self.down_sample = nn.Conv1d(
                config.d_model, config.d_model,
                kernel_size=config.avg_pooler, stride=config.avg_pooler,
                bias=False,
            )
            self.down_sample_norm = _layer_norm(config.ln_type, config.d_model)
        else:
            self.down_sample = None

        # RVQ quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.d_model,
            n_q=config.num_quantizers,
            bins=config.codebook_size,
        )

    def get_output_length(self, mel_len: int) -> int:
        """
        Compute the number of encoder output frames from Mel frame count.

        Conv1(k=3, pad=1): length unchanged
        Conv2(k=3, stride=2, pad=1): ceil(L / 2)
        """
        return (mel_len + 1) // 2

    def _get_features(self, mel: mx.array) -> mx.array:
        """
        mel : mx.array, shape (n_mels, mel_len)  [single utterance]

        Returns
        -------
        mx.array, shape (T_out, d_model)

        NOTE: MLX Conv1d uses NLC format (batch, length, channels),
              NOT PyTorch's NCL format (batch, channels, length).
        """
        # Convert mel (n_mels, L) → NLC (1, L, n_mels)
        x = mel.T[None]  # (1, L, n_mels)
        x = nn.gelu(self.conv1(x))   # (1, L, d_model)
        x = nn.gelu(self.conv2(x))   # (1, L//2, d_model)

        # Flatten batch
        x = x * self.embed_scale
        x = x[0]                     # (T, d_model)

        # RoPE embeddings
        cos, sin = self.position_embedding(x)  # (T, head_dim)

        # Transformer layers with skip connection
        skip_hidden = None
        for i, layer in enumerate(self.layers):
            x = layer(x, rope_cos=cos, rope_sin=sin)
            if self.skip_layer_idx is not None and i == self.skip_layer_idx - 1:
                skip_hidden = x

        if skip_hidden is not None:
            x = x + skip_hidden

        x = self.layer_norm(x)  # (T, d_model)

        # Optional down-sampling
        if self.down_sample is not None:
            T = x.shape[0]
            pool = self.config.avg_pooler
            if T % pool != 0:
                pad_len = pool - (T % pool)
                x = mx.pad(x, ((0, pad_len), (0, 0)))

            # Conv1d NLC: (1, T, d_model)
            x = x[None]
            x = self.down_sample(x)
            # Apply GELU to match HF's Sequential[Conv1d, GELU]
            x = nn.gelu(x)
            x = x[0]
            x = self.down_sample_norm(x)

        return x

    @property
    def num_quantizers(self) -> int:
        return self.config.num_quantizers

    def encode(
        self,
        mel: mx.array,
        n_q: Optional[int] = None,
    ) -> mx.array:
        """
        Encode mel spectrogram to discrete codes.

        Parameters
        ----------
        mel : mx.array, shape (n_mels, mel_len)
        n_q : int, optional – number of quantizer layers (default: all)

        Returns
        -------
        mx.array, shape (n_q, T_tokens), dtype int32
        """
        features = self._get_features(mel)  # (T, d_model)
        codes = self.quantizer.encode(features, n_q=n_q)  # (n_q, T)
        return codes

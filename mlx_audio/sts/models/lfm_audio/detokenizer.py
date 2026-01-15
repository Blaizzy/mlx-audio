# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# LFM2.5-Audio Detokenizer: Converts audio codes to waveforms

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download


@dataclass
class DetokenizerConfig:
    """Configuration for LFM2 Audio Detokenizer."""
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: Tuple[str, ...] = (
        "conv", "conv", "sliding_attention", "conv",
        "sliding_attention", "conv", "sliding_attention", "conv"
    )
    sliding_window: int = 30
    intermediate_size: int = 2304  # Actually 2304 from weights
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    output_size: int = 1282
    num_codebooks: int = 8
    vocab_size: int = 2048
    n_fft: int = 1280
    hop_length: int = 320
    upsample_factor: int = 6

    @classmethod
    def from_dict(cls, d: Dict) -> "DetokenizerConfig":
        layer_types = d.get("layer_types", list(cls.layer_types))
        if isinstance(layer_types, list):
            layer_types = tuple(layer_types)
        return cls(
            hidden_size=d.get("hidden_size", d.get("block_dim", 512)),
            num_hidden_layers=d.get("num_hidden_layers", 8),
            num_attention_heads=d.get("num_attention_heads", d.get("num_heads", 16)),
            num_key_value_heads=d.get("num_key_value_heads", 8),
            layer_types=layer_types,
            sliding_window=d.get("sliding_window", 30),
            intermediate_size=d.get("intermediate_size", 2304),
            norm_eps=d.get("norm_eps", 1e-5),
            rope_theta=d.get("rope_theta", 1000000.0),
            output_size=d.get("output_size", 1282),
        )


class FusedEmbedding(nn.Module):
    """Fused embedding for multiple codebooks.

    Maps tokens from each codebook to offset positions in a single large embedding table.
    """
    def __init__(self, num_codebooks: int, vocab_size: int, dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.dim = dim
        # Single embedding table: (num_codebooks * vocab_size, dim)
        self.emb = nn.Embedding(num_codebooks * vocab_size, dim)

    def __call__(self, codes: mx.array) -> mx.array:
        """
        Args:
            codes: (B, num_codebooks, T) with values in [0, vocab_size)
        Returns:
            embeddings: (B, T, dim) - averaged over codebooks
        """
        B, K, T = codes.shape
        # Add offsets: codebook i gets offset i * vocab_size
        offsets = mx.arange(K)[None, :, None] * self.vocab_size  # (1, K, 1)
        offset_codes = codes + offsets  # (B, K, T)

        # Get embeddings for each codebook
        embeddings = self.emb(offset_codes)  # (B, K, T, dim)

        # Average over codebooks (PyTorch uses .mean(1))
        return embeddings.mean(axis=1)  # (B, T, dim)


class RMSNorm(nn.Module):
    """RMS normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class ConvLayer(nn.Module):
    """1D Convolution layer used in LFM detokenizer.

    LFM2 Short Conv structure:
    1. in_proj: project to B, C, x (3 * dim)
    2. B * x: input gating
    3. conv: depthwise conv on gated input (causal padding)
    4. C * conv_out: output gating
    5. out_proj: project back to dim
    """
    def __init__(self, dim: int):
        super().__init__()
        # in_proj splits into B, C, x (each dim)
        self.in_proj = nn.Linear(dim, dim * 3, bias=False)
        # Depthwise conv with kernel=3, padding=2 to match PyTorch's causal behavior
        # PyTorch pads 2 on each side, then truncates output to original length
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=2, groups=dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: (B, T, dim)
        Returns:
            (B, T, dim)
        """
        seqlen = x.shape[1]

        # Project and split into B, C, x
        BCx = self.in_proj(x)  # (B, T, 3*dim)
        B_gate, C_gate, x_proj = mx.split(BCx, 3, axis=-1)  # each (B, T, dim)

        # Input gating: B * x
        Bx = B_gate * x_proj  # (B, T, dim)

        # Depthwise conv with padding=2 on each side, then truncate
        conv_out = self.conv(Bx)[:, :seqlen, :]  # (B, T, dim)

        # Output gating: C * conv_out
        y = C_gate * conv_out  # (B, T, dim)

        # Final projection
        return self.out_proj(y)


class SlidingWindowAttention(nn.Module):
    """Sliding window self-attention with RoPE."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        sliding_window: int,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.sliding_window = sliding_window
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # QK LayerNorm (common in newer models)
        self.q_layernorm = RMSNorm(self.head_dim)
        self.k_layernorm = RMSNorm(self.head_dim)

        # Precompute RoPE frequencies
        self.rope_theta = rope_theta

    def _rope(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply rotary position embeddings (LLaMA-style split-half)."""
        B, H, T, D = x.shape

        # Compute frequencies for D/2 dimensions
        inv_freq = 1.0 / (self.rope_theta ** (mx.arange(0, D, 2) / D))
        positions = mx.arange(offset, offset + T)
        angles = positions[:, None] * inv_freq[None, :]  # (T, D/2)

        # Compute cos and sin, then repeat to get full D
        cos_half = mx.cos(angles)  # (T, D/2)
        sin_half = mx.sin(angles)  # (T, D/2)
        # PyTorch repeats: [cos0, cos1, ..., cos0, cos1, ...] for full D
        cos = mx.concatenate([cos_half, cos_half], axis=-1)  # (T, D)
        sin = mx.concatenate([sin_half, sin_half], axis=-1)  # (T, D)

        # Expand for batch and heads: (1, 1, T, D)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Split-half rotation (LLaMA style):
        # rotate_half(x) = [-x[D/2:], x[:D/2]]
        # out = x * cos + rotate_half(x) * sin
        x1 = x[..., :D // 2]  # First half
        x2 = x[..., D // 2:]  # Second half

        # Apply rotation
        rotated = mx.concatenate([
            x1 * cos[..., :D // 2] - x2 * sin[..., :D // 2],
            x2 * cos[..., D // 2:] + x1 * sin[..., D // 2:]
        ], axis=-1)

        return rotated

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply QK LayerNorm
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        # Apply RoPE
        q = self._rope(q)
        k = self._rope(k)

        # Expand KV heads if needed (GQA)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Compute attention with sliding window mask
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, T, T)

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v  # (B, H, T, D)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feedforward layer."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class DetokenizerBlock(nn.Module):
    """A block in the detokenizer (either conv or attention based)."""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        layer_type: str,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        sliding_window: int = 30,
        norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.layer_type = layer_type

        # Operator norm (before conv/attention)
        self.operator_norm = RMSNorm(dim, norm_eps)

        # Either conv or attention
        if layer_type == "conv":
            self.conv = ConvLayer(dim)
        else:  # sliding_attention
            self.self_attn = SlidingWindowAttention(
                dim, num_heads, num_kv_heads, sliding_window, rope_theta
            )

        # FFN
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.feed_forward = SwiGLU(dim, hidden_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Operator (conv or attention) with residual
        h = self.operator_norm(x)
        h = self.conv(h, mask) if self.layer_type == "conv" else self.self_attn(h, mask)

        x = x + h

        # FFN with residual
        h = self.ffn_norm(x)
        h = self.feed_forward(h)
        x = x + h

        return x


class LFMDetokenizerModel(nn.Module):
    """The LFM backbone for the detokenizer."""
    def __init__(self, config: DetokenizerConfig):
        super().__init__()
        self.config = config

        # This embedding is not used (codes go through FusedEmbedding)
        # but needed for weight loading compatibility
        self.embed_tokens = nn.Embedding(65536, config.hidden_size)
        # embedding_norm is applied AFTER all layers (final norm)
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)

        # Build layers based on layer_types
        self.layers = []
        for layer_type in config.layer_types:
            self.layers.append(DetokenizerBlock(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                layer_type=layer_type,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                sliding_window=config.sliding_window,
                norm_eps=config.norm_eps,
                rope_theta=config.rope_theta,
            ))

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Apply all layers first
        for layer in self.layers:
            x = layer(x, mask)

        # Apply final normalization (embedding_norm is post-norm in LFM2)
        x = self.embedding_norm(x)

        return x


class LFM2AudioDetokenizer(nn.Module):
    """
    Audio detokenizer that converts audio codes to waveforms.

    Architecture:
    1. Fused embedding: (B, 8, T) -> (B, T, 512)
    2. Upsample 6x: (B, T, 512) -> (B, 6T, 512)
    3. LFM backbone: 8 layers of conv/attention
    4. Linear projection: 512 -> 1282 (641 log-mag + 641 phase)
    5. ISTFT reconstruction
    """

    def __init__(self, config: DetokenizerConfig):
        super().__init__()
        self.config = config

        # Fused embedding for codebooks
        self.emb = FusedEmbedding(
            num_codebooks=config.num_codebooks,
            vocab_size=config.vocab_size,
            dim=config.hidden_size,
        )

        # LFM backbone
        self.lfm = LFMDetokenizerModel(config)

        # Output projection (lin in PyTorch)
        self.lin = nn.Linear(config.hidden_size, config.output_size, bias=True)

        # ISTFT window (will be loaded from weights)
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self._window = None

    @property
    def window(self) -> mx.array:
        """Get or create ISTFT window."""
        if self._window is None:
            # Create hann window
            n = self.n_fft
            self._window = 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(n) / n)
        return self._window

    def _create_sliding_window_mask(self, T: int) -> mx.array:

        idx = mx.arange(T)
        d_idx = idx[:, None] - idx[None, :]  # (T, T)


        valid = (d_idx >= 0) & (d_idx < self.config.sliding_window)
        mask = mx.where(valid, 0.0, float("-inf"))

        # Add batch and head dimensions
        return mask[None, None, :, :]  # (1, 1, T, T)

    def __call__(self, codes: mx.array) -> mx.array:
        """
        Convert audio codes to waveform.

        Args:
            codes: (B, num_codebooks, T) with values in [0, 2047]

        Returns:
            Waveform (B, T_audio) at 24kHz
        """
        B, K, T = codes.shape

        # 1. Embed codes
        x = self.emb(codes)  # (B, T, dim)

        # 2. Upsample 6x using nearest neighbor
        upsample_size = self.config.upsample_factor * T
        # Transpose: (B, T, D) -> (B, D, T)
        x = x.transpose(0, 2, 1)
        # Nearest neighbor upsample
        indices = mx.arange(upsample_size) // self.config.upsample_factor
        x = x[:, :, indices]
        # Transpose back: (B, D, T') -> (B, T', D)
        x = x.transpose(0, 2, 1)

        # 3. Create sliding window causal mask
        T_up = x.shape[1]
        mask = self._create_sliding_window_mask(T_up)

        # 4. Apply LFM backbone
        x = self.lfm(x, mask)

        # 5. Project to spectrogram
        x = self.lin(x)  # (B, T', 1282)

        # 6. Split into log-magnitude and phase
        n_bins = self.n_fft // 2 + 1  # 641
        log_mag = x[:, :, :n_bins]
        phase = x[:, :, n_bins:]

        # 7. Reconstruct magnitude
        mag = mx.exp(log_mag)

        # 8. ISTFT reconstruction
        waveform = self._istft(mag, phase)

        return waveform

    def _istft(self, mag: mx.array, phase: mx.array) -> mx.array:
        """Inverse STFT to reconstruct waveform.

        Uses the dsp.istft with 'same' padding mode to match PyTorch's LFM2 ISTFT.
        """
        from mlx_audio.dsp import istft

        B, T_frames, F = mag.shape

        # Create complex STFT: (B, T, F) -> need (F, T) for istft
        real = mag * mx.cos(phase)
        imag = mag * mx.sin(phase)
        stft_complex = real + 1j * imag

        # Padding for "same" mode (matches PyTorch LFM2 ISTFT)
        pad = (self.n_fft - self.hop_length) // 2

        # Process each batch item
        outputs = []
        for b in range(B):
            # Transpose from (T, F) to (F, T) as expected by istft
            stft_item = stft_complex[b].transpose(1, 0)  # (F, T)

    
            # normalized=True for COLA (window²) normalization to match PyTorch
            output = istft(
                stft_item,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=False,
                normalized=True,
            )

            # Trim for "same" padding: output_len = n_frames * hop_length
            if pad > 0:
                output = output[pad:-pad]

            outputs.append(output)

        return mx.stack(outputs, axis=0)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "LiquidAI/LFM2.5-Audio-1.5B",
    ) -> "LFM2AudioDetokenizer":
        """Load pretrained detokenizer."""
        # Download files
        if Path(model_name_or_path).exists():
            model_path = Path(model_name_or_path)
            config_path = model_path / "audio_detokenizer" / "config.json"
            weights_path = model_path / "audio_detokenizer" / "model.safetensors"
        else:
            config_path = hf_hub_download(
                model_name_or_path,
                "audio_detokenizer/config.json",
            )
            weights_path = hf_hub_download(
                model_name_or_path,
                "audio_detokenizer/model.safetensors",
            )

        # Load config
        with open(config_path) as f:
            config_dict = json.load(f)
        config = DetokenizerConfig.from_dict(config_dict)

        # Load weights first to infer dimensions
        weights = mx.load(weights_path)

        # Infer intermediate_size from actual weights (config may differ due to auto-adjust)
        ffn_key = "lfm.layers.0.feed_forward.w1.weight"
        if ffn_key in weights:
            config.intermediate_size = weights[ffn_key].shape[0]

        # Create model
        model = cls(config)

        # Extract ISTFT window before mapping
        istft_window = weights.pop("istft.window", None)

        # Map weight names
        mapped_weights = cls.sanitize(weights)

        quantization = config_dict.get("quantization", None)
        if quantization:
            from mlx_audio.convert import build_quant_predicate
            final_predicate = build_quant_predicate(model)
            nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"], mode=config_dict.get("quantization_mode", "affine"), class_predicate=final_predicate)

        # Load into model
        model.load_weights(list(mapped_weights.items()))
        model.eval()
        mx.eval(model.parameters())

        # Set ISTFT window separately
        if istft_window is not None:
            model._window = istft_window

        return model

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map PyTorch weight names to MLX names."""
        from mlx_audio.base import check_array_shape
        mapped = {}
        for key, value in weights.items():
            if "conv.conv.weight" in key and not check_array_shape(value):
                value = value.transpose(0, 2, 1)
            mapped[key] = value
        return mapped

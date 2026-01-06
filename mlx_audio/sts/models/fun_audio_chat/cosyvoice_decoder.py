# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""CosyVoice decoder for converting audio tokens to waveforms."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from mlx_audio.dsp import hanning, ISTFTCache


@dataclass
class CosyVoiceDecoderConfig:
    """Configuration for CosyVoice decoder."""
    sample_rate: int = 22050  # CosyVoice uses 22050Hz
    token_mel_ratio: int = 2
    mel_channels: int = 80
    audio_limit: float = 0.99
    vocab_size: int = 6561
    spk_embed_dim: int = 192

    # DiT parameters
    dit_dim: int = 1024
    dit_depth: int = 22
    dit_heads: int = 16
    dit_head_dim: int = 64

    # Flow matching
    n_timesteps: int = 10

    # HiFT parameters
    hift_channels: int = 512
    hift_upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 3])
    hift_upsample_kernels: List[int] = field(default_factory=lambda: [16, 11, 7])

    # Source module parameters
    nb_harmonics: int = 8
    nsf_alpha: float = 0.1  # Sine amplitude
    nsf_sigma: float = 0.003  # Noise std


def rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def mish(x: mx.array) -> mx.array:
    """Mish activation: x * tanh(softplus(x))"""
    return x * mx.tanh(nn.softplus(x))


class ConvPositionEmbedding(nn.Module):
    """Causal convolutional positional embedding."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=0, groups=groups)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=0, groups=groups)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D) -> need causal padding (left only)
        # Pad on left side for causal convolution
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        x = self.conv1(x)
        x = mish(x)
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        x = self.conv2(x)
        return x


def layer_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Apply Layer normalization without learnable parameters."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm Zero - produces modulation parameters for attention and FFN."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        # Only a linear layer to produce 6 modulation parameters
        # Uses layer_norm (not learnable)
        self.linear = nn.Linear(cond_dim, dim * 6)

    def __call__(self, x: mx.array, cond: mx.array) -> Tuple:
        # Apply SiLU to conditioning then linear
        emb = self.linear(nn.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(emb, 6, axis=-1)

        # Apply non-learnable LayerNorm then modulate
        x_normed = layer_norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp


class DiTBlock(nn.Module):
    """DiT Transformer block with adaptive layer norm."""

    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim

        self.attn_norm = AdaLNZero(dim, dim)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim))

        # FFN without learnable norm (use rms_norm function)
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU()),
            nn.Linear(dim * 2, dim),
        )

    def __call__(self, x: mx.array, cond: mx.array, rope_cos: mx.array = None, rope_sin: mx.array = None) -> mx.array:
        B, T, D = x.shape

        # Attention with adaptive layer norm
        h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, cond)

        q = self.to_q(h).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.to_k(h).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.to_v(h).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope_cos is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        attn_out = self.to_out(attn_out)

        x = x + gate_msa[:, None, :] * attn_out

        # Feed-forward with non-learnable LayerNorm and modulation
        h = layer_norm(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        ff_out = self.ff(h)
        x = x + gate_mlp[:, None, :] * ff_out

        return x


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead layer for processing token embeddings before DiT."""

    def __init__(self, in_dim: int = 80, hidden_dim: int = 1024):
        super().__init__()
        # Conv1d: kernel=4, no padding (causal look-ahead)
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=4, padding=0)
        # Conv1d: kernel=3, padding=1 for same output
        self.conv2 = nn.Conv1d(hidden_dim, in_dim, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C) - input token embeddings
        # Pad on the right for causal convolution
        x_padded = mx.pad(x, [(0, 0), (0, 3), (0, 0)])  # Pad 3 on right for kernel=4
        h = self.conv1(x_padded)
        h = nn.silu(h)
        h = self.conv2(h)
        return h


class FlowMatchingDecoder(nn.Module):
    """Flow matching decoder (DiT-based)."""

    def __init__(self, config: CosyVoiceDecoderConfig):
        super().__init__()
        self.config = config
        dim = config.dit_dim

        # Token embedding: vocab_size -> mel_channels
        self.input_embedding = nn.Embedding(config.vocab_size, config.mel_channels)

        # Pre-lookahead layer for processing token embeddings
        self.pre_lookahead_layer = PreLookaheadLayer(config.mel_channels, dim)

        # Speaker projection: spk_dim -> mel_channels
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.mel_channels)

        # Input projection: mel*4 -> dim (x, mu, x-mu, spk all 80-dim)
        self.input_proj = nn.Linear(config.mel_channels * 4, dim)

        # Convolutional position embedding
        self.conv_pos_embed = ConvPositionEmbedding(dim, kernel_size=31, groups=16)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(256, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Rotary embedding frequency (cached)
        self._rope_inv_freq = 1.0 / (10000 ** (mx.arange(0, config.dit_head_dim, 2).astype(mx.float32) / config.dit_head_dim))

        # Transformer blocks
        self.blocks = [
            DiTBlock(dim, config.dit_heads, config.dit_head_dim)
            for _ in range(config.dit_depth)
        ]

        # Output
        self.norm_out = nn.Linear(dim, dim * 2)
        self.proj_out = nn.Linear(dim, config.mel_channels)

    def get_rope(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        """Compute rotary embeddings."""
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self._rope_inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        # Shape: (1, 1, T, head_dim) to broadcast with (B, heads, T, head_dim)
        cos = mx.cos(emb)[None, None, :, :]
        sin = mx.sin(emb)[None, None, :, :]
        return cos, sin

    def get_time_embedding(self, t: mx.array, dim: int = 256) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half) / half)
        args = t[:, None] * freqs[None, :]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    def __call__(
        self,
        x: mx.array,  # Noisy mel (B, C, T)
        mu: mx.array,  # Condition mel from tokens (B, C, T)
        t: mx.array,  # Timestep (B,)
        spk: mx.array,  # Speaker embedding (B, spk_dim)
        cond: mx.array = None,  # Optional conditioning (prompt features)
    ) -> mx.array:
        B, C, T = x.shape

        # Transpose to (B, T, C)
        x = mx.swapaxes(x, 1, 2)
        mu = mx.swapaxes(mu, 1, 2)

        # Use zeros for cond if not provided (zero-shot)
        if cond is None:
            cond = mx.zeros_like(x)
        else:
            cond = mx.swapaxes(cond, 1, 2)

        # Project speaker embedding and expand
        spk_proj = self.spk_embed_affine_layer(spk)  # (B, mel_channels)
        spk_expanded = mx.broadcast_to(spk_proj[:, None, :], (B, T, C))

        # Concatenate [x, cond, mu, spk] -> (B, T, C*4)
        # Note: Original uses [x, cond, text_embed, spk] where text_embed is mu
        h = mx.concatenate([x, cond, mu, spk_expanded], axis=-1)
        h = self.input_proj(h)  # (B, T, dim)

        # Add convolutional position embedding
        h = h + self.conv_pos_embed(h)

        # Time conditioning
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)  # (B, dim)

        # Get rotary embeddings
        rope_cos, rope_sin = self.get_rope(T)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, rope_cos, rope_sin)

        # Output projection with adaptive norm
        norm_out = self.norm_out(t_emb)  # (B, dim*2)
        scale, shift = mx.split(norm_out, 2, axis=-1)
        h = h * (1 + scale[:, None, :]) + shift[:, None, :]
        out = self.proj_out(h)  # (B, T, C)

        return mx.swapaxes(out, 1, 2)  # (B, C, T)

    def inference(
        self,
        tokens: mx.array,
        spk_embedding: mx.array,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        cfg_strength: float = 0.7,
    ) -> mx.array:
        B, T = tokens.shape

        # Clamp tokens to valid range
        tokens = mx.clip(tokens, 0, self.config.vocab_size - 1)

        # Embed tokens and process through pre-lookahead layer
        emb = self.input_embedding(tokens)  # (B, T, mel)
        mu = self.pre_lookahead_layer(emb)  # (B, T, mel) - processed embeddings
        mu = mx.repeat(mu, self.config.token_mel_ratio, axis=1)  # Upsample
        mu = mx.swapaxes(mu, 1, 2)  # (B, mel, T*ratio)

        # L2 normalize speaker embedding before use
        spk_norm = mx.sqrt(mx.sum(spk_embedding * spk_embedding, axis=-1, keepdims=True) + 1e-8)
        spk_embedding = spk_embedding / spk_norm

        T_mel = mu.shape[2]

        # Initialize with noise
        z = mx.random.normal((B, self.config.mel_channels, T_mel)) * temperature

        # Cosine time schedule
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        t_span = 1 - mx.cos(t_span * 0.5 * math.pi)

        # Create null speaker embedding for CFG
        null_spk = mx.zeros_like(spk_embedding)

        # Euler solver with CFG
        x = z
        for i in range(n_timesteps):
            t = mx.full((B,), t_span[i])
            dt = t_span[i + 1] - t_span[i]

            # Run conditioned and unconditioned in parallel via batching
            x_batch = mx.concatenate([x, x], axis=0)
            mu_batch = mx.concatenate([mu, mu], axis=0)
            t_batch = mx.concatenate([t, t], axis=0)
            spk_batch = mx.concatenate([spk_embedding, null_spk], axis=0)

            v_batch = self(x_batch, mu_batch, t_batch, spk_batch)

            # Split and apply CFG
            v_cond, v_uncond = mx.split(v_batch, 2, axis=0)
            v = (1.0 + cfg_strength) * v_cond - cfg_strength * v_uncond

            x = x + dt * v
            mx.eval(x)

        return x


class Snake(nn.Module):
    """Snake activation."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones(channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        alpha = self.alpha[None, None, :]
        return x + (1.0 / (alpha + 1e-9)) * mx.sin(x * alpha) ** 2


class F0Predictor(nn.Module):
    """F0 predictor from mel spectrogram using convolutional network."""

    def __init__(self, in_channels: int = 80, hidden_channels: int = 512):
        super().__init__()
        # Convolutional conditioning network
        # 5 conv layers with kernels: 4, 3, 3, 3, 3
        self.condnet = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=4, padding=2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        ]
        self.classifier = nn.Linear(hidden_channels, 1)

    def __call__(self, mel: mx.array) -> mx.array:
        """Predict F0 from mel spectrogram.

        Args:
            mel: (B, C, T) mel spectrogram
        Returns:
            f0: (B, 1, T) predicted F0 in Hz
        """
        # mel: (B, C, T) -> (B, T, C)
        x = mx.swapaxes(mel, 1, 2)

        for conv in self.condnet:
            x = conv(x)
            x = nn.leaky_relu(x, 0.1)

        # x: (B, T, hidden) -> classifier -> (B, T, 1)
        f0 = self.classifier(x)
        # Convert to Hz and ensure positive
        f0 = mx.abs(f0) * 500  # Scale to typical F0 range (0-500 Hz)
        # (B, T, 1) -> (B, 1, T)
        return mx.swapaxes(f0, 1, 2)


class SineGen(nn.Module):
    """Sine wave generator with harmonics for neural source filter."""

    def __init__(
        self,
        sample_rate: int = 22050,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 10.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array]:
        """Generate sine waves from F0.

        Args:
            f0: (B, T, 1) F0 in Hz
        Returns:
            sine_waves: (B, T, harmonic_num+1) sine waves for each harmonic
            uv: (B, T, 1) unvoiced/voiced mask
        """
        B, T, _ = f0.shape

        # Voiced/unvoiced decision
        uv = (f0 > self.voiced_threshold).astype(mx.float32)

        # Create frequency matrix for all harmonics
        # Shape: (B, T, harmonic_num+1)
        f0_expanded = mx.repeat(f0, self.harmonic_num + 1, axis=-1)
        harmonics = mx.arange(1, self.harmonic_num + 2, dtype=mx.float32)[None, None, :]
        freq_mat = f0_expanded * harmonics / self.sample_rate

        # Compute phase using cumulative sum
        # Phase = 2π * cumsum(freq)
        phase = 2 * math.pi * mx.cumsum(freq_mat, axis=1)

        # Add random phase offset for harmonics (not fundamental)
        phase_offset = mx.random.uniform(
            low=-math.pi, high=math.pi, shape=(B, 1, self.harmonic_num + 1)
        )
        phase_offset = mx.concatenate([
            mx.zeros((B, 1, 1)),
            phase_offset[:, :, 1:]
        ], axis=-1)
        phase = phase + phase_offset

        # Generate sine waves
        sine_waves = self.sine_amp * mx.sin(phase)

        # Apply voicing mask and add noise for unvoiced
        uv_expanded = mx.repeat(uv, self.harmonic_num + 1, axis=-1)
        noise = mx.random.normal(sine_waves.shape) * self.noise_std
        sine_waves = sine_waves * uv_expanded + noise * (1 - uv_expanded)

        return sine_waves, uv


class SourceModuleHnNSF(nn.Module):
    """Harmonic-plus-Noise Source Module using neural source filter."""

    def __init__(self, config: CosyVoiceDecoderConfig):
        super().__init__()
        self.config = config
        # Sine generator
        self.sine_gen = SineGen(
            sample_rate=config.sample_rate,
            harmonic_num=config.nb_harmonics,
            sine_amp=config.nsf_alpha,
            noise_std=config.nsf_sigma,
        )
        # Linear layer to merge harmonics: (harmonic_num+1) -> 1
        self.l_linear = nn.Linear(config.nb_harmonics + 1, 1)

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Generate source signal from F0.

        Args:
            f0: (B, 1, T) F0 in Hz
        Returns:
            source: (B, 1, T) merged harmonic source
            noise: (B, 1, T) noise component
            uv: (B, 1, T) unvoiced/voiced mask
        """
        # f0: (B, 1, T) -> (B, T, 1)
        f0_t = mx.swapaxes(f0, 1, 2)

        # Generate sine waves: (B, T, harmonic_num+1)
        sine_waves, uv = self.sine_gen(f0_t)

        # Merge harmonics: (B, T, harmonic_num+1) -> (B, T, 1)
        source = mx.tanh(self.l_linear(sine_waves))

        # Generate noise component
        noise = mx.random.normal(uv.shape) * self.config.nsf_alpha / 3

        # Transpose back: (B, T, 1) -> (B, 1, T)
        source = mx.swapaxes(source, 1, 2)
        noise = mx.swapaxes(noise, 1, 2)
        uv = mx.swapaxes(uv, 1, 2)

        return source, noise, uv


class SourceResBlock(nn.Module):
    """Residual block for source processing with Snake activation."""

    def __init__(self, channels: int, kernel_size: int = 7, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for d in dilations:
            pad = (kernel_size * d - d) // 2
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=d))
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2))
            self.activations1.append(Snake(channels))
            self.activations2.append(Snake(channels))

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations1, self.activations2):
            h = a1(x)
            h = c1(h)
            h = a2(h)
            h = c2(h)
            x = x + h
        return x


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for d in dilations:
            pad = (kernel_size * d - d) // 2
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=d))
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2))
            self.activations1.append(Snake(channels))
            self.activations2.append(Snake(channels))

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations1, self.activations2):
            h = a1(x)
            h = c1(h)
            h = a2(h)
            h = c2(h)
            x = x + h
        return x


class HiFTGenerator(nn.Module):
    """HiFT vocoder with ISTFT and harmonic source integration."""

    def __init__(self, config: CosyVoiceDecoderConfig):
        super().__init__()
        self.config = config

        self.n_fft = 16
        self.hop_len = 4
        self.n_freq = self.n_fft // 2 + 1  # 9

        ch = config.hift_channels  # 512

        # F0 predictor
        self.f0_predictor = F0Predictor(config.mel_channels, ch)

        # Source module
        self.m_source = SourceModuleHnNSF(config)

        # Compute total upsample factor for F0 upsampling
        self.total_upsample = 1
        for r in config.hift_upsample_rates:
            self.total_upsample *= r
        self.total_upsample *= self.hop_len  # 8 * 5 * 3 * 4 = 480

        self.conv_pre = nn.Conv1d(config.mel_channels, ch, 5, padding=2)

        self.ups = []
        self.resblocks = []

        # Source processing layers
        self.source_downs = []
        self.source_resblocks = []

        # Compute downsample rates for source (reverse of upsample)
        # source_downs: downsample the STFT of source to match each upsample level
        downsample_rates = [1] + config.hift_upsample_rates[::-1][:-1]  # [1, 3, 5]
        downsample_cum = [1]
        for r in downsample_rates[1:]:
            downsample_cum.append(downsample_cum[-1] * r)
        downsample_cum = downsample_cum[::-1]  # [15, 3, 1]

        source_resblock_kernels = [7, 7, 11]

        for i, (rate, kernel) in enumerate(zip(config.hift_upsample_rates, config.hift_upsample_kernels)):
            in_ch = ch // (2 ** i)
            out_ch = ch // (2 ** (i + 1))
            self.ups.append(nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=rate, padding=(kernel - rate) // 2))

            # 3 resblocks per upsample layer with different kernel sizes
            for k in [3, 7, 11]:
                self.resblocks.append(ResBlock(out_ch, k, [1, 3, 5]))

            # Source downsampler and resblock for this level
            ds_rate = downsample_cum[i]
            src_kernel = source_resblock_kernels[i]
            if ds_rate == 1:
                self.source_downs.append(nn.Conv1d(self.n_fft + 2, out_ch, 1, padding=0))
            else:
                self.source_downs.append(nn.Conv1d(self.n_fft + 2, out_ch, ds_rate * 2, stride=ds_rate, padding=ds_rate // 2))
            self.source_resblocks.append(SourceResBlock(out_ch, src_kernel, [1, 3, 5]))

        final_ch = ch // (2 ** len(config.hift_upsample_rates))  # 64
        self.conv_post = nn.Conv1d(final_ch, self.n_fft + 2, 7, padding=3)  # 18 channels

        self._istft = ISTFTCache()
        self._window = hanning(self.n_fft, periodic=True)

    def _stft(self, audio: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute STFT of audio signal.

        Args:
            audio: (B, T) audio waveform
        Returns:
            real, imag: (B, n_freq, T_frames) STFT components
        """
        # Pad audio for STFT
        pad_len = self.n_fft - self.hop_len
        audio_padded = mx.pad(audio, [(0, 0), (pad_len, pad_len)])

        B, T = audio_padded.shape
        n_frames = (T - self.n_fft) // self.hop_len + 1

        # Frame the signal
        frames = []
        for i in range(n_frames):
            start = i * self.hop_len
            frame = audio_padded[:, start:start + self.n_fft]
            frames.append(frame)
        frames = mx.stack(frames, axis=1)  # (B, n_frames, n_fft)

        # Apply window
        windowed = frames * self._window[None, None, :]

        # FFT
        spectrum = mx.fft.rfft(windowed, axis=-1)  # (B, n_frames, n_freq)

        real = mx.real(spectrum)
        imag = mx.imag(spectrum)

        # Transpose to (B, n_freq, n_frames)
        real = mx.swapaxes(real, 1, 2)
        imag = mx.swapaxes(imag, 1, 2)

        return real, imag

    def decode(self, mel: mx.array, source: mx.array) -> mx.array:
        """Decode mel spectrogram with source signal to audio.

        Args:
            mel: (B, C, T) mel spectrogram
            source: (B, 1, T_audio) source signal from harmonic generator
        Returns:
            audio: (B, T_audio) generated audio
        """
        # Compute STFT of source
        source_squeezed = source.squeeze(1)  # (B, T_audio)
        s_real, s_imag = self._stft(source_squeezed)
        # Concatenate real and imag: (B, n_fft+2, T_frames)
        s_stft = mx.concatenate([s_real, s_imag], axis=1)
        # Transpose for Conv1d: (B, T_frames, n_fft+2)
        s_stft = mx.swapaxes(s_stft, 1, 2)

        # mel: (B, C, T) -> (B, T, C)
        x = mx.swapaxes(mel, 1, 2)
        x = self.conv_pre(x)

        idx = 0
        for i, up in enumerate(self.ups):
            x = nn.leaky_relu(x, 0.1)
            x = up(x)

            # Add source signal at this level
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            # Align lengths if needed
            if si.shape[1] > x.shape[1]:
                si = si[:, :x.shape[1], :]
            elif si.shape[1] < x.shape[1]:
                pad_len = x.shape[1] - si.shape[1]
                si = mx.pad(si, [(0, 0), (0, pad_len), (0, 0)])
            x = x + si

            # Apply resblocks
            xs = []
            for _ in range(3):
                xs.append(self.resblocks[idx](x))
                idx += 1
            x = mx.mean(mx.stack(xs), axis=0)

        x = nn.leaky_relu(x, 0.1)
        x = self.conv_post(x)  # (B, T, 18)

        # Split magnitude and phase
        log_mag = x[:, :, :self.n_freq]  # (B, T, 9)
        sin_phase = x[:, :, self.n_freq:]  # (B, T, 9)

        mag = mx.exp(mx.clip(log_mag, -10, 10))
        phase = mx.arcsin(mx.clip(sin_phase, -0.999, 0.999))

        real = mag * mx.cos(phase)
        imag = mag * mx.sin(phase)

        # ISTFT expects (B, n_freq, T)
        real = mx.swapaxes(real, 1, 2)
        imag = mx.swapaxes(imag, 1, 2)

        audio = self._istft.istft(real, imag, self.n_fft, self.hop_len, self.n_fft, self._window, True)
        return mx.clip(audio, -0.99, 0.99)

    def __call__(self, mel: mx.array) -> mx.array:
        """Generate audio from mel spectrogram.

        Args:
            mel: (B, C, T) mel spectrogram
        Returns:
            audio: (B, T_audio) generated audio
        """
        # Predict F0 from mel
        f0 = self.f0_predictor(mel)  # (B, 1, T)

        # Upsample F0 to audio sample rate
        # (B, 1, T) -> (B, 1, T * total_upsample)
        f0_upsampled = mx.repeat(f0, self.total_upsample, axis=2)

        # Generate source signal from F0
        source, noise, uv = self.m_source(f0_upsampled)  # (B, 1, T_audio)

        # Decode with source integration
        audio = self.decode(mel, source)

        return audio


class CosyVoiceDecoder(nn.Module):
    """CosyVoice decoder for audio token to waveform conversion."""

    def __init__(self, config: Optional[CosyVoiceDecoderConfig] = None):
        super().__init__()
        self.config = config or CosyVoiceDecoderConfig()
        self.flow = FlowMatchingDecoder(self.config)
        self.hift = HiFTGenerator(self.config)
        self._default_spk = mx.zeros((1, self.config.spk_embed_dim))

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def decode(self, tokens: mx.array, spk: mx.array = None, n_timesteps: int = 10, temperature: float = 1.0) -> mx.array:
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        B = tokens.shape[0]

        if spk is None:
            spk = mx.broadcast_to(self._default_spk, (B, self.config.spk_embed_dim))

        mel = self.flow.inference(tokens, spk, n_timesteps, temperature)
        mx.eval(mel)

        audio = self.hift(mel)
        mx.eval(audio)

        return audio.squeeze(0) if audio.shape[0] == 1 else audio

    def __call__(self, tokens: mx.array, **kwargs) -> mx.array:
        return self.decode(tokens, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512") -> "CosyVoiceDecoder":
        import torch

        if not Path(model_path).exists():
            model_path = Path(snapshot_download(repo_id=model_path, allow_patterns=["*.pt"]))
        else:
            model_path = Path(model_path)

        config = CosyVoiceDecoderConfig()
        decoder = cls(config)

        # Load flow weights
        flow_path = model_path / "flow.pt"
        if flow_path.exists():
            print(f"Loading flow weights...")
            state = torch.load(flow_path, map_location="cpu", weights_only=False)

            flow_weights = {}
            for k, v in state.items():
                arr = mx.array(v.numpy())

                # Handle convolutions - PyTorch: (out, in, k) -> MLX: (out, k, in)
                if len(arr.shape) == 3 and ("conv" in k.lower() or "Conv" in k):
                    arr = mx.swapaxes(arr, 1, 2)

                # Map weight names
                if k == "input_embedding.weight":
                    flow_weights["input_embedding.weight"] = arr
                elif k == "spk_embed_affine_layer.weight":
                    flow_weights["spk_embed_affine_layer.weight"] = arr
                elif k == "spk_embed_affine_layer.bias":
                    flow_weights["spk_embed_affine_layer.bias"] = arr
                elif k.startswith("pre_lookahead_layer."):
                    # Pre-lookahead layer convolutions
                    flow_weights[k] = arr
                elif k.startswith("decoder.estimator."):
                    new_k = k.replace("decoder.estimator.", "")

                    # Map input embedding
                    if "input_embed.proj" in new_k:
                        new_k = new_k.replace("input_embed.proj", "input_proj")
                    elif "input_embed.conv_pos_embed.conv1.0" in new_k:
                        new_k = new_k.replace("input_embed.conv_pos_embed.conv1.0", "conv_pos_embed.conv1")
                    elif "input_embed.conv_pos_embed.conv2.0" in new_k:
                        new_k = new_k.replace("input_embed.conv_pos_embed.conv2.0", "conv_pos_embed.conv2")
                    # Map transformer blocks
                    elif "transformer_blocks" in new_k:
                        new_k = new_k.replace("transformer_blocks", "blocks")
                        new_k = new_k.replace("attn.to_out.0", "to_out.layers.0")
                        new_k = new_k.replace("attn.to_q", "to_q")
                        new_k = new_k.replace("attn.to_k", "to_k")
                        new_k = new_k.replace("attn.to_v", "to_v")
                        new_k = new_k.replace("ff.ff.0.0", "ff.layers.0.layers.0")
                        new_k = new_k.replace("ff.ff.2", "ff.layers.1")
                    elif "time_embed.time_mlp.0" in new_k:
                        new_k = new_k.replace("time_embed.time_mlp.0", "time_mlp.layers.0")
                    elif "time_embed.time_mlp.2" in new_k:
                        new_k = new_k.replace("time_embed.time_mlp.2", "time_mlp.layers.2")
                    elif "norm_out.linear" in new_k:
                        new_k = new_k.replace("norm_out.linear", "norm_out")
                    elif "proj_out" in new_k:
                        pass  # Keep as is
                    elif "rotary_embed" in new_k:
                        continue  # Skip - we compute this dynamically
                    else:
                        continue

                    flow_weights[new_k] = arr

            try:
                decoder.flow.load_weights(list(flow_weights.items()), strict=False)
                print(f"  Loaded {len(flow_weights)} flow weights")
            except Exception as e:
                print(f"  Warning: {e}")

        # Load HiFT weights
        hift_path = model_path / "hift.pt"
        if hift_path.exists():
            print(f"Loading HiFT weights...")
            state = torch.load(hift_path, map_location="cpu", weights_only=False)

            hift_weights = {}

            def apply_weight_norm(g, v):
                """Apply weight normalization: g * v / ||v||"""
                v_norm = (v ** 2).sum(axis=tuple(range(1, len(v.shape))), keepdims=True) ** 0.5
                return g * v / (v_norm + 1e-8)

            # Process all weights
            for k, v in state.items():
                # Handle weight normalization for any parametrized weight
                if "parametrizations.weight.original1" in k:
                    base_k = k.replace(".parametrizations.weight.original1", "")
                    g_k = k.replace("original1", "original0")
                    if g_k in state:
                        g = state[g_k].numpy()
                        v_arr = v.numpy()
                        weight = apply_weight_norm(g, v_arr)
                        arr = mx.array(weight)
                        # Transpose for MLX Conv1d: (out, in, k) -> (out, k, in)
                        if len(arr.shape) == 3:
                            arr = mx.swapaxes(arr, 1, 2)
                        hift_weights[base_k + ".weight"] = arr
                elif "parametrizations.weight.original0" in k:
                    continue  # Handled above with original1
                elif "alpha" in k:
                    # Snake activation
                    hift_weights[k] = mx.array(v.numpy())
                elif "bias" in k:
                    hift_weights[k] = mx.array(v.numpy())
                elif ".weight" in k and "parametrizations" not in k:
                    # Regular weights (not weight-normalized)
                    arr = mx.array(v.numpy())
                    if len(arr.shape) == 3:
                        arr = mx.swapaxes(arr, 1, 2)
                    hift_weights[k] = arr

            # Map f0_predictor.condnet indices to list indices
            # Original: condnet.0, condnet.2, condnet.4, condnet.6, condnet.8
            # Target: condnet.0, condnet.1, condnet.2, condnet.3, condnet.4
            condnet_map = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4}
            mapped_weights = {}
            for k, v in hift_weights.items():
                new_k = k
                if "f0_predictor.condnet." in k:
                    # Extract index and map
                    parts = k.split(".")
                    for i, part in enumerate(parts):
                        if part == "condnet" and i + 1 < len(parts):
                            old_idx = int(parts[i + 1])
                            if old_idx in condnet_map:
                                parts[i + 1] = str(condnet_map[old_idx])
                                new_k = ".".join(parts)
                                break
                mapped_weights[new_k] = v

            try:
                decoder.hift.load_weights(list(mapped_weights.items()), strict=False)
                print(f"  Loaded {len(mapped_weights)} HiFT weights")
            except Exception as e:
                print(f"  Warning: {e}")

        # Load speaker embedding
        spk_path = model_path / "spk2info.pt"
        if spk_path.exists():
            spk_data = torch.load(spk_path, map_location="cpu", weights_only=False)
            if isinstance(spk_data, dict):
                for key in spk_data:
                    if "embedding" in spk_data[key]:
                        emb = spk_data[key]["embedding"]
                        decoder._default_spk = mx.array(emb.numpy())[None, :]
                        print(f"  Loaded speaker embedding")
                        break

        mx.eval(decoder.parameters())
        print("CosyVoice decoder loaded")
        return decoder


def decode_audio_tokens(tokens: mx.array, decoder: CosyVoiceDecoder = None, **kwargs) -> Tuple[mx.array, int]:
    if decoder is None:
        decoder = CosyVoiceDecoder()
    return decoder.decode(tokens, **kwargs), decoder.sample_rate

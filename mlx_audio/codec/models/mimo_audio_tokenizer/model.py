"""Offline MiMo Audio Tokenizer with all twenty residual codebooks."""

import re

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.dsp import hanning
from mlx_audio.utils import get_model_path, load_config, load_weights

from .audio import load_waveform, log_mel_spectrogram, same_istft
from .config import ModelConfig
from .quantization import ResidualVectorQuantizer


def norm(kind, dim):
    if kind == "LayerNorm":
        return nn.LayerNorm(dim, eps=1e-5)
    if kind == "RMSNorm":
        return nn.RMSNorm(dim, eps=1e-6)
    raise ValueError(f"Unsupported tokenizer normalization: {kind}")


class Attention(nn.Module):
    def __init__(self, dim, heads, theta, causal, window):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.theta = theta
        self.causal = causal
        self.window = window
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _rope(self, x):
        # The reference casts its nonpersistent frequency buffer along with the
        # codec, then computes cos/sin in FP32 and casts them to activation dtype.
        freqs = (
            (
                self.theta
                ** (-mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
            )
            .astype(x.dtype)
            .astype(mx.float32)
        )
        angles = mx.arange(x.shape[-2], dtype=mx.float32)[:, None] * freqs
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        first, second = mx.split(x, 2, axis=-1)
        return mx.concatenate(
            [first * cos - second * sin, second * cos + first * sin], axis=-1
        )

    def __call__(self, x):
        batch, length, dim = x.shape
        q, k, v = [
            layer(x)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
            for layer in (self.q_proj, self.k_proj, self.v_proj)
        ]
        q, k = self._rope(q), self._rope(k)
        mask = "causal" if self.causal else None
        left, right = self.window
        if left >= 0 or right >= 0:
            positions = mx.arange(length)
            difference = positions[None, :] - positions[:, None]
            mask = mx.ones((length, length), dtype=mx.bool_)
            if left >= 0:
                mask = mask & (difference >= -left)
            if right >= 0:
                mask = mask & (difference <= right)
            if self.causal:
                mask = mask & (difference <= 0)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        return self.out_proj(y.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, ffn, config, causal, window):
        super().__init__()
        self.self_attn = Attention(dim, heads, config.rope_theta, causal, window)
        self.self_attn_layer_norm = norm(config.ln_type, dim)
        self.final_layer_norm = norm(config.ln_type, dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)

    def __call__(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.d_model
        self.conv1 = nn.Conv1d(config.n_mels, dim, config.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            dim, dim, config.kernel_size, stride=config.stride_size, padding=1
        )
        self.layers = [
            TransformerLayer(
                dim,
                config.encoder_attention_heads,
                config.encoder_ffn_dim,
                config,
                config.encoder_causal,
                config.encoder_attn_window_size,
            )
            for _ in range(config.encoder_layers)
        ]
        self.layer_norm = norm(config.ln_type, dim)
        if config.avg_pooler != 1:
            self.down_sample_layer = nn.Sequential(
                nn.Conv1d(
                    dim, dim, config.avg_pooler, stride=config.avg_pooler, bias=False
                ),
                nn.GELU(),
            )
            self.down_sample_norm = norm(config.ln_type, dim)
        self.quantizer = ResidualVectorQuantizer(dim, config.codebook_size)

    def __call__(self, mel, length=None):
        length = mel.shape[1] if length is None else length
        c = self.config
        valid = (length + 3 - c.kernel_size + 2 - c.kernel_size) // c.stride_size + 1
        x = nn.gelu(self.conv1(mel.astype(self.conv1.weight.dtype)))
        x = nn.gelu(self.conv2(x))
        padded_length = x.shape[1]
        x = x[:, :valid]
        skip = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 == c.encoder_skip_layer_id:
                skip = x
        if skip is not None:
            x = x + skip
        x = self.layer_norm(x)
        if c.avg_pooler != 1:
            # Upstream unpacking repeats the final valid hidden state before
            # strided downsampling; only padding to a pool multiple is zero.
            if valid < padded_length:
                x = mx.concatenate(
                    [x, mx.repeat(x[:, -1:], padded_length - valid, axis=1)], axis=1
                )
            extra = -padded_length % c.avg_pooler
            if extra:
                x = mx.pad(x, ((0, 0), (0, extra), (0, 0)))
            x = self.down_sample_norm(self.down_sample_layer(x))
            x = x[:, : (valid + c.avg_pooler - 1) // c.avg_pooler]
        return x


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel, stride=stride)
        self.norm = nn.GroupNorm(1, out_dim, eps=1e-5, pytorch_compatible=True)
        self.trim = max(0, kernel - stride)

    def __call__(self, x):
        x = self.norm(self.conv(x))
        return x[:, : -self.trim] if self.trim else x


class ISTFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window = hanning(config.nfft, periodic=True)
        self.n_fft, self.hop_length = config.nfft, config.hop_length

    def __call__(self, spectrum):
        return mx.stack(
            [same_istft(s, self.n_fft, self.hop_length, self.window) for s in spectrum]
        )


class ISTFTHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out = nn.Linear(config.vocoder_dim, config.nfft + 2)
        self.istft = ISTFT(config)

    def __call__(self, x):
        magnitude, phase = mx.split(self.out(x), 2, axis=-1)
        magnitude = mx.minimum(mx.exp(magnitude), 100).astype(mx.float32)
        spectrum = magnitude * (
            mx.cos(phase).astype(mx.float32) + 1j * mx.sin(phase).astype(mx.float32)
        )
        return self.istft(spectrum).astype(x.dtype)


class TransformerVocos(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Linear(config.n_mels, config.vocoder_dim, bias=False)
        self.layers = [
            TransformerLayer(
                config.vocoder_dim,
                config.vocoder_attention_heads,
                config.vocoder_intermediate_dim,
                config,
                False,
                config.vocoder_attn_window_size,
            )
            for _ in range(config.vocoder_num_layers)
        ]
        self.layer_norm = norm(config.ln_type, config.vocoder_dim)
        self.head = ISTFTHead(config)

    def __call__(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.layer_norm(x))


class AudioDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dconv1 = (
            CausalConvTranspose1d(
                config.d_model, config.d_model, config.avg_pooler, config.avg_pooler
            )
            if config.avg_pooler != 1
            else None
        )
        self.layers = [
            TransformerLayer(
                config.d_model,
                config.decoder_attention_heads,
                config.decoder_ffn_dim,
                config,
                config.decoder_causal,
                config.decoder_attn_window_size,
            )
            for _ in range(config.decoder_layers)
        ]
        self.layer_norm = norm(config.ln_type, config.d_model)
        self.dconv2 = CausalConvTranspose1d(
            config.d_model,
            config.n_mels,
            config.decoder_kernel_size,
            config.decoder_stride_size,
        )
        self.vocoder = TransformerVocos(config)

    def __call__(self, x):
        x = x.astype(self.layer_norm.weight.dtype)
        if self.dconv1 is not None:
            x = self.dconv1(x)
        for layer in self.layers:
            x = layer(x)
        return self.vocoder(self.dconv2(self.layer_norm(x)))


class MiMoAudioTokenizer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)

    @property
    def sample_rate(self):
        return self.config.sampling_rate

    @property
    def num_quantizers(self):
        return self.config.num_quantizers

    def sanitize(self, weights):
        if self.config.weights_format == "mlx":
            return weights
        result = {}
        for key, value in weights.items():
            match = re.fullmatch(
                r"encoder.quantizer.vq.layers.(\d+)._codebook.(\w+)", key
            )
            if match:
                index, field = match.groups()
                if field in {"cluster_size", "embed_avg", "inited"}:
                    continue
                if field == "embed":
                    key = f"encoder.quantizer.codebooks.{index}.weight"
            elif key.endswith("weight") and value.ndim == 3:
                value = (
                    value.transpose(1, 2, 0)
                    if ".dconv" in key
                    else value.transpose(0, 2, 1)
                )
            key = key.replace(
                "encoder.down_sample_layer.0.", "encoder.down_sample_layer.layers.0."
            )
            result[key] = value
        return result

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo="XiaomiMiMo/MiMo-Audio-Tokenizer",
        *,
        dtype=mx.bfloat16,
        revision=None,
    ):
        path = get_model_path(str(path_or_repo), revision=revision)
        model = cls(ModelConfig.from_dict(load_config(path)))
        weights = model.sanitize(load_weights(path))
        if dtype is not None:
            weights = {k: v.astype(dtype) for k, v in weights.items()}
        # Match the reference: cast the codec, then restore RVQ to FP32.
        weights = {
            k: v.astype(mx.float32) if ".quantizer." in k else v
            for k, v in weights.items()
        }
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
        return model

    def encode_mels(self, mels, *, num_quantizers=None, segment_size=6000):
        """Encode (mel_frames, n_mels), returning (codebooks, codec_frames)."""
        if mels.ndim != 2 or mels.shape[-1] != self.config.n_mels or not len(mels):
            raise ValueError("Expected nonempty (frames, n_mels) features")
        if segment_size <= 0:
            raise ValueError("segment_size must be positive")
        padded_length = min(len(mels), segment_size)
        chunks = []
        for start in range(0, len(mels), segment_size):
            part = mels[start : start + segment_size]
            length = len(part)
            part = mx.pad(part, ((0, padded_length - length), (0, 0)))
            features = self.encoder(part[None], length=length)[0]
            codes = self.encoder.quantizer.encode(features, num_quantizers)
            mx.eval(codes)
            chunks.append(codes)
        return mx.concatenate(chunks, axis=-1)

    def encode(self, audio, *, sample_rate=None, num_quantizers=None):
        waveform = load_waveform(audio, sample_rate, self.sample_rate)
        return self.encode_mels(
            log_mel_spectrogram(waveform, self.config), num_quantizers=num_quantizers
        )

    def decode(self, codes, *, segment_size=1500, length=None):
        """Decode (codebooks, frames) into a mono waveform at sample_rate."""
        codes = mx.array(codes)
        if codes.ndim != 2 or not 1 <= codes.shape[0] <= self.num_quantizers:
            raise ValueError("Expected codes with shape (codebooks, frames)")
        if not mx.issubdtype(codes.dtype, mx.integer):
            raise ValueError("Audio codes must be integers")
        if segment_size <= 0 or (length is not None and length < 0):
            raise ValueError("Invalid segment size or output length")
        if codes.shape[-1] == 0:
            return mx.zeros((0,), dtype=mx.float32)
        limits = mx.array(self.config.codebook_size[: codes.shape[0]])[:, None]
        if bool(mx.any((codes < 0) | (codes >= limits))):
            raise ValueError(
                "Audio code is outside its codebook (padding is not audio)"
            )
        parts = []
        for start in range(0, codes.shape[-1], segment_size):
            features = self.encoder.quantizer.decode(
                codes[:, start : start + segment_size]
            )
            audio = self.decoder(features[None])[0].astype(mx.float32)
            mx.eval(audio)
            parts.append(audio)
        result = mx.concatenate(parts)
        return result[:length] if length is not None else result


Model = MiMoAudioTokenizer

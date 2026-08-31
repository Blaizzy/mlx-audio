"""MLX implementation of Granite Speech 5.0 TurboCTC."""

import math
import time
from pathlib import Path
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio import dsp
from mlx_audio.stt.models.base import STTOutput

from .config import EncoderConfig, ModelConfig

SAMPLING_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
DELTA_WIN_LENGTH = 3
LOGMEL_FLOOR_DB = 8.0
FRAME_STACKING = 2

_WINDOW_PAD_LEFT = (N_FFT - WIN_LENGTH) // 2
_WINDOW = mx.concatenate(
    [
        mx.zeros((_WINDOW_PAD_LEFT,)),
        dsp.hanning(WIN_LENGTH, periodic=True),
        mx.zeros((N_FFT - WIN_LENGTH - _WINDOW_PAD_LEFT,)),
    ]
)
_MEL_FILTERS_T = dsp.mel_filters(
    SAMPLING_RATE,
    N_FFT,
    80,
    mel_scale="htk",
    precise=True,
).T
mx.eval(_WINDOW, _MEL_FILTERS_T)


class EvalBatchNorm(nn.Module):
    """Inference-only BatchNorm1d over the final channel dimension."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        mean = self.running_mean.astype(mx.float32)
        var = self.running_var.astype(mx.float32)
        weight = self.weight.astype(mx.float32)
        bias = self.bias.astype(mx.float32)
        return (((x - mean) * mx.rsqrt(var + self.eps)) * weight + bias).astype(dtype)


class EncoderFeedForward(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.linear2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(hidden_states)))


class EncoderAttention(nn.Module):
    """Block-local attention with Shaw relative positional embeddings."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        inner_dim = config.num_attention_heads * config.head_dim
        self.context_size = config.context_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.hidden_size, bias=True)
        self.rel_pos_emb = nn.Embedding(
            2 * config.max_position_embeddings + 1,
            config.head_dim,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_dists: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_length, hidden_size = hidden_states.shape
        context_size = self.context_size
        num_padded = (-seq_length) % context_size

        if num_padded:
            hidden_states = mx.pad(
                hidden_states,
                [(0, 0), (0, num_padded), (0, 0)],
            )
            if attention_mask is None:
                attention_mask = mx.ones((batch_size, seq_length), dtype=mx.bool_)
            attention_mask = mx.pad(
                attention_mask.astype(mx.bool_),
                [(0, 0), (0, num_padded)],
                constant_values=False,
            )

        padded_length = hidden_states.shape[1]
        num_blocks = padded_length // context_size
        projected_shape = (
            batch_size,
            num_blocks,
            context_size,
            self.num_heads,
            self.head_dim,
        )

        def split_heads(projection):
            return projection.reshape(projected_shape).transpose(0, 1, 3, 2, 4)

        query = split_heads(self.q_proj(hidden_states))
        key = split_heads(self.k_proj(hidden_states))
        value = split_heads(self.v_proj(hidden_states))

        relative = self.rel_pos_emb(attention_dists) * self.scale
        position_bias = mx.einsum("bmhcd,crd->bmhcr", query, relative)

        if attention_mask is not None:
            key_mask = attention_mask.reshape(
                batch_size,
                num_blocks,
                context_size,
            )
            mask_value = mx.array(mx.finfo(position_bias.dtype).min)
            position_bias = mx.where(
                key_mask[:, :, None, None, :],
                position_bias,
                mask_value,
            )

        logits = (query @ key.transpose(0, 1, 2, 4, 3)) * self.scale + position_bias
        probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(query.dtype)
        output = probs @ value
        output = output.transpose(0, 1, 3, 2, 4).reshape(
            batch_size,
            padded_length,
            hidden_size,
        )
        return self.o_proj(output[:, :seq_length])


class EncoderConvolutionModule(nn.Module):
    def __init__(self, config: EncoderConfig, stride: int = 1):
        super().__init__()
        inner_dim = config.hidden_size * config.conv_expansion_factor
        self.padding = (config.conv_kernel_size - 1) // 2
        self.pointwise_lin1 = nn.Linear(config.hidden_size, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            config.conv_kernel_size,
            stride=stride,
            padding=0,
            groups=inner_dim,
            bias=False,
        )
        self.norm = EvalBatchNorm(inner_dim)
        self.pointwise_lin2 = nn.Linear(inner_dim, config.hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.pointwise_lin1(hidden_states)
        value, gate = mx.split(hidden_states, 2, axis=-1)
        hidden_states = value * mx.sigmoid(gate)

        if attention_mask is not None:
            hidden_states = mx.where(
                attention_mask[:, :, None].astype(mx.bool_),
                hidden_states,
                mx.zeros_like(hidden_states),
            )

        hidden_states = mx.pad(
            hidden_states,
            [(0, 0), (self.padding, self.padding), (0, 0)],
        )
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = nn.silu(self.norm(hidden_states))
        return self.pointwise_lin2(hidden_states)


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig, subsample: bool = False):
        super().__init__()
        self.subsample = subsample
        self.feed_forward1 = EncoderFeedForward(config)
        self.self_attn = EncoderAttention(config)
        self.conv = EncoderConvolutionModule(config, stride=2 if subsample else 1)
        self.feed_forward2 = EncoderFeedForward(config)

        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_dists: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states

        attention_output = self.self_attn(
            self.norm_self_att(hidden_states),
            attention_dists,
            attention_mask,
        )
        hidden_states = hidden_states + attention_output

        convolution_output = self.conv(
            self.norm_conv(hidden_states),
            attention_mask,
        )
        if self.subsample:
            half_length = hidden_states.shape[1] // 2
            residual = hidden_states[:, : 2 * half_length].reshape(
                hidden_states.shape[0],
                half_length,
                2,
                hidden_states.shape[2],
            )
            hidden_states = residual.mean(axis=2) + convolution_output[:, :half_length]
        else:
            hidden_states = hidden_states + convolution_output

        hidden_states = hidden_states + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden_states)
        )
        return self.norm_out(hidden_states)


def _downsample_attention_mask(attention_mask: mx.array) -> mx.array:
    half_length = attention_mask.shape[1] // 2
    return (
        attention_mask[:, : 2 * half_length]
        .reshape(
            attention_mask.shape[0],
            half_length,
            2,
        )
        .all(axis=2)
    )


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.input_linear = nn.Linear(config.num_mel_bins * 4, config.hidden_size)
        self.layers = [
            EncoderBlock(config, subsample=index in config.subsample_layers)
            for index in range(config.num_hidden_layers)
        ]
        self.out = nn.Linear(config.hidden_size, config.vocab_size)
        self.out_mid = nn.Linear(config.vocab_size, config.hidden_size)

        positions = mx.arange(config.context_size)
        relative = positions[:, None] - positions[None, :]
        self._attention_dists = (
            mx.clip(
                relative,
                -config.context_size,
                config.context_size,
            )
            + config.max_position_embeddings
        )
        mx.eval(self._attention_dists)

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.input_linear(
            input_features.astype(self.input_linear.weight.dtype)
        )
        if attention_mask is not None:
            attention_mask = attention_mask.astype(mx.bool_)
            hidden_states = mx.where(
                attention_mask[:, :, None],
                hidden_states,
                mx.zeros_like(hidden_states),
            )

        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                self._attention_dists,
                attention_mask,
            )
            if index in self.config.subsample_layers and attention_mask is not None:
                attention_mask = _downsample_attention_mask(attention_mask)

            if index + 1 == self.config.num_hidden_layers // 2:
                mid_logits = self.out(hidden_states)
                probabilities = mx.softmax(
                    mid_logits.astype(mx.float32),
                    axis=-1,
                ).astype(hidden_states.dtype)
                hidden_states = hidden_states + self.out_mid(probabilities)

        return hidden_states


def compute_deltas(features: mx.array, win_length: int = DELTA_WIN_LENGTH):
    """Match torchaudio's replicate-padded delta computation."""
    if win_length < 3 or win_length % 2 == 0:
        raise ValueError("delta win_length must be an odd integer of at least 3")
    half = (win_length - 1) // 2
    denominator = 2 * sum(index * index for index in range(1, half + 1))
    padded = mx.concatenate(
        [
            mx.broadcast_to(features[:1], (half, features.shape[1])),
            features,
            mx.broadcast_to(features[-1:], (half, features.shape[1])),
        ],
        axis=0,
    )
    deltas = mx.zeros_like(features)
    for index in range(1, half + 1):
        deltas = deltas + index * (
            padded[half + index : half + index + features.shape[0]]
            - padded[half - index : half - index + features.shape[0]]
        )
    return deltas / denominator


def compute_features(waveform: mx.array, num_mel_bins: int = 80) -> mx.array:
    """Convert a mono 16 kHz waveform to stacked log-mel+delta features."""
    waveform = waveform.reshape(-1).astype(mx.float32)
    mel_frames = waveform.shape[0] // HOP_LENGTH
    num_frames = FRAME_STACKING * math.ceil(mel_frames / FRAME_STACKING)
    if num_frames == 0:
        raise ValueError("audio must contain at least one 10 ms frame")

    num_samples_needed = (num_frames - 1) * HOP_LENGTH + 1
    if waveform.shape[0] < num_samples_needed:
        waveform = mx.pad(
            waveform,
            [(0, num_samples_needed - waveform.shape[0])],
        )

    spectrum = dsp.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=_WINDOW,
        center=True,
        pad_mode="reflect",
    )[:num_frames]
    power = mx.abs(spectrum) ** 2
    mel_filters_t = (
        _MEL_FILTERS_T
        if num_mel_bins == 80
        else dsp.mel_filters(
            SAMPLING_RATE,
            N_FFT,
            num_mel_bins,
            mel_scale="htk",
            precise=True,
        ).T
    )
    logmel = mx.log10(mx.maximum(power @ mel_filters_t, 1e-10))
    logmel = mx.maximum(logmel, mx.max(logmel) - LOGMEL_FLOOR_DB) / 4.0 + 1.0
    features = mx.concatenate(
        [logmel, compute_deltas(logmel)],
        axis=-1,
    )
    return features.reshape(
        -1,
        FRAME_STACKING * features.shape[-1],
    )


def ctc_collapse(token_ids: mx.array, blank_id: int = 0) -> mx.array:
    """Collapse adjacent repeats, then remove CTC blank tokens."""
    token_ids = token_ids.reshape(-1).astype(mx.int32)
    if token_ids.shape[0] == 0:
        return token_ids
    distinct = mx.concatenate(
        [
            mx.array([True]),
            token_ids[1:] != token_ids[:-1],
        ]
    )
    keep = distinct & (token_ids != blank_id)
    num_kept = int(mx.sum(keep).item())
    if num_kept == 0:
        return token_ids[:0]

    # MLX does not support boolean indexing. Compact kept values into unique
    # positions and send discarded values to a trailing scratch slot.
    positions = mx.cumsum(keep.astype(mx.int32)) - 1
    scatter_indices = mx.where(keep, positions, num_kept)
    output = mx.zeros((num_kept + 1,), dtype=token_ids.dtype)
    output[scatter_indices] = token_ids
    return output[:num_kept]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder_config)
        self._tokenizer = None

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.encoder(input_features, attention_mask)
        # The final CTC head is tied to the encoder's self-conditioning head.
        return self.encoder.out(hidden_states)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            if key.endswith("num_batches_tracked"):
                continue
            if (
                key.endswith("conv.depthwise_conv.weight")
                and value.ndim == 3
                and value.shape[-1] > value.shape[-2]
            ):
                # PyTorch Conv1d: [out, in/groups, kernel].
                # MLX Conv1d: [out, kernel, in/groups].
                value = value.transpose(0, 2, 1)
            sanitized[key] = value
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from tokenizers import Tokenizer

        tokenizer_path = Path(model_path) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        model._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return model

    def model_quant_predicate(self, path: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    @staticmethod
    def _load_audio(
        audio: Union[str, Path, mx.array, np.ndarray],
    ) -> mx.array:
        if isinstance(audio, (str, Path)):
            from mlx_audio.stt.utils import load_audio

            return load_audio(str(audio), sr=SAMPLING_RATE)
        if isinstance(audio, mx.array):
            return audio.reshape(-1).astype(mx.float32)
        if isinstance(audio, np.ndarray):
            return mx.array(audio.reshape(-1), dtype=mx.float32)
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def generate(
        self,
        audio: Union[str, Path, mx.array, np.ndarray],
        *,
        verbose: bool = False,
        generation_stream=None,
    ) -> STTOutput:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded; load the model with stt.load().")

        start_time = time.time()
        waveform = self._load_audio(audio)
        input_features = compute_features(
            waveform,
            num_mel_bins=self.config.encoder_config.num_mel_bins,
        )[None]

        if verbose:
            print("Encoding audio and running greedy CTC decoding...")

        logits = self(input_features)
        token_ids = ctc_collapse(
            mx.argmax(logits[0], axis=-1),
            blank_id=self.config.pad_token_id,
        )
        mx.eval(token_ids)
        tokens = [int(token) for token in token_ids.tolist()]
        text = self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
        elapsed = time.time() - start_time

        return STTOutput(
            text=text,
            segments=[],
            language="en",
            generation_tokens=len(tokens),
            total_tokens=len(tokens),
            total_time=elapsed,
            generation_tps=len(tokens) / elapsed if elapsed else 0.0,
        )

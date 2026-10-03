"""MLX inference for Nemotron 3 Diarization's feature-stacking RoPE encoder.

Architecture and AOSC semantics follow NVIDIA NeMo Speech (Apache-2.0):
https://github.com/NVIDIA-NeMo/Speech
"""

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_audio.dsp import hanning, mel_filters
from mlx_audio.vad.models.sortformer.sortformer import DiarizationOutput
from mlx_audio.vad.models.sortformer.sortformer import Model as Sortformer
from mlx_audio.vad.models.sortformer.sortformer import SortformerModules

from .config import ModelConfig


class MelFeatures(nn.Module):
    """Deterministic NeMo log-mels, retaining the checkpoint's filter buffers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window = hanning(config.win_length)
        self.fb = mel_filters(
            config.sampling_rate,
            config.n_fft,
            config.feature_size,
            norm="slaney",
            mel_scale="slaney",
        )[None]

    def __call__(self, audio, start=0, count=None, sample_offset=0, total_samples=None):
        """Compute global frames from a waveform (or a retained streaming suffix)."""
        cfg = self.config
        if total_samples is None:
            total_samples = audio.shape[0]
        valid_frames = total_samples // cfg.hop_length
        if count is None:
            count = valid_frames
        if count == 0:
            return mx.zeros((1, cfg.feature_size, 0))
        frame_ids = mx.arange(start, start + count)
        positions = (
            frame_ids[:, None] * cfg.hop_length
            + mx.arange(cfg.n_fft)[None, :]
            - cfg.n_fft // 2
        )

        # Preemphasize before STFT padding, including the sample before each window.
        def gather(indices):
            local = indices - sample_offset
            values = audio[mx.clip(local, 0, max(0, audio.shape[0] - 1))]
            return mx.where((indices >= 0) & (indices < total_samples), values, 0)

        frames = gather(positions) - cfg.preemphasis * gather(positions - 1)
        frames = mx.where((positions >= 0) & (positions < total_samples), frames, 0)
        pad = cfg.n_fft - cfg.win_length
        window = mx.pad(self.window.astype(mx.float32), [(pad // 2, pad - pad // 2)])
        power = mx.abs(mx.fft.rfft(frames * window)) ** 2
        features = mx.log(power @ self.fb[0].astype(mx.float32).T + 2**-24)
        features = mx.where(frame_ids[:, None] < valid_frames, features, 0)
        return mx.contiguous(features.T)[None]


class FeatureStacking(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.factor = config.subsampling_factor
        self.proj = nn.Linear(config.feat_in * self.factor, config.d_model, bias=False)

    def __call__(self, features, lengths):
        x = features.transpose(0, 2, 1)
        b, t, c = x.shape
        x = mx.pad(x, [(0, 0), (0, -t % self.factor), (0, 0)])
        x = self.proj(x.reshape(b, -1, c * self.factor))
        return x, (lengths + self.factor - 1) // self.factor


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.w_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.rope = nn.RoPE(
            int(self.head_dim * config.rotary_fraction),
            traditional=False,
            base=config.rope_base,
        )
        if config.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

    def __call__(self, x, mask):
        b, t, d = x.shape
        qkv = self.w_qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q), self.rope(k)
        x = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        return self.out_proj(x.transpose(0, 2, 1, 3).reshape(b, t, d))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(config.d_model * config.ff_expansion)
        self.linear1 = nn.Linear(config.d_model, hidden)
        self.linear2 = nn.Linear(hidden, config.d_model)

    def __call__(self, x):
        return self.linear2(nn.gelu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_encode = FeatureStacking(config)
        self.embed_norm = (
            nn.LayerNorm(config.d_model) if config.pre_block_norm else nn.Identity()
        )
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.final_norm = nn.LayerNorm(config.d_model)
        self._scale = math.sqrt(config.d_model) if config.xscaling else 1.0

    def __call__(self, x, lengths):
        valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
        mask = valid[:, None, None, :]
        x = self.embed_norm(x * self._scale)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class SpeakerModules(SortformerModules):
    def __init__(self, config):
        super().__init__(config)
        self.subpixel_upsample = nn.Conv1d(
            config.tf_d_model,
            config.tf_d_model * config.subsampling_factor,
            kernel_size=3,
            padding=1,
        )
        self.learnable_sil_emb = mx.zeros((config.fc_d_model,))
        if config.use_activity_head:
            self.activity_head = nn.Sequential(
                nn.LayerNorm(config.tf_d_model), nn.Linear(config.tf_d_model, 3)
            )

    def __call__(self, x):
        x = self.encoder_proj(x)
        b, _, h = x.shape
        x = self.subpixel_upsample(x).reshape(b, -1, h)
        return self.forward_speaker_sigmoids(x)


@dataclass
class StreamingState:
    """AOSC/FIFO at 80 ms, with bounded PCM history for exact STFT boundaries."""

    spkcache: mx.array
    spkcache_preds: mx.array
    fifo: mx.array
    fifo_preds: mx.array
    spkcache_compressed: bool = False
    frames_processed: int = 0  # Native 10 ms frames, before output downsampling.
    samples_received: int = 0
    sample_offset: int = 0
    audio_buffer: mx.array = field(default_factory=lambda: mx.zeros((0,)))
    finished: bool = False


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder_config)
        self.sortformer_modules = SpeakerModules(config.modules_config)
        self.preprocessor = MelFeatures(config.processor_config)
        self._processor_config = config.processor_config

    @property
    def dtype(self):
        projection = self.encoder.pre_encode.proj
        if hasattr(projection, "scales"):
            return projection.scales.dtype
        return projection.weight.dtype

    @property
    def sample_rate(self):
        return self.config.processor_config.sampling_rate

    def __call__(self, audio_signal, audio_signal_length):
        """Single-window forward: (B, mel, T) -> (B, ceil(T/8)*8, speakers)."""
        x, lengths = self.encoder.pre_encode(
            audio_signal.astype(self.dtype), audio_signal_length
        )
        probs = self.sortformer_modules(self.encoder(x, lengths))
        valid = mx.arange(probs.shape[1])[None, :] < audio_signal_length[:, None]
        return probs * valid[:, :, None]

    def init_streaming_state(self):
        cfg = self.config
        return StreamingState(
            spkcache=mx.zeros((1, 0, cfg.encoder_config.d_model), dtype=self.dtype),
            spkcache_preds=mx.zeros((1, 0, cfg.num_speakers)),
            fifo=mx.zeros((1, 0, cfg.encoder_config.d_model), dtype=self.dtype),
            fifo_preds=mx.zeros((1, 0, cfg.num_speakers)),
        )

    def set_streaming_config(self, preset):
        """Select an NVIDIA latency preset before starting a new recording.

        Presets describe input-buffer latency, excluding compute and the STFT
        window: offline=30.4s, low=1.04s, very_low=0.64s, ultra_low=0.32s.
        """
        presets = {
            "offline": (340, 40, 40, 300),
            "low": (9, 4, 264, 222),
            "very_low": (6, 2, 264, 222),
            "ultra_low": (3, 1, 264, 222),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset {preset!r}; choose from {list(presets)}")
        chunk, right, fifo, period = presets[preset]
        modules = replace(
            self.config.modules_config,
            chunk_len=chunk,
            chunk_right_context=right,
            fifo_len=fifo,
            spkcache_len=264,
            spkcache_update_period=period,
        )
        self.config = replace(self.config, modules_config=modules)

    def streaming_step(self, features, state, central_frames, feature_length=None):
        """Process one feature window; context predictions never enter the FIFO."""
        cfg = self.config.modules_config
        factor = self.config.encoder_config.subsampling_factor
        if feature_length is None:
            feature_length = features.shape[-1]
        chunk, lengths = self.encoder.pre_encode(
            features.astype(self.dtype), mx.array([feature_length])
        )
        cache_len, fifo_len = state.spkcache.shape[1], state.fifo.shape[1]
        combined = mx.concatenate([state.spkcache, state.fifo, chunk], axis=1)
        high = self.sortformer_modules(
            self.encoder(combined, lengths + cache_len + fifo_len)
        )
        valid = mx.arange(high.shape[1]) < (lengths[0] + cache_len + fifo_len) * factor
        high = high * valid[None, :, None]
        # Cache scoring always operates at encoder resolution, using mean probabilities.
        low = (
            high.astype(mx.float32)
            .reshape(1, -1, factor, self.config.num_speakers)
            .mean(axis=2)
        )
        n = (central_frames + factor - 1) // factor
        start = cache_len + fifo_len
        result = high[0, start * factor : start * factor + central_frames].astype(
            mx.float32
        )
        state.fifo = mx.concatenate([state.fifo, chunk[:, :n]], axis=1)
        state.fifo_preds = mx.concatenate(
            [low[:, cache_len:start], low[:, start : start + n]], axis=1
        )
        if state.fifo.shape[1] > cfg.fifo_len:
            pop = min(
                state.fifo.shape[1],
                max(cfg.spkcache_update_period, state.fifo.shape[1] - cfg.fifo_len),
            )
            state.spkcache = mx.concatenate(
                [state.spkcache, state.fifo[:, :pop]], axis=1
            )
            previous = (
                state.spkcache_preds
                if state.spkcache_compressed
                else low[:, :cache_len]
            )
            state.spkcache_preds = mx.concatenate(
                [previous, state.fifo_preds[:, :pop]], axis=1
            )
            state.fifo, state.fifo_preds = (
                state.fifo[:, pop:],
                state.fifo_preds[:, pop:],
            )
            if state.spkcache.shape[1] > cfg.spkcache_len:
                state.spkcache, state.spkcache_preds = (
                    Sortformer._compress_spkcache_aosc(
                        state.spkcache,
                        state.spkcache_preds,
                        self.sortformer_modules.learnable_sil_emb[None],
                        cfg,
                    )
                )
                state.spkcache_compressed = True
        state.frames_processed += central_frames
        mx.eval(
            result, state.spkcache, state.spkcache_preds, state.fifo, state.fifo_preds
        )
        return result

    def _output(
        self,
        probs,
        offset=0.0,
        threshold=0.5,
        min_duration=0.0,
        merge_gap=0.0,
        state=None,
    ):
        factor = self.config.output_subsampling_factor
        if factor > 1 and probs.shape[0]:
            length = probs.shape[0]
            padded = mx.pad(probs, [(0, -length % factor), (0, 0)])
            counts = mx.minimum(
                factor, length - mx.arange(padded.shape[0] // factor) * factor
            )
            probs = (
                padded.reshape(-1, factor, self.config.num_speakers).sum(axis=1)
                / counts[:, None]
            )
        stride = self._processor_config.hop_length / self.sample_rate
        segments = Sortformer._preds_to_segments(
            probs, stride * factor, threshold, min_duration, merge_gap
        )
        for seg in segments:
            seg.start += offset
            seg.end += offset
            if state is not None:
                seg.end = min(seg.end, state.frames_processed * stride)
        return DiarizationOutput(
            segments=segments,
            speaker_probs=probs,
            num_speakers=len({s.speaker for s in segments}),
            state=state,
        )

    def feed(
        self,
        audio,
        state,
        sample_rate=16000,
        *,
        final=False,
        threshold=0.5,
        min_duration=0.0,
        merge_gap=0.0,
    ):
        """Feed arbitrary mono PCM chunks. Call once with final=True to flush lookahead.

        Audio must already have the model sample rate. Empty output means more
        samples are needed for the configured chunk, right context and STFT window.
        """
        if state.finished:
            raise ValueError("This stream is finished; initialize a new state")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"feed requires {self.sample_rate} Hz audio; resample before streaming"
            )
        chunk = mx.array(audio).astype(mx.float32)
        if chunk.ndim != 1:
            raise ValueError("feed expects one-dimensional mono audio")
        state.audio_buffer = mx.concatenate([state.audio_buffer, chunk])
        state.samples_received += chunk.shape[0]
        cfg, proc = self.config.modules_config, self._processor_config
        factor = self.config.encoder_config.subsampling_factor
        central, right = cfg.chunk_len * factor, cfg.chunk_right_context * factor
        offset = state.frames_processed * proc.hop_length / self.sample_rate
        outputs = []
        while True:
            available = (
                state.samples_received // proc.hop_length - state.frames_processed
            )
            needed = (
                state.frames_processed + central + right - 1
            ) * proc.hop_length + proc.n_fft // 2
            if available <= 0 or (not final and state.samples_received < needed):
                break
            n = min(central, available)
            if final:
                # NeMo masks the extra centered STFT frame, then pads to pad_to.
                total_frames = state.samples_received // proc.hop_length + 1
                pad_to = getattr(proc, "pad_to", 16)
                total_frames += -total_frames % pad_to if pad_to else 0
                window_frames = min(
                    central + right, total_frames - state.frames_processed
                )
            else:
                window_frames = central + right
            features = self.preprocessor(
                state.audio_buffer,
                start=state.frames_processed,
                count=window_frames,
                sample_offset=state.sample_offset,
                total_samples=state.samples_received,
            )
            outputs.append(
                self.streaming_step(features, state, n, min(window_frames, available))
            )
            keep_from = max(
                0, state.frames_processed * proc.hop_length - proc.n_fft // 2 - 1
            )
            state.audio_buffer = state.audio_buffer[keep_from - state.sample_offset :]
            state.sample_offset = keep_from
            mx.eval(state.audio_buffer)
        state.finished = final
        if final:
            state.audio_buffer = mx.zeros((0,))
        probs = (
            mx.concatenate(outputs)
            if outputs
            else mx.zeros((0, self.config.num_speakers))
        )
        return (
            self._output(probs, offset, threshold, min_duration, merge_gap, state),
            state,
        )

    _load_audio = Sortformer._load_audio
    _resample = staticmethod(Sortformer._resample)

    def generate_stream(
        self,
        audio,
        sample_rate=16000,
        *,
        threshold=0.5,
        min_duration=0.0,
        merge_gap=0.0,
        verbose=False,
    ):
        """Yield chunk results for a file/array or iterable of mono PCM chunks."""
        state = self.init_streaming_state()
        if isinstance(audio, (str, Path, np.ndarray, mx.array)):
            waveform, sample_rate = self._load_audio(
                str(audio) if isinstance(audio, Path) else audio, sample_rate
            )
            # Feed one configured window at a time; arbitrary input partitions are equivalent.
            step = (
                self.config.modules_config.chunk_len
                * self.config.encoder_config.subsampling_factor
                * self._processor_config.hop_length
            )
            audio = (waveform[i : i + step] for i in range(0, waveform.shape[0], step))
        for chunk in audio:
            result, state = self.feed(
                chunk,
                state,
                sample_rate,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=merge_gap,
            )
            if result.speaker_probs.shape[0]:
                if verbose:
                    print(result.text)
                yield result
        result, state = self.feed(
            mx.zeros((0,)),
            state,
            sample_rate,
            final=True,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )
        if result.speaker_probs.shape[0]:
            if verbose:
                print(result.text)
            yield result

    def generate(
        self,
        audio,
        sample_rate=16000,
        threshold=0.5,
        min_duration=0.0,
        merge_gap=0.0,
        verbose=False,
    ):
        """Diarize audio with bounded AOSC context, including long recordings."""
        start = time.perf_counter()
        results = list(self.generate_stream(audio, sample_rate, threshold=threshold))
        probs = (
            mx.concatenate([r.speaker_probs for r in results])
            if results
            else mx.zeros((0, self.config.num_speakers))
        )
        stride = (
            self._processor_config.hop_length
            * self.config.output_subsampling_factor
            / self.sample_rate
        )
        segments = Sortformer._preds_to_segments(
            probs, stride, threshold, min_duration, merge_gap
        )
        if results:
            end = (
                results[-1].state.frames_processed
                * self._processor_config.hop_length
                / self.sample_rate
            )
            for seg in segments:
                seg.end = min(seg.end, end)
        result = DiarizationOutput(
            segments,
            probs,
            len({s.speaker for s in segments}),
            time.perf_counter() - start,
        )
        if verbose:
            print(result.text)
        return result

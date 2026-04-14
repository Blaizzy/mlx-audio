"""MeloTTS (VITS2-based) TTS model for MLX."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import BaseModelArgs, GenerationResult
from .bert import BertConfig, BertModel
from .hifigan import Generator
from .modules import (
    DurationPredictor,
    Flip,
    PosteriorEncoder,
    StochasticDurationPredictor,
    TextEncoder,
    TransformerCouplingLayer,
)


@dataclass
class ModelConfig(BaseModelArgs):
    sampling_rate: int = 44100
    filter_length: int = 2048
    hop_length: int = 512
    segment_size: int = 16384
    add_blank: bool = True
    n_speakers: int = 256
    spk2id: Dict[str, int] = field(default_factory=dict)

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    n_layers_trans_flow: int = 3
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 8, 2, 2])
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 256
    use_spk_conditioned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_transformer_flow: bool = True

    num_tones: int = 16
    num_languages: int = 10
    n_vocab: int = 219
    bert_hidden_size: int = 1024

    @property
    def sample_rate(self):
        return self.sampling_rate


class Model(nn.Module):
    """MeloTTS model — VITS2-based end-to-end TTS."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.enc_p = TextEncoder(
            n_vocab=config.n_vocab,
            out_channels=config.inter_channels,
            hidden_channels=config.hidden_channels,
            filter_channels=config.filter_channels,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            kernel_size=config.kernel_size,
            p_dropout=config.p_dropout,
            gin_channels=(
                config.gin_channels if config.use_spk_conditioned_encoder else 0
            ),
            num_tones=config.num_tones,
            num_languages=config.num_languages,
        )

        self.dec = Generator(
            initial_channel=config.inter_channels,
            resblock=config.resblock,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            upsample_rates=config.upsample_rates,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            gin_channels=config.gin_channels,
        )

        self.enc_q = PosteriorEncoder(
            in_channels=config.filter_length // 2 + 1,  # spec_channels
            out_channels=config.inter_channels,
            hidden_channels=config.inter_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=config.gin_channels,
        )

        self.flow_layers = []
        for i in range(4):
            self.flow_layers.append(
                TransformerCouplingLayer(
                    config.inter_channels,
                    config.inter_channels,
                    kernel_size=5,
                    n_layers=config.n_layers_trans_flow,
                    n_heads=config.n_heads,
                    p_dropout=config.p_dropout,
                    filter_channels=config.filter_channels,
                    mean_only=True,
                    gin_channels=config.gin_channels,
                )
            )
            self.flow_layers.append(Flip())

        self.dp = DurationPredictor(
            config.hidden_channels,
            256,
            3,
            config.p_dropout,
            gin_channels=config.gin_channels,
        )
        self.sdp = StochasticDurationPredictor(
            config.hidden_channels,
            192,
            3,
            config.p_dropout,
            4,
            gin_channels=config.gin_channels,
        )

        self.emb_g = nn.Embedding(config.n_speakers, config.gin_channels)

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def infer(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert=None,
        noise_scale=0.667,
        length_scale=1.0,
        noise_scale_w=0.8,
        sdp_ratio=0.0,
    ):
        """Run TTS inference. Returns (B, 1, T_audio) waveform."""
        g = mx.expand_dims(self.emb_g(sid), -1)

        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert=ja_bert, g=g
        )

        logw_dp = self.dp(x, x_mask, g=g)
        if sdp_ratio > 0:
            logw_sdp = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
            logw = sdp_ratio * logw_sdp + (1 - sdp_ratio) * logw_dp
        else:
            logw = logw_dp
        w = mx.exp(logw) * x_mask * length_scale

        w_ceil = mx.ceil(w)
        y_lengths = mx.clip(mx.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(
            mx.int32
        )
        y_mask = self._sequence_mask(y_lengths, int(y_lengths.max()))[:, None, :]

        attn_mask = x_mask[:, :, :, None] * y_mask[:, :, None, :]
        attn = self._generate_path(w_ceil, attn_mask)

        m_p = mx.matmul(m_p, attn.squeeze(1))
        logs_p = mx.matmul(logs_p, attn.squeeze(1))

        z_p = m_p + mx.random.normal(m_p.shape) * mx.exp(logs_p) * noise_scale

        z = z_p
        for layer in reversed(self.flow_layers):
            z = layer(z, y_mask, g=g, reverse=True)

        audio = self.dec(z * y_mask, g=g)
        return audio

    def _sequence_mask(self, lengths, max_len=None):
        if max_len is None:
            max_len = int(lengths.max())
        return (mx.arange(max_len)[None, :] < lengths[:, None]).astype(mx.float32)

    def _generate_path(self, duration, mask):
        """Generate alignment path (B, 1, t_x, t_y) from durations."""
        b, _, t_x = duration.shape
        t_y = mask.shape[-1]

        dur = duration.squeeze(1)
        cum_dur = mx.cumsum(dur, axis=-1)
        cum_dur_shifted = mx.pad(cum_dur[:, :-1], [(0, 0), (1, 0)])

        y_pos = mx.arange(t_y)[None, :]
        start = cum_dur_shifted[:, :, None]
        end = cum_dur[:, :, None]

        path = ((y_pos[None, :, :] >= start) & (y_pos[None, :, :] < end)).astype(
            mx.float32
        )
        path = path[:, None, :, :]
        return path * mask

    def _prepare_inputs(
        self, text, voice, lang_code, speed, noise_scale, noise_scale_w, sdp_ratio
    ):
        """Text processing and latent z computation (everything before decoding)."""
        from .text import process_text

        spk2id = self.config.spk2id
        if voice and voice in spk2id:
            sid = spk2id[voice]
        elif lang_code in spk2id:
            sid = spk2id[lang_code]
        else:
            sid = spk2id.get("EN-Default", 0)

        result = process_text(
            text,
            bert_model=self.bert if hasattr(self, "bert") else None,
            language="EN",
            add_blank=self.config.add_blank,
        )

        phone_ids = mx.array([result["phone_ids"]])
        tone_ids = mx.array([result["tone_ids"]])
        lang_ids = mx.array([result["lang_ids"]])
        n_phones = len(result["phone_ids"])
        bert_zeros = mx.zeros((1, 1024, n_phones))
        ja_bert_features = mx.expand_dims(result["bert_features"], 0)
        x_lengths = mx.array([n_phones])
        sid_tensor = mx.array([sid])

        # Run encoder + duration + flow (everything except decoder)
        g = mx.expand_dims(self.emb_g(sid_tensor), -1)
        x, m_p, logs_p, x_mask = self.enc_p(
            phone_ids,
            x_lengths,
            tone_ids,
            lang_ids,
            bert_zeros,
            ja_bert=ja_bert_features,
            g=g,
        )

        logw_dp = self.dp(x, x_mask, g=g)
        if sdp_ratio > 0:
            logw_sdp = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
            logw = sdp_ratio * logw_sdp + (1 - sdp_ratio) * logw_dp
        else:
            logw = logw_dp
        w = mx.exp(logw) * x_mask * (1.0 / speed)

        w_ceil = mx.ceil(w)
        y_lengths = mx.clip(mx.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(
            mx.int32
        )
        y_mask = self._sequence_mask(y_lengths, int(y_lengths.max()))[:, None, :]

        attn_mask = x_mask[:, :, :, None] * y_mask[:, :, None, :]
        attn = self._generate_path(w_ceil, attn_mask)

        m_p = mx.matmul(m_p, attn.squeeze(1))
        logs_p = mx.matmul(logs_p, attn.squeeze(1))

        z_p = m_p + mx.random.normal(m_p.shape) * mx.exp(logs_p) * noise_scale

        z = z_p
        for layer in reversed(self.flow_layers):
            z = layer(z, y_mask, g=g, reverse=True)

        return z * y_mask, g, result

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "EN-US",
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.0,
        stream: bool = False,
        streaming_interval: float = 1.0,
        **kwargs,
    ):
        """Generate audio from text. Set stream=True for chunked output."""
        start_time = time.time()

        z, g, result = self._prepare_inputs(
            text,
            voice,
            lang_code,
            speed,
            noise_scale,
            noise_scale_w,
            sdp_ratio,
        )

        if not stream:
            audio = self.dec(z, g=g).squeeze(0).squeeze(0)
            mx.eval(audio)
            elapsed = time.time() - start_time
            samples = audio.shape[0]
            audio_duration = samples / self.sample_rate
            yield self._make_result(audio, samples, result, elapsed, segment_idx=0)
        else:
            # Streaming: chunk z along time dim, decode with overlap, yield
            # HiFi-GAN upsample factor
            hop = 1
            for r in self.config.upsample_rates:
                hop *= r
            # Context frames for overlap (avoid boundary artifacts)
            context_frames = 16
            # Chunk size in latent frames from streaming_interval
            chunk_frames = max(1, int(self.sample_rate * streaming_interval / hop))
            t_total = z.shape[2]

            segment_idx = 0
            pos = 0
            while pos < t_total:
                chunk_end = min(pos + chunk_frames, t_total)
                # Add context from previous chunk
                ctx_start = max(0, pos - context_frames)
                z_chunk = z[:, :, ctx_start:chunk_end]

                audio_chunk = self.dec(z_chunk, g=g).squeeze(0).squeeze(0)
                mx.eval(audio_chunk)

                # Trim context samples from the front
                trim_samples = (pos - ctx_start) * hop
                audio_chunk = audio_chunk[trim_samples:]

                elapsed = time.time() - start_time
                samples = audio_chunk.shape[0]
                is_final = chunk_end >= t_total

                yield self._make_result(
                    audio_chunk,
                    samples,
                    result,
                    elapsed,
                    segment_idx=segment_idx,
                    is_streaming_chunk=True,
                    is_final_chunk=is_final,
                )

                segment_idx += 1
                pos = chunk_end

    def _make_result(
        self,
        audio,
        samples,
        text_result,
        elapsed,
        segment_idx=0,
        is_streaming_chunk=False,
        is_final_chunk=False,
    ):
        audio_duration = samples / self.sample_rate
        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=segment_idx,
            token_count=len(text_result["phone_ids"]),
            audio_duration=f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}.{int((audio_duration % 1) * 1000):03d}",
            real_time_factor=(
                round(elapsed / audio_duration, 2) if audio_duration > 0 else 0
            ),
            prompt={
                "tokens": len(text_result["phone_ids"]),
                "tokens-per-sec": (
                    round(len(text_result["phone_ids"]) / elapsed, 2)
                    if elapsed > 0
                    else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

    def sanitize(self, weights):
        """Map PyTorch weight names to MLX structure."""
        sanitized = {}
        for key, value in weights.items():
            if any(key.startswith(p) for p in ["net_dur_disc", "net_d"]):
                continue

            new_key = key.replace("flow.flows.", "flow_layers.")

            if new_key.endswith(".weight_g"):
                continue
            if new_key.endswith(".weight_v"):
                base = new_key[: -len(".weight_v")]
                g_key = key[: -len(".weight_v")] + ".weight_g"
                if g_key in weights:
                    wv = value
                    wg = weights[g_key]
                    norm_dims = tuple(range(1, wv.ndim))
                    norm = (wv**2).sum(axis=norm_dims, keepdims=True) ** 0.5
                    weight = wg * wv / norm
                    sanitized[base + ".weight"] = weight
                else:
                    sanitized[new_key] = value
                continue

            if new_key.endswith(".gamma"):
                new_key = new_key[:-6] + ".weight"
            elif new_key.endswith(".beta"):
                new_key = new_key[:-5] + ".bias"

            sanitized[new_key] = value
        return sanitized

    @classmethod
    def post_load_hook(cls, model, model_path):
        """Load BERT model and symbols after main weights are loaded."""
        import json
        from pathlib import Path

        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            if "symbols" in config_data:
                from .text import load_symbols_from_config

                load_symbols_from_config(config_data["symbols"])

        bert_weights_path = model_path / "bert_weights.npz"
        if not bert_weights_path.exists():
            return model

        import numpy as np

        config = BertConfig()
        bert = BertModel(config)
        weights = dict(np.load(str(bert_weights_path)))
        weights = {k: mx.array(v) for k, v in weights.items()}
        weights = bert.sanitize(weights)
        bert.load_weights(list(weights.items()))
        model.bert = bert
        return model

import json
import sys
import time
from dataclasses import dataclass
from numbers import Number
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from ..base import BaseModelArgs, GenerationResult, check_array_shape
from .istftnet import Decoder
from .modules import AlbertModelArgs, CustomAlbert, ProsodyPredictor, TextEncoder
from .pipeline import KokoroPipeline

# Force reset logger configuration at the top of your file
logger.remove()  # Remove all handlers
logger.configure(
    handlers=[{"sink": sys.stderr, "level": "DEBUG"}]
)  # Add back with explicit level


def sanitize_lstm_weights(key: str, state_dict: mx.array) -> dict:
    """Convert PyTorch LSTM weight keys to MLX LSTM weight keys."""
    base_key = key.rsplit(".", 1)[0]

    # Mapping of PyTorch LSTM weight suffixes to MLX weight names
    weight_map = {
        "weight_ih_l0_reverse": "Wx_backward",
        "weight_hh_l0_reverse": "Wh_backward",
        "bias_ih_l0_reverse": "bias_ih_backward",
        "bias_hh_l0_reverse": "bias_hh_backward",
        "weight_ih_l0": "Wx_forward",
        "weight_hh_l0": "Wh_forward",
        "bias_ih_l0": "bias_ih_forward",
        "bias_hh_l0": "bias_hh_forward",
    }

    for suffix, new_suffix in weight_map.items():
        if key.endswith(suffix):
            return {f"{base_key}.{new_suffix}": state_dict}

    return {key: state_dict}


@dataclass
class ModelConfig(BaseModelArgs):
    istftnet: dict
    dim_in: int
    dropout: float
    hidden_dim: int
    max_conv_dim: int
    max_dur: int
    multispeaker: bool
    n_layer: int
    n_mels: int
    n_token: int
    style_dim: int
    text_encoder_kernel_size: int
    plbert: dict
    vocab: Dict[str, int]
    sample_rate: int = 24000


class Model(nn.Module):
    """
    KokoroModel is a mlx.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KokoroModel instance, and it can be reused across
    multiple KokoroPipelines to avoid redundant memory allocation.

    Unlike KokoroPipeline, KokoroModel is language-blind.

    KokoroModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KokoroModel.
    """

    REPO_ID = "prince-canuma/Kokoro-82M"

    def __init__(self, config: ModelConfig, repo_id: str = None):
        super().__init__()
        self.repo_id = repo_id
        self.config = config
        self.vocab = config.vocab
        self.bert = CustomAlbert(
            AlbertModelArgs(vocab_size=config.n_token, **config.plbert)
        )

        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config.hidden_dim)
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_dim,
            nlayers=config.n_layer,
            max_dur=config.max_dur,
            dropout=config.dropout,
        )
        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )
        self.decoder = Decoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            dim_out=config.n_mels,
            **config.istftnet,
        )
        self._pipelines: Dict[str, KokoroPipeline] = {}  # Cache for pipelines

    @dataclass
    class Output:
        audio: mx.array
        pred_dur: Optional[mx.array] = None

    def __call__(
        self,
        phonemes: str,
        ref_s: mx.array,
        speed: Number = 1,
        return_output: bool = False,  # MARK: BACKWARD COMPAT
        decoder=None,
    ) -> Union["KokoroModel.Output", mx.array]:
        input_ids = list(
            filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes))
        )
        assert len(input_ids) + 2 <= self.context_length, (
            len(input_ids) + 2,
            self.context_length,
        )
        input_ids = mx.array([[0, *input_ids, 0]])
        input_lengths = mx.array([input_ids.shape[-1]])
        text_mask = mx.arange(int(input_lengths.max()))[None, ...]
        text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(
            input_lengths.dtype
        )
        text_mask = text_mask + 1 > input_lengths[:, None]
        bert_dur, _ = self.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))
        d_en = self.bert_encoder(bert_dur).transpose(0, 2, 1)
        ref_s = ref_s
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = mx.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=None).astype(mx.int32)[0]
        indices = mx.concatenate(
            [mx.repeat(mx.array(i), int(n)) for i, n in enumerate(pred_dur)]
        )
        pred_aln_trg = mx.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, mx.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg[None, :]
        en = d.transpose(0, 2, 1) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(
            input_ids, input_lengths, text_mask
        )  # Working fine in MLX
        asr = t_en @ pred_aln_trg

        decoder = mx.compile(decoder) if decoder is not None else self.decoder
        audio = decoder(asr, F0_pred, N_pred, ref_s[:, :128])[0]  # Working fine in MLX

        # Evaluate the computation graph for audio and pred_dur before returning
        mx.eval(audio, pred_dur)

        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, state_dict in weights.items():

            if key.startswith("bert"):
                if "position_ids" in key:
                    # Remove unused position_ids
                    continue
                else:
                    # print(k, v.shape)
                    sanitized_weights[key] = state_dict

            if key.startswith("bert_encoder"):
                sanitized_weights[key] = state_dict

            if key.startswith("text_encoder"):

                if key.endswith((".gamma", ".beta")):
                    base_key = key.rsplit(".", 1)[0]
                    if key.endswith(".gamma"):
                        new_key = f"{base_key}.weight"
                    else:
                        new_key = f"{base_key}.bias"

                    sanitized_weights[new_key] = state_dict
                elif "weight_v" in key:
                    if check_array_shape(state_dict):
                        sanitized_weights[key] = state_dict
                    else:
                        sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                # Replace weight_ih_l0_reverse and weight_hh_l0_reverse with Wx and Wh
                elif key.endswith(
                    (
                        ".weight_ih_l0_reverse",
                        ".weight_hh_l0_reverse",
                        ".bias_ih_l0_reverse",
                        ".bias_hh_l0_reverse",
                        ".weight_ih_l0",
                        ".weight_hh_l0",
                        ".bias_ih_l0",
                        ".bias_hh_l0",
                    )
                ):
                    sanitized_weights.update(sanitize_lstm_weights(key, state_dict))
                else:
                    sanitized_weights[key] = state_dict

            if key.startswith("predictor"):
                if "F0_proj.weight" in key:
                    sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                elif "N_proj.weight" in key:
                    sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                elif "weight_v" in key:
                    if check_array_shape(state_dict):
                        sanitized_weights[key] = state_dict
                    else:
                        sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                    # Replace weight_ih_l0_reverse and weight_hh_l0_reverse with Wx and Wh
                elif key.endswith(
                    (
                        ".weight_ih_l0_reverse",
                        ".weight_hh_l0_reverse",
                        ".bias_ih_l0_reverse",
                        ".bias_hh_l0_reverse",
                        ".weight_ih_l0",
                        ".weight_hh_l0",
                        ".bias_ih_l0",
                        ".bias_hh_l0",
                    )
                ):
                    sanitized_weights.update(sanitize_lstm_weights(key, state_dict))
                else:
                    sanitized_weights[key] = state_dict

            if key.startswith("decoder"):
                sanitized_weights[key] = self.decoder.sanitize(key, state_dict)
        return sanitized_weights

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def _get_pipeline(self, lang_code: str) -> KokoroPipeline:
        """Retrieves or creates a cached KokoroPipeline for the given language code."""
        if lang_code not in self._pipelines:
            logger.info(f"Creating new KokoroPipeline for language: {lang_code}")
            self._pipelines[lang_code] = KokoroPipeline(
                model=self,
                repo_id=self.REPO_ID if self.repo_id is None else self.repo_id,
                lang_code=lang_code,
            )
        return self._pipelines[lang_code]

    def generate(
        self,
        text: str,
        voice: str = None,
        speed: float = 1.0,
        lang_code: str = "a",
        split_pattern: str = r"\n+",
        **kwargs,
    ):
        pipeline = self._get_pipeline(lang_code)

        pipeline.voices = {}  # Reset voices

        if voice is None:
            voice = "af_heart"

        # Track overall generation time
        start_time = time.time()

        for segment_idx, (graphenes, phonemes, audio) in enumerate(
            pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
        ):
            # Track per-segment generation time
            segment_time = time.time() - start_time

            samples = audio.shape[0] if audio is not None else 0
            assert samples > 0, "No audio generated"

            # Calculate token count
            token_count = len(phonemes) if phonemes is not None else 0

            # Calculate audio duration in seconds
            sample_rate = self.config.sample_rate

            audio_duration_seconds = samples / sample_rate * audio.shape[1]

            # Calculate real-time factor (RTF)
            rtf = (
                segment_time / audio_duration_seconds
                if audio_duration_seconds > 0
                else 0
            )

            # Format duration as HH:MM:SS.mmm
            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=audio[0],
                samples=samples,
                sample_rate=sample_rate,
                segment_idx=segment_idx,
                token_count=token_count,
                audio_duration=duration_str,
                real_time_factor=round(rtf, 2),
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": (
                        round(token_count / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                processing_time_seconds=segment_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            # Clear cache after each segment to avoid memory leaks
            mx.clear_cache()

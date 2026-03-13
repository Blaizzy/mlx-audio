import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.utils import load_audio

from .config import ModelConfig


def _compute_fbank(
    waveform: mx.array,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
    window: str = "hamming",
) -> mx.array:
    from mlx_audio.dsp import compute_fbank_kaldi

    win_len = int(sample_rate * frame_length_ms / 1000)
    win_inc = int(sample_rate * frame_shift_ms / 1000)

    waveform = waveform * (1 << 15)

    return compute_fbank_kaldi(
        waveform,
        sample_rate=sample_rate,
        win_len=win_len,
        win_inc=win_inc,
        num_mels=n_mels,
        win_type=window,
        preemphasis=0.97,
        dither=0.0,
        snip_edges=True,
        low_freq=20.0,
        high_freq=0.0,
    )


def _apply_lfr(feats: mx.array, lfr_m: int = 7, lfr_n: int = 6) -> mx.array:
    T, D = feats.shape
    T_lfr = math.ceil(T / lfr_n)

    # left-pad with copies of the first frame
    left_pad = (lfr_m - 1) // 2
    if left_pad > 0:
        feats = mx.concatenate([mx.tile(feats[:1], (left_pad, 1)), feats], axis=0)
    T_padded = feats.shape[0]

    frames = []
    for i in range(T_lfr):
        start = i * lfr_n
        end = start + lfr_m
        if end <= T_padded:
            stacked = feats[start:end].reshape(-1)
        else:
            # right-pad with copies of the last frame
            available = feats[start:T_padded]
            pad_count = lfr_m - available.shape[0]
            padded = mx.concatenate(
                [available, mx.tile(feats[-1:], (pad_count, 1))], axis=0
            )
            stacked = padded.reshape(-1)
        frames.append(stacked)
    return mx.stack(frames, axis=0)


def _apply_cmvn(
    feats: mx.array,
    means: mx.array,
    istd: mx.array,
) -> mx.array:
    return (feats + means) * istd


def _parse_am_mvn(path: str) -> Tuple[List[float], List[float]]:
    with open(path) as f:
        text = f.read()

    # Extract AddShift values (negative means)
    shift_match = re.search(
        r"<AddShift>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]", text, re.DOTALL
    )
    if not shift_match:
        raise ValueError("Could not parse AddShift from am.mvn")
    means = [float(x) for x in shift_match.group(1).split()]

    # Extract Rescale values (inverse std)
    rescale_match = re.search(
        r"<Rescale>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]", text, re.DOTALL
    )
    if not rescale_match:
        raise ValueError("Could not parse Rescale from am.mvn")
    istd = [float(x) for x in rescale_match.group(1).split()]

    return means, istd


class SinusoidalPositionEncoder(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        batch_size, timesteps, input_dim = x.shape
        positions = mx.arange(1, timesteps + 1)[None, :]  # (1, T)

        # Compute inverse timescales
        half_dim = input_dim // 2
        log_timescale_increment = math.log(10000) / (half_dim - 1)
        inv_timescales = mx.exp(mx.arange(half_dim) * (-log_timescale_increment))
        inv_timescales = mx.broadcast_to(
            inv_timescales[None, :], (batch_size, half_dim)
        )

        scaled_time = positions[:, :, None] * inv_timescales[:, None, :]
        encoding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=2)

        return x + encoding


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int, dropout_rate: float = 0.0):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(nn.relu(self.w_1(x)))


class MultiHeadedAttentionSANM(nn.Module):
    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        kernel_size: int = 11,
        sanm_shift: int = 0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)

        self.fsmn_block = nn.Conv1d(
            n_feat,
            n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=n_feat,
            bias=False,
        )

        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        self.left_padding = left_padding
        self.right_padding = kernel_size - 1 - left_padding

    def _forward_fsmn(self, inputs: mx.array) -> mx.array:
        x = mx.pad(
            inputs,
            pad_width=((0, 0), (self.left_padding, self.right_padding), (0, 0)),
        )
        x = self.fsmn_block(x)
        x = x + inputs
        return x

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape

        q_k_v = self.linear_q_k_v(x)
        q, k, v = mx.split(q_k_v, 3, axis=-1)

        fsmn_memory = self._forward_fsmn(v)

        q = q.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        v_h = v.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)

        scores = (q * (self.d_k**-0.5)) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(scores, axis=-1)
        att_out = attn @ v_h

        att_out = att_out.transpose(0, 2, 1, 3).reshape(B, T, self.n_feat)
        att_out = self.linear_out(att_out)

        return att_out + fsmn_memory


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size: int,
        size: int,
        self_attn: MultiHeadedAttentionSANM,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: float = 0.0,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(size)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        attn_out = self.self_attn(x)

        if self.in_size == self.size:
            x = residual + attn_out
        else:
            x = attn_out

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x = residual + self.feed_forward(x)
        return x


class SenseVoiceEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        enc = config.encoder_conf
        input_size = config.input_size
        output_size = enc.output_size

        self.embed = SinusoidalPositionEncoder()
        self._output_size = output_size

        self.encoders0 = [
            EncoderLayerSANM(
                in_size=input_size,
                size=output_size,
                self_attn=MultiHeadedAttentionSANM(
                    n_head=enc.attention_heads,
                    in_feat=input_size,
                    n_feat=output_size,
                    dropout_rate=enc.attention_dropout_rate,
                    kernel_size=enc.kernel_size,
                    sanm_shift=enc.sanm_shift,
                ),
                feed_forward=PositionwiseFeedForward(
                    idim=output_size,
                    hidden_units=enc.linear_units,
                    dropout_rate=enc.dropout_rate,
                ),
                dropout_rate=enc.dropout_rate,
                normalize_before=enc.normalize_before,
            )
        ]

        self.encoders = [
            EncoderLayerSANM(
                in_size=output_size,
                size=output_size,
                self_attn=MultiHeadedAttentionSANM(
                    n_head=enc.attention_heads,
                    in_feat=output_size,
                    n_feat=output_size,
                    dropout_rate=enc.attention_dropout_rate,
                    kernel_size=enc.kernel_size,
                    sanm_shift=enc.sanm_shift,
                ),
                feed_forward=PositionwiseFeedForward(
                    idim=output_size,
                    hidden_units=enc.linear_units,
                    dropout_rate=enc.dropout_rate,
                ),
                dropout_rate=enc.dropout_rate,
                normalize_before=enc.normalize_before,
            )
            for _ in range(enc.num_blocks - 1)
        ]

        self.after_norm = nn.LayerNorm(output_size)

        self.tp_encoders = [
            EncoderLayerSANM(
                in_size=output_size,
                size=output_size,
                self_attn=MultiHeadedAttentionSANM(
                    n_head=enc.attention_heads,
                    in_feat=output_size,
                    n_feat=output_size,
                    dropout_rate=enc.attention_dropout_rate,
                    kernel_size=enc.kernel_size,
                    sanm_shift=enc.sanm_shift,
                ),
                feed_forward=PositionwiseFeedForward(
                    idim=output_size,
                    hidden_units=enc.linear_units,
                    dropout_rate=enc.dropout_rate,
                ),
                dropout_rate=enc.dropout_rate,
                normalize_before=enc.normalize_before,
            )
            for _ in range(enc.tp_blocks)
        ]

        self.tp_norm = nn.LayerNorm(output_size)

    def __call__(self, xs_pad: mx.array) -> mx.array:
        xs_pad = xs_pad * (self._output_size**0.5)
        xs_pad = self.embed(xs_pad)

        for layer in self.encoders0:
            xs_pad = layer(xs_pad)

        for layer in self.encoders:
            xs_pad = layer(xs_pad)

        xs_pad = self.after_norm(xs_pad)

        for layer in self.tp_encoders:
            xs_pad = layer(xs_pad)

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad


class SenseVoiceSmall(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = SenseVoiceEncoder(config)

        self.ctc_lo = nn.Linear(config.encoder_conf.output_size, config.vocab_size)
        self.embed = nn.Embedding(16, config.input_size)

        self.lid_dict = {
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        }
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.emo_dict = {
            "unk": 25009,
            "happy": 25001,
            "sad": 25002,
            "angry": 25003,
            "neutral": 25004,
        }

        self.blank_id = 0

        # CMVN stats (loaded in post_load_hook)
        self._cmvn_means: Optional[mx.array] = None
        self._cmvn_istd: Optional[mx.array] = None

        # Tokenizer (loaded in post_load_hook)
        self._tokenizer = None
        self._token_list: Optional[List[str]] = None

    def _extract_features(self, audio: mx.array) -> mx.array:
        fc = self.config.frontend_conf

        fbank = _compute_fbank(
            audio,
            sample_rate=fc.fs,
            n_mels=fc.n_mels,
            frame_length_ms=fc.frame_length,
            frame_shift_ms=fc.frame_shift,
            window=fc.window,
        )

        feats = _apply_lfr(fbank, lfr_m=fc.lfr_m, lfr_n=fc.lfr_n)

        if self._cmvn_means is not None and self._cmvn_istd is not None:
            feats = _apply_cmvn(feats, self._cmvn_means, self._cmvn_istd)

        return feats

    def _build_query(
        self,
        batch_size: int,
        language: str = "auto",
        use_itn: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        lid = self.lid_dict.get(language, 0)
        language_query = self.embed(mx.array([[lid]]))

        textnorm = "withitn" if use_itn else "woitn"
        textnorm_id = self.textnorm_dict[textnorm]
        textnorm_query = self.embed(mx.array([[textnorm_id]]))

        event_emo_query = self.embed(mx.array([[1, 2]]))

        if batch_size > 1:
            language_query = mx.broadcast_to(
                language_query, (batch_size,) + language_query.shape[1:]
            )
            textnorm_query = mx.broadcast_to(
                textnorm_query, (batch_size,) + textnorm_query.shape[1:]
            )
            event_emo_query = mx.broadcast_to(
                event_emo_query, (batch_size,) + event_emo_query.shape[1:]
            )

        input_query = mx.concatenate([language_query, event_emo_query], axis=1)
        return textnorm_query, input_query

    def __call__(
        self, feats: mx.array, language: str = "auto", use_itn: bool = False
    ) -> mx.array:
        B = feats.shape[0]
        textnorm_query, input_query = self._build_query(B, language, use_itn)

        speech = mx.concatenate([textnorm_query, feats], axis=1)
        speech = mx.concatenate([input_query, speech], axis=1)

        encoder_out = self.encoder(speech)
        logits = self.ctc_lo(encoder_out)
        return nn.log_softmax(logits, axis=-1)

    def _decode_tokens(self, token_ids: List[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)
        if self._token_list is not None:
            pieces = [
                self._token_list[t] for t in token_ids if 0 <= t < len(self._token_list)
            ]
            text = "".join(pieces).replace("▁", " ").strip()
            return text
        return " ".join(str(t) for t in token_ids)

    def _greedy_ctc_decode(self, log_probs: mx.array) -> Tuple[List[int], str]:
        pred = mx.argmax(log_probs, axis=-1)
        pred_list = pred.tolist()

        deduped = []
        prev = None
        for t in pred_list:
            if t != prev:
                deduped.append(t)
                prev = t

        token_ids = [t for t in deduped if t != self.blank_id]
        text = self._decode_tokens(token_ids)
        return token_ids, text

    def _extract_rich_info(self, log_probs: mx.array) -> Dict[str, str]:
        info = {}

        lid_pred = mx.argmax(log_probs[0]).item()
        lid_map = {
            24884: "zh",
            24885: "en",
            24888: "yue",
            24892: "ja",
            24896: "ko",
            24992: "nospeech",
        }
        info["language"] = lid_map.get(lid_pred, "unknown")

        emo_pred = mx.argmax(log_probs[1]).item()
        emo_map = {
            25001: "happy",
            25002: "sad",
            25003: "angry",
            25004: "neutral",
            25005: "fearful",
            25006: "disgusted",
            25007: "surprised",
            25008: "other",
            25009: "unk",
        }
        info["emotion"] = emo_map.get(emo_pred, f"token_{emo_pred}")

        event_pred = mx.argmax(log_probs[2]).item()
        event_map = {
            24993: "Speech",
            24995: "BGM",
            24997: "Laughter",
            24999: "Applause",
        }
        info["event"] = event_map.get(event_pred, f"token_{event_pred}")

        return info

    def generate(
        self,
        audio: Union[str, Path, mx.array, np.ndarray],
        *,
        language: str = "auto",
        use_itn: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        kwargs.pop("max_tokens", None)
        kwargs.pop("generation_stream", None)
        kwargs.pop("dtype", None)

        if isinstance(audio, (str, Path)):
            audio_data = load_audio(str(audio), sr=self.config.frontend_conf.fs)
        elif isinstance(audio, np.ndarray):
            audio_data = mx.array(audio, dtype=mx.float32)
        elif isinstance(audio, mx.array):
            audio_data = audio
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        feats = self._extract_features(audio_data)
        feats = feats[None, :, :]

        log_probs = self(feats, language=language, use_itn=use_itn)
        log_probs = log_probs[0]

        rich_info = self._extract_rich_info(log_probs[:4])
        token_ids, text = self._greedy_ctc_decode(log_probs[4:])

        if verbose:
            print(f"Language: {rich_info.get('language', '?')}")
            print(f"Emotion: {rich_info.get('emotion', '?')}")
            print(f"Event: {rich_info.get('event', '?')}")
            print(f"Text: {text}")

        return STTOutput(
            text=text,
            language=rich_info.get("language"),
            segments=[
                {
                    "text": text,
                    "language": rich_info.get("language"),
                    "emotion": rich_info.get("emotion"),
                    "event": rich_info.get("event"),
                }
            ],
        )

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        new_weights = {}
        for k, v in weights.items():
            new_key = k
            new_key = new_key.replace("ctc.ctc_lo.", "ctc_lo.")

            if "fsmn_block.weight" in new_key and v.ndim == 3:
                v = v.transpose(0, 2, 1)

            new_weights[new_key] = v
        return new_weights

    @staticmethod
    def post_load_hook(model: "SenseVoiceSmall", model_path: Path) -> "SenseVoiceSmall":
        model_path = Path(model_path)

        mvn_path = model_path / "am.mvn"
        if mvn_path.exists():
            means, istd = _parse_am_mvn(str(mvn_path))
            model._cmvn_means = mx.array(means)
            model._cmvn_istd = mx.array(istd)

        if model._cmvn_means is None and model.config.cmvn_means is not None:
            model._cmvn_means = mx.array(model.config.cmvn_means)
            model._cmvn_istd = mx.array(model.config.cmvn_istd)

        bpe_path = model_path / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        tokens_path = model_path / "tokens.json"

        if bpe_path.exists():
            try:
                import sentencepiece as spm

                sp = spm.SentencePieceProcessor()
                sp.Load(str(bpe_path))
                model._tokenizer = sp
            except ImportError:
                pass

        if model._tokenizer is None and tokens_path.exists():
            with open(tokens_path) as f:
                model._token_list = json.load(f)

        return model

# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""Confucius4-TTS: multilingual, cross-lingual, zero-shot voice-cloning TTS in MLX.

Pipeline: w2v-bert semantic features + CAMPPlus speaker emb -> T2S (GPT-2) ->
S2A flow-matching (DiT+WaveNet) -> BigVGAN vocoder. All MLX; ref-mel DSP numpy.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult
from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus

from .prefix import T2SPrefixMLX
from .s2a import S2AEstimator
from .t2s import T2SMLX
from .vocoder import BigVGANMLX
from .w2vbert import W2VBertMLX

LANGUAGE_TOKEN = {  # subset; matches Confucius LANGUAGE_TOKEN_MAP
    "zh": "请用中文朗读接下来的文字",
    "en": "请用英文朗读接下来的文字",
    "vi": "请用越南语朗读接下来的文字",
    "ja": "请用日语朗读接下来的文字",
    "ko": "请用韩语朗读接下来的文字",
    "th": "请用泰语朗读接下来的文字",
}


@dataclass
class ModelConfig(BaseModelArgs):
    model_path: str = ""
    sample_rate: int = 22050
    model_type: str = "confucius4"
    quant_bits: int = 8  # bits used if T2S weights are quantized (int8/int4)
    quant_group_size: int = 64


def _slaney_mel(sr, n_fft, n_mels):
    """librosa.filters.mel(htk=False, norm="slaney") reimplemented in numpy."""
    n_freqs = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_freqs)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(f):
        f = np.asarray(f, dtype=float)
        mel = f / f_sp
        log = f >= min_log_hz
        mel[log] = min_log_mel + np.log(f[log] / min_log_hz) / logstep
        return mel

    def mel_to_hz(m):
        f = f_sp * m
        log = m >= min_log_mel
        f[log] = min_log_hz * np.exp(logstep * (m[log] - min_log_mel))
        return f

    mpts = np.linspace(0.0, hz_to_mel([sr / 2])[0], n_mels + 2)
    fpts = mel_to_hz(mpts)
    fdiff = np.diff(fpts)
    ramps = fpts[:, None] - fftfreqs[None, :]
    w = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (fpts[2 : n_mels + 2] - fpts[:n_mels]))[:, None]  # slaney norm
    return w.astype(np.float32)


_REF_MEL_FB = _slaney_mel(22050, 1024, 80)


def _ref_mel(audio16k):
    from mlx_audio.utils import resample_audio

    SR, NFFT, HOP, WIN = 22050, 1024, 256, 1024
    a = np.asarray(resample_audio(audio16k, 16000, SR))
    hann = np.hanning(WIN + 1)[:-1].astype(np.float32)
    pad = (NFFT - HOP) // 2
    y = np.pad(a, (pad, pad), mode="reflect")
    nfr = 1 + (len(y) - NFFT) // HOP
    fr = np.stack([y[i * HOP : i * HOP + NFFT] * hann for i in range(nfr)], 0)
    spec = np.sqrt(np.abs(np.fft.rfft(fr, NFFT, axis=1)).T ** 2 + 1e-9)
    return np.log(np.clip(_REF_MEL_FB @ spec, 1e-5, None)).T[None].astype(np.float32)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        d = Path(config.model_path)
        self.w2v = W2VBertMLX(
            str(d / "w2vbert_mlx.safetensors"),
            group_size=config.quant_group_size,
            bits=config.quant_bits,
        )
        self.prefix = T2SPrefixMLX(str(d / "t2s_model.safetensors"))
        self.t2s = T2SMLX(
            str(d / "t2s_model.safetensors"),
            group_size=config.quant_group_size,
            bits=config.quant_bits,
        )
        self.s2a = S2AEstimator(str(d / "s2a_mlx.safetensors"))
        self.voc = BigVGANMLX(str(d / "bigvgan_mlx.safetensors"))
        self.stats = np.load(str(d / "w2v_stats.npz"))
        from mlx.utils import tree_unflatten

        self.camp = CAMPPlus(feat_dim=80, embedding_size=192)
        self.camp.update(
            tree_unflatten(list(mx.load(str(d / "campplus.safetensors")).items()))
        )
        mx.eval(self.camp.parameters())
        self.camp.eval()
        # torch-free preprocessing: numpy fbank + `tokenizers` (Rust) tokenizer
        from tokenizers import Tokenizer

        from .features import fbank_160

        self._fbank = fbank_160
        ff = np.load(str(d / "fbank_filters.npz"))
        self._mel, self._win = mx.array(ff["mel"]), mx.array(ff["window"])
        self._tok = Tokenizer.from_file(str(d / "checkpoints" / "tokenizer.json"))

    def sanitize(self, weights):
        return {}  # sub-weights are loaded from the model dir in __init__

    def load_weights(self, *args, **kwargs):
        # Components self-load from the model dir in __init__; the standard
        # tree loader is bypassed. No-op keeps mlx_audio.tts.utils.load() happy.
        return self

    def generate(
        self,
        text: str,
        ref_audio: str,
        lang: str = "vi",
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.8,
        repetition_penalty: float = 10.0,
        seed: int = 0,
        **kwargs,
    ):
        from mlx_audio.utils import load_audio

        t0 = time.time()
        # load_audio decodes via audio_io and resamples to 16 kHz mono with the
        # repo resampler; fbank_160, CAMPPlus and _ref_mel all assume 16 kHz.
        # Without the resample a 44.1/48 kHz ref is misread as 16 kHz and the
        # downstream mel is off by sr/16000, producing garbled audio.
        audio = np.asarray(load_audio(ref_audio, sample_rate=16000))

        feats = self._fbank(mx.array(audio), self._mel, self._win)
        h17 = np.array(self.w2v.hidden17(feats))
        cond_vec = mx.array((h17 - self.stats["mean"]) / self.stats["std"])
        style = mx.array(np.array(self.camp.inference(mx.array(audio))).reshape(1, 192))
        ref_mel = mx.array(_ref_mel(audio))

        lt = LANGUAGE_TOKEN.get(lang, LANGUAGE_TOKEN["en"])
        ids = self._tok.encode(f"You are a helpful assistant. {lt}:{text}").ids
        cond_emb = self.prefix.cond_emb(cond_vec)
        text_emb = self.prefix.text_emb(mx.array([ids]))
        codes, latent = self.t2s.generate(
            cond_emb,
            text_emb,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_pen=repetition_penalty,
            seed=seed,
        )

        T_ref = ref_mel.shape[1]
        mu = self.s2a.build_mu(mx.array(codes[None]), mx.array(latent), T_ref)
        mx.random.seed(seed)
        z = mx.random.normal((1, 80, mu.shape[1]))
        mel = self.s2a.solve_euler(
            z,
            mx.transpose(ref_mel, (0, 2, 1)),
            mu,
            style,
            mx.linspace(0, 1, 26),
            cfg=0.7,
        )[:, :, T_ref:]
        wav = self.voc(mel)
        mx.eval(wav)
        wav = mx.array(np.array(wav).reshape(-1))

        samples = wav.shape[0]
        dt = time.time() - t0
        dur = samples / self.sample_rate
        yield GenerationResult(
            audio=wav,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(codes),
            audio_duration=f"{dur:.2f}s",
            real_time_factor=round(dt / dur, 2) if dur else 0.0,
            prompt={"tokens": len(codes)},
            audio_samples={"samples": samples},
            processing_time_seconds=round(dt, 2),
            peak_memory_usage=0.0,
            is_final_chunk=True,
        )

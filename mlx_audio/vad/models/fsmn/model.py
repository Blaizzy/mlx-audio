"""
FSMN-VAD: frontend + encoder + postprocess.

Usage:
    from mlx_audio.vad import load
    model = load("mlx-community/fsmn-vad")
    segments = model.detect("test.wav")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import FSMNEncoderConfig, ModelConfig
from .encoder import FSMNEncoder
from .frontend import extract_features
from .postprocess import VADPostProcess, VADXOptions


class Model(nn.Module):
    """FSMN-VAD: complete VAD pipeline."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = FSMNEncoder(config.encoder)

        opts = VADXOptions(
            sample_rate=config.sample_rate,
            frame_in_ms=config.frame_in_ms,
            frame_length_ms=config.frame_length,
            window_size_ms=config.window_size_ms,
            sil_to_speech_time_thres=config.sil_to_speech_time_thres,
            speech_to_sil_time_thres=config.speech_to_sil_time_thres,
            speech_noise_thres=config.speech_noise_thres,
            max_end_silence_time=config.max_end_silence_time,
            max_start_silence_time=config.max_start_silence_time,
            sil_pdf_ids=config.sil_pdf_ids,
        )
        self.postprocess = VADPostProcess(opts)

        self._cmvn_shift: Optional[np.ndarray] = None
        self._cmvn_scale: Optional[np.ndarray] = None

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        for k, v in weights.items():
            if not k.startswith("encoder."):
                k = f"encoder.{k}"
            sanitized[k] = v
        return sanitized

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], **kwargs) -> "Model":
        """Load model from a local directory or HuggingFace repo.

        Args:
            model_path: local path or HuggingFace repo ID
                (e.g. "mlx-community/fsmn-vad")
        """
        from mlx_audio.utils import get_model_path

        model_path = get_model_path(str(model_path), **kwargs)

        with open(model_path / "config.json") as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)

        model = cls(config)

        weights = mx.load(str(model_path / "model.safetensors"))
        model.encoder.load_weights(list(weights.items()))

        cmvn_json = model_path / "cmvn.json"
        cmvn_mvn = model_path / "am.mvn"
        if cmvn_json.exists():
            with open(cmvn_json) as f:
                cmvn = json.load(f)
            model._cmvn_shift = np.array(cmvn["shift"], dtype=np.float32)
            model._cmvn_scale = np.array(cmvn["scale"], dtype=np.float32)
        elif cmvn_mvn.exists():
            from .frontend import load_cmvn

            shift, scale = load_cmvn(str(cmvn_mvn))
            model._cmvn_shift = shift
            model._cmvn_scale = scale

        return model

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        cmvn_path = Path(model_path) / "cmvn.json"
        if cmvn_path.exists():
            with open(cmvn_path) as f:
                cmvn = json.load(f)
            model._cmvn_shift = np.array(cmvn["shift"], dtype=np.float32)
            model._cmvn_scale = np.array(cmvn["scale"], dtype=np.float32)
        return model

    def detect(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: int = 16000,
    ) -> List[List[int]]:
        """
        Detect speech segments in audio.

        Args:
            audio: path to audio file or float32 numpy waveform
            sample_rate: audio sample rate (used only when audio is numpy)

        Returns:
            [[start_ms, end_ms], ...] speech segment timestamps
        """
        if isinstance(audio, str):
            from mlx_audio.audio_io import read as audio_read
            from mlx_audio.utils import resample_audio

            waveform, sr = audio_read(audio, dtype="float32")
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=-1)
            if sr != self.config.sample_rate:
                waveform = resample_audio(waveform, sr, self.config.sample_rate).astype(
                    np.float32
                )
        else:
            waveform = audio.astype(np.float32)

        features = extract_features(
            waveform,
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            frame_length_ms=self.config.frame_length,
            frame_shift_ms=self.config.frame_shift,
            lfr_m=self.config.lfr_m,
            lfr_n=self.config.lfr_n,
            cmvn_shift=self._cmvn_shift,
            cmvn_scale=self._cmvn_scale,
        )

        x = mx.array(features[np.newaxis, :, :])
        scores = self.encoder(x)
        mx.eval(scores)
        scores_np = np.array(scores)

        cache = self.postprocess.init_cache()
        segments = self.postprocess.forward(
            scores=scores_np,
            waveform=waveform,
            cache=cache,
            is_final=True,
        )

        return segments

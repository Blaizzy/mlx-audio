from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.audio_io import read as audio_read
from mlx_audio.utils import resample_audio

from .config import BranchConfig, ModelConfig


@dataclass
class SileroVADState:
    state: Optional[mx.array]
    context: mx.array
    sample_rate: int


@dataclass
class VADOutput:
    timestamps: List[dict]
    probabilities: mx.array
    sample_rate: int


def _reflect_pad_right(x: mx.array, pad: int) -> mx.array:
    if pad <= 0:
        return x
    if x.shape[-1] <= pad:
        raise ValueError(
            f"Reflect padding of {pad} requires more than {pad} samples, "
            f"got {x.shape[-1]}"
        )

    indices = mx.arange(x.shape[-1] - 2, x.shape[-1] - pad - 2, -1)
    reflected = mx.take(x, indices, axis=-1)
    return mx.concatenate([x, reflected], axis=-1)


class SileroVADBranch(nn.Module):
    def __init__(self, config: BranchConfig):
        super().__init__()
        self.config = config
        self.stft_conv = nn.Conv1d(
            1,
            config.cutoff * 2,
            kernel_size=config.filter_length,
            stride=config.hop_length,
            padding=0,
            bias=False,
        )
        self.conv1 = nn.Conv1d(config.cutoff, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 128)
        self.final_conv = nn.Conv1d(128, 1, kernel_size=1)

    def __call__(
        self,
        x: mx.array,
        state: Optional[Union[mx.array, Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, mx.array]:
        if x.ndim == 1:
            x = x[None, :]

        hidden, cell = self._split_state(state)
        x = _reflect_pad_right(x, self.config.pad)
        x = self.stft_conv(x[..., None])

        real = x[..., : self.config.cutoff]
        imag = x[..., self.config.cutoff :]
        x = mx.sqrt(real * real + imag * imag)

        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = nn.relu(self.conv4(x))

        hidden_seq, cell_seq = self.lstm(x, hidden=hidden, cell=cell)
        hidden = hidden_seq[:, -1, :]
        cell = cell_seq[:, -1, :]
        new_state = mx.stack([hidden, cell], axis=0)

        x = nn.relu(hidden_seq)
        x = mx.sigmoid(self.final_conv(x))
        x = mx.mean(mx.squeeze(x, axis=-1), axis=1, keepdims=True)
        return x, new_state

    @staticmethod
    def _split_state(
        state: Optional[Union[mx.array, Tuple[mx.array, mx.array]]],
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        if state is None:
            return None, None
        if isinstance(state, tuple):
            return state[0], state[1]
        if state.ndim != 3 or state.shape[0] != 2:
            raise ValueError(f"Expected state shape (2, batch, 128), got {state.shape}")
        return state[0], state[1]


class Model(nn.Module):
    """
    Silero voice activity detector.

    The low-level call expects one audio window that already includes the model
    context: 576 samples for 16 kHz, or 288 samples for 8 kHz. Use
    ``predict_proba`` or ``feed`` to manage context automatically.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dtype = mx.float16 if config.dtype == "float16" else mx.float32
        self.vad_16k = SileroVADBranch(config.branch_16k)
        self.vad_8k = SileroVADBranch(config.branch_8k)

    def __call__(
        self,
        x: Union[np.ndarray, mx.array],
        state: Optional[Union[mx.array, Tuple[mx.array, mx.array]]] = None,
        sample_rate: int = 16000,
    ) -> Tuple[mx.array, mx.array]:
        branch = self._branch(sample_rate)
        x = (
            mx.array(x, dtype=self.dtype)
            if not isinstance(x, mx.array)
            else x.astype(self.dtype)
        )
        if state is not None and not isinstance(state, tuple):
            state = (
                mx.array(state, dtype=self.dtype)
                if not isinstance(state, mx.array)
                else state.astype(self.dtype)
            )
        elif isinstance(state, tuple):
            state = tuple(
                (
                    mx.array(part, dtype=self.dtype)
                    if not isinstance(part, mx.array)
                    else part.astype(self.dtype)
                )
                for part in state
            )
        return branch(x, state=state)

    def initial_state(
        self, batch_size: int = 1, sample_rate: int = 16000
    ) -> SileroVADState:
        branch = self._branch(sample_rate)
        context = mx.zeros((batch_size, branch.config.context_size), dtype=self.dtype)
        return SileroVADState(state=None, context=context, sample_rate=sample_rate)

    def reset_state(
        self, batch_size: int = 1, sample_rate: int = 16000
    ) -> SileroVADState:
        return self.initial_state(batch_size=batch_size, sample_rate=sample_rate)

    def feed(
        self,
        chunk: Union[np.ndarray, mx.array],
        state: Optional[SileroVADState] = None,
        sample_rate: int = 16000,
    ) -> Tuple[mx.array, SileroVADState]:
        branch = self._branch(sample_rate)
        chunk_mx = (
            mx.array(chunk, dtype=self.dtype)
            if not isinstance(chunk, mx.array)
            else chunk.astype(self.dtype)
        )
        if chunk_mx.ndim == 1:
            chunk_mx = chunk_mx[None, :]
        if chunk_mx.shape[-1] != branch.config.chunk_size:
            raise ValueError(
                f"Expected {branch.config.chunk_size} samples at {sample_rate} Hz, "
                f"got {chunk_mx.shape[-1]}"
            )

        if state is None:
            state = self.initial_state(chunk_mx.shape[0], sample_rate=sample_rate)
        if state.sample_rate != sample_rate:
            raise ValueError(
                f"Streaming state is for {state.sample_rate} Hz, got {sample_rate} Hz"
            )

        window = mx.concatenate([state.context, chunk_mx], axis=-1)
        probability, lstm_state = self(
            window, state=state.state, sample_rate=sample_rate
        )
        new_context = chunk_mx[:, -branch.config.context_size :]
        return probability, SileroVADState(
            state=lstm_state, context=new_context, sample_rate=sample_rate
        )

    def predict(self, audio, sample_rate: Optional[int] = None) -> mx.array:
        return self.predict_proba(audio, sample_rate=sample_rate)

    def predict_proba(
        self,
        audio: Union[str, np.ndarray, mx.array],
        sample_rate: Optional[int] = None,
    ) -> mx.array:
        audio_array, sr = self._prepare_audio_array(audio, sample_rate=sample_rate)
        return self._predict_proba_array(audio_array, sr)

    def get_speech_timestamps(
        self,
        audio: Union[str, np.ndarray, mx.array],
        sample_rate: Optional[int] = None,
        threshold: Optional[float] = None,
        min_speech_duration_ms: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
        speech_pad_ms: Optional[int] = None,
        return_seconds: bool = False,
    ) -> List[dict]:
        audio_array, sr = self._prepare_audio_array(audio, sample_rate=sample_rate)
        probabilities = self._predict_proba_array(audio_array, sr)
        mx.eval(probabilities)
        return self._probs_to_timestamps(
            probabilities,
            audio_len=audio_array.shape[-1],
            sample_rate=sr,
            threshold=self.config.threshold if threshold is None else threshold,
            min_speech_duration_ms=(
                self.config.min_speech_duration_ms
                if min_speech_duration_ms is None
                else min_speech_duration_ms
            ),
            min_silence_duration_ms=(
                self.config.min_silence_duration_ms
                if min_silence_duration_ms is None
                else min_silence_duration_ms
            ),
            speech_pad_ms=(
                self.config.speech_pad_ms if speech_pad_ms is None else speech_pad_ms
            ),
            return_seconds=return_seconds,
        )

    def generate(self, audio, sample_rate: Optional[int] = None, **kwargs) -> VADOutput:
        audio_array, sr = self._prepare_audio_array(audio, sample_rate=sample_rate)
        probabilities = self._predict_proba_array(audio_array, sr)
        mx.async_eval(probabilities)
        timestamps = self._probs_to_timestamps(
            probabilities,
            audio_len=audio_array.shape[-1],
            sample_rate=sr,
            threshold=kwargs.pop("threshold", self.config.threshold),
            min_speech_duration_ms=kwargs.pop(
                "min_speech_duration_ms", self.config.min_speech_duration_ms
            ),
            min_silence_duration_ms=kwargs.pop(
                "min_silence_duration_ms", self.config.min_silence_duration_ms
            ),
            speech_pad_ms=kwargs.pop("speech_pad_ms", self.config.speech_pad_ms),
            return_seconds=kwargs.pop("return_seconds", False),
        )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        return VADOutput(
            timestamps=timestamps, probabilities=probabilities, sample_rate=sr
        )

    def _predict_proba_array(
        self,
        audio_array: Union[np.ndarray, mx.array],
        sample_rate: int,
        eval_every: int = 16,
    ) -> mx.array:
        if eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {eval_every}")

        branch = self._branch(sample_rate)
        chunk_size = branch.config.chunk_size
        context_size = branch.config.context_size
        audio_mx = (
            audio_array.astype(self.dtype)
            if isinstance(audio_array, mx.array)
            else mx.array(audio_array, dtype=self.dtype)
        )
        original_ndim = audio_mx.ndim

        if original_ndim == 1:
            audio_mx = audio_mx[None, :]

        if audio_mx.shape[-1] == 0:
            return (
                mx.zeros((0,), dtype=self.dtype)
                if original_ndim == 1
                else mx.zeros((audio_mx.shape[0], 0), dtype=self.dtype)
            )

        pad = (chunk_size - audio_mx.shape[-1] % chunk_size) % chunk_size
        if pad:
            audio_mx = mx.pad(audio_mx, [(0, 0), (0, pad)])

        context = mx.zeros((audio_mx.shape[0], context_size), dtype=self.dtype)
        audio_mx = mx.concatenate([context, audio_mx], axis=-1)

        outputs = []
        state = None
        for step, pos in enumerate(
            range(context_size, audio_mx.shape[-1], chunk_size), start=1
        ):
            window = audio_mx[:, pos - context_size : pos + chunk_size]
            out, state = self(window, state=state, sample_rate=sample_rate)
            outputs.append(out)
            if step % eval_every == 0:
                mx.async_eval(out, state)

        if outputs and len(outputs) % eval_every:
            mx.async_eval(outputs[-1], state)

        probabilities = mx.concatenate(outputs, axis=1)
        if original_ndim == 1:
            probabilities = probabilities[0]
        return probabilities

    def _prepare_audio_array(
        self,
        audio: Union[str, np.ndarray, mx.array],
        sample_rate: Optional[int] = None,
    ) -> Tuple[mx.array, int]:
        if isinstance(audio, str):
            waveform, sr = audio_read(audio, dtype="float32")
            audio_array = mx.array(waveform, dtype=mx.float32)
        elif isinstance(audio, mx.array):
            audio_array = audio.astype(mx.float32)
            sr = 16000 if sample_rate is None else int(sample_rate)
        else:
            audio_array = mx.array(audio, dtype=mx.float32)
            sr = 16000 if sample_rate is None else int(sample_rate)

        if audio_array.ndim == 2 and audio_array.shape[-1] <= 8 < audio_array.shape[0]:
            audio_array = mx.mean(audio_array, axis=-1)
        if audio_array.ndim not in (1, 2):
            raise ValueError(
                f"Expected mono or batched audio, got shape {audio_array.shape}"
            )

        target_sr = sr if sr in (8000, 16000) else 16000
        if sr != target_sr:
            if audio_array.size:
                audio_array = resample_audio(audio_array, sr, target_sr, axis=-1)
            sr = target_sr

        return audio_array.astype(mx.float32), sr

    def _branch(self, sample_rate: int) -> SileroVADBranch:
        if int(sample_rate) == 16000:
            return self.vad_16k
        if int(sample_rate) == 8000:
            return self.vad_8k
        raise ValueError("Silero VAD supports 8000 Hz and 16000 Hz audio")

    @staticmethod
    def _probs_to_timestamps(
        probabilities: Union[np.ndarray, mx.array],
        audio_len: int,
        sample_rate: int,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        return_seconds: bool,
    ) -> List[dict]:
        probs = probabilities[0] if probabilities.ndim == 2 else probabilities
        chunk_size = 512 if sample_rate == 16000 else 256
        min_speech_samples = sample_rate * min_speech_duration_ms / 1000
        min_silence_samples = sample_rate * min_silence_duration_ms / 1000
        speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)
        neg_threshold = max(threshold - 0.15, 0.01)

        speeches = []
        triggered = False
        current_start = 0
        temp_end = 0

        for idx, prob in enumerate(probs.tolist()):
            chunk_start = idx * chunk_size

            if prob >= threshold and not triggered:
                triggered = True
                current_start = chunk_start
                temp_end = 0
                continue

            if triggered and prob >= threshold:
                temp_end = 0
                continue

            if triggered and prob < neg_threshold:
                if temp_end == 0:
                    temp_end = chunk_start
                if chunk_start - temp_end >= min_silence_samples:
                    if temp_end - current_start >= min_speech_samples:
                        speeches.append({"start": current_start, "end": temp_end})
                    triggered = False
                    temp_end = 0

        if triggered:
            end = min(audio_len, len(probs) * chunk_size)
            if end - current_start >= min_speech_samples:
                speeches.append({"start": current_start, "end": end})

        padded = []
        for speech in speeches:
            start = max(0, speech["start"] - speech_pad_samples)
            end = min(audio_len, speech["end"] + speech_pad_samples)
            if padded and start <= padded[-1]["end"]:
                padded[-1]["end"] = max(padded[-1]["end"], end)
            else:
                padded.append({"start": start, "end": end})

        if return_seconds:
            return [
                {
                    "start": round(speech["start"] / sample_rate, 3),
                    "end": round(speech["end"] / sample_rate, 3),
                }
                for speech in padded
            ]
        return padded

    @classmethod
    def sanitize(cls, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return {k: v for k, v in weights.items() if not k.startswith("val_")}

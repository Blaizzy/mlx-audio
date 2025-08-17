import io
import json
import os
from dataclasses import asdict
from typing import Union

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import SAMPLE_RATE as WHISPER_SAMPLE_RATE
from mlx_audio.stt.utils import load_model, resample_audio

from .dac_interface import DacInterface
from .prompt_processor import PromptProcessor


def calculate_pitch(
    audio_array: mx.array,
    sr: int,
    min_freq: float = 75.0,
    max_freq: float = 600.0,
    frame_length: int = 400,
    hop_length: int = 160,
    threshold: float = 0.3,
) -> mx.array:
    """
    Calculate pitch frequencies for short audio clips using autocorrelation.

    Args:
        audio_array: Input audio array (1D or 2D [channels, samples])
        sr: Sampling rate
        min_freq: Minimum detectable frequency (Hz)
        max_freq: Maximum detectable frequency (Hz)
        frame_length: Analysis frame length in samples
        hop_length: Hop size in samples
        threshold: Voicing threshold (0.0-1.0)

    Returns:
        Array of pitch values (Hz) per frame
    """
    audio_np = np.array(audio_array)

    # convert to mono and ensure 1D
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=0)
    audio_np = np.squeeze(audio_np)

    num_samples = audio_np.shape[-1]
    pad_len = (frame_length - (num_samples % hop_length)) % hop_length
    audio_np = np.pad(audio_np, (0, pad_len))

    num_frames = (len(audio_np) - frame_length) // hop_length + 1
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        frames[i] = audio_np[i * hop_length : i * hop_length + frame_length]

    window = np.hanning(frame_length)
    frames_windowed = frames * window

    # compute autocorrelation using FFT
    fft_frames = np.fft.rfft(frames_windowed, n=2 * frame_length, axis=1)
    power_spectrum = fft_frames.real**2 + fft_frames.imag**2
    autocorr = np.fft.irfft(power_spectrum, axis=1)[:, :frame_length]

    # find valid frequency range indices
    min_idx = max(1, int(sr / max_freq))
    max_idx = min(frame_length, int(sr / min_freq))

    # find peak indices in valid range
    relevant_autocorr = autocorr[:, min_idx:max_idx]
    peak_indices = np.argmax(relevant_autocorr, axis=1) + min_idx
    peak_values = np.array([autocorr[i, peak_indices[i]] for i in range(num_frames)])

    # parabolic interpolation for sub-sample accuracy
    indices = np.clip(peak_indices, 1, frame_length - 2)
    alpha = np.array([autocorr[i, indices[i] - 1] for i in range(num_frames)])
    beta = np.array([autocorr[i, indices[i]] for i in range(num_frames)])
    gamma = np.array([autocorr[i, indices[i] + 1] for i in range(num_frames)])

    delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-8)
    valid_mask = (peak_indices > 0) & (peak_indices < frame_length - 1)
    delta = np.where(valid_mask, delta, 0.0)

    # calculate final periods and pitches
    best_period = (peak_indices + delta) / sr
    pitch = np.where(best_period > 0, 1.0 / best_period, 0.0)

    # apply voicing threshold
    autocorr_0 = autocorr[:, 0]
    voiced = (peak_values / (autocorr_0 + 1e-8)) > threshold
    pitch = np.where(voiced, pitch, 0.0)

    # clamp valid frequencies
    pitch = np.clip(pitch, min_freq, max_freq)

    return mx.array(pitch)


def extract_single_pitch_value(
    audio_array: mx.array,
    sr: int,
    min_freq: float = 75.0,
    max_freq: float = 600.0,
    frame_length: int = 400,
    hop_length: int = 160,
    threshold: float = 0.3,
) -> float:
    """
    Calculates the average pitch of an audio array and normalizes it to 0-1 range.

    Args:
        audio_array: Input audio array (1D or 2D [channels, samples])
        sr: Sampling rate
        min_freq: Minimum detectable frequency (Hz)
        max_freq: Maximum detectable frequency (Hz)
        frame_length: Analysis frame length in samples
        hop_length: Hop size in samples
        threshold: Voicing threshold (0.0-1.0)

    Returns:
        A single float value representing the normalized average pitch (0.0-1.0).
    """
    pitch_array = calculate_pitch(
        audio_array, sr, min_freq, max_freq, frame_length, hop_length, threshold
    )

    # calculate the average pitch across frames
    average_pitch = float(mx.mean(pitch_array))

    # normalize to 0-1 range
    normalized_pitch = (average_pitch - min_freq) / (max_freq - min_freq)

    # clamp to ensure it's strictly within 0-1
    normalized_pitch = min(max(normalized_pitch, 0.0), 1.0)

    return normalized_pitch


class Features:
    def __init__(self):
        self.eps = 1e-10

    def scale_values(self, value: float) -> int:
        """
        Scale a value from [0,1] to [0,100] and round to nearest integer
        """
        return round(value * 100)

    def features_to_tokens(self, features: dict) -> list:
        """
        Convert features to token strings in format <|feature_value|>
        """
        return [f"<|{name}_{value}|>" for name, value in features.items()]

    def validate_audio(self, audio: mx.array) -> bool:
        if audio is None or not isinstance(audio, mx.array):
            return False
        if audio.size == 0:  # Check if array is empty
            return False
        audio_np = np.array(audio)
        if np.isnan(audio_np).any() or np.isinf(audio_np).any():
            return False
        return True

    def get_default_features(self) -> dict:
        """
        Return default feature values when audio is invalid
        """
        return {"energy": 0, "spectral_centroid": 0, "pitch": 0}

    def extract_audio_features(self, audio: mx.array, sr: int) -> dict:
        """
        Extract fast-to-compute features from audio segments.
        Each feature is normalized to [0, 1] range.

        Args:
            audio: Audio array of shape [channels, samples]
            sr: Sample rate

        Returns:
            Dictionary of features, each as a single float value
        """
        if not self.validate_audio(audio):
            return self.get_default_features()

        audio_np = np.array(audio)

        # convert to mono if stereo
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            audio_np = np.mean(audio_np, axis=0, keepdims=True)

        audio = mx.array(audio_np)

        features = {}

        # rms energy (loudness) - normalized to [0, 1]
        features["energy"] = float(mx.sqrt(mx.mean(audio**2)))

        # spectral centroid - normalized to [0, 1]
        spec_np = np.abs(np.fft.rfft(audio_np))
        freqs_np = np.linspace(0, sr / 2, spec_np.shape[-1])
        spec_sum = np.sum(spec_np) + self.eps
        centroid = np.sum(freqs_np * spec_np.squeeze()) / spec_sum
        features["spectral_centroid"] = float(centroid / (sr / 2))

        # pitch - normalized to [0, 1]
        features["pitch"] = extract_single_pitch_value(audio, sr)

        # scale values to 0-100 range
        for name, value in features.items():
            features[name] = self.scale_values(value)

        return features


class AudioProcessor:
    def __init__(
        self, audio_codec_path: str = "mlx-community/dac-speech-24khz-1.5kbps"
    ):
        self.features = Features()
        self.audio_codec = DacInterface(audio_codec_path)

    def create_speaker_from_whisper(
        self,
        audio: str,
        whisper_model: str = "mlx-community/whisper-large-v3-turbo",
    ):
        if isinstance(audio, str):
            audio = self.audio_codec.load_audio(audio)
        else:
            # resample audio to 16000 for whisper
            resampled_audio = resample_audio(
                audio[..., None], self.audio_codec.sr, WHISPER_SAMPLE_RATE
            )
            resampled_audio = mx.array(resampled_audio, dtype=mx.float32).mean(axis=1)

            # convert to 2d array
            audio = audio[None, None, ...]

        seconds = audio.flatten().shape[0] / self.audio_codec.sr
        if seconds > 20:
            print(
                "Speaker audio is longer than 20 seconds. Use a shorter clip for best results."
            )
        if seconds > 15:
            print(
                "Speaker audio is longer than 15 seconds. For best results, consider using an audio clip up to 15 seconds."
            )

        # load whisper model
        whisper_model = load_model(whisper_model)

        # transcribe audio
        data = whisper_model.generate(resampled_audio.flatten(), word_timestamps=True)
        data = asdict(data)

        # clear memory
        del whisper_model
        mx.clear_cache()

        text = PromptProcessor.text_normalizations(data["text"])
        words = []
        for s in data["segments"]:
            words.extend(
                [
                    {
                        "word": i["word"].strip(),
                        "start": float(i["start"]),
                        "end": float(i["end"]),
                    }
                    for i in s["words"]
                ]
            )

        return self.create_speaker_from_dict(
            {"audio": {"bytes": audio}, "text": text, "words": words}
        )

    def create_speaker_from_dict(self, data: dict):
        audio = data["audio"]["bytes"]
        if isinstance(audio, str):
            audio = io.BytesIO(audio)
            audio = self.audio_codec.load_audio(audio)

        full_codes = self.audio_codec.encode(audio, verbose=True).tolist()[0]

        c1 = full_codes[0]
        c2 = full_codes[1]

        sr = self.audio_codec.sr
        text = data["text"]
        words = data["words"]

        tps = 75

        audio = audio.squeeze(0)
        global_features = self.features.extract_audio_features(audio, sr)

        start = None
        word_codes = []
        max_extension = 20

        for idx, i in enumerate(words):
            if start is None:
                start = max(0, int(i["start"] * tps) - max_extension)
            word = i["word"].strip()
            if idx == len(words) - 1:
                end = min(len(c1), int(i["end"] * tps) + max_extension)
            else:
                end = int(i["end"] * tps)

            word_c1 = c1[start:end]
            word_c2 = c2[start:end]

            word_audio = audio[:, int(i["start"] * sr) : int(i["end"] * sr)]
            features = self.features.extract_audio_features(word_audio, sr)

            start = end

            word_codes.append(
                {
                    "word": word,
                    "duration": round(len(word_c1) / tps, 2),
                    "c1": word_c1,
                    "c2": word_c2,
                    "features": features,
                }
            )

        return {"text": text, "words": word_codes, "global_features": global_features}

    def save_speaker(self, speaker: dict, path: str):
        # Expand ~ to home directory to save in ~/.cache/mlx_audio/voices
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(speaker, f)

        print(f"Speaker saved to: {path}")

    def load_speaker(self, path: str):
        # Expand ~ to home directory to load from ~/.cache/mlx_audio/voices
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Speaker file not found: {path}")

        with open(path, "r") as f:
            return json.load(f)

        print(f"Speaker loaded from: {path}")

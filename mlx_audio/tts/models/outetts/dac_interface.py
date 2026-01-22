import math

import mlx.core as mx
import numpy as np
import pyloudnorm as pyln
import scipy.signal

from mlx_audio.audio_io import read as audio_read
from mlx_audio.codec import DAC


def process_audio_array(
    audio: mx.array,
    sample_rate: int = 24000,
    target_loudness: float = -18.0,
    peak_limit: float = -1,
    block_size: float = 0.400,
) -> mx.array:
    audio_np = np.array(audio)

    # handle multi-channel audio
    if len(audio_np.shape) > 1:
        if audio_np.shape[1] > 1:
            audio_np = np.mean(audio_np, axis=1)
        else:
            audio_np = np.squeeze(audio_np)

    original_length = len(audio_np)
    min_samples = int(block_size * sample_rate)

    if original_length < min_samples:
        pad_length = min_samples - original_length
        audio_padded = np.pad(audio_np, (0, pad_length), mode="constant")
    else:
        audio_padded = audio_np

    # measure and normalize loudness
    meter = pyln.Meter(sample_rate, block_size=block_size)
    measured_loudness = meter.integrated_loudness(audio_padded)
    normalized = pyln.normalize.loudness(
        audio_padded, measured_loudness, target_loudness
    )

    # apply peak limiting if necessary
    peak_value = np.max(np.abs(normalized))
    threshold_value = 10 ** (peak_limit / 20)
    if peak_value > threshold_value:
        normalized = pyln.normalize.peak(normalized, peak_limit)

    if original_length < min_samples:
        normalized = normalized[:original_length]

    normalized_array = mx.array(normalized).reshape(1, 1, -1)
    return normalized_array


class DacInterface:
    def __init__(self, repo_id: str = "mlx-community/dac-speech-24khz-1.5kbps"):
        self.model = DAC.from_pretrained(repo_id)
        self.sr = 24000

    def convert_audio(
        self, audio: mx.array, sr: int, target_sr: int, target_channels: int
    ):
        audio_np = np.array(audio)

        if len(audio_np.shape) < 2:
            audio_np = audio_np.reshape(1, -1)

        channels, length = audio_np.shape[-2:]

        if target_channels == 1:
            if channels > 1:
                audio_np = np.mean(audio_np, axis=-2, keepdims=True)
        elif target_channels == 2:
            if channels == 1:
                audio_np = np.repeat(audio_np, 2, axis=-2)
            elif channels > 2:
                audio_np = audio_np[..., :2, :]

        if sr != target_sr:
            new_length = int(length * target_sr / sr)
            resampled = np.zeros((target_channels, new_length))

            for ch in range(target_channels):
                resampled[ch] = scipy.signal.resample(audio_np[ch], new_length)

            audio_np = resampled

        return mx.array(audio_np)

    def convert_audio_array(self, audio: mx.array, sr):
        return self.convert_audio(audio, sr, self.sr, 1)

    def load_audio(self, path):
        audio_np, sr = audio_read(path)
        audio = mx.array(audio_np)
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)
        # if stereo, reshape to channels-first format
        elif len(audio.shape) > 1 and audio.shape[0] > audio.shape[1]:
            audio = audio.T
        return self.convert_audio_array(audio, sr).reshape(1, 1, -1)

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        hop_length = self.model.hop_length
        right_pad = math.ceil(length / hop_length) * hop_length - length
        audio_data = mx.pad(audio_data, [(0, 0), (0, 0), (0, right_pad)])
        return audio_data

    def encode(self, x: mx.array, win_duration: int = 5.0, verbose: bool = False):
        x = process_audio_array(x)
        nb, nac, nt = x.shape
        x = x.reshape(nb * nac, 1, nt)
        n_samples = int(win_duration * self.sr)
        n_samples = int(
            math.ceil(n_samples / self.model.hop_length) * self.model.hop_length
        )
        hop = n_samples
        codes_list = []

        if verbose:
            from tqdm import trange

            range_fn = trange
        else:
            range_fn = range

        for i in range_fn(0, nt, hop):
            chunk = x[..., i : i + n_samples]
            audio_data = self.preprocess(chunk)
            _, c, _, _, _ = self.model.encode(audio_data, None)
            codes_list.append(c)

        codes = mx.concatenate(codes_list, axis=-1)
        return codes

    def decode(self, codes: mx.array, verbose: bool = False) -> mx.array:
        model = self.model
        chunk_length = 4096
        recons = []

        if verbose:
            from tqdm import trange

            range_fn = trange
        else:
            range_fn = range

        @mx.compile
        def decode_chunk(codes):
            z = model.quantizer.from_codes(codes)[0]
            r = model.decode(z)
            return r

        for i in range_fn(0, codes.shape[-1], chunk_length):
            c = codes[..., i : i + chunk_length]
            recons.append(decode_chunk(c))

        recons = mx.concatenate(recons, axis=-1)
        return process_audio_array(recons.swapaxes(1, 2))

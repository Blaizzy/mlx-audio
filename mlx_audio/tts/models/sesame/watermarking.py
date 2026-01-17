import argparse

import mlx.core as mx
import numpy as np
import silentcipher
from scipy import signal

from mlx_audio.audio_io import read as audio_read

# This watermark key is public, it is not secure.
# If using CSM 1B in another application, use a new private key and keep it secret.
CSM_1B_GH_WATERMARK = [212, 211, 146, 56, 201]


def cli_check_audio() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()
    check_audio_from_file(args.audio_path)


def load_watermarker() -> silentcipher.server.Model:
    model = silentcipher.get_model(
        model_type="44.1k",
    )
    return model


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled


def watermark(
    watermarker: silentcipher.server.Model,
    audio_array: mx.array,
    sample_rate: int,
    watermark_key: list[int],
) -> tuple[mx.array, int]:
    audio_array = np.array(audio_array, dtype=np.float32)

    if sample_rate != 44100:
        audio_array_44khz = resample_audio(audio_array, sample_rate, 44100)
    else:
        audio_array_44khz = audio_array

    encoded, *_ = watermarker.encode_wav(
        audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=36
    )

    if sample_rate != 44100:
        encoded = resample_audio(encoded, 44100, sample_rate)

    return encoded


def verify(
    watermarker: silentcipher.server.Model,
    watermarked_audio: mx.array,
    sample_rate: int,
    watermark_key: list[int],
) -> bool:
    if sample_rate != 44100:
        watermarked_audio_44khz = resample_audio(watermarked_audio, sample_rate, 44100)
    else:
        watermarked_audio_44khz = watermarked_audio

    result = watermarker.decode_wav(
        watermarked_audio_44khz, 44100, phase_shift_decoding=True
    )

    is_watermarked = result["status"]
    if is_watermarked:
        is_csm_watermarked = result["messages"][0] == watermark_key
    else:
        is_csm_watermarked = False

    return is_watermarked and is_csm_watermarked


def check_audio_from_file(audio_path: str) -> None:
    watermarker = load_watermarker()
    audio_array, sample_rate = load_audio(audio_path)
    is_watermarked = verify(watermarker, audio_array, sample_rate, CSM_1B_GH_WATERMARK)
    outcome = "Watermarked" if is_watermarked else "Not watermarked"
    print(f"{outcome}: {audio_path}")


def load_audio(audio_path: str) -> tuple[mx.array, int]:
    audio_array_np, sample_rate = audio_read(audio_path, always_2d=True)

    if audio_array_np.shape[1] > 1:
        audio_array_np = audio_array_np.mean(axis=1)
    else:
        audio_array_np = audio_array_np.squeeze()

    audio_array = mx.array(audio_array_np)

    return audio_array, int(sample_rate)


if __name__ == "__main__":
    cli_check_audio()

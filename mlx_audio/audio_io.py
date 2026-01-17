"""Audio I/O utilities using miniaudio.

This module provides functions for reading and writing audio files.
- Reading: Uses miniaudio to support WAV, MP3, FLAC, and Vorbis formats.
- Writing: Uses miniaudio for WAV/FLAC and ffmpeg for MP3 encoding.
"""

import io
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

# Format mapping for miniaudio
_FORMAT_MAP = {
    "wav": "wav",
    "mp3": "mp3",
    "flac": "flac",
    "ogg": "vorbis",
    "vorbis": "vorbis",
}

# Sample format mapping
_SAMPLE_FORMAT_MAP = {
    "int16": "SIGNED16",
    "int32": "SIGNED32",
    "float32": "FLOAT32",
}


def _detect_format_from_bytes(data: bytes) -> str:
    """Detect audio format from bytes data using magic bytes."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    elif data[:3] == b"ID3" or (data[0:2] == b"\xff\xfb" or data[0:2] == b"\xff\xfa"):
        return "mp3"
    elif data[:4] == b"fLaC":
        return "flac"
    elif data[:4] == b"OggS":
        return "vorbis"
    else:
        raise ValueError("Unable to detect audio format from bytes")


def read(
    file: Union[str, Path, io.BytesIO],
    always_2d: bool = False,
    dtype: str = "float64",
) -> Tuple[np.ndarray, int]:
    """Read an audio file using miniaudio.

    Args:
        file: Path to the audio file or a BytesIO object.
        always_2d: If True, always return a 2D array (samples, channels).
        dtype: Data type for the output array. Supports 'float32', 'float64', 'int16'.

    Returns:
        Tuple of (audio_data, sample_rate).
        audio_data is a numpy array with shape (samples,) for mono or (samples, channels) for multi-channel.
    """
    import miniaudio

    if isinstance(file, (str, Path)):
        # Get file info to preserve original sample rate and channels
        info = miniaudio.get_file_info(str(file))
        decoded = miniaudio.decode_file(
            str(file),
            nchannels=info.nchannels,
            sample_rate=info.sample_rate,
        )
    elif isinstance(file, io.BytesIO):
        file.seek(0)
        data = file.read()
        # Detect format and get info to preserve original sample rate and channels
        fmt = _detect_format_from_bytes(data)
        if fmt == "wav":
            info = miniaudio.wav_get_info(data)
        elif fmt == "mp3":
            info = miniaudio.mp3_get_info(data)
        elif fmt == "flac":
            info = miniaudio.flac_get_info(data)
        elif fmt == "vorbis":
            info = miniaudio.vorbis_get_info(data)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        decoded = miniaudio.decode(
            data,
            nchannels=info.nchannels,
            sample_rate=info.sample_rate,
        )
    else:
        raise TypeError(f"Unsupported file type: {type(file)}")

    sample_rate = decoded.sample_rate
    nchannels = decoded.nchannels

    # Convert to numpy array
    # miniaudio returns samples as array of signed 16-bit integers interleaved
    samples = np.array(decoded.samples, dtype=np.int16)

    # Reshape to (samples, channels) if multi-channel
    if nchannels > 1:
        samples = samples.reshape(-1, nchannels)

    # Convert to requested dtype
    if dtype in ("float32", "float64"):
        samples = samples.astype(dtype) / 32768.0
    elif dtype == "int16":
        pass  # Already int16
    else:
        samples = samples.astype(dtype)

    # Handle always_2d
    if always_2d and samples.ndim == 1:
        samples = samples[:, np.newaxis]

    return samples, sample_rate


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable.

    Returns:
        Path to ffmpeg executable.

    Raises:
        RuntimeError: If ffmpeg is not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "\n"
            "========================================\n"
            "  ffmpeg not found!\n"
            "========================================\n"
            "\n"
            "ffmpeg is required for MP3 and FLAC audio encoding.\n"
            "\n"
            "Install ffmpeg:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "\n"
            "Alternatively, use WAV format which doesn't require ffmpeg.\n"
        )
    return ffmpeg_path


def _encode_ffmpeg(
    data: np.ndarray,
    samplerate: int,
    nchannels: int,
    output: Union[str, Path, io.BytesIO],
    format: str = "mp3",
    bitrate: str = "128k",
) -> None:
    """Encode audio using ffmpeg.

    Args:
        data: Audio data as int16 numpy array (samples,) or (samples, channels)
        samplerate: Sample rate in Hz
        nchannels: Number of channels
        output: Output file path or BytesIO object
        format: Output format (mp3, flac, etc.)
        bitrate: Audio bitrate for lossy formats (default: 128k)
    """
    ffmpeg_path = _get_ffmpeg_path()

    # Prepare raw PCM bytes (interleaved for stereo)
    if data.ndim == 1:
        pcm_bytes = data.tobytes()
    else:
        pcm_bytes = data.flatten().tobytes()

    # Build ffmpeg command
    cmd = [
        ffmpeg_path,
        "-y",  # Overwrite output
        "-f",
        "s16le",  # Input: signed 16-bit little-endian PCM
        "-ar",
        str(samplerate),  # Sample rate
        "-ac",
        str(nchannels),  # Channels
        "-i",
        "pipe:0",  # Read from stdin
    ]

    # Add format-specific options
    if format == "mp3":
        cmd.extend(["-b:a", bitrate])

    cmd.extend(["-f", format])

    if isinstance(output, io.BytesIO):
        cmd.append("pipe:1")
        result = subprocess.run(
            cmd,
            input=pcm_bytes,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        output.write(result.stdout)
        output.seek(0)
    else:
        cmd.append(str(output))
        result = subprocess.run(
            cmd,
            input=pcm_bytes,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")


def write(
    file: Union[str, Path, io.BytesIO],
    data: np.ndarray,
    samplerate: int,
    format: Optional[str] = None,
) -> None:
    """Write audio data to a file.

    Args:
        file: Path to the output file or a BytesIO object.
        data: Audio data as numpy array. Shape can be (samples,) for mono
              or (samples, channels) for multi-channel.
        samplerate: Sample rate in Hz.
        format: Output format. Supports 'wav', 'flac', 'mp3'. If None, inferred from file extension.

    Note:
        WAV and FLAC use miniaudio for encoding.
        MP3 uses ffmpeg (must be installed: brew install ffmpeg).
    """
    import miniaudio

    # Determine format
    if format is None:
        if isinstance(file, (str, Path)):
            format = Path(file).suffix.lstrip(".").lower()
        else:
            format = "wav"  # Default to WAV for BytesIO

    format = format.lower()

    # Ensure data is numpy array (handle MLX arrays and other array-like types)
    if not isinstance(data, np.ndarray):
        # Check for MLX array (has tolist but not __array__)
        if hasattr(data, "tolist") and not hasattr(data, "__array__"):
            data = np.array(data.tolist())
        elif hasattr(data, "__array__"):
            data = np.asarray(data)
        else:
            data = np.array(data)

    # Convert to int16 for encoding
    if data.dtype in (np.float32, np.float64):
        # Clip to [-1, 1] range and convert to int16
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)

    # Get number of channels
    if data.ndim == 1:
        nchannels = 1
    else:
        nchannels = data.shape[1]

    if format == "wav":
        import array

        # Flatten for miniaudio (interleaved)
        if data.ndim == 1:
            samples_flat = data
        else:
            samples_flat = data.flatten()

        # Convert to array.array for miniaudio
        samples_array = array.array("h", samples_flat.tolist())

        # Create DecodedSoundFile
        sound = miniaudio.DecodedSoundFile(
            name="output",
            nchannels=nchannels,
            sample_rate=samplerate,
            sample_format=miniaudio.SampleFormat.SIGNED16,
            samples=samples_array,
        )

        if isinstance(file, io.BytesIO):
            # Write to temp file, read back to BytesIO
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                miniaudio.wav_write_file(tmp_path, sound)
                with open(tmp_path, "rb") as f:
                    file.write(f.read())
                file.seek(0)
            finally:
                import os

                os.unlink(tmp_path)
        else:
            miniaudio.wav_write_file(str(file), sound)

    elif format in ("flac", "mp3"):
        # Check for ffmpeg early to provide a clear error message
        if not _check_ffmpeg_available():
            import warnings

            warnings.warn(
                f"ffmpeg is required for {format.upper()} encoding but was not found. "
                "Install with: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Ubuntu). "
                "Falling back will fail - consider using WAV format instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        _encode_ffmpeg(data, samplerate, nchannels, file, format=format)
    else:
        raise ValueError(f"Unsupported output format: {format}")


# Convenience aliases to match soundfile API
def sf_read(
    file: Union[str, Path, io.BytesIO],
    always_2d: bool = False,
) -> Tuple[np.ndarray, int]:
    """Read audio file (soundfile-compatible API).

    This is a drop-in replacement for soundfile.read().

    Args:
        file: Path to audio file or BytesIO object.
        always_2d: If True, always return 2D array.

    Returns:
        Tuple of (data, samplerate).
    """
    return read(file, always_2d=always_2d, dtype="float64")


def sf_write(
    file: Union[str, Path, io.BytesIO],
    data: np.ndarray,
    samplerate: int,
    format: Optional[str] = None,
) -> None:
    """Write audio file (soundfile-compatible API).

    This is a drop-in replacement for soundfile.write().

    Args:
        file: Path to output file or BytesIO object.
        data: Audio data as numpy array.
        samplerate: Sample rate in Hz.
        format: Output format (wav, flac, mp3).
    """
    write(file, data, samplerate, format=format)

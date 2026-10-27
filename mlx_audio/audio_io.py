"""Audio I/O utilities using miniaudio.

This module provides functions for reading and writing audio files.
- Reading: Uses miniaudio to support WAV, MP3, and FLAC formats.
           Uses ffmpeg for M4A/AAC, OGG (Vorbis, Opus), WebM, CAF, AIFF, and AIFC format support.
- Writing: Uses miniaudio for WAV and ffmpeg for MP3, FLAC, OGG, Opus, Vorbis, and WebM encoding.
"""

import io
import math
import os
import shutil
import subprocess
import tempfile
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
    "m4a": "m4a",
    "aac": "m4a",
    "webm": "webm",
}

# Sample format mapping
_SAMPLE_FORMAT_MAP = {
    "int16": "SIGNED16",
    "int32": "SIGNED32",
    "float32": "FLOAT32",
}

_RESAMPLE_CHUNK_DURATION_SECONDS = 1.0


# Formats ffmpeg decodes, named by _detect_format or by file extension.
# miniaudio decodes the rest.
_FFMPEG_FORMATS = (
    "m4a",
    "m4b",
    "mp4",
    "aac",
    "ogg",
    "opus",
    "webm",
    "caf",
    "aiff",
    "aifc",
)

# Extensions that pick a miniaudio decoder.
_MINIAUDIO_FORMATS = ("wav", "mp3", "flac", "vorbis")

# How far _detect_format looks past an ID3v2 tag: tags in a row, bytes of padding.
_MAX_ID3_TAGS = 8
_MAX_ID3_PADDING = 1 << 16


def _detect_format_from_bytes(data: bytes) -> Optional[str]:
    """Detect audio format from bytes data using magic bytes."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    elif data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return data[8:12].decode().lower()
    elif data[:3] == b"ID3":
        return "mp3"
    elif len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        # MPEG audio frame sync. ADTS AAC has the same sync and layer bits 00.
        return "aac" if data[1] & 0x06 == 0 else "mp3"
    elif data[:4] == b"fLaC":
        return "flac"
    elif data[:4] == b"OggS":
        return "ogg"
    elif data[4:8] == b"ftyp":
        # M4A/MP4/AAC container format
        return "m4a"
    elif data[:4] == b"\x1a\x45\xdf\xa3":
        # WebM/Matroska container (EBML header)
        return "webm"
    elif data[:4] == b"caff":
        # Core Audio Format (macOS default recording container)
        return "caf"
    return None


def _detect_format(f, after_id3: str = "mp3") -> Optional[str]:
    """Detect the audio format of a binary file, read from offset 0.

    after_id3 is the answer when ID3v2 tags are followed by unknown bytes.
    """
    head = f.read(12)
    if head[:3] != b"ID3":
        return _detect_format_from_bytes(head)
    # MP3, AAC, FLAC and Ogg files can start with ID3v2 tags, and zero padding
    # can follow a tag. Look past both.
    offset = 0
    for _ in range(_MAX_ID3_TAGS):
        if len(head) < 10:
            break
        size = (head[6] & 0x7F) << 21 | (head[7] & 0x7F) << 14
        size |= (head[8] & 0x7F) << 7 | head[9] & 0x7F
        offset += 10 + size + (10 if head[5] & 0x10 else 0)
        f.seek(offset)
        head = f.read(12)
        if head[:1] == b"\x00":
            block = head + f.read(_MAX_ID3_PADDING)
            offset += len(block) - len(block.lstrip(b"\x00"))
            f.seek(offset)
            head = f.read(12)
        if head[:3] != b"ID3":
            return _detect_format_from_bytes(head) or after_id3
    return after_id3


def _backend(fmt: str) -> str:
    """Name the decoder for a format: ffmpeg, or one of miniaudio's."""
    if fmt in _FFMPEG_FORMATS:
        return "ffmpeg"
    return fmt


def _decode_ffmpeg(
    input_data: Union[str, Path, bytes],
    sample_rate: Optional[int] = None,
    nchannels: Optional[int] = None,
    duration: Optional[float] = None,
    input_format: Optional[str] = None,
) -> Tuple[np.ndarray, int, int]:
    """Decode audio using ffmpeg (for formats not supported by miniaudio like M4A).

    Args:
        input_data: Path to the audio file or raw bytes data.
        sample_rate: Optional target sample rate. Preserves the input rate when omitted.
        nchannels: Optional target channel count. Preserves the input channels when omitted.
        duration: Optional number of seconds to decode from the start.
        input_format: Optional demuxer hint when automatic probing fails.

    Returns:
        Tuple of (samples as int16 numpy array, sample_rate, nchannels).

    Raises:
        RuntimeError: If ffmpeg is not found or decoding fails.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "\n"
            "========================================\n"
            "  ffmpeg not found!\n"
            "========================================\n"
            "\n"
            "ffmpeg is required to decode this audio format.\n"
            "\n"
            "Install ffmpeg:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
        )

    # First, get audio info using ffprobe
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError("ffprobe not found (usually installed with ffmpeg)")

    if isinstance(input_data, bytes):
        # ffmpeg cannot seek in a pipe, so a container with its index after the
        # audio (an M4A with moov last) decodes to nothing. A file is seekable.
        fd, tmp_path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(input_data)
            return _decode_ffmpeg(
                tmp_path,
                sample_rate=sample_rate,
                nchannels=nchannels,
                duration=duration,
                input_format=input_format,
            )
        finally:
            os.unlink(tmp_path)

    probe_cmd = [
        ffprobe_path,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "a:0",
        str(input_data),
    ]
    probe_result = subprocess.run(
        probe_cmd, stdin=subprocess.DEVNULL, capture_output=True
    )
    input_options = []
    if probe_result.returncode != 0 and input_format is not None:
        input_options = ["-f", input_format]
        probe_cmd[-1:-1] = input_options
        probe_result = subprocess.run(
            probe_cmd, stdin=subprocess.DEVNULL, capture_output=True
        )

    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe_result.stderr.decode()}")

    import json

    probe_info = json.loads(probe_result.stdout.decode())
    if not probe_info.get("streams"):
        raise RuntimeError("No audio streams found in file")

    stream = probe_info["streams"][0]
    output_sample_rate = sample_rate or int(stream.get("sample_rate", 44100))
    output_nchannels = nchannels or int(stream.get("channels", 2))

    # Decode to raw PCM using ffmpeg
    decode_cmd = [
        ffmpeg_path,
        "-v",
        "error",
        *input_options,
        "-i",
        str(input_data),
        "-f",
        "s16le",  # Output: signed 16-bit little-endian PCM
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(output_sample_rate),
        "-ac",
        str(output_nchannels),
    ]
    max_frames = None
    if duration is not None:
        max_frames = round(duration * output_sample_rate)
        if max_frames == 0:
            return np.empty(0, dtype=np.int16), output_sample_rate, output_nchannels
        # One extra frame, so rounding in ffmpeg cannot leave the result short.
        # ffmpeg needs fixed-point seconds that fit in int64 microseconds.
        seconds = min((max_frames + 1) / output_sample_rate, 1e12)
        # -t cuts by timestamp, and a file's timestamps can jump. asetpts sets
        # them to the sample count, so the cut lands on the frame count.
        decode_cmd += ["-af", "asetpts=N/SR/TB", "-t", f"{seconds:.6f}"]
    decode_cmd.append("pipe:1")
    decode_result = subprocess.run(
        decode_cmd, stdin=subprocess.DEVNULL, capture_output=True
    )

    if decode_result.returncode != 0:
        raise RuntimeError(f"ffmpeg decoding failed: {decode_result.stderr.decode()}")
    # ffmpeg can exit with status 0 without writing any audio.
    if not decode_result.stdout:
        message = "ffmpeg decoded no audio samples"
        errors = decode_result.stderr.decode().strip()
        raise RuntimeError(f"{message}: {errors}" if errors else message)

    # Convert raw PCM bytes to numpy array
    samples = np.frombuffer(decode_result.stdout, dtype=np.int16)
    if max_frames is not None:
        samples = samples[: max_frames * output_nchannels]

    return samples, output_sample_rate, output_nchannels


def _stream_miniaudio(
    input_data: Union[str, Path, bytes], sample_rate: int, nchannels: int
):
    """Stream interleaved int16 PCM from miniaudio in chunks of about one second."""
    import miniaudio

    stream_kwargs = {
        "output_format": miniaudio.SampleFormat.SIGNED16,
        "nchannels": nchannels,
        "sample_rate": sample_rate,
        "frames_to_read": max(1, int(_RESAMPLE_CHUNK_DURATION_SECONDS * sample_rate)),
    }
    if isinstance(input_data, bytes):
        return miniaudio.stream_memory(input_data, **stream_kwargs)
    return miniaudio.stream_file(str(input_data), **stream_kwargs)


def _decode_miniaudio_head(
    input_data: Union[str, Path, bytes],
    sample_rate: int,
    nchannels: int,
    max_frames: int,
) -> np.ndarray:
    """Decode only the first max_frames frames as interleaved int16 PCM."""
    pcm_stream = _stream_miniaudio(input_data, sample_rate, nchannels)
    chunks, frames = [], 0
    try:
        for raw_samples in pcm_stream:
            chunks.append(np.asarray(raw_samples, dtype=np.int16))
            frames += chunks[-1].shape[0] // nchannels
            if frames >= max_frames:
                break
    finally:
        pcm_stream.close()
    samples = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
    return samples[: max_frames * nchannels]


def _decode_miniaudio_resampled(
    input_data: Union[str, Path, bytes],
    input_sample_rate: int,
    input_num_frames: int,
    output_sample_rate: int,
    output_nchannels: int,
    max_output_frames: Optional[int] = None,
) -> np.ndarray:
    """Stream native-rate mono/stereo PCM through the high-quality FIR."""
    from mlx_audio.resample import resample_audio_chunks

    pcm_stream = _stream_miniaudio(input_data, input_sample_rate, output_nchannels)

    def normalized_chunks():
        for raw_samples in pcm_stream:
            chunk = np.asarray(raw_samples, dtype=np.int16)
            if output_nchannels > 1:
                chunk = chunk.reshape(-1, output_nchannels)
            chunk = chunk.astype(np.float32)
            chunk /= 32768.0
            yield chunk

    try:
        return resample_audio_chunks(
            normalized_chunks(),
            input_sample_rate,
            output_sample_rate,
            input_num_frames,
            chunk_duration_seconds=_RESAMPLE_CHUNK_DURATION_SECONDS,
            max_output_frames=max_output_frames,
        )
    finally:
        pcm_stream.close()


def read(
    file: Union[str, Path, io.BytesIO],
    always_2d: bool = False,
    dtype: str = "float64",
    sample_rate: Optional[int] = None,
    nchannels: Optional[int] = None,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Read audio using miniaudio or ffmpeg.

    Args:
        file: Path to the audio file or a BytesIO object. The format comes from the
            first bytes, or from the extension when the bytes are not recognized
            or do not decode.
        always_2d: If True, always return a 2D array (samples, channels).
        dtype: Data type for the output array. Supports 'float32', 'float64', 'int16'.
        sample_rate: Optional target sample rate. Preserves the input rate when omitted.
            Miniaudio-backed resampling uses the high-quality chunked FIR path.
        nchannels: Optional target channel count. Preserves the input channels when omitted.
        duration: Optional number of seconds to read from the start. Decoding stops
            there, so time and memory follow the limit. Reads the whole file when omitted.

    Returns:
        Tuple of (audio_data, sample_rate).
        audio_data is a numpy array with shape (samples,) for mono or (samples, channels) for multi-channel.
    """
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if nchannels is not None and nchannels <= 0:
        raise ValueError(f"nchannels must be positive, got {nchannels}")
    if duration is not None and not (duration > 0 and math.isfinite(duration)):
        raise ValueError(f"duration must be a positive finite number, got {duration}")

    # The first bytes name the format. The extension is the fallback.
    fallback = None
    if isinstance(file, (str, Path)):
        ext = Path(file).suffix.lstrip(".").lower()
        if ext in _FFMPEG_FORMATS + _MINIAUDIO_FORMATS:
            fallback = ext
        try:
            with open(file, "rb") as f:
                fmt = _detect_format(f, after_id3=fallback or "mp3")
        except OSError:
            fmt = None
        fmt = fmt or ext
    elif isinstance(file, io.BytesIO):
        file.seek(0)
        fmt = _detect_format(file)
        file.seek(0)
        if fmt is None:
            raise ValueError("Unable to detect audio format from bytes")
        if file.read(3) == b"ID3":
            fallback = "mp3"
    else:
        raise TypeError(f"Unsupported file type: {type(file)}")

    def decode(as_format):
        return _read_as(
            file, as_format, always_2d, dtype, sample_rate, nchannels, duration
        )

    if fallback is None or _backend(fallback) == _backend(fmt):
        return decode(fmt)

    import miniaudio

    decode_errors = (RuntimeError, miniaudio.DecodeError)
    try:
        return decode(fmt)
    except decode_errors as error:
        try:
            return decode(fallback)
        except decode_errors:
            raise error from None


def _read_as(
    file: Union[str, Path, io.BytesIO],
    fmt: str,
    always_2d: bool,
    dtype: str,
    sample_rate: Optional[int],
    nchannels: Optional[int],
    duration: Optional[float],
) -> Tuple[np.ndarray, int]:
    """Decode a file as the given format. See read() for the other arguments."""
    decoder = _backend(fmt)
    samples_are_normalized = False
    if decoder == "ffmpeg":
        if isinstance(file, io.BytesIO):
            file.seek(0)
            input_data = file.read()
        else:
            input_data = file
        samples, sample_rate, nchannels = _decode_ffmpeg(
            input_data,
            sample_rate=sample_rate,
            nchannels=nchannels,
            duration=duration,
            input_format="ogg" if fmt == "ogg" else None,
        )
    else:
        # Use miniaudio for other formats
        import miniaudio

        # Get file info to preserve original sample rate and channels.
        if isinstance(file, io.BytesIO):
            file.seek(0)
            miniaudio_input = file.read()
            get_info = {
                "wav": miniaudio.wav_get_info,
                "mp3": miniaudio.mp3_get_info,
                "flac": miniaudio.flac_get_info,
            }[decoder]
            info = get_info(miniaudio_input)
        else:
            miniaudio_input = file
            get_info = {
                "wav": miniaudio.wav_get_file_info,
                "mp3": miniaudio.mp3_get_file_info,
                "flac": miniaudio.flac_get_file_info,
            }.get(decoder, miniaudio.get_file_info)
            info = get_info(str(file))

        output_nchannels = nchannels or info.nchannels
        output_sample_rate = sample_rate or info.sample_rate
        max_frames = None
        if duration is not None:
            max_frames = round(duration * output_sample_rate)
        if sample_rate is not None and sample_rate != info.sample_rate:
            samples = _decode_miniaudio_resampled(
                miniaudio_input,
                info.sample_rate,
                info.num_frames,
                sample_rate,
                output_nchannels,
                max_output_frames=max_frames,
            )
            nchannels = output_nchannels
            samples_are_normalized = True
        elif max_frames is not None:
            samples = _decode_miniaudio_head(
                miniaudio_input, output_sample_rate, output_nchannels, max_frames
            )
            sample_rate = output_sample_rate
            nchannels = output_nchannels
        else:
            if isinstance(miniaudio_input, bytes):
                decoded = miniaudio.decode(
                    miniaudio_input,
                    nchannels=output_nchannels,
                    sample_rate=sample_rate or info.sample_rate,
                )
            else:
                decoded = miniaudio.decode_file(
                    str(miniaudio_input),
                    nchannels=output_nchannels,
                    sample_rate=sample_rate or info.sample_rate,
                )
            sample_rate = decoded.sample_rate
            nchannels = decoded.nchannels

            # miniaudio returns signed 16-bit interleaved samples.
            samples = np.array(decoded.samples, dtype=np.int16)

    # Reshape to (samples, channels) if multi-channel
    if nchannels > 1:
        samples = samples.reshape(-1, nchannels)

    # Convert to requested dtype
    if dtype in ("float32", "float64"):
        samples = samples.astype(dtype, copy=False)
        if not samples_are_normalized:
            samples /= 32768.0
    elif dtype == "int16" and samples_are_normalized:
        samples = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(np.int16)
    elif dtype == "int16":
        pass
    else:
        if samples_are_normalized:
            samples = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(
                np.int16
            )
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
            "ffmpeg is required for MP3/FLAC/WebM encoding and M4A/AAC/WebM decoding.\n"
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
        format: Output format (mp3, flac, ogg, opus, vorbis, etc.)
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
    elif format == "opus":
        cmd.extend(["-c:a", "libopus", "-b:a", bitrate])
    elif format == "webm":
        cmd.extend(["-c:a", "libopus", "-b:a", bitrate])
    elif format in ("ogg", "vorbis"):
        # Use FLAC codec in OGG container for maximum compatibility
        # Native vorbis encoder has limitations (experimental, stereo-only)
        # FLAC in OGG provides lossless compression and works with any channels
        cmd.extend(["-c:a", "flac"])

    # Map format to output container format
    if format == "vorbis":
        format = "ogg"  # Vorbis uses OGG container

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
        format: Output format. Supports 'wav', 'flac', 'mp3', 'ogg', 'opus', 'vorbis', 'webm'.
                If None, inferred from file extension.

    Note:
        WAV uses miniaudio for encoding.
        MP3, FLAC, OGG, Opus, Vorbis, and WebM use ffmpeg (must be installed: brew install ffmpeg).
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
        samples_flat = data
    else:
        nchannels = data.shape[1]
        samples_flat = data.flatten()

    if format == "pcm" or format == "raw":
        # Flatten for miniaudio (interleaved)
        pcm_bytes = samples_flat.tobytes()

        if isinstance(file, io.BytesIO):
            file.write(pcm_bytes)
            file.seek(0)
        else:
            with open(file, "wb") as f:
                f.write(pcm_bytes)

    elif format == "wav":
        import array

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

    elif format in ("flac", "mp3", "ogg", "opus", "vorbis", "webm"):
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

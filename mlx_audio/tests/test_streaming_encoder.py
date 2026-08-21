"""Tests for mlx_audio.streaming_encoder.

These cover the defect the module exists to fix: encoding a streamed response
one chunk at a time used to produce one complete container file per chunk,
each carrying its own header, encoder delay and end padding, which showed up
as audible silence at every chunk seam.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av", reason="PyAV not installed")

from mlx_audio.streaming_encoder import SUPPORTED, StreamingEncoder

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
FFPROBE_AVAILABLE = shutil.which("ffprobe") is not None

SAMPLE_RATE = 24000

# One container magic per format, used to assert a single header per response.
CONTAINER_MAGIC = {
    "mp3": b"ID3",
    "webm": bytes.fromhex("1a45dfa3"),
    "flac": b"fLaC",
    "wav": b"RIFF",
}


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def _encode_chunks(fmt: str, chunks) -> bytes:
    encoder = StreamingEncoder(fmt, SAMPLE_RATE, 1)
    out = b"".join(encoder.encode(chunk) for chunk in chunks)
    return out + encoder.finalize()


def _duration(data: bytes, suffix: str) -> float:
    """Decode to WAV and measure that.

    Several muxers report no container-level duration when written to a
    non-seekable buffer, so ``ffprobe`` on the encoded bytes can return "N/A".
    Decoding first gives a reliable sample count either way.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        src = handle.name
    dst = src + ".probe.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                src,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                dst,
            ],
            capture_output=True,
            check=True,
        )
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                dst,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    finally:
        Path(src).unlink(missing_ok=True)
        Path(dst).unlink(missing_ok=True)


class TestStreamingEncoder:
    def test_supported_formats_all_constructible(self):
        """Every advertised format must actually build an encoder."""
        for fmt in SUPPORTED:
            encoder = StreamingEncoder(fmt, SAMPLE_RATE, 1)
            encoder.finalize()

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError):
            StreamingEncoder("not-a-format", SAMPLE_RATE, 1)

    @pytest.mark.parametrize("fmt", sorted(CONTAINER_MAGIC))
    def test_single_container_header(self, fmt):
        """A streamed response is one container, not one per chunk."""
        data = _encode_chunks(fmt, [_tone(1.0) for _ in range(5)])
        assert data.count(CONTAINER_MAGIC[fmt]) == 1

    @pytest.mark.skipif(
        not (FFMPEG_AVAILABLE and FFPROBE_AVAILABLE),
        reason="ffmpeg/ffprobe not installed",
    )
    @pytest.mark.parametrize(
        "fmt,tolerance",
        [
            ("mp3", 0.05),  # mp3 pads to a whole final frame
            ("webm", 0.01),
            ("opus", 0.01),
            ("flac", 0.01),
            ("wav", 0.01),
        ],
    )
    def test_duration_matches_input(self, fmt, tolerance):
        """The regression test: no silence accumulates at chunk seams."""
        chunks = [_tone(1.0) for _ in range(5)]
        expected = sum(len(chunk) for chunk in chunks) / SAMPLE_RATE
        suffix = "." + ("ogg" if fmt in ("ogg", "vorbis") else fmt)
        actual = _duration(_encode_chunks(fmt, chunks), suffix)
        assert abs(actual - expected) < tolerance, (
            f"{fmt}: {actual:.3f}s from {expected:.3f}s of input "
            f"({(actual - expected) * 1000:+.0f}ms)"
        )

    @pytest.mark.skipif(
        not (FFMPEG_AVAILABLE and FFPROBE_AVAILABLE),
        reason="ffmpeg/ffprobe not installed",
    )
    def test_many_small_chunks_do_not_accumulate_silence(self):
        """Halving the chunk size must not change the output duration."""
        few = _duration(_encode_chunks("mp3", [_tone(1.0) for _ in range(4)]), ".mp3")
        many = _duration(
            _encode_chunks("mp3", [_tone(0.25) for _ in range(16)]), ".mp3"
        )
        assert abs(few - many) < 0.05

    def test_finalize_is_idempotent(self):
        encoder = StreamingEncoder("mp3", SAMPLE_RATE, 1)
        encoder.encode(_tone(0.5))
        assert encoder.finalize() is not None
        assert encoder.finalize() == b""

    def test_encode_after_finalize_raises(self):
        encoder = StreamingEncoder("mp3", SAMPLE_RATE, 1)
        encoder.encode(_tone(0.1))
        encoder.finalize()
        with pytest.raises(RuntimeError):
            encoder.encode(_tone(0.1))

    def test_empty_chunk_is_ignored(self):
        encoder = StreamingEncoder("mp3", SAMPLE_RATE, 1)
        assert encoder.encode(np.array([], dtype=np.float32)) == b""
        encoder.finalize()

    def test_int16_input_passes_through(self):
        """int16 input must not be rescaled."""
        pcm = (_tone(0.5) * 32767).astype(np.int16)
        assert len(_encode_chunks("wav", [pcm])) > 0

    @pytest.mark.skipif(
        not (FFMPEG_AVAILABLE and FFPROBE_AVAILABLE),
        reason="ffmpeg/ffprobe not installed",
    )
    def test_loud_float_is_not_silenced(self):
        """Float input above 1.0 clips, rather than being read as int16 scale."""
        loud = (_tone(0.5) * 2.8).astype(np.float32)
        data = _encode_chunks("wav", [loud])
        peak = np.abs(np.frombuffer(data[44:], dtype=np.int16)).max()
        assert peak > 30000, f"loud audio encoded near-silent (peak {peak})"

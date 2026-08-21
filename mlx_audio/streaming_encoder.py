"""Incremental audio encoder for streaming responses.

The batch encoder (``audio_io._encode_ffmpeg``) spawns one ffmpeg per call and
waits for it to exit, so every streamed chunk becomes a complete standalone
container file: its own header, its own encoder delay, its own end padding.
Concatenated on the wire that yields one container per chunk and an audible
gap at every seam.

This keeps a single libav container + encoder open for the whole response and
returns only the bytes produced by each chunk, so a response is one container
with one header.

The incremental-encoding approach (hold one container open, drain the buffer
after each chunk, flush the encoder on finalize) follows Kokoro-FastAPI's
``StreamingAudioWriter``:

  https://github.com/remsky/Kokoro-FastAPI
  Copyright (c) remsky, licensed under the Apache License 2.0
  api/src/services/streaming_audio_writer.py

including its ordering fix for Ogg muxers, which write their final page during
close (Kokoro-FastAPI #497).
"""

from io import BytesIO
from typing import Optional

import numpy as np

# Codec + container per response_format, mirroring ``audio_io._encode_ffmpeg``
# so streamed and non-streamed output stay byte-compatible in codec choice.
# ogg/vorbis deliberately use FLAC-in-Ogg: the native vorbis encoder is
# experimental and stereo-only, per the comment in _encode_ffmpeg.
_CODECS = {
    "mp3": "mp3",
    "opus": "libopus",
    "webm": "libopus",
    "ogg": "flac",
    "vorbis": "flac",
    "flac": "flac",
    "aac": "aac",
    "wav": "pcm_s16le",
}

# Container (libav muxer) name when it differs from the requested format.
_CONTAINERS = {
    "aac": "adts",
    "vorbis": "ogg",
}

# Formats that carry a bitrate setting; the rest are lossless or PCM.
_BITRATE_FORMATS = ("mp3", "opus", "webm", "aac")

SUPPORTED = frozenset(_CODECS)


class StreamingEncoder:
    """One open container per response; ``encode()`` returns incremental bytes."""

    def __init__(
        self, fmt: str, sample_rate: int, channels: int = 1, bit_rate: int = 128000
    ):
        import av

        self.format = fmt.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.pts = 0
        self._closed = False

        codec = _CODECS.get(self.format)
        if codec is None:
            raise ValueError(f"StreamingEncoder: unsupported format {fmt!r}")

        self.buffer = BytesIO()
        container_fmt = _CONTAINERS.get(self.format, self.format)
        self.container = av.open(self.buffer, mode="w", format=container_fmt)
        self.stream = self.container.add_stream(
            codec,
            rate=self.sample_rate,
            layout="mono" if channels == 1 else "stereo",
        )
        if self.format in _BITRATE_FORMATS:
            self.stream.bit_rate = bit_rate

    def _drain(self) -> bytes:
        data = self.buffer.getvalue()
        self.buffer.seek(0)
        self.buffer.truncate(0)
        return data

    def encode(self, audio: np.ndarray) -> bytes:
        import av

        if self._closed:
            raise RuntimeError("StreamingEncoder already finalized")
        if audio is None or len(audio) == 0:
            return b""

        # Match audio_io.write()'s conversion exactly so streamed and
        # non-streamed output are byte-comparable: float input is treated as
        # normalized, clipped to [-1, 1], then scaled. No peak guessing.
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        frame = av.AudioFrame.from_ndarray(
            audio.reshape(1, -1),
            format="s16",
            layout="mono" if self.channels == 1 else "stereo",
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        self.pts += frame.samples

        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        return self._drain()

    def finalize(self) -> bytes:
        """Flush the encoder tail and close. Safe to call more than once."""
        if self._closed:
            return b""
        self._closed = True
        try:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        except Exception:
            pass
        # Muxers differ in when their trailing bytes land in the buffer:
        # Ogg-family and Matroska/WebM write their final page/cluster during
        # close(), so the buffer must be read AFTER closing. Other muxers only
        # seek back to patch headers at position 0, which would be lost if the
        # buffer were truncated first, so read BEFORE closing.
        # (cf. Kokoro-FastAPI #497)
        if self.format in ("ogg", "vorbis", "opus", "webm"):
            self.container.close()
            data = self.buffer.getvalue()
        else:
            data = self.buffer.getvalue()
            self.container.close()
        self.buffer.close()
        return data

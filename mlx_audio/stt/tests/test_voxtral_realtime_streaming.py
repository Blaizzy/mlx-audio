"""Unit tests for Voxtral Realtime streaming ingestion primitives.

Covers the pieces that do not require downloaded model weights:
- StreamingAudioSource: thread-safe blocking queue of audio samples.
- StreamingMel: parity with compute_mel_spectrogram under many
  chunking regimes.
- StreamingCausalConv1d: parity with CausalConv1d on a randomly
  initialized module (no quantized checkpoint needed).

The end-to-end (encoder + session) parity lives in a weight-gated
test class that skips cleanly when the model checkpoint is absent.
"""

import os
import threading
import time
import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.voxtral_realtime.audio import (
    compute_mel_filters,
    compute_mel_spectrogram,
)
from mlx_audio.stt.models.voxtral_realtime.encoder import CausalConv1d
from mlx_audio.stt.models.voxtral_realtime.streaming import (
    StreamingAudioSource,
    StreamingCausalConv1d,
    StreamingMel,
)


def _make_audio(n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 16000.0
    sig = (
        0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2 * np.pi * 1100.0 * t)
        + 0.05 * rng.standard_normal(n_samples)
    )
    return sig.astype(np.float32)


class TestStreamingAudioSource(unittest.TestCase):
    def test_append_then_read_coalesces(self):
        src = StreamingAudioSource()
        src.append(np.ones(100, dtype=np.float32))
        src.append(np.full(50, 2.0, dtype=np.float32))
        samples, closed = src.read(timeout=0.1)
        self.assertFalse(closed)
        self.assertEqual(samples.shape, (150,))
        self.assertTrue(np.allclose(samples[:100], 1.0))
        self.assertTrue(np.allclose(samples[100:], 2.0))

    def test_close_flushes_and_signals(self):
        src = StreamingAudioSource()
        src.append(np.arange(10, dtype=np.float32))
        src.close()
        samples, closed = src.read(timeout=0.1)
        self.assertEqual(samples.shape, (10,))
        self.assertTrue(closed)

    def test_empty_append_is_noop(self):
        src = StreamingAudioSource()
        src.append(np.zeros(0, dtype=np.float32))
        src.close()
        samples, closed = src.read(timeout=0.1)
        self.assertEqual(samples.size, 0)
        self.assertTrue(closed)

    def test_dtype_coerced_to_float32(self):
        src = StreamingAudioSource()
        src.append(np.arange(5, dtype=np.float64))
        src.close()
        samples, _ = src.read(timeout=0.1)
        self.assertEqual(samples.dtype, np.float32)

    def test_read_blocks_until_producer_appends(self):
        src = StreamingAudioSource()
        observed: list[tuple[np.ndarray, bool]] = []

        def consumer():
            observed.append(src.read(timeout=2.0))

        t = threading.Thread(target=consumer)
        t.start()
        # Give the consumer a moment to block on the empty queue.
        time.sleep(0.05)
        src.append(np.ones(8, dtype=np.float32))
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(observed), 1)
        samples, closed = observed[0]
        self.assertEqual(samples.shape, (8,))
        self.assertFalse(closed)


class TestStreamingMelParity(unittest.TestCase):
    """StreamingMel ∘ close == compute_mel_spectrogram for any chunking."""

    @classmethod
    def setUpClass(cls):
        cls.mel_filters = mx.array(compute_mel_filters(), dtype=mx.float32)

    def _batch(self, audio: np.ndarray) -> np.ndarray:
        mel = compute_mel_spectrogram(mx.array(audio), self.mel_filters)
        mx.eval(mel)
        return np.array(mel)

    def _streaming(self, audio: np.ndarray, chunk: int) -> np.ndarray:
        sm = StreamingMel(self.mel_filters)
        pieces: list[np.ndarray] = []
        for start in range(0, len(audio), chunk):
            out = sm.append(audio[start : start + chunk])
            if out is not None:
                pieces.append(np.array(out))
        tail = sm.close()
        if tail is not None:
            pieces.append(np.array(tail))
        if not pieces:
            return np.zeros((128, 0), dtype=np.float32)
        return np.concatenate(pieces, axis=1)

    def test_parity_matrix(self):
        cases = [
            (16000, 160),
            (16000, 800),
            (16000, 3200),
            (16001, 3200),
            (1600, 200),
            (24157, 1000),
        ]
        tol = 1e-4
        for n_samples, chunk in cases:
            with self.subTest(n_samples=n_samples, chunk=chunk):
                audio = _make_audio(n_samples)
                batch = self._batch(audio)
                stream = self._streaming(audio, chunk)
                self.assertEqual(batch.shape, stream.shape)
                self.assertLess(float(np.abs(batch - stream).max()), tol)

    def test_close_is_idempotent(self):
        sm = StreamingMel(self.mel_filters)
        sm.append(_make_audio(1000))
        sm.close()
        # Second close must be a no-op and must not raise.
        self.assertIsNone(sm.close())


class TestStreamingCausalConv1dParity(unittest.TestCase):
    """Streaming wrapper matches the batch CausalConv1d it wraps."""

    def _run_case(self, kernel_size: int, stride: int, n_in: int, chunk: int):
        mx.random.seed(0)
        in_channels, out_channels = 8, 12
        conv = CausalConv1d(in_channels, out_channels, kernel_size, stride=stride)
        # Force-initialize params (MLX Modules lazy init on first call).
        _ = conv(mx.zeros((1, kernel_size, in_channels)))

        x_np = (
            np.random.default_rng(1)
            .standard_normal((n_in, in_channels))
            .astype(np.float32)
        )
        x = mx.array(x_np)

        # Batch reference: CausalConv1d expects [batch, seq, channels].
        batch_out = conv(x[None, :, :]).squeeze(0)
        mx.eval(batch_out)
        batch_np = np.array(batch_out)

        # Streaming: feed chunks, concatenate outputs.
        sc = StreamingCausalConv1d(conv)
        pieces: list[mx.array] = []
        for start in range(0, n_in, chunk):
            piece = sc.step(x[start : start + chunk])
            if piece.shape[0] > 0:
                pieces.append(piece)
        stream_out = (
            mx.concatenate(pieces, axis=0) if pieces else mx.zeros((0, out_channels))
        )
        mx.eval(stream_out)
        stream_np = np.array(stream_out)

        self.assertEqual(batch_np.shape, stream_np.shape)
        self.assertLess(float(np.abs(batch_np - stream_np).max()), 1e-5)

    def test_stride_1(self):
        for chunk in (1, 3, 8, 17):
            with self.subTest(chunk=chunk):
                self._run_case(kernel_size=5, stride=1, n_in=40, chunk=chunk)

    def test_stride_2(self):
        for chunk in (1, 2, 4, 8, 16):
            with self.subTest(chunk=chunk):
                self._run_case(kernel_size=6, stride=2, n_in=40, chunk=chunk)

    def test_kernel_equals_stride(self):
        # Edge case: no state carried between calls.
        for chunk in (4, 8, 12):
            with self.subTest(chunk=chunk):
                self._run_case(kernel_size=4, stride=4, n_in=40, chunk=chunk)


def _voxtral_weights_path() -> str:
    return os.environ.get(
        "VOXTRAL_REALTIME_MODEL",
        os.path.expanduser("~/.omlx/models/Voxtral-Mini-4B-Realtime-2602-4bit"),
    )


def _voxtral_weights_available() -> bool:
    path = _voxtral_weights_path()
    return os.path.isdir(path) and any(
        f.endswith(".safetensors") for f in os.listdir(path)
    )


@unittest.skipUnless(
    _voxtral_weights_available(),
    "Voxtral Realtime weights not available (set VOXTRAL_REALTIME_MODEL to enable)",
)
class TestVoxtralStreamingEndToEnd(unittest.TestCase):
    """Parity vs batch, using real quantized weights. Skipped in CI."""

    @classmethod
    def setUpClass(cls):
        from mlx_audio.stt.utils import load_model

        cls.model = load_model(_voxtral_weights_path())

    def test_encoder_streaming_parity(self):
        from mlx_audio.stt.models.voxtral_realtime.streaming import (
            StreamingConvStem,
            StreamingDownsampler,
            StreamingEncoder,
        )

        encoder = self.model.encoder
        rng = np.random.default_rng(0)
        # 320 mel frames is divisible by 8 -> conv_stem out divisible by ds=4,
        # which keeps the streaming output aligned with the batch output (no
        # front-trunc mismatch).
        mel = mx.array((rng.standard_normal((128, 320)) * 0.5).astype(np.float32))

        batch_out = encoder(mel)
        mx.eval(batch_out)

        cs = StreamingConvStem(encoder)
        enc = StreamingEncoder(encoder)
        ds = StreamingDownsampler(encoder)
        pieces: list[mx.array] = []
        chunk = 40
        for start in range(0, mel.shape[1], chunk):
            mel_chunk = mel[:, start : start + chunk]
            out = ds.step(enc.step(cs.step(mel_chunk)))
            if out.shape[0] > 0:
                pieces.append(out)
        stream_out = mx.concatenate(pieces, axis=0)
        mx.eval(stream_out)

        self.assertEqual(tuple(batch_out.shape), tuple(stream_out.shape))
        diff = float(mx.max(mx.abs(batch_out - stream_out)).item())
        # Permissive: 4-bit quant + many ops. Scripts observe ~1e-2.
        self.assertLess(diff, 2e-2)

    def test_session_runs_to_completion(self):
        """Drip-feed silence; session must terminate and yield str deltas."""
        audio = np.zeros(16000 * 2, dtype=np.float32)

        sess = self.model.create_streaming_session()
        chunk = 3200  # 200 ms
        for start in range(0, len(audio), chunk):
            sess.feed(audio[start : start + chunk])
        sess.close()

        pieces: list[str] = []
        steps = 0
        while not sess.done:
            pieces.extend(sess.step(max_decode_tokens=8))
            steps += 1
            self.assertLess(steps, 2000, "session.step made no progress")
        self.assertTrue(all(isinstance(p, str) for p in pieces))


if __name__ == "__main__":
    unittest.main()

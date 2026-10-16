import unittest
from threading import Thread
from unittest import mock

import numpy as np
import sounddevice as sd

from mlx_audio.tts.audio_player import AudioPlayer


class _FakeOutputStream:
    """Model callback consumption separately from completion at the device."""

    def __init__(
        self,
        callback,
        blocksize,
        played,
        finished_callback=None,
        auto_pump=True,
        auto_finish=True,
        **kwargs,
    ):
        self.callback = callback
        self.blocksize = blocksize
        self.played = played
        self.finished_callback = finished_callback
        self.auto_pump = auto_pump
        self.auto_finish = auto_finish
        self.started = False
        self.closed = False
        self.aborted = False
        self.pending_output = False

    def start(self):
        self.started = True
        if self.auto_pump:
            # PortAudio can invoke callbacks before start() returns.
            self.pump()

    def pump(self):
        while True:
            outdata = np.zeros((self.blocksize, 1), dtype=np.float32)
            try:
                self.callback(outdata, self.blocksize, None, None)
            except sd.CallbackStop:
                self.played.append(outdata[:, 0].copy())
                self.pending_output = True
                if self.auto_finish:
                    self.finish()
                return
            self.played.append(outdata[:, 0].copy())

    def finish(self):
        self.pending_output = False
        if self.finished_callback:
            self.finished_callback()

    def stop(self):
        self.finish()

    def abort(self):
        self.aborted = True
        self.finish()

    def close(self):
        self.closed = True


class TestAudioPlayer(unittest.TestCase):
    def setUp(self):
        self.player = AudioPlayer(sample_rate=24000)
        self.streams = []
        self.played = []
        self.auto_pump = True
        self.auto_finish = True
        stream_patch = mock.patch(
            "mlx_audio.tts.audio_player.sd.OutputStream",
            side_effect=self._create_stream,
        )
        self.stream_cls = stream_patch.start()
        self.addCleanup(stream_patch.stop)
        # A lost drain notification must fail the test instead of hanging CI.
        wait = self.player.drain_event.wait

        def bounded_wait(timeout=None):
            self.assertTrue(wait(timeout=1), "Audio player lost its drain notification")
            return True

        wait_patch = mock.patch.object(
            self.player.drain_event, "wait", side_effect=bounded_wait
        )
        wait_patch.start()
        self.addCleanup(wait_patch.stop)

    def _create_stream(self, **kwargs):
        self.assertTrue(
            all(stream.closed for stream in self.streams),
            "Previous stream must be closed before opening another",
        )
        stream = _FakeOutputStream(
            played=self.played,
            auto_pump=self.auto_pump,
            auto_finish=self.auto_finish,
            **kwargs,
        )
        self.streams.append(stream)
        return stream

    def _assert_played(self, samples):
        played = np.concatenate(self.played)
        np.testing.assert_array_equal(played[: len(samples)], samples)
        np.testing.assert_array_equal(played[len(samples) :], 0)

    def test_callback_accepts_column_vector_audio(self):
        samples = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)

        self.player.queue_audio(samples)
        outdata = np.zeros((4, 1), dtype=np.float32)
        self.player.callback(outdata, frames=4, time=None, status=None)

        np.testing.assert_allclose(outdata[:, 0], samples[:, 0])

    def test_queue_audio_downmixes_stereo_audio(self):
        samples = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)

        self.player.queue_audio(samples)

        self.assertEqual(len(self.player.audio_buffer), 1)
        np.testing.assert_allclose(self.player.audio_buffer[0], np.array([0.1, 0.5]))

    def test_stop_plays_short_audio_below_buffer_threshold(self):
        # Include exact callback block sizes and silence within the utterance.
        for size in (3, 2048, 4096, 12000):
            with self.subTest(size=size):
                self.played.clear()
                samples = np.linspace(-0.5, 0.5, size, dtype=np.float32)
                self.player.queue_audio(samples)
                self.assertFalse(self.player.playing)

                self.player.stop()

                self._assert_played(samples)
                self.assertEqual(self.player.buffered_samples(), 0)
                self.assertFalse(self.player.playing)
                self.assertIsNone(self.player.stream)
                self.assertTrue(self.streams[-1].closed)

    def test_stop_with_nothing_buffered_does_not_start_stream(self):
        self.player.stop()
        self.player.stop()

        self.stream_cls.assert_not_called()
        self.assertFalse(self.player.playing)

    def test_auto_start_does_not_overwrite_synchronous_callback_state(self):
        samples = np.full(36000, 0.25, dtype=np.float32)
        self.player.queue_audio(samples)

        self.assertFalse(self.player.playing)
        self.assertTrue(self.player.drain_event.is_set())
        self.player.stop()
        self._assert_played(samples)
        self.assertEqual(len(self.streams), 1)
        self.assertTrue(self.streams[0].closed)

    def test_stop_plays_audio_from_callback_thread(self):
        workers = []
        errors = []

        def create_threaded_stream(**kwargs):
            stream = self._create_stream(**kwargs)

            def pump():
                try:
                    stream.pump()
                except BaseException as error:
                    errors.append(error)

            def start():
                worker = Thread(target=pump, daemon=True)
                workers.append(worker)
                worker.start()

            stream.start = start
            return stream

        self.stream_cls.side_effect = create_threaded_stream
        samples = np.linspace(-0.5, 0.5, 12000, dtype=np.float32)
        self.player.queue_audio(samples)
        try:
            self.player.stop()
        finally:
            for worker in workers:
                worker.join(timeout=1)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(errors, [])
        self._assert_played(samples)
        self.assertTrue(self.streams[0].closed)

    def test_wait_for_drain_then_stop_closes_stream(self):
        self.player.queue_audio(np.ones(3, dtype=np.float32))
        self.player.wait_for_drain()
        self.player.stop()

        self.assertEqual(len(self.streams), 1)
        self.assertTrue(self.streams[0].closed)
        self.assertIsNone(self.player.stream)

    def test_wait_for_drain_waits_for_device_after_buffer_is_empty(self):
        self.auto_pump = False
        self.auto_finish = False
        self.player.queue_audio(np.ones(3, dtype=np.float32))
        self.player.start_stream()
        stream = self.streams[0]
        stream.pump()

        self.assertEqual(self.player.buffered_samples(), 0)
        self.assertFalse(self.player.playing)
        self.assertFalse(self.player.drain_event.is_set())

        def finish_on_wait():
            self.assertTrue(stream.pending_output)
            stream.finish()
            return True

        with mock.patch.object(
            self.player.drain_event, "wait", side_effect=finish_on_wait
        ) as wait:
            self.player.wait_for_drain()
            wait.assert_called_once_with()
        self.player.stop()

    def test_underrun_closes_old_stream_before_playing_tail(self):
        self.auto_pump = False
        first = np.full(3, 0.25, dtype=np.float32)
        self.player.queue_audio(first)
        self.player.start_stream()
        self.streams[0].pump()
        self._assert_played(first)

        self.auto_pump = True
        self.played.clear()
        tail = np.full(5, 0.5, dtype=np.float32)
        self.player.queue_audio(tail)
        self.player.stop()

        self._assert_played(tail)
        self.assertEqual(len(self.streams), 2)
        self.assertTrue(all(stream.closed for stream in self.streams))

    def test_flush_discards_audio_before_auto_start(self):
        self.player.queue_audio(np.ones(3, dtype=np.float32))
        self.player.flush()
        self.assertEqual(self.player.buffered_samples(), 0)
        self.player.stop()

        self.stream_cls.assert_not_called()
        self.assertTrue(self.player.drain_event.is_set())

    def test_flush_aborts_active_stream(self):
        self.auto_pump = False
        self.player.queue_audio(np.ones(3, dtype=np.float32))
        self.player.start_stream()
        self.player.flush()
        self.player.stop()

        self.assertEqual(self.player.buffered_samples(), 0)
        self.assertTrue(self.streams[0].aborted)
        self.assertTrue(self.streams[0].closed)
        self.assertIsNone(self.player.stream)
        self.assertFalse(self.player.playing)
        self.assertTrue(self.player.drain_event.is_set())
        self.assertEqual(self.played, [])

    def test_flush_closes_stream_after_underrun(self):
        self.auto_pump = False
        self.player.queue_audio(np.ones(3, dtype=np.float32))
        self.player.start_stream()
        self.streams[0].pump()
        self.assertFalse(self.player.playing)

        self.player.flush()

        self.assertTrue(self.streams[0].closed)
        self.assertIsNone(self.player.stream)

    def test_start_failure_closes_stream_and_allows_retry(self):
        self.player.queue_audio(np.ones(3, dtype=np.float32))
        stream = mock.Mock()
        stream.start.side_effect = RuntimeError("start failed")
        with mock.patch(
            "mlx_audio.tts.audio_player.sd.OutputStream", return_value=stream
        ):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                self.player.stop()

        stream.close.assert_called_once_with()
        self.assertIsNone(self.player.stream)
        self.assertFalse(self.player.playing)
        self.player.stop()
        self._assert_played(np.ones(3, dtype=np.float32))

    def test_stop_closes_stream_when_drain_is_interrupted(self):
        self.auto_pump = False
        self.player.queue_audio(np.ones(3, dtype=np.float32))
        self.player.start_stream()
        with mock.patch.object(
            self.player, "wait_for_drain", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.player.stop()

        self.assertTrue(self.streams[0].closed)
        self.assertIsNone(self.player.stream)
        self.assertFalse(self.player.playing)

    def test_stop_stream_closes_even_when_stop_fails(self):
        stream = mock.Mock()
        stream.stop.side_effect = RuntimeError("stop failed")
        self.player.stream = stream
        self.player.playing = True

        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            self.player.stop_stream()

        stream.close.assert_called_once_with()
        self.assertIsNone(self.player.stream)
        self.assertFalse(self.player.playing)
        self.assertTrue(self.player.drain_event.is_set())


if __name__ == "__main__":
    unittest.main()

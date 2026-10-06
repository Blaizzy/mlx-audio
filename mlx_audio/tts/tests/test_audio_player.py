import threading
import time
import unittest
from unittest import mock

import numpy as np

from mlx_audio.tts.audio_player import AudioPlayer


class _FakeOutputStream:
    """Stands in for sd.OutputStream: start() pumps the callback to completion."""

    def __init__(self, callback, blocksize, played, **kwargs):
        self.callback = callback
        self.blocksize = blocksize
        self.played = played

    def start(self):
        # Play from a thread, like PortAudio does, so start_stream() can finish
        # setting its state before the callback drains the buffer.
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        import sounddevice as sd

        time.sleep(0.05)
        while True:
            outdata = np.zeros((self.blocksize, 1), dtype=np.float32)
            try:
                self.callback(outdata, self.blocksize, None, None)
            except sd.CallbackStop:
                self.played.append(outdata[:, 0].copy())
                return
            self.played.append(outdata[:, 0].copy())

    def stop(self):
        pass

    def close(self):
        pass


class TestAudioPlayer(unittest.TestCase):
    def test_callback_accepts_column_vector_audio(self):
        player = AudioPlayer(sample_rate=24000)
        samples = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)

        player.queue_audio(samples)
        outdata = np.zeros((4, 1), dtype=np.float32)
        player.callback(outdata, frames=4, time=None, status=None)

        np.testing.assert_allclose(outdata[:, 0], samples[:, 0])

    def test_queue_audio_downmixes_stereo_audio(self):
        player = AudioPlayer(sample_rate=24000)
        samples = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)

        player.queue_audio(samples)

        self.assertEqual(len(player.audio_buffer), 1)
        np.testing.assert_allclose(player.audio_buffer[0], np.array([0.1, 0.5]))

    def _play(self, chunks):
        played = []
        player = AudioPlayer(sample_rate=24000)
        factory = lambda **kw: _FakeOutputStream(played=played, **kw)
        with (
            mock.patch(
                "mlx_audio.tts.audio_player.sd.OutputStream", side_effect=factory
            ) as stream_cls,
            mock.patch("mlx_audio.tts.audio_player.sd.sleep"),
        ):
            for chunk in chunks:
                player.queue_audio(chunk)
            player.stop()
        return player, stream_cls, (np.concatenate(played) if played else np.array([]))

    def test_stop_plays_short_audio_below_buffer_threshold(self):
        # 0.5 s is below min_buffer_seconds, so the stream never auto-starts.
        samples = np.full(12000, 0.25, dtype=np.float32)

        player, stream_cls, played = self._play([samples])

        stream_cls.assert_called_once()
        self.assertEqual(int(np.count_nonzero(played)), len(samples))
        self.assertEqual(player.buffered_samples(), 0)
        self.assertFalse(player.playing)
        self.assertIsNone(player.stream)

    def test_stop_with_nothing_buffered_does_not_start_stream(self):
        player, stream_cls, _ = self._play([])

        stream_cls.assert_not_called()
        self.assertFalse(player.playing)


if __name__ == "__main__":
    unittest.main()

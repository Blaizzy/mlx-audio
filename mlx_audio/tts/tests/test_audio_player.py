import unittest

import numpy as np

from mlx_audio.tts.audio_player import AudioPlayer


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


if __name__ == "__main__":
    unittest.main()

import unittest

import mlx.core as mx

from ..models.mimi.mimi import Mimi, mimi_202407


class TestMimi(unittest.TestCase):
    def test_mimi_model(self):
        """Test Mimi model encoding and decoding."""
        model = Mimi(mimi_202407(32))

        audio = mx.zeros((1, 1, 120_000))
        codes = model.encode(audio)
        self.assertEqual(codes.shape, (1, 32, 63))

        audio_out = model.decode(codes)
        self.assertEqual(audio_out.shape, (1, 1, 120_960))


if __name__ == "__main__":
    unittest.main()

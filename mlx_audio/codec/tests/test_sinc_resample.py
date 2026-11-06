import tracemalloc
import unittest

import numpy as np

from ..models.higgs_audio.higgs_audio import _sinc_resample


class TestSincResample(unittest.TestCase):
    def test_coprime_rates_match_torchaudio_in_little_memory(self):
        # torchaudio 2.8.0 in float64: functional.resample(x.double(), 499, 500).
        # In float32, torchaudio's own kernel drifts by up to 2e-5 at this ratio.
        expected = {
            0: [-0.3124536, -0.005088, 0.3010581, -0.388916]
            + [-0.0923166, 0.2225614, -0.4652202, -0.180068],
            246: [-0.281015, 0.3780479, 0.0068574, -0.3535656]
            + [0.2936403, -0.0691907, -0.4379988, 0.2211303],
            492: [0.1187831, 0.4067028, -0.2859417, 0.0339388]
            + [0.3272217, -0.3622061, -0.0544791, 0.2518446],
        }
        n = np.arange(499, dtype=np.int64)
        x = (((20077 * n + 12345) % 65536 - 32768) // 2 / 32768).astype(np.float32)
        # Measured from here, also when tracing already runs (PYTHONTRACEMALLOC).
        started = not tracemalloc.is_tracing()
        if started:
            tracemalloc.start()
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        try:
            y = _sinc_resample(x, 499, 500)
            peak = tracemalloc.get_traced_memory()[1] - before
        finally:
            if started:
                tracemalloc.stop()
        self.assertEqual(len(y), 500)
        for start, values in expected.items():
            np.testing.assert_allclose(y[start : start + 8], values, atol=1e-6)
        # torchaudio's dense kernel for 499 -> 500 alone takes 2 MB; rates
        # such as 44099 -> 44100 would take 15 GB.
        self.assertLess(peak, 512 * 1024)


if __name__ == "__main__":
    unittest.main()

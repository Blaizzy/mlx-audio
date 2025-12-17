"""Test that lazy imports work correctly for modular installation."""

import sys
import unittest


class TestLazyImports(unittest.TestCase):
    """Test utils modules don't eagerly import optional dependencies."""

    def test_stt_utils_no_eager_imports(self):
        """Importing stt.utils should not import soundfile or scipy."""
        if "mlx_audio.stt.utils" in sys.modules:
            self.skipTest("stt.utils already imported")

        import mlx_audio.stt.utils  # noqa: F401

        self.assertNotIn("soundfile", sys.modules)
        self.assertNotIn("scipy", sys.modules)
        self.assertNotIn("scipy.signal", sys.modules)

    def test_tts_utils_no_eager_imports(self):
        """Importing tts.utils should not import transformers or mlx_lm."""
        if "mlx_audio.tts.utils" in sys.modules:
            self.skipTest("tts.utils already imported")

        import mlx_audio.tts.utils  # noqa: F401

        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("mlx_lm", sys.modules)
        self.assertNotIn("mlx_lm.utils", sys.modules)
        self.assertNotIn("mlx_lm.convert", sys.modules)


if __name__ == "__main__":
    unittest.main()

"""Tests for MossFormer2 SE model."""

import unittest

import mlx.core as mx


class TestMossFormer2SEConfig(unittest.TestCase):
    """Tests for MossFormer2 SE configuration."""

    def test_config_defaults(self):
        """Test MossFormer2SEConfig default values."""
        from mlx_audio.sts.models.mossformer2_se.config import MossFormer2SEConfig

        config = MossFormer2SEConfig()
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.win_len, 1920)
        self.assertEqual(config.win_inc, 384)
        self.assertEqual(config.fft_len, 1920)
        self.assertEqual(config.num_mels, 60)
        self.assertEqual(config.win_type, "hamming")
        self.assertEqual(config.preemphasis, 0.97)
        self.assertEqual(config.in_channels, 180)
        self.assertEqual(config.out_channels, 512)
        self.assertEqual(config.out_channels_final, 961)

    def test_config_from_dict(self):
        """Test MossFormer2SEConfig.from_dict method."""
        from mlx_audio.sts.models.mossformer2_se.config import MossFormer2SEConfig

        config_dict = {
            "sample_rate": 44100,
            "num_mels": 80,
        }

        config = MossFormer2SEConfig.from_dict(config_dict)
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.num_mels, 80)
        # Default values should be preserved
        self.assertEqual(config.win_len, 1920)

    def test_config_to_dict(self):
        """Test MossFormer2SEConfig.to_dict method."""
        from mlx_audio.sts.models.mossformer2_se.config import MossFormer2SEConfig

        config = MossFormer2SEConfig()
        config_dict = config.to_dict()

        self.assertIn("sample_rate", config_dict)
        self.assertIn("win_len", config_dict)
        self.assertIn("num_mels", config_dict)
        self.assertEqual(config_dict["sample_rate"], 48000)

    def test_config_sampling_rate_alias(self):
        """Test sampling_rate property alias."""
        from mlx_audio.sts.models.mossformer2_se.config import MossFormer2SEConfig

        config = MossFormer2SEConfig()
        self.assertEqual(config.sampling_rate, config.sample_rate)


class TestSTFT(unittest.TestCase):
    """Tests for STFT utilities."""

    def test_create_window_hamming(self):
        """Test hamming window creation."""
        from mlx_audio.sts.models.mossformer2_se.stft import create_window

        window = create_window("hamming", 1920, periodic=False)
        mx.eval(window)

        self.assertEqual(window.shape[0], 1920)
        # Hamming window should have non-zero edges
        self.assertGreater(float(window[0]), 0)

    def test_create_window_hann(self):
        """Test hann window creation."""
        from mlx_audio.sts.models.mossformer2_se.stft import create_window

        window = create_window("hann", 1920, periodic=False)
        mx.eval(window)

        self.assertEqual(window.shape[0], 1920)

    def test_stft_shape(self):
        """Test STFT output shape."""
        from mlx_audio.sts.models.mossformer2_se.stft import create_window, stft

        window = create_window("hamming", 1920, periodic=False)
        audio = mx.zeros((1, 48000))

        real_part, imag_part = stft(audio, 1920, 384, 1920, window, center=False)
        mx.eval(real_part, imag_part)

        # Check output shapes
        self.assertEqual(real_part.shape[0], 1)
        self.assertEqual(imag_part.shape[0], 1)
        self.assertEqual(real_part.shape[1], 961)  # n_fft // 2 + 1

    def test_istft_cache(self):
        """Test ISTFTCache caching behavior."""
        from mlx_audio.sts.models.mossformer2_se.stft import ISTFTCache, create_window

        cache = ISTFTCache()
        window = create_window("hamming", 1920, periodic=False)

        # First call should create cache entries
        norm_buffer = cache.get_norm_buffer(1920, 384, 1920, window, 10)
        positions = cache.get_positions(10, 1920, 384)
        mx.eval(norm_buffer, positions)

        info = cache.cache_info()
        self.assertEqual(info["norm_buffers"], 1)
        self.assertEqual(info["position_indices"], 1)

        # Clear cache
        cache.clear_cache()
        info = cache.cache_info()
        self.assertEqual(info["total_cached_items"], 0)


class TestFeatures(unittest.TestCase):
    """Tests for feature extraction."""

    def test_compute_deltas_shape(self):
        """Test compute_deltas output shape."""
        from mlx_audio.sts.models.mossformer2_se.deltas import compute_deltas

        # Input shape: (freq, time)
        specgram = mx.zeros((60, 100))
        deltas = compute_deltas(specgram, win_length=5)
        mx.eval(deltas)

        self.assertEqual(deltas.shape, specgram.shape)

    def test_fbank_computation(self):
        """Test fbank computation."""
        import numpy as np

        from mlx_audio.sts.models.mossformer2_se.config import MossFormer2SEConfig
        from mlx_audio.sts.models.mossformer2_se.fbank import compute_fbank

        config = MossFormer2SEConfig()
        audio = mx.array(np.random.randn(24000).astype(np.float32))
        fbank = compute_fbank(audio, config)
        mx.eval(fbank)

        # Should return (time, num_mels)
        self.assertEqual(fbank.shape[1], config.num_mels)


class TestModelComponents(unittest.TestCase):
    """Tests for model components."""

    def test_scale_norm(self):
        """Test ScaleNorm layer."""
        from mlx_audio.sts.models.mossformer2_se.scalenorm import ScaleNorm

        layer = ScaleNorm(dim=512)
        x = mx.random.normal((1, 100, 512))
        out = layer(x)
        mx.eval(out)

        self.assertEqual(out.shape, x.shape)

    def test_global_layer_norm_3d(self):
        """Test GlobalLayerNorm for 3D tensors."""
        from mlx_audio.sts.models.mossformer2_se.globallayernorm import GlobalLayerNorm

        layer = GlobalLayerNorm(dim=64, shape=3)
        x = mx.random.normal((1, 64, 100))
        out = layer(x)
        mx.eval(out)

        self.assertEqual(out.shape, x.shape)

    def test_clayer_norm(self):
        """Test CLayerNorm layer."""
        from mlx_audio.sts.models.mossformer2_se.gated_fsmn_block import CLayerNorm

        layer = CLayerNorm(normalized_shape=256)
        x = mx.random.normal((1, 100, 256))
        out = layer(x)
        mx.eval(out)

        self.assertEqual(out.shape, x.shape)

    def test_scaled_sinu_embedding(self):
        """Test ScaledSinuEmbedding."""
        from mlx_audio.sts.models.mossformer2_se.scaledsinuembedding import (
            ScaledSinuEmbedding,
        )

        emb = ScaledSinuEmbedding(dim=512)
        x = mx.random.normal((1, 100, 512))
        out = emb(x)
        mx.eval(out)

        self.assertEqual(out.shape[0], 100)
        self.assertEqual(out.shape[1], 512)

    def test_offset_scale(self):
        """Test OffsetScale."""
        from mlx_audio.sts.models.mossformer2_se.offsetscale import OffsetScale

        layer = OffsetScale(dim=128, heads=4)
        x = mx.random.normal((1, 100, 128))
        outputs = layer(x)
        mx.eval(outputs[0])

        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[0].shape, x.shape)


class TestMossFormer2SE(unittest.TestCase):
    """Tests for main MossFormer2 SE model."""

    def test_model_initialization(self):
        """Test MossFormer2SE initialization."""
        from mlx_audio.sts.models.mossformer2_se.mossformer2_se_wrapper import (
            MossFormer2SE,
        )

        model = MossFormer2SE()
        self.assertIsNotNone(model.model)

    def test_masknet_output_shape(self):
        """Test MossFormer_MaskNet output shape."""
        from mlx_audio.sts.models.mossformer2_se.mossformer_masknet import (
            MossFormer_MaskNet,
        )

        # Create smaller model for testing
        masknet = MossFormer_MaskNet(
            in_channels=180,
            out_channels=64,  # Reduced for testing
            out_channels_final=961,
            num_blocks=2,  # Reduced for testing
        )

        # Input: (batch, channels, time)
        x = mx.random.normal((1, 180, 100))
        out = masknet(x)
        mx.eval(out)

        # Output should be (batch, time, out_channels_final)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[2], 961)


if __name__ == "__main__":
    unittest.main()

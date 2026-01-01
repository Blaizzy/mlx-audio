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
        from mlx_audio.dsp import hamming

        window = hamming(1920, periodic=False)
        mx.eval(window)

        self.assertEqual(window.shape[0], 1920)
        # Hamming window should have non-zero edges
        self.assertGreater(float(window[0]), 0)

    def test_create_window_hann(self):
        """Test hann window creation."""
        from mlx_audio.dsp import hanning

        window = hanning(1920, periodic=False)
        mx.eval(window)

        self.assertEqual(window.shape[0], 1920)

    def test_stft_shape(self):
        """Test STFT output shape."""
        from mlx_audio.dsp import hamming, stft

        window = hamming(1920, periodic=False)
        audio = mx.zeros((48000,))

        stft_complex = stft(audio, 1920, 384, 1920, window, center=False)
        mx.eval(stft_complex)

        # Check output shape: (time, freq)
        self.assertEqual(stft_complex.shape[1], 961)  # n_fft // 2 + 1

    def test_istft_cache(self):
        """Test ISTFTCache caching behavior."""
        from mlx_audio.dsp import ISTFTCache, hamming

        cache = ISTFTCache()
        window = hamming(1920, periodic=False)

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
        """Test compute_deltas_kaldi output shape."""
        from mlx_audio.dsp import compute_deltas_kaldi

        # Input shape: (freq, time)
        specgram = mx.zeros((60, 100))
        deltas = compute_deltas_kaldi(specgram, win_length=5)
        mx.eval(deltas)

        self.assertEqual(deltas.shape, specgram.shape)

    def test_fbank_computation(self):
        """Test compute_fbank_kaldi."""
        import numpy as np

        from mlx_audio.dsp import compute_fbank_kaldi

        audio = mx.array(np.random.randn(24000).astype(np.float32))
        fbank = compute_fbank_kaldi(
            audio, sample_rate=48000, win_len=1920, win_inc=384, num_mels=60
        )
        mx.eval(fbank)

        self.assertEqual(fbank.shape[1], 60)

    def test_compute_deltas_vs_torchaudio(self):
        """Compare compute_deltas_kaldi with torchaudio."""
        try:
            import torch
            import torchaudio
        except ImportError:
            self.skipTest("torchaudio not installed")

        import numpy as np

        from mlx_audio.dsp import compute_deltas_kaldi

        np.random.seed(42)
        specgram_np = np.random.randn(60, 100).astype(np.float32)

        mlx_deltas = compute_deltas_kaldi(mx.array(specgram_np), win_length=5)
        mx.eval(mlx_deltas)

        torch_deltas = torchaudio.functional.compute_deltas(
            torch.from_numpy(specgram_np), win_length=5
        )

        max_diff = np.max(np.abs(np.array(mlx_deltas) - torch_deltas.numpy()))
        self.assertLess(max_diff, 1e-5)

    def test_fbank_vs_torchaudio(self):
        """Compare compute_fbank_kaldi with torchaudio.compliance.kaldi.fbank."""
        try:
            import torch
            import torchaudio
        except ImportError:
            self.skipTest("torchaudio not installed")

        import numpy as np

        from mlx_audio.dsp import compute_fbank_kaldi

        np.random.seed(42)
        audio_np = np.random.randn(24000).astype(np.float32)

        mlx_fbank = compute_fbank_kaldi(
            mx.array(audio_np),
            sample_rate=48000,
            win_len=1920,
            win_inc=384,
            num_mels=60,
            dither=0.0,
        )
        mx.eval(mlx_fbank)

        torch_fbank = torchaudio.compliance.kaldi.fbank(
            torch.from_numpy(audio_np).unsqueeze(0),
            sample_frequency=48000,
            frame_length=40.0,
            frame_shift=8.0,
            num_mel_bins=60,
            dither=0.0,
        )

        self.assertEqual(mlx_fbank.shape, tuple(torch_fbank.shape))


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

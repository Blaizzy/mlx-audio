"""Tests for SAM-Audio model."""

import unittest
from unittest.mock import patch

import mlx.core as mx


class TestSAMAudioConfig(unittest.TestCase):
    """Tests for SAM-Audio configuration classes."""

    def test_dacvae_config_defaults(self):
        """Test DACVAEConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import DACVAEConfig

        config = DACVAEConfig()
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.latent_dim, 1024)
        self.assertEqual(config.codebook_dim, 128)
        self.assertEqual(config.encoder_rates, [2, 8, 10, 12])
        self.assertEqual(config.decoder_rates, [12, 10, 8, 2])
        self.assertEqual(config.encoder_dim, 64)
        self.assertEqual(config.decoder_dim, 1536)

    def test_dacvae_config_hop_length(self):
        """Test DACVAEConfig hop_length calculation."""
        from mlx_audio.sts.models.sam_audio.config import DACVAEConfig

        config = DACVAEConfig()
        # hop_length = product of encoder_rates = 2*8*10*12 = 1920
        expected_hop = 2 * 8 * 10 * 12
        self.assertEqual(config.hop_length, expected_hop)

    def test_t5_encoder_config_defaults(self):
        """Test T5EncoderConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import T5EncoderConfig

        config = T5EncoderConfig()
        self.assertEqual(config.name, "t5-base")
        self.assertEqual(config.dim, 768)
        self.assertEqual(config.max_length, 512)

    def test_transformer_config_defaults(self):
        """Test TransformerConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import TransformerConfig

        config = TransformerConfig()
        self.assertEqual(config.dim, 2816)
        self.assertEqual(config.n_layers, 22)
        self.assertEqual(config.n_heads, 22)
        self.assertEqual(config.out_channels, 256)

    def test_sam_audio_config_defaults(self):
        """Test SAMAudioConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import SAMAudioConfig

        config = SAMAudioConfig()
        self.assertEqual(config.in_channels, 768)
        self.assertIsNotNone(config.audio_codec)
        self.assertIsNotNone(config.text_encoder)
        self.assertIsNotNone(config.transformer)

    def test_sam_audio_config_from_dict(self):
        """Test SAMAudioConfig.from_dict method."""
        from mlx_audio.sts.models.sam_audio.config import SAMAudioConfig

        config_dict = {
            "in_channels": 768,
            "audio_codec": {"sample_rate": 44100},
            "text_encoder": {"name": "t5-small"},
        }

        config = SAMAudioConfig.from_dict(config_dict)
        self.assertEqual(config.in_channels, 768)
        self.assertEqual(config.audio_codec.sample_rate, 44100)
        self.assertEqual(config.text_encoder.name, "t5-small")


class TestSeparationResult(unittest.TestCase):
    """Tests for SeparationResult dataclass."""

    def test_separation_result_creation(self):
        """Test SeparationResult dataclass creation."""
        from mlx_audio.sts.models.sam_audio.model import SeparationResult

        target = [mx.zeros((1000, 1))]
        residual = [mx.zeros((1000, 1))]
        noise = mx.zeros((1, 10, 256))

        result = SeparationResult(
            target=target, residual=residual, noise=noise, peak_memory=1.5
        )

        self.assertEqual(len(result.target), 1)
        self.assertEqual(len(result.residual), 1)
        self.assertEqual(result.noise.shape, (1, 10, 256))
        self.assertEqual(result.peak_memory, 1.5)


class TestDACVAECodec(unittest.TestCase):
    """Tests for DACVAE codec."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.codec.models.dacvae import DACVAE
        from mlx_audio.sts.models.sam_audio.config import DACVAEConfig

        self.config = DACVAEConfig()
        self.codec = DACVAE(self.config)
        self.codec.hop_length = 192

    def test_codec_initialization(self):
        """Test DACVAE initialization."""
        self.assertIsNotNone(self.codec.encoder)
        self.assertIsNotNone(self.codec.decoder)
        self.assertIsNotNone(self.codec.quantizer_in_proj)
        self.assertIsNotNone(self.codec.quantizer_out_proj)
        self.assertEqual(self.codec.sample_rate, 48000)

    def test_codec_hop_length(self):
        """Test DACVAE hop_length property."""
        expected_hop = 192
        self.assertEqual(self.codec.hop_length, expected_hop)

    def test_codec_encode_shape(self):
        """Test DACVAE encode output shape."""
        # Create test audio: (batch, 1, samples)
        batch_size = 1
        samples = self.codec.hop_length * 1  # 1 frames
        audio = mx.zeros((batch_size, 1, samples))

        # Encode
        encoded = self.codec(audio)
        mx.eval(encoded)

        # Expected shape: (batch, codebook_dim, frames)
        expected_frames = samples // self.codec.hop_length
        self.assertEqual(encoded.shape[0], batch_size)
        self.assertEqual(encoded.shape[1], self.config.codebook_dim)
        self.assertEqual(encoded.shape[2], expected_frames)

    def test_codec_decode_shape(self):
        """Test DACVAE decode output shape."""
        batch_size = 1
        codebook_dim = self.config.codebook_dim
        frames = 1

        # Create encoded features
        encoded = mx.zeros((batch_size, codebook_dim, frames))

        # Decode
        decoded = self.codec.decode(encoded)
        mx.eval(decoded)

        # Expected shape: (batch, samples, 1)
        self.assertEqual(decoded.shape[0], batch_size)
        self.assertEqual(decoded.shape[2], 1)

    def test_codec_chunked_decode(self):
        """Test DACVAE chunked decode."""
        batch_size = 1
        codebook_dim = self.config.codebook_dim
        frames = 5

        encoded = mx.zeros((batch_size, codebook_dim, frames))

        # Decode with chunking
        decoded = self.codec.decode(encoded, chunk_size=50)
        mx.eval(decoded)

        # Should produce output
        self.assertEqual(decoded.shape[0], batch_size)
        self.assertEqual(decoded.shape[2], 1)

    def test_wav_idx_to_feature_idx(self):
        """Test waveform to feature index conversion."""
        # 1920 samples should be 1 frame
        wav_idx = self.codec.hop_length
        feature_idx = self.codec.wav_idx_to_feature_idx(wav_idx)
        self.assertEqual(feature_idx, 1)

    def test_feature_idx_to_wav_idx(self):
        """Test feature to waveform index conversion."""
        feature_idx = 1
        wav_idx = self.codec.feature_idx_to_wav_idx(feature_idx)
        self.assertEqual(wav_idx, self.codec.hop_length)


class TestSAMAudioModel(unittest.TestCase):
    """Tests for SAMAudio model."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.sts.models.sam_audio.config import SAMAudioConfig
        from mlx_audio.sts.models.sam_audio.model import SAMAudio

        self.config = SAMAudioConfig()
        self.model = SAMAudio(self.config)
        self.model.eval()

    def test_model_initialization(self):
        """Test SAMAudio initialization."""
        self.assertIsNotNone(self.model.audio_codec)
        self.assertIsNotNone(self.model.text_encoder)
        self.assertIsNotNone(self.model.transformer)
        self.assertIsNotNone(self.model.proj)
        self.assertIsNotNone(self.model.memory_proj)

    def test_model_sample_rate(self):
        """Test SAMAudio sample_rate property."""
        self.assertEqual(self.model.sample_rate, 48000)

    def test_sanitize_weights(self):
        """Test SAMAudio sanitize method."""
        # Create mock weights with keys that should be filtered
        # The sanitize method removes: text_encoder., span_predictor., visual_ranker.,
        # text_ranker., vision_encoder., align_masked_video., and keys with wm_rates
        weights = {
            "transformer.layers.0.weight": mx.zeros((10, 10)),
            "text_encoder.some_param": mx.zeros((10,)),  # Should be removed
            "span_predictor.weight": mx.zeros((10,)),  # Should be removed
            "some.wm_rates.param": mx.zeros((10,)),  # Should be removed
            "audio_codec.encoder.weight": mx.zeros((10, 10)),
        }

        sanitized = self.model.sanitize(weights)

        # Filtered prefixes should be removed
        self.assertNotIn("text_encoder.some_param", sanitized)
        self.assertNotIn("span_predictor.weight", sanitized)
        self.assertNotIn("some.wm_rates.param", sanitized)
        # Other weights should remain
        self.assertIn("transformer.layers.0.weight", sanitized)
        self.assertIn("audio_codec.encoder.weight", sanitized)

    def test_get_audio_features_shape(self):
        """Test _get_audio_features output shape."""
        batch_size = 1
        samples = self.model.audio_codec.hop_length * 10
        audio = mx.zeros((batch_size, 1, samples))

        features = self.model._get_audio_features(audio)
        mx.eval(features)

        # Output shape: (batch, frames, 2*codebook_dim)
        expected_frames = samples // self.model.audio_codec.hop_length
        self.assertEqual(features.shape[0], batch_size)
        self.assertEqual(features.shape[1], expected_frames)
        self.assertEqual(features.shape[2], 256)  # 2 * 128


class TestSAMAudioProcessor(unittest.TestCase):
    """Tests for SAMAudioProcessor."""

    def test_processor_initialization(self):
        """Test SAMAudioProcessor initialization."""
        from mlx_audio.sts.models.sam_audio.processor import SAMAudioProcessor

        processor = SAMAudioProcessor(
            audio_hop_length=1920,
            audio_sampling_rate=48000,
        )
        self.assertEqual(processor.audio_hop_length, 1920)
        self.assertEqual(processor.audio_sampling_rate, 48000)

    def test_batch_dataclass(self):
        """Test Batch dataclass."""
        from mlx_audio.sts.models.sam_audio.processor import Batch

        batch = Batch(
            descriptions=["speech", "music"],
            audios=mx.zeros((2, 1, 48000)),
            sizes=mx.array([25, 25]),
            wav_sizes=mx.array([48000, 48000]),
            anchor_ids=mx.zeros((2, 3), dtype=mx.int32),
            anchor_alignment=mx.zeros((2, 25), dtype=mx.int32),
        )

        self.assertEqual(len(batch.descriptions), 2)
        self.assertEqual(batch.audios.shape[0], 2)

    def test_wav_to_feature_idx(self):
        """Test wav_to_feature_idx conversion."""
        from mlx_audio.sts.models.sam_audio.processor import SAMAudioProcessor

        processor = SAMAudioProcessor(
            audio_hop_length=192,
            audio_sampling_rate=48000,
        )

        # 1920 samples = 1 frame
        result = processor.wav_to_feature_idx(192)
        self.assertEqual(result, 1)

        # 3840 samples = 2 frames
        result = processor.wav_to_feature_idx(384)
        self.assertEqual(result, 2)


class TestODEStepFunctions(unittest.TestCase):
    """Tests for ODE step functions."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.sts.models.sam_audio.config import SAMAudioConfig
        from mlx_audio.sts.models.sam_audio.model import SAMAudio

        self.config = SAMAudioConfig()
        self.model = SAMAudio(self.config)
        self.model.eval()

    def test_ode_step_euler_shape(self):
        """Test Euler ODE step output shape."""
        batch_size = 1
        seq_len = 2
        channels = 256

        noisy_audio = mx.zeros((batch_size, seq_len, channels))
        audio_features = mx.zeros((batch_size, seq_len, channels))
        text_features = mx.zeros((batch_size, 5, 768))

        result = self.model._ode_step_euler(
            t=0.0,
            dt=0.0625,
            noisy_audio=noisy_audio,
            audio_features=audio_features,
            text_features=text_features,
            text_mask=None,
            anchor_ids=None,
            anchor_alignment=None,
            audio_pad_mask=None,
        )
        mx.eval(result)

        self.assertEqual(result.shape, noisy_audio.shape)

    def test_ode_step_midpoint_shape(self):
        """Test Midpoint ODE step output shape."""
        batch_size = 1
        seq_len = 2
        channels = 256

        noisy_audio = mx.zeros((batch_size, seq_len, channels))
        audio_features = mx.zeros((batch_size, seq_len, channels))
        text_features = mx.zeros((batch_size, 5, 768))

        result = self.model._ode_step_midpoint(
            t=0.0,
            dt=0.0625,
            noisy_audio=noisy_audio,
            audio_features=audio_features,
            text_features=text_features,
            text_mask=None,
            anchor_ids=None,
            anchor_alignment=None,
            audio_pad_mask=None,
        )
        mx.eval(result)

        self.assertEqual(result.shape, noisy_audio.shape)

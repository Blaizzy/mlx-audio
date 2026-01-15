import unittest


DEFAULT_LFM_CONFIG = {
    "model_type": "lfm2",
    "vocab_size": 65536,
    "hidden_size": 512,  
    "num_hidden_layers": 4,  
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": 512,
    "block_ff_dim": 1536,
    "block_multiple_of": 256,
    "block_ffn_dim_multiplier": None,
    "block_auto_adjust_ff_dim": True,
    "rope_theta": 10000.0,
    # layer_types: one per layer - "conv" or "full_attention"
    "layer_types": ["conv", "full_attention", "conv", "full_attention"],
}


def get_test_config():
    """Create a test config dict with all required fields."""
    return {
        "model_type": "lfm_audio",
        "sample_rate": 24000,
        "codebooks": 8,
        "audio_vocab_size": 2049,
        "preprocessor": {"sample_rate": 16000, "features": 128},
        "encoder": {"d_model": 512, "n_layers": 2},  # Small for testing
        "depthformer": {"layers": 2, "dim": 256},  # Small for testing
        "lfm": DEFAULT_LFM_CONFIG,
    }


class TestPreprocessorConfig(unittest.TestCase):

    def test_defaults(self):
        """Test PreprocessorConfig default values."""
        from mlx_audio.sts.models.lfm_audio.config import PreprocessorConfig

        config = PreprocessorConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.features, 128)
        self.assertEqual(config.n_fft, 512)
        self.assertEqual(config.hop_length, 160)
        self.assertEqual(config.win_length, 400)


class TestConformerEncoderConfig(unittest.TestCase):

    def test_defaults(self):
        from mlx_audio.sts.models.lfm_audio.config import ConformerEncoderConfig

        config = ConformerEncoderConfig()
        self.assertEqual(config.feat_in, 128)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_layers, 17)
        self.assertEqual(config.n_heads, 8)


class TestDepthformerConfig(unittest.TestCase):

    def test_defaults(self):
        from mlx_audio.sts.models.lfm_audio.config import DepthformerConfig

        config = DepthformerConfig()
        self.assertEqual(config.layers, 6)
        self.assertEqual(config.dim, 1024)
        self.assertEqual(config.num_heads, 32)
        self.assertEqual(config.num_kv_heads, 8)


class TestLFM2AudioConfig(unittest.TestCase):

    def test_from_dict(self):
        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig

        config_dict = get_test_config()
        config = LFM2AudioConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "lfm_audio")
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.codebooks, 8)
        self.assertEqual(config.audio_vocab_size, 2049)
        self.assertIsNotNone(config.preprocessor)
        self.assertIsNotNone(config.encoder)
        self.assertIsNotNone(config.depthformer)
        self.assertIsNotNone(config.lfm)


class TestLFM2AudioModelOutput(unittest.TestCase):

    def test_call_returns_text_and_audio_logits(self):
        import mlx.core as mx
        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

        config = LFM2AudioConfig.from_dict(get_test_config())
        model = LFM2AudioModel(config)
        model.eval()

        # Create dummy text input
        batch_size = 1
        seq_len = 5
        text_tokens = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        # Call model
        text_logits, audio_logits = model(text_tokens=text_tokens)
        mx.eval(text_logits)

        # Check text_logits is an array with correct shape
        self.assertIsInstance(text_logits, mx.array)
        self.assertEqual(text_logits.shape[0], batch_size)
        self.assertEqual(text_logits.shape[1], seq_len)

        # Check audio_logits is a list of 8 arrays (one per codebook)
        self.assertIsInstance(audio_logits, list)
        self.assertEqual(len(audio_logits), config.codebooks)

        for logit in audio_logits:
            mx.eval(logit)
            self.assertIsInstance(logit, mx.array)
            self.assertEqual(logit.shape[0], batch_size)
            self.assertEqual(logit.shape[1], seq_len)
            self.assertEqual(logit.shape[2], config.audio_vocab_size)


class TestLFM2AudioModelDtype(unittest.TestCase):

    def test_model_components_exist(self):
        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

        config = LFM2AudioConfig.from_dict(get_test_config())
        model = LFM2AudioModel(config)

        # Check components exist
        self.assertIsNotNone(model.audio_encoder)
        self.assertIsNotNone(model.audio_adapter)
        self.assertIsNotNone(model.lfm)
        self.assertIsNotNone(model.audio_embedding)
        self.assertIsNotNone(model.audio_head)
        self.assertIsNotNone(model.depth_embeddings)
        self.assertIsNotNone(model.depth_linear)

        # Check depth_embeddings count matches codebooks
        self.assertEqual(len(model.depth_embeddings), config.codebooks)

    def test_sample_rate_property(self):
        from mlx_audio.sts.models.lfm_audio.config import LFM2AudioConfig
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel

        config = LFM2AudioConfig.from_dict(get_test_config())
        model = LFM2AudioModel(config)

        self.assertEqual(model.sample_rate, 24000)


class TestLFMModality(unittest.TestCase):

    def test_modality_values(self):
        from mlx_audio.sts.models.lfm_audio.model import LFMModality

        self.assertEqual(LFMModality.TEXT, 1)
        self.assertEqual(LFMModality.AUDIO_IN, 2)
        self.assertEqual(LFMModality.AUDIO_OUT, 3)


class TestSpecialTokens(unittest.TestCase):

    def test_token_values(self):
        from mlx_audio.sts.models.lfm_audio.model import (
            AUDIO_START_TOKEN,
            IM_END_TOKEN,
            TEXT_END_TOKEN,
            AUDIO_EOS_TOKEN,
        )

        self.assertEqual(AUDIO_START_TOKEN, 128)
        self.assertEqual(IM_END_TOKEN, 7)
        self.assertEqual(TEXT_END_TOKEN, 130)
        self.assertEqual(AUDIO_EOS_TOKEN, 2048)


class TestGenerationConfig(unittest.TestCase):

    def test_defaults(self):
        from mlx_audio.sts.models.lfm_audio.model import GenerationConfig

        config = GenerationConfig()
        self.assertEqual(config.max_new_tokens, 512)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.audio_temperature, 1.0)
        self.assertEqual(config.audio_top_k, 4)


if __name__ == "__main__":
    unittest.main()

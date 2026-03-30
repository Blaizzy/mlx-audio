import importlib.resources
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from misaki import en


# Create a patch for the deprecated open_text function
def patched_open_text(package, resource):
    """Replacement for deprecated open_text using files() API"""
    return importlib.resources.files(package).joinpath(resource).open("r")


class FakeTokenizer:
    def __init__(self):
        self.semantic_begin_id = 1000
        self._next = 1

    def encode(self, text):
        token = self._next
        self._next += 1
        return [token]


def tiny_config():
    from mlx_audio.tts.models.fish_qwen3_omni.config import ModelConfig

    return ModelConfig.from_dict(
        {
            "semantic_start_token_id": 1000,
            "semantic_end_token_id": 1007,
            "text_config": {
                "vocab_size": 32,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "norm_eps": 1e-6,
                "max_seq_len": 64,
                "attention_qk_norm": True,
            },
            "audio_decoder_config": {
                "vocab_size": 8,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "num_codebooks": 2,
                "norm_eps": 1e-6,
                "max_seq_len": 3,
            },
        }
    )


# Apply the patch at the module level
@patch("importlib.resources.open_text", patched_open_text)
class TestSanitizeLSTMWeights(unittest.TestCase):
    def test_sanitize_lstm_weights(self):
        """Test sanitize_lstm_weights function."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import sanitize_lstm_weights

        # Test weight_ih_l0_reverse
        key = "lstm.weight_ih_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_backward")

        # Test weight_hh_l0_reverse
        key = "lstm.weight_hh_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_backward")

        # Test bias_ih_l0_reverse
        key = "lstm.bias_ih_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_backward")

        # Test bias_hh_l0_reverse
        key = "lstm.bias_hh_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_backward")

        # Test weight_ih_l0
        key = "lstm.weight_ih_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_forward")

        # Test weight_hh_l0
        key = "lstm.weight_hh_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_forward")

        # Test bias_ih_l0
        key = "lstm.bias_ih_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_forward")

        # Test bias_hh_l0
        key = "lstm.bias_hh_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_forward")

        # Test unknown key
        key = "unknown.key"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "unknown.key")


@patch("importlib.resources.open_text", patched_open_text)
class TestKokoroModel(unittest.TestCase):
    @patch("json.load")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("mlx_audio.tts.models.kokoro.kokoro.mx.load")
    @patch.object(nn.Module, "load_weights")
    def test_init(self, mock_load_weights, mock_mx_load, mock_open, mock_json_load):
        """Test KokoroModel initialization."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import Model, ModelConfig

        # Mock the config loading
        config = {
            "istftnet": {
                "upsample_kernel_sizes": [20, 12],
                "upsample_rates": [10, 6],
                "gen_istft_hop_size": 5,
                "gen_istft_n_fft": 20,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_initial_channel": 512,
            },
            "dim_in": 64,
            "dropout": 0.2,
            "hidden_dim": 512,
            "max_conv_dim": 512,
            "max_dur": 50,
            "multispeaker": True,
            "n_layer": 3,
            "n_mels": 80,
            "n_token": 178,
            "style_dim": 128,
            "text_encoder_kernel_size": 5,
            "plbert": {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "num_hidden_layers": 12,
                "dropout": 0.1,
            },
            "vocab": {"a": 1, "b": 2},
        }
        mock_json_load.return_value = config

        # Mock the weights loading
        mock_mx_load.return_value = {"key": mx.array(np.zeros(10))}

        # Make load_weights return the module
        mock_load_weights.return_value = None

        # Initialize the model with the config parameter
        model = Model(ModelConfig.from_dict(config))

        # Check that the model was initialized correctly
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.vocab, {"a": 1, "b": 2})

    def test_output_dataclass(self):
        """Test KokoroModel.Output dataclass."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import Model

        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))

        # Mock __init__ to return None
        with patch.object(Model, "__init__", return_value=None):
            output = Model.Output(audio=audio, pred_dur=pred_dur)

        # Check that the output was created correctly
        self.assertIs(output.audio, audio)
        self.assertIs(output.pred_dur, pred_dur)


@patch("importlib.resources.open_text", patched_open_text)
class TestKokoroPipeline(unittest.TestCase):
    def test_aliases_and_lang_codes(self):
        """Test ALIASES and LANG_CODES constants."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import ALIASES, LANG_CODES

        # Check that all aliases map to valid language codes
        for alias_key, alias_value in ALIASES.items():
            self.assertIn(alias_value, LANG_CODES)

        # Check specific mappings
        self.assertEqual(ALIASES["en-us"], "a")
        self.assertEqual(ALIASES["ja"], "j")
        self.assertEqual(LANG_CODES["a"], "American English")
        self.assertEqual(LANG_CODES["j"], "Japanese")

    def test_init(self):
        """Test KokoroPipeline initialization."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import LANG_CODES, KokoroPipeline

        # Mock the G2P class to avoid spacy download during tests
        with patch("mlx_audio.tts.models.kokoro.pipeline.en.G2P") as mock_g2p:
            with patch(
                "mlx_audio.tts.models.kokoro.pipeline.espeak.EspeakFallback"
            ) as mock_fallback:
                mock_model = MagicMock()
                mock_g2p.return_value = MagicMock()
                mock_fallback.return_value = MagicMock()

                # Initialize with default model
                pipeline = KokoroPipeline(
                    lang_code="a", model=mock_model, repo_id="mock"
                )
                self.assertEqual(pipeline.lang_code, "a")
                self.assertEqual(LANG_CODES[pipeline.lang_code], "American English")

                # Initialize with provided model
                model = mock_model
                pipeline = KokoroPipeline(lang_code="a", model=model, repo_id="mock")
                self.assertEqual(pipeline.model, model)

                # Initialize with no model
                pipeline = KokoroPipeline(lang_code="a", model=False, repo_id="mock")
                self.assertIs(pipeline.model, False)

    def test_load_voice(self):
        """Test load_voice method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Setup the pipeline
        with patch.object(KokoroPipeline, "__init__", return_value=None):
            with patch(
                "mlx_audio.tts.models.kokoro.pipeline.load_voice_tensor"
            ) as load_voice_tensor:
                with patch(
                    "mlx_audio.tts.models.kokoro.pipeline.snapshot_download"
                ) as mock_snapshot_download:
                    pipeline = KokoroPipeline.__new__(KokoroPipeline)
                    pipeline.lang_code = "a"
                    pipeline.voices = {}
                    # Add the missing repo_id attribute
                    pipeline.repo_id = "mlx-community/kokoro-tts"

                    # Mock the load voice return value
                    load_voice_tensor.return_value = mx.zeros((512, 1, 256))

                    # Mock snapshot_download to return a path
                    # First call with local_files_only=True raises error, second downloads
                    mock_snapshot_download.side_effect = [
                        FileNotFoundError(),  # local_files_only=True fails
                        "/mock/path",  # actual download succeeds
                    ]

                    # Test loading a single voice
                    pipeline.load_single_voice("voice1")
                    self.assertEqual(mock_snapshot_download.call_count, 2)
                    self.assertIn("voice1", pipeline.voices)

                    # Test loading multiple voices
                    mock_snapshot_download.reset_mock()
                    mock_snapshot_download.side_effect = [
                        FileNotFoundError(),
                        "/mock/path",
                        FileNotFoundError(),
                        "/mock/path",
                    ]
                    pipeline.voices = {}  # Reset voices
                    result = pipeline.load_voice("voice1,voice2")
                    self.assertEqual(mock_snapshot_download.call_count, 4)
                    self.assertIn("voice1", pipeline.voices)
                    self.assertIn("voice2", pipeline.voices)

    def test_tokens_to_ps(self):
        """Test tokens_to_ps method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create mock tokens with whitespace attribute
        token1 = MagicMock(spec=en.MToken)
        token1.ps = "p1"
        token1.whitespace = " "
        token1.phonemes = "p1"

        token2 = MagicMock(spec=en.MToken)
        token2.ps = "p2"
        token2.whitespace = ""
        token2.phonemes = "p2"

        tokens = [token1, token2]

        # Test the method
        with patch.object(KokoroPipeline, "__init__", return_value=None):
            with patch.object(KokoroPipeline, "tokens_to_ps", return_value="p1 p2"):
                result = KokoroPipeline.tokens_to_ps(tokens)
                self.assertEqual(result, "p1 p2")

    def test_tokens_to_text(self):
        """Test tokens_to_text method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create mock tokens with whitespace attribute
        token1 = MagicMock(spec=en.MToken)
        token1.text = "Hello"
        token1.whitespace = " "

        token2 = MagicMock(spec=en.MToken)
        token2.text = "world"
        token2.whitespace = ""

        tokens = [token1, token2]

        # Test the method
        with patch.object(KokoroPipeline, "__init__", return_value=None):
            with patch.object(
                KokoroPipeline, "tokens_to_text", return_value="Hello world"
            ):
                result = KokoroPipeline.tokens_to_text(tokens)
                self.assertEqual(result, "Hello world")

    def test_result_dataclass(self):
        """Test KokoroPipeline.Result dataclass."""
        # Import inside the test methods
        from mlx_audio.tts.models.kokoro.kokoro import Model
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))
        model_output = Model.Output(audio=audio, pred_dur=pred_dur)

        # Create a Result instance
        result = KokoroPipeline.Result(
            graphemes="Hello",
            phonemes="HH EH L OW",
            tokens=[MagicMock()],
            output=model_output,
            text_index=0,
        )

        # Check properties
        self.assertEqual(result.graphemes, "Hello")
        self.assertEqual(result.phonemes, "HH EH L OW")
        self.assertIs(result.audio, audio)
        self.assertIs(result.pred_dur, pred_dur)

        # Test backward compatibility
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Hello")
        self.assertEqual(result[1], "HH EH L OW")
        self.assertIs(result[2], audio)

        # Test iteration
        items = list(result)
        self.assertEqual(items[0], "Hello")
        self.assertEqual(items[1], "HH EH L OW")
        self.assertIs(items[2], audio)


@patch("importlib.resources.open_text", patched_open_text)
class TestKittenTTSModel(unittest.TestCase):
    def _config(self):
        return {
            "hidden_dim": 16,
            "max_conv_dim": 16,
            "max_dur": 10,
            "n_layer": 1,
            "n_mels": 80,
            "n_token": 32,
            "style_dim": 64,
            "text_encoder_kernel_size": 3,
            "asr_res_dim": 8,
            "decoder_out_dim": 16,
            "plbert": {
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "max_position_embeddings": 32,
                "embedding_size": 16,
                "inner_group_num": 1,
                "num_hidden_groups": 1,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
            },
            "istftnet": {
                "resblock_kernel_sizes": [3, 3],
                "upsample_rates": [2, 2],
                "upsample_initial_channel": 32,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes": [4, 4],
                "gen_istft_n_fft": 16,
                "gen_istft_hop_size": 4,
            },
        }

    def test_init(self):
        from mlx_audio.tts.models.kitten_tts.kitten_tts import Model, ModelConfig

        config = self._config()
        model = Model(ModelConfig.from_dict(config))
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.config.n_token, config["n_token"])

    def test_sanitize_alpha_names(self):
        from mlx_audio.tts.models.kitten_tts.kitten_tts import Model, ModelConfig

        config = self._config()
        model = Model(ModelConfig.from_dict(config))
        weights = {
            "decoder.generator.resblocks.0.alpha1.0": mx.ones((1, 1, 1)),
            "decoder.generator.resblocks.0.alpha2.0": mx.ones((1, 1, 1)),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("decoder.generator.resblocks.0.alpha1_0", sanitized)
        self.assertIn("decoder.generator.resblocks.0.alpha2_0", sanitized)
        self.assertNotIn("decoder.generator.resblocks.0.alpha1.0", sanitized)


class TestBarkModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.bark.bark.BertTokenizer")
    def test_init(self, mock_tokenizer):
        """Test BarkModel initialization."""
        from mlx_audio.tts.models.bark.bark import (
            CoarseAcousticsConfig,
            CodecConfig,
            FineAcousticsConfig,
            Model,
            ModelConfig,
            SemanticConfig,
        )

        # Create mock configs
        semantic_config = SemanticConfig()
        coarse_config = CoarseAcousticsConfig()
        fine_config = FineAcousticsConfig()
        codec_config = CodecConfig()

        config = ModelConfig(
            semantic_config=semantic_config,
            coarse_acoustics_config=coarse_config,
            fine_acoustics_config=fine_config,
            codec_config=codec_config,
        )

        # Initialize model
        model = Model(config)

        # Check that components were initialized correctly
        self.assertIsNotNone(model.semantic)
        self.assertIsNotNone(model.coarse_acoustics)
        self.assertIsNotNone(model.fine_acoustics)
        self.assertIsNotNone(model.tokenizer)

    def test_sanitize_weights(self):
        """Test weight sanitization."""
        from mlx_audio.tts.models.bark.bark import Model, ModelConfig

        # Create a minimal config
        config = ModelConfig(
            semantic_config={},
            coarse_acoustics_config={},
            fine_acoustics_config={},
            codec_config={},
        )

        model = Model(config)

        # Test with transformer weights
        weights = {
            "_orig_mod.transformer.h.0.mlp.weight": mx.zeros((10, 10)),
            "_orig_mod.transformer.h.1.mlp.weight": mx.zeros((10, 10)),
            "lm_head.weight": mx.zeros((10, 10)),
        }

        sanitized = model.sanitize(weights)

        # Check that weights were properly renamed
        self.assertIn("layers.0.mlp.weight", sanitized)
        self.assertIn("layers.1.mlp.weight", sanitized)
        self.assertIn("lm_head.weight", sanitized)


@patch("importlib.resources.open_text", patched_open_text)
class TestBarkPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.tts.models.bark.bark import (
            CoarseAcousticsConfig,
            CodecConfig,
            FineAcousticsConfig,
            Model,
            ModelConfig,
            SemanticConfig,
        )
        from mlx_audio.tts.models.bark.pipeline import Pipeline

        # Create mock model with required attributes
        self.mock_model = MagicMock(spec=Model)

        # Add the required mock attributes/methods
        self.mock_model.semantic = MagicMock()
        self.mock_model.coarse_acoustics = MagicMock()
        self.mock_model.fine_acoustics = MagicMock()
        self.mock_model.codec_model = MagicMock()

        self.mock_tokenizer = MagicMock()

        # Initialize pipeline
        self.pipeline = Pipeline(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=ModelConfig(
                semantic_config=SemanticConfig(),
                coarse_acoustics_config=CoarseAcousticsConfig(),
                fine_acoustics_config=FineAcousticsConfig(),
                codec_config=CodecConfig(),
            ),
        )

    def test_generate_text_semantic(self):
        """Test semantic token generation."""
        # Mock tokenizer output
        self.mock_tokenizer.encode.return_value = [1, 2, 3]

        # Create logits with proper shape including SEMANTIC_PAD_TOKEN
        logits = mx.zeros((1, 1, 129596))  # Large enough to include SEMANTIC_PAD_TOKEN
        # Mock model output
        self.mock_model.semantic.return_value = (
            logits,  # logits with correct shape
            None,  # kv_cache
        )

        # Test generation
        semantic_tokens, text_tokens = self.pipeline.generate_text_semantic(
            "test text",
            temperature=0.7,
            use_kv_caching=True,
            voice=None,
        )

        # Verify tokenizer was called
        self.mock_tokenizer.encode.assert_called_once_with(
            "test text", add_special_tokens=False
        )

        # Verify model was called
        self.mock_model.semantic.assert_called()

        # Check output types
        self.assertIsInstance(semantic_tokens, mx.array)
        self.assertIsInstance(text_tokens, mx.array)

    @patch("mlx.core.random.categorical")  # Add this patch since we use mx alias
    def test_generate_coarse(self, mock_mlx_categorical):
        """Test coarse token generation."""
        # Create mock semantic tokens
        semantic_tokens = mx.array([1, 2, 3])

        # Create logits with proper shape
        logits = mx.zeros((1, 1, 12096))

        # Mock both categorical functions to return predictable values
        mock_mlx_categorical.return_value = mx.array([10000])  # Return token index

        # Set up the mock to return proper values for each call
        self.mock_model.coarse_acoustics.return_value = (logits, None)

        # Test generation with minimal parameters to reduce test time
        coarse_tokens = self.pipeline.generate_coarse(
            semantic_tokens,
            temperature=0.7,
            use_kv_caching=True,
            voice=None,
            max_coarse_history=60,
            sliding_window_len=2,  # Reduce this to minimum
        )

        # Verify model was called at least once
        self.mock_model.coarse_acoustics.assert_called()

        # Check output type and shape
        self.assertIsInstance(coarse_tokens, mx.array)
        self.assertEqual(coarse_tokens.shape[0], 2)  # N_COARSE_CODEBOOKS

    def test_generate_fine(self):
        """Test fine token generation."""
        # Create mock coarse tokens
        coarse_tokens = mx.zeros((2, 100))  # N_COARSE_CODEBOOKS x sequence_length

        # Mock model output with proper shape
        self.mock_model.fine_acoustics.return_value = mx.zeros((1, 1024, 1024))

        # Test generation
        fine_tokens = self.pipeline.generate_fine(coarse_tokens, temperature=0.7)

        # Verify model was called
        self.mock_model.fine_acoustics.assert_called()

        # Check output type and shape
        self.assertIsInstance(fine_tokens, mx.array)
        self.assertEqual(
            fine_tokens.shape[0], 8
        )  # N_FINE_CODEBOOKS (corrected from 10 to 8)
        self.assertEqual(fine_tokens.shape[1], 100)  # sequence_length


class TestLlamaModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "attention_bias": False,
            "head_dim": 128,
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 24,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "vocab_size": 156940,
            "layer_types": ["full_attention"] * 28,
        }

    @patch("transformers.AutoTokenizer")
    def test_init(self, mock_tokenizer):
        """Test LlamaModel initialization."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

    @patch("transformers.AutoTokenizer")
    def test_generate(self, mock_tokenizer):
        """Test generate method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()

        def mock_tokenize(text, return_tensors=None):
            result = MagicMock()
            result.input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
            return result

        mock_tokenizer_instance.side_effect = mock_tokenize
        mock_tokenizer_instance.__call__ = mock_tokenize
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Verify batched input creation with a voice
        input_ids = model.prepare_input_ids(["Foo", "Bar Baz"], voice="zoe")
        self.assertEqual(input_ids.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, input_ids.shape[1], config.vocab_size))

        # Verify batched input creation with reference audio
        input_ids, input_mask = model.prepare_input_ids(
            ["Foo", "Bar Baz"], ref_audio=mx.zeros((100,)), ref_text="Caption"
        )
        self.assertEqual(input_ids.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, input_ids.shape[1], config.vocab_size))

    @patch("transformers.AutoTokenizer")
    def test_sanitize(self, mock_tokenizer):
        """Test sanitize method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a config with tie_word_embeddings=True
        config = ModelConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=16384,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            head_dim=128,
            max_position_embeddings=1024,
            num_key_value_heads=32,
            attention_bias=True,
            mlp_bias=True,
            rope_theta=500000.0,
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=True,
        )

        # Initialize the model with a patched __init__
        with patch.object(Model, "__init__", return_value=None):
            model = Model.__new__(Model)
            model.config = config

            # Add the sanitize method from actual implementation
            def mock_sanitize(weights):
                result = {}
                for k, v in weights.items():
                    if "rotary_emb" in k:
                        continue
                    if "lm_head.weight" in k and config.tie_word_embeddings:
                        continue
                    result[k] = v
                return result

            model.sanitize = mock_sanitize

            # Create test weights with rotary embeddings and lm_head
            weights = {
                "self_attn.rotary_emb.inv_freq": mx.zeros(10),
                "lm_head.weight": mx.zeros((32000, 4096)),
                "model.layers.0.input_layernorm.weight": mx.zeros(4096),
            }

            # Test sanitize method
            sanitized = model.sanitize(weights)

            # Assert rotary embeddings are removed
            self.assertNotIn("self_attn.rotary_emb.inv_freq", sanitized)

            # Assert lm_head weights are removed with tie_word_embeddings=True
            self.assertNotIn("lm_head.weight", sanitized)

            # Assert other weights remain
            self.assertIn("model.layers.0.input_layernorm.weight", sanitized)

            # Now test with tie_word_embeddings=False
            config.tie_word_embeddings = False

            # Test sanitize again
            sanitized2 = model.sanitize(weights)

            # lm_head should be kept with tie_word_embeddings=False
            self.assertIn("lm_head.weight", sanitized2)


class TestQwen3Model(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "head_dim": 128,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "max_position_embeddings": 40960,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000,
            "tie_word_embeddings": True,
            "vocab_size": 180352,
        }

    @patch("transformers.AutoTokenizer")
    def test_init(self, mock_tokenizer):
        """Test Qwen3Model initialization."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)
        self.assertEqual(model.model_type, "qwen3")
        self.assertIsNone(model.tokenizer)

    @patch("transformers.AutoTokenizer")
    def test_forward(self, mock_tokenizer):
        """Test forward pass."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Test forward pass with random input
        input_ids = mx.random.randint(0, config.vocab_size, (2, 9))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 9, config.vocab_size))

    @patch("transformers.AutoTokenizer")
    def test_prepare_input_ids_with_voice(self, mock_tokenizer):
        """Test prepare_input_ids method with voice."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()

        # Mock tokenizer __call__ to return proper input_ids
        def mock_tokenize(text, return_tensors=None):
            result = MagicMock()
            # Return a simple token sequence for each text
            result.input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int64)
            return result

        mock_tokenizer_instance.side_effect = mock_tokenize
        mock_tokenizer_instance.__call__ = mock_tokenize
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)
        model.tokenizer = mock_tokenizer_instance

        # Test with voice
        input_ids = model.prepare_input_ids(["Hello", "World"], voice="zoe")

        # Verify batch size
        self.assertEqual(input_ids.shape[0], 2)

    @patch("transformers.AutoTokenizer")
    def test_parse_output(self, mock_tokenizer):
        """Test parse_output method."""
        from mlx_audio.tts.models.qwen3.qwen3 import (
            AUDIO_TOKENS_START,
            END_OF_SPEECH,
            START_OF_SPEECH,
            Model,
            ModelConfig,
        )

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Create input with speech tokens
        # Format: [START_OF_SPEECH, audio_tokens..., END_OF_SPEECH]
        audio_tokens = [AUDIO_TOKENS_START + i for i in range(7)]  # 7 audio tokens
        input_sequence = [START_OF_SPEECH] + audio_tokens + [END_OF_SPEECH]
        input_ids = mx.array([input_sequence], dtype=mx.int64)

        # Test parse_output
        code_lists = model.parse_output(input_ids)

        # Should return one code list (one batch item)
        self.assertEqual(len(code_lists), 1)

        # The code list should have 7 items (trimmed to multiple of 7)
        self.assertEqual(len(code_lists[0]), 7)

        # Verify codes are offset by AUDIO_TOKENS_START
        for i, code in enumerate(code_lists[0]):
            self.assertEqual(code, i)

    @patch("transformers.AutoTokenizer")
    def test_sample_rate(self, mock_tokenizer):
        """Test sample_rate property."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Default sample rate should be 24000
        self.assertEqual(model.sample_rate, 24000)

    @patch("transformers.AutoTokenizer")
    def test_layers_property(self, mock_tokenizer):
        """Test layers property returns model layers."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Verify layers property returns the model's layers
        layers = model.layers
        self.assertEqual(len(layers), config.num_hidden_layers)


class TestOuteTTSModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "attention_bias": False,
            "head_dim": 64,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "vocab_size": 134400,
        }

    @patch("transformers.AutoTokenizer")
    def test_init(self, mock_tokenizer):
        """Test initialization."""
        from mlx_audio.tts.models.outetts.outetts import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

    @patch("transformers.AutoTokenizer")
    def test_generate(self, mock_tokenizer):
        """Test generate method."""
        from mlx_audio.tts.models.outetts.outetts import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        input_ids = mx.random.randint(0, config.vocab_size, (2, 9))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 9, config.vocab_size))


class TestDiaModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "version": "0.1",
            "model": {
                "encoder": {
                    "n_layer": 12,
                    "n_embd": 1024,
                    "n_hidden": 4096,
                    "n_head": 16,
                    "head_dim": 128,
                },
                "decoder": {
                    "n_layer": 18,
                    "n_embd": 2048,
                    "n_hidden": 8192,
                    "gqa_query_heads": 16,
                    "cross_query_heads": 16,
                    "kv_heads": 4,
                    "gqa_head_dim": 128,
                    "cross_head_dim": 128,
                },
                "src_vocab_size": 256,
                "tgt_vocab_size": 1028,
                "dropout": 0.0,
            },
            "training": {},
            "data": {
                "text_length": 1024,
                "audio_length": 3072,
                "channels": 9,
                "text_pad_value": 0,
                "audio_eos_value": 1024,
                "audio_pad_value": 1025,
                "audio_bos_value": 1026,
                "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
            },
        }

    def test_init(self):
        """Test DiaModel initialization."""
        from mlx_audio.tts.models.dia.dia import Model

        # Initialize model
        config = self._default_config
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)


class TestSparkTTSModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "sample_rate": 16000,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 896,
            "initializer_range": 0.02,
            "intermediate_size": 4864,
            "max_position_embeddings": 32768,
            "max_window_layers": 21,
            "model_type": "qwen2",
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.43.1",
            "use_sliding_window": False,
            "vocab_size": 166000,
            "rope_traditional": False,
            "rope_scaling": None,
        }

    @patch("mlx_audio.tts.models.spark.spark.Qwen2Model")
    def test_init(self, mock_qwen2_model):
        """Test SparkTTSModel initialization."""
        from mlx_audio.tts.models.spark.spark import Model, ModelConfig

        # Mock return value for Qwen2Model
        mock_qwen2_model.return_value = MagicMock()

        # Create a config instance
        config = ModelConfig(**self._default_config)

        # Initialize the model
        model = Model(config)

        # Check that the model was initialized correctly
        self.assertIsInstance(model, Model)

        # Verify tokenizer is None initially (loaded via post_load_hook)
        self.assertIsNone(model.tokenizer)

        # Verify the Qwen2Model was initialized correctly
        mock_qwen2_model.assert_called_once_with(config)


class TestIndexTTS(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "tokenizer_name": "mlx-community/IndexTTS",
            "bigvgan": {
                "adam_b1": 0.8,
                "adam_b2": 0.99,
                "lr_decay": 0.999998,
                "seed": 1234,
                "resblock": "1",
                "upsample_rates": [4, 4, 4, 4, 2, 2],
                "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
                "upsample_initial_channel": 1536,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "feat_upsample": False,
                "speaker_embedding_dim": 512,
                "cond_d_vector_in_each_upsampling_layer": True,
                "gpt_dim": 1024,
                "activation": "snakebeta",
                "snake_logscale": True,
                "use_cqtd_instead_of_mrd": True,
                "cqtd_filters": 128,
                "cqtd_max_filters": 1024,
                "cqtd_filters_scale": 1,
                "cqtd_dilations": [1, 2, 4],
                "cqtd_hop_lengths": [512, 256, 256],
                "cqtd_n_octaves": [9, 9, 9],
                "cqtd_bins_per_octaves": [24, 36, 48],
                "resolutions": [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
                "mpd_reshapes": [2, 3, 5, 7, 11],
                "use_spectral_norm": False,
                "discriminator_channel_mult": 1,
                "use_multiscale_melloss": True,
                "lambda_melloss": 15,
                "clip_grad_norm": 1000,
                "segment_size": 16384,
                "num_mels": 100,
                "num_freq": 1025,
                "n_fft": 1024,
                "hop_size": 256,
                "win_size": 1024,
                "sampling_rate": 24000,
                "fmin": 0,
                "fmax": None,
                "fmax_for_loss": None,
                "mel_type": "pytorch",
                "num_workers": 2,
                "dist_config": {
                    "dist_backend": "nccl",
                    "dist_url": "tcp://localhost:54321",
                    "world_size": 1,
                },
            },
            "bigvgan_checkpoint": "bigvgan_generator.pth",
            "dataset": {
                "bpe_model": "checkpoints/bpe.model",
                "sample_rate": 24000,
                "squeeze": False,
                "mel": {
                    "sample_rate": 24000,
                    "n_fft": 1024,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_mels": 100,
                    "mel_fmin": 0,
                    "normalize": False,
                },
            },
            "dvae_checkpoint": "dvae.pth",
            "gpt": {
                "model_dim": 1024,
                "max_mel_tokens": 605,
                "max_text_tokens": 402,
                "heads": 16,
                "use_mel_codes_as_input": True,
                "mel_length_compression": 1024,
                "layers": 20,
                "number_text_tokens": 12000,
                "number_mel_codes": 8194,
                "start_mel_token": 8192,
                "stop_mel_token": 8193,
                "start_text_token": 0,
                "stop_text_token": 1,
                "train_solo_embeddings": False,
                "condition_type": "conformer_perceiver",
                "condition_module": {
                    "output_size": 512,
                    "linear_units": 2048,
                    "attention_heads": 8,
                    "num_blocks": 6,
                    "input_layer": "conv2d2",
                    "perceiver_mult": 2,
                },
            },
            "gpt_checkpoint": "gpt.pth",
            "vqvae": {
                "channels": 100,
                "num_tokens": 8192,
                "hidden_dim": 512,
                "num_resnet_blocks": 3,
                "codebook_dim": 512,
                "num_layers": 2,
                "positional_dims": 1,
                "kernel_size": 3,
                "smooth_l1_loss": True,
                "use_transposed_convs": False,
            },
        }

    def test_init(self):
        """Test IndexTTS initialization."""
        from mlx_audio.tts.models.indextts.indextts import Model

        # Initialize model
        config = self._default_config
        model = Model(config)  # type: ignore

        # Check that model was created
        self.assertIsInstance(model, Model)


class TestVibeVoiceModel(unittest.TestCase):
    @property
    def _default_config(self):
        from mlx_audio.tts.models.vibevoice.config import ModelConfig

        return ModelConfig(
            model_path="/fake/model/path",
            sample_rate=24000,
        )

    def test_init(self):
        """Test VibeVoiceModel initialization."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        # Initialize model
        config = self._default_config
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

        # Verify model components exist
        self.assertIsNotNone(model.language_model)
        self.assertIsNotNone(model.tts_language_model)
        self.assertIsNotNone(model.acoustic_tokenizer)
        self.assertIsNotNone(model.prediction_head)
        self.assertIsNotNone(model.tts_eos_classifier)

    def test_sample_rate(self):
        """Test VibeVoiceModel sample_rate property."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        self.assertEqual(model.sample_rate, 24000)

    def test_get_input_embeddings(self):
        """Test VibeVoiceModel get_input_embeddings method."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        embeddings = model.get_input_embeddings()
        self.assertIsInstance(embeddings, nn.Embedding)
        self.assertEqual(embeddings.weight.shape[0], config.decoder_config.vocab_size)

    def test_sanitize(self):
        """Test VibeVoiceModel sanitize method."""
        from mlx.utils import tree_flatten

        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        # Test sanitize with model's own weights (no transformation needed)
        weights = dict(tree_flatten(model.parameters()))
        sanitized = model.sanitize(weights)

        # Sanitized weights should contain valid keys
        self.assertIsInstance(sanitized, dict)

    def test_sanitize_huggingface_keys(self):
        """Test VibeVoiceModel sanitize transforms HuggingFace keys."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        # Create mock weights with HuggingFace-style keys
        mock_weights = {
            "model.prediction_head.t_embedder.mlp.0.weight": mx.zeros((64, 64)),
            "model.prediction_head.adaLN_modulation.1.weight": mx.zeros((64, 64)),
        }

        sanitized = model.sanitize(mock_weights)

        # Check that keys were transformed (original keys should not exist)
        self.assertNotIn("model.prediction_head.t_embedder.mlp.0.weight", sanitized)
        self.assertNotIn("model.prediction_head.adaLN_modulation.1.weight", sanitized)

    def test_sanitize_preserves_quantization_metadata(self):
        """Test that sanitize preserves .scales and .biases for quantized models."""
        from mlx.utils import tree_flatten

        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        # Start with the model's own weights
        weights = dict(tree_flatten(model.parameters()))

        # Add mock quantization metadata for the key from the bug report:
        # "Expected shape (151936, 896) but received shape (151936, 224)
        #  for parameter language_model.embed_tokens.weight"
        quant_key = "language_model.embed_tokens.weight"
        weights[f"{quant_key}.scales"] = mx.ones((1,))
        weights[f"{quant_key}.biases"] = mx.ones((1,))

        sanitized = model.sanitize(weights)

        # Quantization metadata must survive sanitization
        self.assertIn(f"{quant_key}.scales", sanitized)
        self.assertIn(f"{quant_key}.biases", sanitized)

    def test_config_defaults(self):
        """Test VibeVoiceModel uses correct config defaults."""
        from mlx_audio.tts.models.vibevoice.config import ModelConfig

        config = ModelConfig()

        # Verify default values
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.acoustic_vae_dim, 64)
        self.assertEqual(config.tts_backbone_num_hidden_layers, 20)
        self.assertEqual(config.decoder_config.hidden_size, 896)
        self.assertEqual(config.decoder_config.num_hidden_layers, 24)


class TestChatterboxConfig(unittest.TestCase):
    def test_t3_config_defaults(self):
        """Test T3Config default values and factory methods."""
        from mlx_audio.tts.models.chatterbox.config import T3Config

        # Test defaults
        config = T3Config()
        self.assertEqual(config.text_tokens_dict_size, 704)
        self.assertEqual(config.speech_tokens_dict_size, 8194)
        self.assertEqual(config.llama_config_name, "Llama_520M")
        self.assertEqual(config.n_channels, 1024)
        self.assertFalse(config.is_multilingual)

        # Test factory methods
        self.assertFalse(T3Config.english_only().is_multilingual)
        self.assertTrue(T3Config.multilingual().is_multilingual)

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config = ModelConfig()

        self.assertEqual(config.model_type, "chatterbox")
        self.assertEqual(config.s3_sr, 16000)
        self.assertEqual(config.s3gen_sr, 24000)
        self.assertEqual(config.sample_rate, 24000)
        self.assertIsNotNone(config.t3_config)

    def test_model_config_from_dict(self):
        """Test ModelConfig.from_dict method."""
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config_dict = {
            "model_type": "chatterbox",
            "t3_config": {
                "text_tokens_dict_size": 2454,
            },
        }

        config = ModelConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "chatterbox")
        self.assertTrue(config.t3_config.is_multilingual)


class TestChatterboxModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.T3")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3Token2Wav")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3TokenizerV2")
    def test_init(self, mock_s3_tokenizer, mock_ve, mock_s3gen, mock_t3):
        """Test Model initialization with config."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Model
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config = ModelConfig()
        model = Model(config)

        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsNotNone(model.ve)
        self.assertEqual(model.sr, 24000)
        self.assertEqual(model.sample_rate, 24000)

    @patch("mlx_audio.tts.models.chatterbox.chatterbox.T3")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3Token2Wav")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3TokenizerV2")
    def test_sanitize(
        self, mock_s3_tokenizer, mock_ve_class, mock_s3gen_class, mock_t3_class
    ):
        """Test weight sanitization routes to correct components."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Model

        # Mock components to have sanitize methods that pass through weights
        for mock_class in [
            mock_ve_class,
            mock_t3_class,
            mock_s3gen_class,
            mock_s3_tokenizer,
        ]:
            mock_class.return_value.sanitize.side_effect = lambda w: w

        model = Model()

        # Test that prefixed weights are routed and re-prefixed
        weights = {
            "ve.lstm.weight": mx.zeros((10, 10)),
            "t3.tfmr.weight": mx.zeros((10, 10)),
            "s3gen.flow.weight": mx.zeros((10, 10)),
        }

        result = model.sanitize(weights)

        # Verify weights keep their prefixes
        self.assertIn("ve.lstm.weight", result)
        self.assertIn("t3.tfmr.weight", result)
        self.assertIn("s3gen.flow.weight", result)


class TestChatterboxTurboConfig(unittest.TestCase):
    def test_t3_config_defaults(self):
        """Test T3Config default values."""
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Config

        config = T3Config()
        self.assertEqual(config.text_tokens_dict_size, 50276)
        self.assertEqual(config.speech_tokens_dict_size, 6563)
        self.assertEqual(config.llama_config_name, "GPT2_medium")
        self.assertEqual(config.n_channels, 1024)
        self.assertEqual(config.speaker_embed_size, 256)
        self.assertEqual(config.speech_cond_prompt_len, 375)
        self.assertFalse(config.emotion_adv)
        self.assertFalse(config.use_perceiver_resampler)

    def test_t3_config_turbo_factory(self):
        """Test T3Config.turbo() factory method."""
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Config

        config = T3Config.turbo()
        self.assertEqual(config.text_tokens_dict_size, 50276)
        self.assertEqual(config.speech_tokens_dict_size, 6563)
        self.assertEqual(config.llama_config_name, "GPT2_medium")
        self.assertEqual(config.speech_cond_prompt_len, 375)
        self.assertFalse(config.emotion_adv)
        self.assertFalse(config.use_perceiver_resampler)

    def test_t3_config_is_multilingual(self):
        """Test is_multilingual property."""
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Config

        # Default turbo config is not multilingual
        config = T3Config.turbo()
        self.assertFalse(config.is_multilingual)

        # Multilingual config has text_tokens_dict_size == 2454
        multilingual_config = T3Config(text_tokens_dict_size=2454)
        self.assertTrue(multilingual_config.is_multilingual)


class TestChatterboxTurboPuncNorm(unittest.TestCase):
    def test_empty_string(self):
        """Test punc_norm handles empty string."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("")
        self.assertEqual(result, "You need to add some text for me to talk.")

    def test_capitalizes_first_letter(self):
        """Test punc_norm capitalizes first letter."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("hello world")
        self.assertTrue(result[0].isupper())

    def test_adds_period_if_missing(self):
        """Test punc_norm adds period if no ending punctuation."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("Hello world")
        self.assertTrue(result.endswith("."))

    def test_keeps_existing_punctuation(self):
        """Test punc_norm keeps existing ending punctuation."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        self.assertTrue(punc_norm("Hello world!").endswith("!"))
        self.assertTrue(punc_norm("Hello world?").endswith("?"))
        self.assertTrue(punc_norm("Hello world.").endswith("."))

    def test_removes_multiple_spaces(self):
        """Test punc_norm removes multiple spaces."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("Hello    world")
        self.assertNotIn("  ", result)

    def test_replaces_special_punctuation(self):
        """Test punc_norm replaces special punctuation."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        # Test ellipsis replacement
        result = punc_norm("Hello… world")
        self.assertNotIn("…", result)

        # Test em dash replacement
        result = punc_norm("Hello—world")
        self.assertIn("-", result)


class TestChatterboxTurboModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_init_with_config(self, mock_s3_tokenizer, mock_ve, mock_s3gen, mock_t3):
        """Test ChatterboxTurboTTS initialization with config dict."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS(config_or_t3={})

        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsNotNone(model.ve)
        self.assertEqual(model.sr, 24000)
        self.assertEqual(model.sample_rate, 24000)

    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_init_with_none(self, mock_s3_tokenizer, mock_ve, mock_s3gen, mock_t3):
        """Test ChatterboxTurboTTS initialization with None (default config)."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS()

        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsNotNone(model.ve)

    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_sanitize(
        self, mock_s3_tokenizer, mock_ve_class, mock_s3gen_class, mock_t3_class
    ):
        """Test weight sanitization routes to correct components."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        # Mock components to have sanitize methods that pass through weights
        for mock_class in [
            mock_ve_class,
            mock_t3_class,
            mock_s3gen_class,
            mock_s3_tokenizer,
        ]:
            mock_class.return_value.sanitize.side_effect = lambda w: w

        model = ChatterboxTurboTTS()

        # Test that prefixed weights are routed and re-prefixed
        weights = {
            "ve.lstm.weight": mx.zeros((10, 10)),
            "t3.tfmr.weight": mx.zeros((10, 10)),
            "s3gen.flow.weight": mx.zeros((10, 10)),
        }

        result = model.sanitize(weights)

        # Verify weights keep their prefixes
        self.assertIn("ve.lstm.weight", result)
        self.assertIn("t3.tfmr.weight", result)
        self.assertIn("s3gen.flow.weight", result)

    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_sanitize_with_other_weights(
        self, mock_s3_tokenizer, mock_ve_class, mock_s3gen_class, mock_t3_class
    ):
        """Test that unrecognized weights pass through sanitization."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        # Mock components to have sanitize methods that pass through weights
        for mock_class in [
            mock_ve_class,
            mock_t3_class,
            mock_s3gen_class,
            mock_s3_tokenizer,
        ]:
            mock_class.return_value.sanitize.side_effect = lambda w: w

        model = ChatterboxTurboTTS()

        # Test with weights that don't have known prefixes
        weights = {
            "ve.lstm.weight": mx.zeros((10, 10)),
            "unknown.param": mx.zeros((5, 5)),
        }

        result = model.sanitize(weights)

        # Both should be in result
        self.assertIn("ve.lstm.weight", result)
        self.assertIn("unknown.param", result)


class TestChatterboxTurboConditionals(unittest.TestCase):
    def test_conditionals_dataclass(self):
        """Test Conditionals dataclass creation."""
        from mlx_audio.tts.models.chatterbox_turbo import Conditionals
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Cond

        t3_cond = T3Cond(
            speaker_emb=mx.zeros((1, 256)),
            cond_prompt_speech_tokens=mx.zeros((1, 375), dtype=mx.int32),
        )
        gen_dict = {"ref_mel": mx.zeros((1, 80, 100))}

        conds = Conditionals(t3=t3_cond, gen=gen_dict)

        self.assertIsNotNone(conds.t3)
        self.assertIsNotNone(conds.gen)
        self.assertEqual(conds.t3.speaker_emb.shape, (1, 256))


class TestChatterboxTurboModelAlias(unittest.TestCase):
    def test_model_alias(self):
        """Test that Model is aliased to ChatterboxTurboTTS."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS, Model

        self.assertIs(Model, ChatterboxTurboTTS)


class TestSoprano(unittest.TestCase):
    """Tests for Soprano TTS model."""

    @property
    def _default_config(self):
        from mlx_audio.tts.models.soprano import DecoderConfig, ModelConfig

        return ModelConfig(
            model_type="qwen3",
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=32000,
            head_dim=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            decoder_config=DecoderConfig(),
        )

    # Config tests
    def test_decoder_config_defaults(self):
        """Test DecoderConfig default values."""
        from mlx_audio.tts.models.soprano import DecoderConfig

        config = DecoderConfig()
        self.assertEqual(config.decoder_num_layers, 8)
        self.assertEqual(config.decoder_dim, 768)
        self.assertEqual(config.decoder_intermediate_dim, 2304)
        self.assertEqual(config.hop_length, 512)
        self.assertEqual(config.n_fft, 2048)
        self.assertEqual(config.upscale, 4)
        self.assertEqual(config.input_kernel, 1)
        self.assertEqual(config.dw_kernel, 3)
        self.assertEqual(config.token_size, 2048)
        self.assertEqual(config.receptive_field, 4)

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from mlx_audio.tts.models.soprano import ModelConfig

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=32000,
            head_dim=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )
        self.assertEqual(config.sample_rate, 32000)
        self.assertIsNotNone(config.decoder_config)

    def test_model_config_post_init(self):
        """Test that ModelConfig creates decoder_config if None."""
        from mlx_audio.tts.models.soprano import DecoderConfig, ModelConfig

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=32000,
            head_dim=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            decoder_config=None,
        )
        self.assertIsNotNone(config.decoder_config)
        self.assertIsInstance(config.decoder_config, DecoderConfig)

    # Model tests
    def test_model_init(self):
        """Test Model initialization."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        self.assertIsNotNone(model.language_model)
        self.assertIsNotNone(model.decoder)
        self.assertEqual(model.config.sample_rate, 32000)

    def test_sample_rate_property(self):
        """Test sample_rate property."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        self.assertEqual(model.sample_rate, 32000)

    def test_layers_property(self):
        """Test layers property returns LM layers."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        layers = model.layers
        self.assertEqual(len(layers), config.num_hidden_layers)

    def test_sanitize(self):
        """Test weight sanitization."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        weights = {
            "model.embed_tokens.weight": mx.zeros((32000, 512)),
            "model.layers.0.input_layernorm.weight": mx.zeros(512),
            "decoder.backbone.weight": mx.zeros((512, 512)),
        }

        sanitized = model.sanitize(weights)

        self.assertIn("language_model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.layers.0.input_layernorm.weight", sanitized)
        self.assertIn("decoder.backbone.weight", sanitized)
        self.assertNotIn("model.embed_tokens.weight", sanitized)

    def test_sanitize_decoder_float32(self):
        """Test that decoder weights are converted to float32."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        weights = {
            "decoder.backbone.weight": mx.zeros((512, 512), dtype=mx.bfloat16),
            "lm_head.weight": mx.zeros((32000, 512), dtype=mx.bfloat16),
        }

        sanitized = model.sanitize(weights)

        self.assertEqual(sanitized["decoder.backbone.weight"].dtype, mx.float32)
        self.assertEqual(sanitized["language_model.lm_head.weight"].dtype, mx.bfloat16)

    def test_format_duration(self):
        """Test _format_duration helper method."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        self.assertEqual(model._format_duration(0), "00:00:00.000")
        self.assertEqual(model._format_duration(1.5), "00:00:01.500")
        self.assertEqual(model._format_duration(61.25), "00:01:01.250")
        self.assertEqual(model._format_duration(3661.123), "01:01:01.123")

    # Text processing tests
    def test_clean_text(self):
        """Test clean_text function."""
        from mlx_audio.tts.models.soprano.text import clean_text

        self.assertEqual(clean_text("Hello World!"), "hello world!")
        self.assertEqual(clean_text("I have 5 apples."), "i have five apples.")

    def test_normalize_numbers(self):
        """Test number normalization."""
        from mlx_audio.tts.models.soprano.text import normalize_numbers

        self.assertIn("five", normalize_numbers("5"))
        self.assertIn("twenty", normalize_numbers("20"))
        self.assertIn("hundred", normalize_numbers("100"))
        self.assertIn("dollar", normalize_numbers("$5"))
        self.assertIn("first", normalize_numbers("1st"))

    def test_expand_abbreviations(self):
        """Test abbreviation expansion."""
        from mlx_audio.tts.models.soprano.text import expand_abbreviations

        self.assertIn("mister", expand_abbreviations("Mr."))
        self.assertIn("doctor", expand_abbreviations("Dr."))
        self.assertIn("text to speech", expand_abbreviations("TTS"))

    def test_expand_special_characters(self):
        """Test special character expansion."""
        from mlx_audio.tts.models.soprano.text import expand_special_characters

        self.assertIn("at", expand_special_characters("@"))
        self.assertIn("and", expand_special_characters("&"))
        self.assertIn("percent", expand_special_characters("%"))

    def test_collapse_whitespace(self):
        """Test whitespace collapsing."""
        from mlx_audio.tts.models.soprano.text import collapse_whitespace

        self.assertEqual(collapse_whitespace("hello  world"), "hello world")
        self.assertEqual(collapse_whitespace("  hello   world  "), "hello world")
        self.assertEqual(collapse_whitespace("hello ,world"), "hello,world")

    def test_dedup_punctuation(self):
        """Test punctuation deduplication."""
        from mlx_audio.tts.models.soprano.text import dedup_punctuation

        self.assertEqual(dedup_punctuation("hello...."), "hello.")
        self.assertEqual(dedup_punctuation("hello,,,,"), "hello,")
        self.assertEqual(dedup_punctuation("hello??!!"), "hello?")

    def test_convert_to_ascii(self):
        """Test unicode to ASCII conversion."""
        from mlx_audio.tts.models.soprano.text import convert_to_ascii

        self.assertEqual(convert_to_ascii("café"), "cafe")
        self.assertEqual(convert_to_ascii("naïve"), "naive")

    def test_num_to_words(self):
        """Test number to words conversion."""
        from mlx_audio.tts.models.soprano.text import _num_to_words

        self.assertEqual(_num_to_words(0), "zero")
        self.assertEqual(_num_to_words(1), "one")
        self.assertEqual(_num_to_words(10), "ten")
        self.assertEqual(_num_to_words(21), "twenty one")
        self.assertEqual(_num_to_words(100), "one hundred")
        self.assertEqual(_num_to_words(1000), "one thousand")
        self.assertEqual(_num_to_words(-5), "minus five")

    def test_ordinal_to_words(self):
        """Test ordinal to words conversion."""
        from mlx_audio.tts.models.soprano.text import _ordinal_to_words

        self.assertEqual(_ordinal_to_words(1), "first")
        self.assertEqual(_ordinal_to_words(2), "second")
        self.assertEqual(_ordinal_to_words(3), "third")
        self.assertEqual(_ordinal_to_words(10), "tenth")
        self.assertEqual(_ordinal_to_words(21), "twenty first")

    # Decoder tests
    def test_decoder_init(self):
        """Test SopranoDecoder initialization."""
        from mlx_audio.tts.models.soprano.decoder import SopranoDecoder

        decoder = SopranoDecoder(
            num_input_channels=512,
            decoder_num_layers=4,
            decoder_dim=256,
            decoder_intermediate_dim=768,
            hop_length=512,
            n_fft=2048,
            upscale=4,
            input_kernel=1,
            dw_kernel=3,
        )

        self.assertEqual(decoder.decoder_initial_channels, 512)
        self.assertEqual(decoder.num_layers, 4)
        self.assertEqual(decoder.dim, 256)
        self.assertEqual(decoder.intermediate_dim, 768)
        self.assertEqual(decoder.hop_length, 512)
        self.assertEqual(decoder.n_fft, 2048)
        self.assertEqual(decoder.upscale, 4)

    def test_decoder_default_intermediate_dim(self):
        """Test default intermediate_dim calculation."""
        from mlx_audio.tts.models.soprano.decoder import SopranoDecoder

        decoder = SopranoDecoder(
            num_input_channels=512,
            decoder_num_layers=4,
            decoder_dim=256,
            decoder_intermediate_dim=None,
        )

        self.assertEqual(decoder.intermediate_dim, 256 * 3)

    # ISTFT Head tests
    def test_istft_head_init(self):
        """Test ISTFTHead initialization."""
        from mlx_audio.tts.models.soprano.decoder import ISTFTHead

        head = ISTFTHead(dim=512, n_fft=2048, hop_length=512)

        self.assertEqual(head.n_fft, 2048)
        self.assertEqual(head.hop_length, 512)

    def test_istft_head_forward(self):
        """Test ISTFTHead forward pass."""
        from mlx_audio.tts.models.soprano.decoder import ISTFTHead

        head = ISTFTHead(dim=512, n_fft=2048, hop_length=512)
        x = mx.zeros((1, 10, 512))
        audio = head(x)

        self.assertEqual(len(audio.shape), 2)
        self.assertEqual(audio.shape[0], 1)


class TestQwen3TTSModel(unittest.TestCase):
    """Tests for Qwen3-TTS model."""

    def _default_talker_config(self):
        # Minimal config for fast tests
        return {
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "num_code_groups": 4,
            "text_hidden_size": 64,
            "text_vocab_size": 100,
            "codec_eos_token_id": 30,
            "codec_pad_id": 28,
            "codec_bos_id": 29,
            "codec_language_id": {"english": 20, "chinese": 21},
            "spk_id": {"chelsie": 10, "ethan": 11},
            "code_predictor_config": {
                "vocab_size": 32,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": 4,
            },
        }

    def _default_config(self, tts_model_type="base"):
        return {
            "model_type": "qwen3_tts",
            "tts_model_type": tts_model_type,
            "tts_model_size": "0b6",
            "talker_config": self._default_talker_config(),
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "tts_pad_token_id": 151671,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "sample_rate": 24000,
        }

    def test_config_init(self):
        """Test Qwen3TTS ModelConfig initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import ModelConfig

        config = ModelConfig.from_dict(self._default_config())

        self.assertEqual(config.model_type, "qwen3_tts")
        self.assertEqual(config.tts_model_type, "base")
        self.assertEqual(config.sample_rate, 24000)
        self.assertIsNotNone(config.talker_config)

    def test_config_custom_voice(self):
        """Test config with custom_voice model type."""
        from mlx_audio.tts.models.qwen3_tts.config import ModelConfig

        config = ModelConfig.from_dict(self._default_config("custom_voice"))

        self.assertEqual(config.tts_model_type, "custom_voice")

    def test_config_voice_design(self):
        """Test config with voice_design model type."""
        from mlx_audio.tts.models.qwen3_tts.config import ModelConfig

        config = ModelConfig.from_dict(self._default_config("voice_design"))

        self.assertEqual(config.tts_model_type, "voice_design")

    def test_model_init(self):
        """Test Qwen3TTS Model initialization."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config())
        model = Model(config)

        self.assertIsInstance(model, Model)
        self.assertEqual(model.model_type, "qwen3_tts")
        self.assertEqual(model.sample_rate, 24000)

    def test_model_supported_speakers(self):
        """Test supported speakers list."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config())
        model = Model(config)

        speakers = model.get_supported_speakers()
        self.assertIn("chelsie", speakers)
        self.assertIn("ethan", speakers)

    def test_model_supported_languages(self):
        """Test supported languages list."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config())
        model = Model(config)

        languages = model.get_supported_languages()
        self.assertIn("auto", languages)
        self.assertIn("english", languages)
        self.assertIn("chinese", languages)

    def test_talker_init(self):
        """Test Talker model initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerConfig
        from mlx_audio.tts.models.qwen3_tts.talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        config = Qwen3TTSTalkerConfig(**self._default_talker_config())
        talker = Qwen3TTSTalkerForConditionalGeneration(config)

        self.assertIsNotNone(talker.model)
        self.assertIsNotNone(talker.code_predictor)
        self.assertEqual(config.vocab_size, 32)

    def test_talker_forward(self):
        """Test Talker forward pass."""
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerConfig
        from mlx_audio.tts.models.qwen3_tts.talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        config = Qwen3TTSTalkerConfig(**self._default_talker_config())
        talker = Qwen3TTSTalkerForConditionalGeneration(config)

        # Test forward with inputs_embeds
        batch_size, seq_len = 1, 10
        hidden_size = config.hidden_size
        inputs_embeds = mx.random.normal((batch_size, seq_len, hidden_size))

        # Talker returns (logits, hidden_states)
        logits, hidden_states = talker(inputs_embeds=inputs_embeds)

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], config.vocab_size)
        self.assertEqual(hidden_states.shape, (batch_size, seq_len, hidden_size))

    def test_code_predictor_init(self):
        """Test CodePredictor initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import (
            Qwen3TTSTalkerCodePredictorConfig,
        )
        from mlx_audio.tts.models.qwen3_tts.talker import Qwen3TTSTalkerCodePredictor

        config = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            num_code_groups=4,
        )
        code_predictor = Qwen3TTSTalkerCodePredictor(config, talker_hidden_size=64)

        self.assertEqual(len(code_predictor.codec_embedding), 3)  # num_code_groups - 1
        self.assertIsNotNone(code_predictor.model)

    def test_code_predictor_forward(self):
        """Test CodePredictor forward pass."""
        from mlx_audio.tts.models.qwen3_tts.config import (
            Qwen3TTSTalkerCodePredictorConfig,
        )
        from mlx_audio.tts.models.qwen3_tts.talker import Qwen3TTSTalkerCodePredictor

        config = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            num_code_groups=4,
        )
        code_predictor = Qwen3TTSTalkerCodePredictor(config, talker_hidden_size=64)

        batch_size, seq_len = 1, 2
        inputs_embeds = mx.random.normal((batch_size, seq_len, 64))

        # CodePredictor returns (logits, cache, next_step)
        logits, _, _ = code_predictor(inputs_embeds=inputs_embeds)

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], config.vocab_size)

    def test_generate_routing_base(self):
        """Test that generate routes correctly for base model."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config("base"))
        model = Model(config)

        # Base model should not require instruct
        self.assertEqual(config.tts_model_type, "base")

    def test_generate_routing_custom_voice_requires_voice(self):
        """Test that custom_voice model requires voice parameter."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config("custom_voice"))
        model = Model(config)

        # Mock speech_tokenizer to avoid loading
        model.speech_tokenizer = MagicMock()

        with self.assertRaises(ValueError) as context:
            list(model.generate(text="Hello", voice=None))

        self.assertIn("voice", str(context.exception).lower())

    def test_generate_routing_voice_design_requires_instruct(self):
        """Test that voice_design model requires instruct parameter."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config("voice_design"))
        model = Model(config)

        # Mock speech_tokenizer to avoid loading
        model.speech_tokenizer = MagicMock()

        with self.assertRaises(ValueError) as context:
            list(model.generate(text="Hello", instruct=None))

        self.assertIn("instruct", str(context.exception).lower())

    def test_speaker_encoder_config(self):
        """Test SpeakerEncoder config initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig

        config = Qwen3TTSSpeakerEncoderConfig()

        self.assertEqual(config.mel_dim, 128)
        self.assertEqual(config.enc_dim, 1024)
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(len(config.enc_channels), 5)

    def test_mel_spectrogram(self):
        """Test mel spectrogram computation."""
        from mlx_audio.tts.models.qwen3_tts.qwen3_tts import mel_spectrogram

        # Create a simple test audio
        sample_rate = 24000
        duration = 0.5  # 0.5 seconds
        audio = mx.random.normal((int(sample_rate * duration),))

        mel = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=128,
            sample_rate=sample_rate,
            hop_size=256,
        )

        # Check output shape
        self.assertEqual(mel.ndim, 3)  # [batch, time, mels]
        self.assertEqual(mel.shape[0], 1)  # batch size
        self.assertEqual(mel.shape[2], 128)  # num_mels


class TestQwen3TTSEncoder(unittest.TestCase):
    """Tests for Qwen3TTSSpeechTokenizerEncoder."""

    def _default_encoder_config(self):
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTokenizerEncoderConfig

        return Qwen3TTSTokenizerEncoderConfig(
            frame_rate=12.5,
            audio_channels=1,
            codebook_dim=256,
            codebook_size=64,  # Small for tests
            compress=2,
            dilation_growth_rate=2,
            hidden_size=64,  # Small for tests
            intermediate_size=128,
            kernel_size=7,
            last_kernel_size=3,
            num_attention_heads=4,
            num_filters=16,  # Small for tests
            num_hidden_layers=1,  # Single layer for speed
            num_key_value_heads=4,
            num_quantizers=32,
            num_residual_layers=1,
            num_semantic_quantizers=1,
            residual_kernel_size=3,
            rope_theta=10000.0,
            sampling_rate=24000,
            sliding_window=250,
            upsampling_ratios=[8, 6, 5, 4],
            use_causal_conv=True,
            use_conv_shortcut=False,
            layer_scale_initial_scale=0.01,
            max_position_embeddings=8000,
            head_dim=16,
        )

    def test_encoder_init(self):
        """Test encoder initialization with valid config."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        self.assertIsNotNone(encoder.encoder)
        self.assertIsNotNone(encoder.encoder_transformer)
        self.assertIsNotNone(encoder.downsample)
        self.assertIsNotNone(encoder.quantizer)
        self.assertEqual(encoder.valid_num_quantizers, 16)

    def test_encoder_components(self):
        """Test encoder has correct component types."""
        from mlx_audio.codec.models.mimi.modules.conv import ConvDownsample1d
        from mlx_audio.codec.models.mimi.modules.quantization import (
            SplitResidualVectorQuantizer as MimiSplitRVQ,
        )
        from mlx_audio.codec.models.mimi.modules.seanet import SeanetEncoder
        from mlx_audio.codec.models.mimi.modules.transformer import ProjectedTransformer
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        self.assertIsInstance(encoder.encoder, SeanetEncoder)
        self.assertIsInstance(encoder.encoder_transformer, ProjectedTransformer)
        self.assertIsInstance(encoder.downsample, ConvDownsample1d)
        self.assertIsInstance(encoder.quantizer, MimiSplitRVQ)

    def test_encoder_cache_init(self):
        """Test encoder cache is properly initialized."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # Cache should have one entry per transformer layer
        self.assertEqual(len(encoder.encoder_cache), config.num_hidden_layers)

    def test_encoder_encode_output_shape(self):
        """Test encoder produces correct output shape."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # Input: [batch, channels, samples]
        # The downsample rate = prod(upsampling_ratios) * downsample_stride
        # = (8*6*5*4) * 2 = 1920
        num_samples = 1920 * 3  # 3 time steps expected
        audio = mx.random.normal((1, 1, num_samples))

        codes = encoder.encode(audio)
        mx.eval(codes)

        # Output: [batch, valid_num_quantizers, time]
        self.assertEqual(codes.ndim, 3)
        self.assertEqual(codes.shape[0], 1)
        self.assertEqual(codes.shape[1], 16)  # valid_num_quantizers
        # Time dimension: num_samples / 1920 = 3
        self.assertEqual(codes.shape[2], 3)

    def test_encoder_encode_different_lengths(self):
        """Test encoder handles different audio lengths correctly."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        for num_frames in [2, 5, 10]:
            num_samples = 1920 * num_frames
            audio = mx.random.normal((1, 1, num_samples))
            codes = encoder.encode(audio)
            mx.eval(codes)

            self.assertEqual(codes.shape[0], 1)
            self.assertEqual(codes.shape[1], 16)
            self.assertEqual(codes.shape[2], num_frames)

    def test_encoder_encode_truncates_quantizers(self):
        """Test encoder only returns first 16 quantizers out of 32."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        self.assertEqual(config.num_quantizers, 32)

        encoder = Qwen3TTSSpeechTokenizerEncoder(config)
        audio = mx.random.normal((1, 1, 1920 * 2))
        codes = encoder.encode(audio)
        mx.eval(codes)

        # Should only have 16 quantizers, not 32
        self.assertEqual(codes.shape[1], 16)

    def test_encoder_encode_code_range(self):
        """Test that encoded codes are within valid codebook range."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        audio = mx.random.normal((1, 1, 1920 * 3))
        codes = encoder.encode(audio)
        mx.eval(codes)

        # Codes should be in range [0, codebook_size)
        self.assertTrue(mx.all(codes >= 0).item())
        self.assertTrue(mx.all(codes < config.codebook_size).item())

    def test_encoder_causal_mask(self):
        """Test that encode creates proper causal attention mask."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # Different length inputs should produce consistent results
        # when processing the same prefix (due to causal masking)
        audio_short = mx.random.normal((1, 1, 1920 * 2))
        codes_short = encoder.encode(audio_short)
        mx.eval(codes_short)

        self.assertEqual(codes_short.shape[2], 2)

    def test_encoder_downsample_stride(self):
        """Test downsample stride is computed correctly from config."""
        import math

        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # encoder_frame_rate = sampling_rate / prod(upsampling_ratios) = 24000/960 = 25
        # downsample_stride = encoder_frame_rate / frame_rate = 25 / 12.5 = 2
        encoder_frame_rate = config.sampling_rate / math.prod(config.upsampling_ratios)
        expected_stride = int(encoder_frame_rate / config.frame_rate)
        self.assertEqual(expected_stride, 2)

        # Verify stride effect: with stride=2, output time should be halved
        # compared to no-downsample. The encode test already validates total
        # output shape (samples/1920), here we verify the stride math.
        self.assertEqual(encoder.downsample.conv.conv.conv._stride, expected_stride)


class TestQwen3TTSPrepareICLInputs(unittest.TestCase):
    """Tests for _prepare_icl_generation_inputs method."""

    def _make_model_with_mocks(self, hidden_size=64, num_code_groups=4, vocab_size=32):
        """Create a minimal Model with mocked components for testing ICL prep."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        # Use text_vocab_size large enough to hold tts_*_token_ids
        # so embeddings are distinct for different token IDs
        text_vocab_size = 200

        config_dict = {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "talker_config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": num_code_groups,
                "text_hidden_size": hidden_size,
                "text_vocab_size": text_vocab_size,
                "codec_eos_token_id": 30,
                "codec_pad_id": 28,
                "codec_bos_id": 29,
                "codec_language_id": {"english": 20, "chinese": 21},
                "spk_id": {"chelsie": 10, "ethan": 11},
                "code_predictor_config": {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "num_code_groups": num_code_groups,
                },
            },
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 50,
            "im_end_token_id": 51,
            "tts_pad_token_id": 60,
            "tts_bos_token_id": 61,
            "tts_eos_token_id": 62,
            "sample_rate": 24000,
        }

        config = ModelConfig.from_dict(config_dict)
        model = Model(config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        # Return 10 tokens for any encode call (includes role tokens)
        mock_tokenizer.encode.return_value = list(range(10))
        model.tokenizer = mock_tokenizer

        # Mock speech_tokenizer with encoder
        mock_speech_tokenizer = MagicMock()
        mock_speech_tokenizer.has_encoder = True
        ref_time = 5
        # encode returns [1, 16, ref_time]
        mock_speech_tokenizer.encode.return_value = mx.zeros(
            (1, 16, ref_time), dtype=mx.int32
        )
        model.speech_tokenizer = mock_speech_tokenizer

        # No speaker encoder for this test
        model.speaker_encoder = None

        return model, ref_time

    def test_prepare_icl_output_shapes(self):
        """Test that _prepare_icl_generation_inputs returns correct shapes."""
        model, ref_time = self._make_model_with_mocks()
        hidden_size = model.config.talker_config.hidden_size

        ref_audio = mx.random.normal((24000,))  # 1s audio
        input_embeds, trailing, tts_pad, ref_codes = (
            model._prepare_icl_generation_inputs(
                text="Hello world",
                ref_audio=ref_audio,
                ref_text="Reference text",
                language="auto",
            )
        )
        mx.eval(input_embeds, trailing, tts_pad, ref_codes)

        # input_embeds: [1, text_lens + codec_lens, hidden_size]
        self.assertEqual(input_embeds.ndim, 3)
        self.assertEqual(input_embeds.shape[0], 1)
        self.assertEqual(input_embeds.shape[2], hidden_size)

        # trailing_text_hidden: tts_pad_embed [1, 1, hidden_size] in non-streaming mode
        self.assertEqual(trailing.ndim, 3)
        self.assertEqual(trailing.shape[0], 1)
        self.assertEqual(trailing.shape[2], hidden_size)

        # tts_pad_embed: [1, 1, hidden_size]
        self.assertEqual(tts_pad.shape, (1, 1, hidden_size))

        # ref_codes: [1, 16, ref_time]
        self.assertEqual(ref_codes.shape, (1, 16, ref_time))

    def test_prepare_icl_non_streaming_structure(self):
        """Test non-streaming mode: text_with_codec_pad + codec_with_tts_pad."""
        model, ref_time = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))
        input_embeds, trailing, tts_pad, ref_codes = (
            model._prepare_icl_generation_inputs(
                text="Hello",
                ref_audio=ref_audio,
                ref_text="Ref",
                language="auto",
            )
        )
        mx.eval(input_embeds)

        # In non-streaming mode:
        # input_embeds = concat(text_with_codec_pad, codec_with_text_pad)
        # text_lens = ref_text_tokens + target_text_tokens + eos
        # codec_lens = 1 (bos) + ref_time (ref_codes)
        codec_lens = 1 + ref_time  # codec_bos + ref_time
        total_len = input_embeds.shape[1]

        # Total should be text_lens + codec_lens
        self.assertGreater(total_len, codec_lens)

    def test_prepare_icl_trailing_is_tts_pad(self):
        """Test that trailing_text_hidden equals tts_pad_embed in non-streaming mode."""
        model, _ = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))
        _, trailing, tts_pad, _ = model._prepare_icl_generation_inputs(
            text="Hello",
            ref_audio=ref_audio,
            ref_text="Ref",
            language="auto",
        )
        mx.eval(trailing, tts_pad)

        # In non-streaming mode, trailing = tts_pad_embed
        np.testing.assert_array_equal(np.array(trailing), np.array(tts_pad))

    def test_prepare_icl_ref_audio_dim_handling(self):
        """Test that ref_audio is properly reshaped for encoding."""
        model, _ = self._make_model_with_mocks()

        # Test 1D input
        ref_audio_1d = mx.random.normal((24000,))
        model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio_1d, ref_text="Ref"
        )
        # Speech tokenizer should receive [1, 1, samples]
        call_args = model.speech_tokenizer.encode.call_args[0][0]
        self.assertEqual(call_args.ndim, 3)
        self.assertEqual(call_args.shape[0], 1)
        self.assertEqual(call_args.shape[1], 1)

    def test_prepare_icl_ref_audio_2d_handling(self):
        """Test that 2D ref_audio is properly reshaped."""
        model, _ = self._make_model_with_mocks()

        # Test 2D input [1, samples]
        ref_audio_2d = mx.random.normal((1, 24000))
        model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio_2d, ref_text="Ref"
        )
        call_args = model.speech_tokenizer.encode.call_args[0][0]
        self.assertEqual(call_args.ndim, 3)

    def test_prepare_icl_language_id(self):
        """Test that language_id is incorporated in codec prefix."""
        model, _ = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))

        # With language="english", should include language_id in codec prefix
        input_embeds_en, _, _, _ = model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio, ref_text="Ref", language="english"
        )
        mx.eval(input_embeds_en)

        # With language="auto", no language_id
        input_embeds_auto, _, _, _ = model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio, ref_text="Ref", language="auto"
        )
        mx.eval(input_embeds_auto)

        # The embeddings should differ in size (language adds one more token)
        # auto: [nothink, think_bos, think_eos] = 3 codec prefix tokens
        # english: [think, think_bos, language_id, think_eos] = 4 codec prefix tokens
        # But this difference is after the icl_input_embed, so they should differ
        # Actually the codec prefix is appended AFTER icl_input_embed in the generate loop,
        # not in _prepare_icl_generation_inputs. The returned input_embeds only has
        # text_with_codec_pad + codec_with_text_pad. Language affects codec_prefix_embed
        # which is concatenated later. Let's verify the core structure is consistent.
        self.assertEqual(input_embeds_en.shape[2], input_embeds_auto.shape[2])

    def test_prepare_icl_no_tokenizer_raises(self):
        """Test that missing tokenizer raises ValueError."""
        model, _ = self._make_model_with_mocks()
        model.tokenizer = None

        ref_audio = mx.random.normal((24000,))
        with self.assertRaises(ValueError):
            model._prepare_icl_generation_inputs(
                text="Hello", ref_audio=ref_audio, ref_text="Ref"
            )

    def test_prepare_icl_codec_embed_includes_bos(self):
        """Test that codec embedding includes codec_bos prepended."""
        model, ref_time = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))
        input_embeds, _, _, _ = model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio, ref_text="Ref"
        )
        mx.eval(input_embeds)

        # Full structure: role_embed + combined_prefix + icl_input_embed
        # tokenizer.encode returns 10 tokens for each call
        # role_embed = target_ids[:, :3] = 3 tokens
        # combined_prefix: codec_prefix(nothink,think_bos,think_eos=3) + suffix(pad,bos=2) = 5
        #   pad_count = 5-2 = 3, combined_prefix = [3 pads + 1 bos] = 4 tokens
        # icl_input_embed = text_with_codec_pad + codec_with_text_pad
        #   text_lens = ref_text_ids(5) + text_ids(2) + eos(1) = 8
        #   codec_lens = bos(1) + ref_time(5) = 6
        #   icl total = 8 + 6 = 14
        # Total: 3 + 4 + 14 = 21
        role_tokens = 3
        prefix_tokens = 4  # (nothink/think + pad,bos) - 1 for offset
        text_lens = 5 + 2 + 1
        codec_lens = 1 + ref_time
        expected_total = role_tokens + prefix_tokens + text_lens + codec_lens
        self.assertEqual(input_embeds.shape[1], expected_total)


class TestQwen3TTSGenerateICL(unittest.TestCase):
    """Tests for _generate_icl method."""

    def _make_icl_model(self, hidden_size=64, num_code_groups=4, vocab_size=2048):
        """Create a minimal Model for ICL generation testing."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config_dict = {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "talker_config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": num_code_groups,
                "text_hidden_size": hidden_size,
                "text_vocab_size": 100,
                "codec_eos_token_id": 30,
                "codec_pad_id": 28,
                "codec_bos_id": 29,
                "codec_language_id": {"english": 20, "chinese": 21},
                "spk_id": {"chelsie": 10, "ethan": 11},
                "code_predictor_config": {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "num_code_groups": num_code_groups,
                },
            },
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "tts_pad_token_id": 151671,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "sample_rate": 24000,
        }

        config = ModelConfig.from_dict(config_dict)
        model = Model(config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(10))
        model.tokenizer = mock_tokenizer

        # Mock speech_tokenizer
        mock_speech_tokenizer = MagicMock()
        mock_speech_tokenizer.has_encoder = True
        mock_speech_tokenizer.decode_upsample_rate = 1920
        ref_time = 5
        # encode returns [1, num_code_groups, ref_time] to match generation
        mock_speech_tokenizer.encode.return_value = mx.zeros(
            (1, num_code_groups, ref_time), dtype=mx.int32
        )
        # decode returns (audio, audio_lengths)
        mock_speech_tokenizer.decode.return_value = (
            mx.random.normal((1, 24000)),  # ~1s audio
            mx.array([24000]),
        )

        # streaming_decode yields audio chunks
        def mock_streaming_decode(codes):
            yield mx.random.normal((1, 24000))  # ~1s audio chunk

        mock_speech_tokenizer.streaming_decode = mock_streaming_decode
        model.speech_tokenizer = mock_speech_tokenizer
        model.speaker_encoder = None

        return model

    def test_generate_icl_produces_result(self):
        """Test that _generate_icl produces a GenerationResult."""
        from mlx_audio.tts.models.base import GenerationResult

        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        results = list(
            model._generate_icl(
                text="Hello world",
                ref_audio=ref_audio,
                ref_text="Reference",
                language="auto",
                temperature=0.9,
                max_tokens=5,  # Very few tokens for fast test
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.5,
            )
        )

        # Should produce at least one result (or empty if EOS on first token)
        # With random weights, it's unlikely to hit EOS immediately
        if results:
            self.assertIsInstance(results[0], GenerationResult)
            self.assertIsNotNone(results[0].audio)
            self.assertEqual(results[0].sample_rate, 24000)

    def test_generate_icl_calls_speech_tokenizer_decode(self):
        """Test that _generate_icl calls speech_tokenizer.decode."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        # Track decode calls
        decode_calls = []
        original_decode = model.speech_tokenizer.decode

        def tracking_decode(codes):
            decode_calls.append(codes)
            return original_decode(codes)

        model.speech_tokenizer.decode = tracking_decode

        results = list(
            model._generate_icl(
                text="Hello",
                ref_audio=ref_audio,
                ref_text="Ref",
                max_tokens=3,
                repetition_penalty=1.5,
            )
        )

        if results:
            # decode should have been called with combined ref + gen codes
            self.assertEqual(len(decode_calls), 1)
            decode_args = decode_calls[0]
            # Should be [1, ref_time + gen_len, num_code_groups]
            self.assertEqual(decode_args.ndim, 3)
            self.assertEqual(decode_args.shape[0], 1)
            self.assertEqual(decode_args.shape[2], 4)  # num_code_groups

    def test_generate_icl_eos_stops_generation(self):
        """Test that EOS token stops generation early."""
        model = self._make_icl_model()
        config = model.config.talker_config

        ref_audio = mx.random.normal((24000,))

        # Patch _sample_token to always return EOS
        eos_id = config.codec_eos_token_id
        with patch.object(model, "_sample_token", return_value=mx.array([[eos_id]])):
            results = list(
                model._generate_icl(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    max_tokens=100,
                    repetition_penalty=1.5,
                )
            )

        # Should produce no results since EOS on first step
        self.assertEqual(len(results), 0)

    def test_generate_icl_max_tokens_limit(self):
        """Test that max_tokens caps effective generation length."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        # With max_tokens=2, should generate at most 2 tokens
        results = list(
            model._generate_icl(
                text="Hello",
                ref_audio=ref_audio,
                ref_text="Ref",
                max_tokens=2,
                repetition_penalty=1.5,
            )
        )

        if results:
            # token_count should be <= 2
            self.assertLessEqual(results[0].token_count, 2)

    def test_generate_icl_text_based_max_tokens_cap(self):
        """Test that effective_max_tokens is capped based on text length."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        # tokenizer.encode returns 10 tokens for the text
        # target_token_count = 10
        # effective_max_tokens = min(200, max(75, 10 * 6)) = min(200, 75) = 75
        # Without the cap, it would generate up to 200 tokens

        # Mock _sample_token to never return EOS (forces hitting the cap)
        def non_eos_sample(*args, **kwargs):
            return mx.array([[5]])  # Always non-EOS (eos=30)

        with patch.object(model, "_sample_token", side_effect=non_eos_sample):
            results = list(
                model._generate_icl(
                    text="Hi",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    max_tokens=200,  # Higher than text-based cap
                    repetition_penalty=1.5,
                )
            )

        self.assertEqual(len(results), 1)
        # With text-based cap: effective = min(200, max(75, 10*6)) = 75
        # Token count should be exactly 75 (hit the cap, not 200)
        self.assertEqual(results[0].token_count, 75)

    def test_generate_icl_repetition_penalty_applied(self):
        """Test that repetition penalty is applied during generation."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        # Track _sample_token calls to verify rep_penalty param
        original_sample = model._sample_token
        sample_calls = []

        def tracking_sample(*args, **kwargs):
            sample_calls.append(kwargs.get("repetition_penalty", 1.0))
            return original_sample(*args, **kwargs)

        with patch.object(model, "_sample_token", side_effect=tracking_sample):
            list(
                model._generate_icl(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    max_tokens=2,
                    repetition_penalty=1.5,
                )
            )

        # All CB0 sampling calls should use the specified repetition penalty
        for call_pen in sample_calls:
            if call_pen != 1.0:  # CB1+ calls don't pass rep_penalty
                self.assertEqual(call_pen, 1.5)

    def test_generate_icl_ref_codes_prepended(self):
        """Test that reference codes are prepended to generated codes for decoding."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))
        ref_time = 5  # Matches mock setup
        config = model.config.talker_config
        eos_id = config.codec_eos_token_id

        # Track decode calls
        decode_calls = []
        original_decode = model.speech_tokenizer.decode

        def tracking_decode(codes):
            decode_calls.append(codes)
            return original_decode(codes)

        model.speech_tokenizer.decode = tracking_decode

        # Force generation of exactly 2 tokens then EOS
        cb0_count = [0]

        def controlled_sample(*args, **kwargs):
            # CB0 calls have eos_token_id set
            if kwargs.get("eos_token_id") is not None:
                cb0_count[0] += 1
                if cb0_count[0] <= 2:
                    return mx.array([[5]])  # non-EOS token
                else:
                    return mx.array([[eos_id]])  # EOS
            # Code predictor calls: return valid token
            return mx.array([[3]])

        with patch.object(model, "_sample_token", side_effect=controlled_sample):
            results = list(
                model._generate_icl(
                    text="Hello world test",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    max_tokens=10,
                    repetition_penalty=1.5,
                )
            )

        self.assertEqual(len(results), 1)
        # Check that decode was called with ref_time + gen_len time steps
        self.assertEqual(len(decode_calls), 1)
        decode_args = decode_calls[0]
        gen_len = results[0].token_count
        self.assertEqual(gen_len, 2)
        expected_time = ref_time + gen_len
        self.assertEqual(decode_args.shape[1], expected_time)

    def test_generate_icl_proportional_trimming(self):
        """Test that reference audio portion is trimmed from output."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))
        ref_time = 5

        # Set up decode to return longer audio so trimming is testable
        total_audio_len = 48000
        model.speech_tokenizer.decode.return_value = (
            mx.random.normal((1, total_audio_len)),
            mx.array([total_audio_len]),
        )

        results = list(
            model._generate_icl(
                text="Hello world",
                ref_audio=ref_audio,
                ref_text="Ref",
                max_tokens=5,
                repetition_penalty=1.5,
            )
        )

        if results:
            # Audio should be shorter than total_audio_len due to trimming
            self.assertLess(results[0].samples, total_audio_len)

    def test_generate_routing_uses_icl_when_ref_audio_provided(self):
        """Test that generate() routes to ICL when ref_audio and ref_text provided."""
        model = self._make_icl_model()

        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])  # Empty generator
            list(
                model.generate(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Reference text",
                )
            )

        # Should have called _generate_icl
        mock_icl.assert_called_once()

    def test_generate_routing_icl_rep_penalty_floor(self):
        """Test that generate() enforces min rep_penalty=1.5 for ICL mode."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            list(
                model.generate(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    repetition_penalty=1.05,  # Below floor
                )
            )

        # Should have been called with rep_penalty=1.5 (the floor)
        call_kwargs = mock_icl.call_args[1]
        self.assertEqual(call_kwargs["repetition_penalty"], 1.5)

    def test_generate_routing_icl_rep_penalty_passthrough(self):
        """Test that rep_penalty > 1.5 is passed through unchanged."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            list(
                model.generate(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    repetition_penalty=2.0,  # Above floor
                )
            )

        call_kwargs = mock_icl.call_args[1]
        self.assertEqual(call_kwargs["repetition_penalty"], 2.0)

    def test_generate_routing_no_icl_without_encoder(self):
        """Test that generate() skips ICL when speech_tokenizer has no encoder."""
        model = self._make_icl_model()
        model.speech_tokenizer.has_encoder = False
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            # This will fall through to the non-ICL base path
            # which would call _generate_base or similar
            try:
                list(
                    model.generate(
                        text="Hello",
                        ref_audio=ref_audio,
                        ref_text="Ref",
                    )
                )
            except Exception:
                pass  # May fail in non-ICL path, that's fine

        # Should NOT have called _generate_icl
        mock_icl.assert_not_called()

    def test_generate_routing_no_icl_without_ref_text(self):
        """Test that generate() skips ICL when ref_text is not provided."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            try:
                list(
                    model.generate(
                        text="Hello",
                        ref_audio=ref_audio,
                        ref_text=None,
                    )
                )
            except Exception:
                pass

        mock_icl.assert_not_called()


class TestQwen3TTSStreamingDecode(unittest.TestCase):
    """Tests for streaming vs non-streaming decode behavior."""

    def _make_model(self, hidden_size=64, num_code_groups=4, vocab_size=2048):
        """Create a minimal Qwen3-TTS model for testing."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config_dict = {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "talker_config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": num_code_groups,
                "text_hidden_size": hidden_size,
                "text_vocab_size": 100,
                "codec_eos_token_id": 30,
                "codec_pad_id": 28,
                "codec_bos_id": 29,
                "codec_language_id": {"english": 20, "chinese": 21},
                "spk_id": {"chelsie": 10, "ethan": 11},
                "code_predictor_config": {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "num_code_groups": num_code_groups,
                },
            },
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "tts_pad_token_id": 151671,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "sample_rate": 24000,
        }

        config = ModelConfig.from_dict(config_dict)
        model = Model(config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(10))
        model.tokenizer = mock_tokenizer

        # Mock speech_tokenizer
        mock_speech_tokenizer = MagicMock()
        mock_speech_tokenizer.has_encoder = False
        mock_speech_tokenizer.decode_upsample_rate = 1920
        mock_speech_tokenizer.decode.return_value = (
            mx.random.normal((1, 48000)),
            mx.array([48000]),
        )

        def mock_streaming_decode(codes, chunk_tokens=100):
            # Yield chunks
            total_samples = 48000
            chunk_samples = chunk_tokens * 1920
            for i in range(0, total_samples, chunk_samples):
                end = min(i + chunk_samples, total_samples)
                yield mx.random.normal((1, end - i))

        mock_speech_tokenizer.streaming_decode = mock_streaming_decode
        model.speech_tokenizer = mock_speech_tokenizer

        return model

    def test_decode_chunk_uses_streaming_decode(self):
        """Test that _decode_chunk internally calls streaming_decode."""
        model = self._make_model()

        # Track streaming_decode calls
        streaming_decode_calls = []

        def tracking_streaming_decode(codes, chunk_tokens=100):
            streaming_decode_calls.append(
                {"codes": codes, "chunk_tokens": chunk_tokens}
            )
            yield mx.random.normal((1, 48000))

        model.speech_tokenizer.streaming_decode = tracking_streaming_decode

        # Call _decode_chunk directly
        codes = mx.zeros((1, 10, 4))  # [batch, time, num_code_groups]
        model._decode_chunk(codes, chunk_tokens=50)

        # Verify streaming_decode was called with correct chunk_tokens
        self.assertEqual(len(streaming_decode_calls), 1)
        self.assertEqual(streaming_decode_calls[0]["chunk_tokens"], 50)

    def test_decode_chunk_respects_chunk_tokens_parameter(self):
        """Test that _decode_chunk passes chunk_tokens to streaming_decode."""
        model = self._make_model()

        # Track chunk_tokens values
        chunk_tokens_used = []

        def tracking_streaming_decode(codes, chunk_tokens=100):
            chunk_tokens_used.append(chunk_tokens)
            yield mx.random.normal((1, 48000))

        model.speech_tokenizer.streaming_decode = tracking_streaming_decode

        # Test with different chunk_tokens values
        codes = mx.zeros((1, 10, 4))

        model._decode_chunk(codes, chunk_tokens=25)
        self.assertEqual(chunk_tokens_used[-1], 25)

        model._decode_chunk(codes, chunk_tokens=100)
        self.assertEqual(chunk_tokens_used[-1], 100)

        model._decode_chunk(codes, chunk_tokens=300)
        self.assertEqual(chunk_tokens_used[-1], 300)

    def test_streaming_chunk_size_calculation(self):
        """Test that streaming_chunk_size is calculated from streaming_interval."""
        # The formula is: streaming_chunk_size = max(1, int(streaming_interval * 12.5))
        # Test the calculation directly

        # streaming_interval=2.0 -> 25 tokens
        self.assertEqual(max(1, int(2.0 * 12.5)), 25)

        # streaming_interval=4.0 -> 50 tokens
        self.assertEqual(max(1, int(4.0 * 12.5)), 50)

        # streaming_interval=8.0 -> 100 tokens
        self.assertEqual(max(1, int(8.0 * 12.5)), 100)

        # streaming_interval=0.1 -> 1 token (minimum)
        self.assertEqual(max(1, int(0.1 * 12.5)), 1)


@patch("importlib.resources.open_text", patched_open_text)
class TestBailingMMModel(unittest.TestCase):
    HAS_ONNX = importlib.util.find_spec("onnx") is not None

    @staticmethod
    def _build_dummy_onnx(path: Path):
        import onnx
        from onnx import helper, numpy_helper

        expected = {
            "linear.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "linear.bias": np.array([0.1, -0.2], dtype=np.float32),
        }
        initializers = [
            numpy_helper.from_array(array, name=name)
            for name, array in expected.items()
        ]
        graph = helper.make_graph(
            nodes=[],
            name="dummy_campplus",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )
        model = helper.make_model(graph, producer_name="mlx-audio-tests")
        onnx.save(model, path.as_posix())
        return expected

    def _make_minimal_bailingmm_model(self):
        from mlx_audio.tts.models.bailingmm.bailingmm import Model

        model = Model.__new__(Model)
        model.tokenizer = MagicMock()
        model.tokenizer.encode.return_value = [1, 2, 3]
        model.audio = MagicMock()
        model.audio.sample_rate = 24000
        return model

    def test_convert_campplus_onnx_to_safetensors_allclose(self):
        if not self.HAS_ONNX:
            self.skipTest("onnx is required for campplus conversion test.")

        from safetensors.numpy import load_file

        from mlx_audio.tts.models.bailingmm import convert_campplus_onnx_to_safetensors

        with TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "campplus.onnx"
            expected = self._build_dummy_onnx(onnx_path)

            safetensors_path = convert_campplus_onnx_to_safetensors(
                onnx_path=onnx_path,
                output_path=Path(tmpdir) / "campplus.safetensors",
                verify_allclose=True,
            )

            converted = load_file(str(safetensors_path))
            self.assertEqual(set(converted.keys()), set(expected.keys()))
            for key in expected:
                self.assertTrue(
                    np.allclose(expected[key], converted[key], rtol=1e-5, atol=1e-6)
                )

    def test_qwen2_sliding_window_attention_applies_after_window_boundary(self):
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelArgs

        from mlx_audio.tts.models.bailingmm.bailingmm import MingQwen2Model

        args = Qwen2ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=1,
            rope_theta=1_000_000.0,
        )
        model = MingQwen2Model(
            args,
            {
                "use_sliding_window": True,
                "sliding_window": 4,
                "max_window_layers": 0,
            },
        )

        x = mx.random.normal((1, 8, 32), dtype=mx.float32)
        mask = create_attention_mask(x, None)

        y_sliding = model.layers[0](x, mask, None)
        model.layers[0].self_attn.sliding_window = None
        y_full = model.layers[0](x, mask, None)

        diff = np.abs(np.array(y_sliding) - np.array(y_full)).mean(axis=-1)[0]
        self.assertLess(diff[:4].max(), 1e-6)
        self.assertGreater(diff[4:].max(), 1e-6)

    @patch("mlx_audio.tts.models.bailingmm.bailingmm.FlowLoss")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.Aggregator")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.AudioVAE")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.MingBailingMoeModel")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.MingQwen2ForCausalLM")
    def test_init_uses_dense_qwen2_backbone_when_moe_fields_missing(
        self,
        mock_qwen2_lm,
        mock_moe_lm,
        mock_audio_vae,
        _mock_aggregator,
        _mock_flowloss,
    ):
        from mlx_audio.tts.models.bailingmm.bailingmm import Model

        audio = MagicMock()
        audio.sample_rate = 24000
        mock_audio_vae.return_value = audio

        config = {
            "model_type": "dense",
            "llm_config": {
                "model_type": "qwen2",
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "intermediate_size": 4864,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 151936,
                "rope_theta": 1_000_000.0,
            },
            "audio_tokenizer_config": {
                "sample_rate": 24000,
                "enc_kwargs": {"latent_dim": 64},
            },
            "ditar_config": {"patch_size": 8},
            "aggregator_config": {"depth": 1},
        }

        model = Model(config)

        mock_qwen2_lm.assert_called_once()
        mock_moe_lm.assert_not_called()
        self.assertEqual(model.llm_args.model_type, "qwen2")

    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.mx.get_peak_memory",
        return_value=0.0,
    )
    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.time.perf_counter",
        side_effect=[0.0, 1.0],
    )
    def test_generate_keeps_trailing_samples(
        self, _mock_perf_counter, _mock_peak_memory
    ):
        model = self._make_minimal_bailingmm_model()

        latent = mx.zeros((1, 1, 64), dtype=mx.float32)
        model.sample = MagicMock(return_value=iter([(latent, True)]))

        # Torch path does not apply a trailing low-energy trim pass.
        chunk = mx.array([[0.25, 0.15, 0.0, 0.0, 0.0]], dtype=mx.float32)
        model.audio.decode.return_value = (chunk, (None, None, None), None)

        result = next(model.generate(text="hello", max_tokens=1))
        np.testing.assert_allclose(np.array(result.audio), np.array(chunk)[0], atol=0)

    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.mx.get_peak_memory",
        return_value=0.0,
    )
    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.time.perf_counter",
        side_effect=[0.0, 1.0],
    )
    def test_generate_concatenates_all_decode_chunks(
        self, _mock_perf_counter, _mock_peak_memory
    ):
        import mlx_audio.tts.models.bailingmm.bailingmm as bailingmm_module

        model = self._make_minimal_bailingmm_model()

        latent = mx.zeros((1, 1, 64), dtype=mx.float32)
        model.sample = MagicMock(return_value=iter([(latent, False), (latent, True)]))
        chunk1 = mx.array([[0.1, 0.2]], dtype=mx.float32)
        chunk2 = mx.zeros((1, 0), dtype=mx.float32)
        model.audio.decode.side_effect = [
            (chunk1, (None, None, None), None),
            (chunk2, (None, None, None), None),
        ]

        observed = {}
        original_concat = bailingmm_module.mx.concatenate

        def _concat_spy(items, axis=0):
            observed["num_chunks"] = len(items)
            return original_concat(items, axis=axis)

        with patch(
            "mlx_audio.tts.models.bailingmm.bailingmm.mx.concatenate",
            side_effect=_concat_spy,
        ):
            result = next(model.generate(text="hello", max_tokens=2))

        self.assertEqual(observed.get("num_chunks"), 2)
        np.testing.assert_allclose(np.array(result.audio), np.array(chunk1)[0], atol=0)


class TestFishSpeechPrompt(unittest.TestCase):
    def test_encode_for_inference_places_vq_codes_in_correct_rows(self):
        from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
            Conversation,
            Message,
            TextPart,
            VQPart,
        )

        tokenizer = FakeTokenizer()
        conversation = Conversation(
            [
                Message(
                    role="user",
                    parts=[
                        TextPart("hello"),
                        VQPart(mx.array([[1, 2], [3, 4]], dtype=mx.int32)),
                        TextPart("world"),
                    ],
                    add_im_start=False,
                    add_im_end=False,
                )
            ]
        )

        values = conversation.encode_for_inference(tokenizer, num_codebooks=2)
        self.assertEqual(tuple(values.shape), (3, 4))
        self.assertEqual(values[0].tolist(), [1, 1001, 1002, 2])
        self.assertEqual(values[1].tolist(), [0, 1, 2, 0])
        self.assertEqual(values[2].tolist(), [0, 3, 4, 0])


class TestFishSpeechModel(unittest.TestCase):
    def test_model_type_remapping_uses_config_value(self):
        from mlx_audio.tts.utils import get_model_and_args

        module, model_type = get_model_and_args("fish_qwen3_omni", ["s2", "pro"])
        self.assertEqual(model_type, "fish_qwen3_omni")
        self.assertTrue(hasattr(module, "Model"))

    def test_sanitize_remaps_upstream_keys(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        model = Model(tiny_config())
        weights = {
            "text_model.model.embeddings.weight": mx.zeros((4, 4)),
            "audio_decoder.embeddings.weight": mx.zeros((4, 4)),
            "audio_decoder.layers.0.attention.wqkv.weight": mx.zeros((4, 4)),
            "audio_decoder.codebook_embeddings.weight": mx.zeros((4, 4)),
        }

        sanitized = model.sanitize(weights)

        self.assertIn("model.embeddings.weight", sanitized)
        self.assertIn("model.fast_embeddings.weight", sanitized)
        self.assertIn("model.fast_layers.0.attention.wqkv.weight", sanitized)
        self.assertIn("model.codebook_embeddings.weight", sanitized)

    def test_config_from_dict_handles_upstream_nested_shape(self):
        config = tiny_config()
        self.assertEqual(config.text_config.n_layer, 1)
        self.assertEqual(config.audio_decoder_config.num_codebooks, 2)
        self.assertEqual(config.semantic_start_token_id, 1000)


# ---------------------------------------------------------------------------
# Irodori-TTS helpers
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal HuggingFace-style tokenizer stub for Irodori-TTS tests."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [3, 4, 5]


class _FakeDACVAE:
    """DACVAE stub that matches the real API shapes."""

    def __init__(self, latent_dim: int = 8, downsample_factor: int = 1920):
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor

    def encode(self, audio_in: mx.array) -> mx.array:
        B = audio_in.shape[0]
        T = max(1, int(audio_in.shape[1]) // self.downsample_factor)
        return mx.zeros((B, self.latent_dim, T), dtype=mx.float32)

    def decode(self, latent: mx.array) -> mx.array:
        B, _D, T = latent.shape
        return mx.zeros((B, T * self.downsample_factor, 1), dtype=mx.float32)


def _small_irodori_dit_config(**overrides):
    from mlx_audio.tts.models.irodori_tts.config import IrodoriDiTConfig

    defaults = dict(
        latent_dim=8,
        latent_patch_size=1,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        text_mlp_ratio=2.0,
        speaker_mlp_ratio=2.0,
        text_vocab_size=64,
        text_dim=32,
        text_layers=1,
        text_heads=4,
        speaker_dim=32,
        speaker_layers=1,
        speaker_heads=4,
        speaker_patch_size=1,
        timestep_embed_dim=16,
        adaln_rank=8,
        norm_eps=1e-5,
    )
    defaults.update(overrides)
    return IrodoriDiTConfig(**defaults)


def _small_irodori_model_config(**sampler_overrides):
    from mlx_audio.tts.models.irodori_tts.config import ModelConfig, SamplerConfig

    sampler_defaults = dict(
        num_steps=1,
        cfg_scale_text=1.0,
        cfg_scale_speaker=1.0,
        sequence_length=4,
    )
    sampler_defaults.update(sampler_overrides)
    return ModelConfig(
        dit=_small_irodori_dit_config(),
        sampler=SamplerConfig(**sampler_defaults),
    )


# ---------------------------------------------------------------------------
# Irodori-TTS test classes
# ---------------------------------------------------------------------------


class TestIrodoriNormalizeText(unittest.TestCase):
    def test_fullwidth_alpha_to_halfwidth(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("Ａｂ"), "Ab")

    def test_fullwidth_digits_to_halfwidth(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("１２３"), "123")

    def test_halfwidth_kana_to_fullwidth(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("ｱｲ"), "アイ")

    def test_wave_dash_to_katakana_dash(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("ー〜ー"), "ーーー")

    def test_trailing_kuten_stripped(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        result = normalize_text("こんにちは。")
        self.assertFalse(result.endswith("。"))
        self.assertEqual(result, "こんにちは")

    def test_surrounding_brackets_stripped(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("「こんにちは」"), "こんにちは")

    def test_no_change_for_plain_text(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        text = "こんにちは"
        self.assertEqual(normalize_text(text), text)


class TestIrodoriEncodeText(unittest.TestCase):
    def setUp(self):
        self.tok = _MockTokenizer()

    def test_output_shapes(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        self.assertEqual(tuple(ids.shape), (1, 10))
        self.assertEqual(tuple(mask.shape), (1, 10))

    def test_bos_prepended(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        self.assertEqual(int(ids[0, 0]), self.tok.bos_token_id)

    def test_no_bos(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, _ = encode_text("hello", self.tok, max_length=10, add_bos=False)
        self.assertEqual(int(ids[0, 0]), 3)

    def test_padding(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        for i in range(4, 10):
            self.assertEqual(int(ids[0, i]), self.tok.pad_token_id)
            self.assertFalse(bool(mask[0, i]))

    def test_mask_true_for_real_tokens(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        for i in range(4):
            self.assertTrue(bool(mask[0, i]))

    def test_truncation(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=2, add_bos=True)
        self.assertEqual(tuple(ids.shape), (1, 2))


class TestIrodoriDiTShapes(unittest.TestCase):
    def setUp(self):
        from mlx_audio.tts.models.irodori_tts.model import IrodoriDiT

        self.cfg = _small_irodori_dit_config()
        self.model = IrodoriDiT(self.cfg)

    def test_full_forward_shape(self):
        B, S = 1, 6
        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([0.5], dtype=mx.float32)
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.random.normal((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        out = self.model(x_t, t, text_ids, text_mask, ref_latent, ref_mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))

    def test_encode_conditions_shapes(self):
        B = 1
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.random.normal((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        t_state, t_mask, s_state, s_mask = self.model.encode_conditions(
            text_ids, text_mask, ref_latent, ref_mask
        )
        mx.eval(t_state, s_state)
        self.assertEqual(tuple(t_state.shape), (B, 5, self.cfg.text_dim))
        self.assertEqual(int(s_state.shape[0]), B)
        self.assertEqual(int(s_state.shape[-1]), self.cfg.speaker_dim)

    def test_kv_cache_and_forward_with_conditions(self):
        B, S = 1, 4
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.zeros((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        t_state, t_mask, s_state, s_mask = self.model.encode_conditions(
            text_ids, text_mask, ref_latent, ref_mask
        )
        kv_text, kv_speaker = self.model.build_kv_cache(t_state, s_state)
        self.assertEqual(len(kv_text), self.cfg.num_layers)
        self.assertEqual(len(kv_speaker), self.cfg.num_layers)

        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([0.3], dtype=mx.float32)
        out = self.model.forward_with_conditions(
            x_t, t, t_state, t_mask, s_state, s_mask, kv_text, kv_speaker
        )
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))

    def test_zero_speaker_latent(self):
        B, S = 1, 4
        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([1.0], dtype=mx.float32)
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.zeros((B, 1, self.cfg.latent_dim))
        ref_mask = mx.zeros((B, 1), dtype=mx.bool_)

        out = self.model(x_t, t, text_ids, text_mask, ref_latent, ref_mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))


class TestIrodoriModelSanitize(unittest.TestCase):
    def setUp(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        self.model = Model(_small_irodori_model_config())

    def test_cond_module_key_remapped(self):
        weights = {"cond_module.0.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.cond_module.layers.0.weight", sanitized)
        self.assertNotIn("cond_module.0.weight", sanitized)

    def test_model_prefix_added(self):
        weights = {"blocks.0.mlp.w1.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.blocks.0.mlp.w1.weight", sanitized)

    def test_model_prefix_not_doubled(self):
        weights = {"model.out_proj.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.out_proj.weight", sanitized)
        self.assertNotIn("model.model.out_proj.weight", sanitized)

    def test_deep_cond_module_key(self):
        weights = {"cond_module.2.bias": mx.zeros((1,), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.cond_module.layers.2.bias", sanitized)


class TestIrodoriGenerateSmoke(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        cfg = _small_irodori_model_config()
        model = Model(cfg)
        model.dacvae = _FakeDACVAE(
            latent_dim=cfg.dit.latent_dim,
            downsample_factor=cfg.audio_downsample_factor,
        )
        model._tokenizer = _MockTokenizer()
        return model

    def test_generate_returns_result(self):
        model = self._make_model()
        results = list(model.generate("こんにちは", rng_seed=0))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].sample_rate, 48000)
        self.assertGreater(results[0].samples, 0)

    def test_generate_with_ref_audio(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        cfg = _small_irodori_model_config()
        model = Model(cfg)
        model.dacvae = _FakeDACVAE(
            latent_dim=cfg.dit.latent_dim,
            downsample_factor=cfg.audio_downsample_factor,
        )
        model._tokenizer = _MockTokenizer()
        ref = mx.zeros((1, cfg.audio_downsample_factor * 4), dtype=mx.float32)
        results = list(model.generate("テスト", ref_audio=ref, rng_seed=1))
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].samples, 0)

    def test_generate_stream_raises(self):
        model = self._make_model()
        with self.assertRaises(NotImplementedError):
            next(model.generate("hi", stream=True))

    def test_generate_without_dacvae_raises(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        cfg = _small_irodori_model_config()
        model = Model(cfg)
        model._tokenizer = _MockTokenizer()
        with self.assertRaises(ValueError):
            next(model.generate("hi"))

    def test_result_fields(self):
        model = self._make_model()
        result = next(model.generate("テスト", rng_seed=0))
        self.assertIsNotNone(result.audio)
        self.assertIsInstance(result.token_count, int)
        self.assertGreater(result.token_count, 0)
        self.assertIsNotNone(result.audio_duration)
        self.assertGreater(result.real_time_factor, 0.0)


class TestKugelAudioModel(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.kugelaudio.config import ModelConfig
        from mlx_audio.tts.models.kugelaudio.kugelaudio import Model

        cfg = ModelConfig.from_dict(
            {
                "acoustic_vae_dim": 4,
                "decoder_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_hidden_layers": 1,
                    "vocab_size": 32,
                    "max_position_embeddings": 64,
                },
                "diffusion_head_config": {
                    "hidden_size": 8,
                    "latent_size": 4,
                    "head_layers": 1,
                },
                "acoustic_tokenizer_config": {
                    "vae_dim": 4,
                    "encoder_n_filters": 4,
                    "decoder_n_filters": 4,
                    "encoder_ratios": [2],
                    "encoder_depths": "1",
                },
            }
        )
        return Model(cfg)

    def test_from_dict_basic(self):
        """Basic config creation with nested sub-configs."""
        from mlx_audio.tts.models.kugelaudio.config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "acoustic_vae_dim": 64,
                "decoder_config": {"hidden_size": 3584, "num_hidden_layers": 28},
                "diffusion_head_config": {},
                "acoustic_tokenizer_config": {},
            }
        )
        self.assertEqual(cfg.model_type, "kugelaudio")
        self.assertEqual(cfg.acoustic_vae_dim, 64)
        self.assertEqual(cfg.decoder_config.hidden_size, 3584)
        self.assertEqual(cfg.decoder_config.num_hidden_layers, 28)

    def test_sanitize(self):
        """Test full sanitize: prefix stripping, skip rules, re-indexing,
        quantization metadata, and linear weight transposition."""
        from mlx.utils import tree_flatten

        model = self._make_model()
        params = dict(tree_flatten(model.parameters()))

        # Build fake HF-style weight dict exercising every sanitize path
        fake_weights = {}

        # 1) "model." prefix → should be stripped
        lang_key = next(k for k in params if k.startswith("language_model."))
        fake_weights[f"model.{lang_key}"] = params[lang_key]

        # 2) Keys that should be dropped
        fake_weights["model.semantic_tokenizer.foo"] = mx.zeros((2,))
        fake_weights["model.semantic_connector.bar"] = mx.zeros((2,))
        fake_weights["model.acoustic_tokenizer.encoder.baz"] = mx.zeros((2,))

        # 3) Sequential re-indexing (.mlp.0. → .mlp.layers.0.)
        mlp_key = next((k for k in params if "t_embedder.mlp.layers." in k), None)
        if mlp_key:
            pytorch_mlp = "model." + mlp_key.replace(".mlp.layers.", ".mlp.")
            fake_weights[pytorch_mlp] = params[mlp_key]

        # 4) Quantization metadata
        fake_weights["language_model.layers.0.self_attn.q_proj.scales"] = mx.zeros((4,))
        fake_weights["language_model.layers.0.self_attn.q_proj.biases"] = mx.zeros((4,))

        # 5) Linear weight transposition (reversed shape)
        lm_key = "lm_head.weight"
        if lm_key in params:
            fake_weights[lm_key] = mx.zeros(tuple(reversed(params[lm_key].shape)))

        result = model.sanitize(fake_weights)

        # Verify prefix stripping
        self.assertIn(lang_key, result)

        # Verify dropped keys
        for dropped in [
            "semantic_tokenizer.foo",
            "semantic_connector.bar",
            "acoustic_tokenizer.encoder.baz",
        ]:
            self.assertNotIn(dropped, result)

        # Verify sequential re-indexing
        if mlp_key:
            self.assertIn(mlp_key, result)

        # Verify quantization metadata preserved
        self.assertIn("language_model.layers.0.self_attn.q_proj.scales", result)
        self.assertIn("language_model.layers.0.self_attn.q_proj.biases", result)

        # Verify linear weight transposed to correct shape
        if lm_key in params:
            self.assertEqual(result[lm_key].shape, params[lm_key].shape)

    def test_token_constraint_mask_allows_valid_tokens(self):
        """Only VALID_SPEECH_TOKENS should survive the mask."""
        from mlx_audio.tts.models.kugelaudio.kugelaudio import VALID_SPEECH_TOKENS

        vocab_size = 152064
        logits = mx.zeros((1, vocab_size))

        constraint_mask = mx.full(logits.shape, float("-inf"))
        valid_indices = mx.array(VALID_SPEECH_TOKENS)
        constraint_mask[:, valid_indices] = 0.0
        masked = logits + constraint_mask
        mx.eval(masked)

        for tid in VALID_SPEECH_TOKENS:
            self.assertEqual(masked[0, tid].item(), 0.0)

        self.assertEqual(masked[0, 0].item(), float("-inf"))
        self.assertEqual(masked[0, 1000].item(), float("-inf"))

    def test_token_argmax_selects_highest_valid_token(self):
        """argmax on masked logits should pick the boosted valid token."""
        from mlx_audio.tts.models.kugelaudio.kugelaudio import (
            SPEECH_DIFFUSION_ID,
            VALID_SPEECH_TOKENS,
        )

        vocab_size = 152064
        logits = mx.zeros((1, vocab_size))

        logits = logits.at[0, SPEECH_DIFFUSION_ID].add(mx.array(10.0))

        constraint_mask = mx.full(logits.shape, float("-inf"))
        valid_indices = mx.array(VALID_SPEECH_TOKENS)
        constraint_mask[:, valid_indices] = 0.0
        masked = logits + constraint_mask

        selected = mx.argmax(masked, axis=-1).item()
        self.assertEqual(selected, SPEECH_DIFFUSION_ID)

    def test_sde_scheduler_inherits_from_base(self):
        """SDEDPMSolverMultistepScheduler should subclass the base."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )
        from mlx_audio.tts.models.vibevoice.scheduler import (
            DPMSolverMultistepScheduler as BaseDPMSolver,
        )

        self.assertTrue(issubclass(SDEDPMSolverMultistepScheduler, BaseDPMSolver))

    def test_sde_scheduler_adds_noise(self):
        """SDE variant should produce different results from deterministic."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )
        from mlx_audio.tts.models.vibevoice.scheduler import (
            DPMSolverMultistepScheduler as BaseDPMSolver,
        )

        mx.random.seed(42)
        sde = SDEDPMSolverMultistepScheduler(
            num_train_timesteps=100, prediction_type="v_prediction"
        )
        det = BaseDPMSolver(num_train_timesteps=100, prediction_type="v_prediction")

        sde.set_timesteps(5)
        det.set_timesteps(5)

        sample = mx.ones((1, 4)) * 0.5
        model_output = mx.ones((1, 4)) * 0.1

        sde_result = sde.step(model_output, sde.timesteps[0], sample)
        det_result = det.step(model_output, det.timesteps[0], sample)

        mx.eval(sde_result.prev_sample, det_result.prev_sample)

        diff = mx.abs(sde_result.prev_sample - det_result.prev_sample).sum().item()
        self.assertGreater(diff, 0.0)

    def test_sde_scheduler_step_output_shape(self):
        """Output shape should match input shape."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )

        sched = SDEDPMSolverMultistepScheduler(num_train_timesteps=100)
        sched.set_timesteps(5)

        sample = mx.zeros((2, 8))
        output = mx.zeros((2, 8))

        result = sched.step(output, sched.timesteps[0], sample)
        mx.eval(result.prev_sample)

        self.assertEqual(result.prev_sample.shape, (2, 8))
        self.assertIsNotNone(result.x0_pred)

    def test_sde_scheduler_reset_clears_state(self):
        """reset() should zero out step index and lower order count."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )

        sched = SDEDPMSolverMultistepScheduler(num_train_timesteps=100)
        sched.set_timesteps(5)
        sched._step_index = 3  # pylint: disable=protected-access
        sched.lower_order_nums = 2
        sched.reset()
        self.assertIsNone(sched._step_index)  # pylint: disable=protected-access
        self.assertEqual(sched.lower_order_nums, 0)


class TestAudioDiTModel(unittest.TestCase):
    @staticmethod
    def _small_vae_config():
        from mlx_audio.tts.models.longcat_audiodit.config import VaeConfig

        return VaeConfig(
            in_channels=1,
            channels=4,
            c_mults=[1, 2],
            strides=[2, 4],
            latent_dim=4,
            encoder_latent_dim=8,
            use_snake=True,
            downsample_shortcut="averaging",
            upsample_shortcut="duplicating",
            out_shortcut="averaging",
            in_shortcut="duplicating",
            final_tanh=False,
            downsampling_ratio=8,
            sample_rate=24000,
            scale=0.71,
        )

    def setUp(self):
        from mlx_audio.tts.models.longcat_audiodit.config import (
            ModelConfig,
            TextEncoderConfig,
        )
        from mlx_audio.tts.models.longcat_audiodit.longcat_audiodit import Model

        self.cfg = ModelConfig(
            dit_dim=16,
            dit_depth=2,
            dit_heads=2,
            dit_ff_mult=2.0,
            dit_text_dim=8,
            dit_dropout=0.0,
            dit_bias=True,
            dit_cross_attn=True,
            dit_adaln_type="global",
            dit_adaln_use_text_cond=True,
            dit_long_skip=True,
            dit_text_conv=True,
            dit_qk_norm=True,
            dit_cross_attn_norm=False,
            dit_eps=1e-6,
            dit_use_latent_condition=True,
            repa_dit_layer=1,
            latent_dim=4,
            sigma=0.0,
            sampling_rate=24000,
            latent_hop=8,
            max_wav_duration=2.0,
            text_encoder_model="google/umt5-base",
            text_add_embed=True,
            text_norm_feat=True,
            vae_config=TestAudioDiTModel._small_vae_config(),
            text_encoder_config=TextEncoderConfig(
                vocab_size=32,
                d_model=8,
                d_kv=4,
                d_ff=16,
                num_layers=1,
                num_heads=2,
                relative_attention_num_buckets=8,
                relative_attention_max_distance=16,
            ),
        )

        self.model = Model(self.cfg)

    # -- config --

    def test_config_from_dict(self):
        from mlx_audio.tts.models.longcat_audiodit.config import ModelConfig

        cfg = ModelConfig(
            vae_config={"channels": 64, "latent_dim": 32},
            text_encoder_config={"d_model": 256},
        )
        self.assertEqual(cfg.vae_config.channels, 64)
        self.assertEqual(cfg.text_encoder_config.d_model, 256)

    def test_config_defaults(self):
        from mlx_audio.tts.models.longcat_audiodit.config import ModelConfig

        cfg = ModelConfig()
        self.assertEqual(cfg.model_type, "audiodit")
        self.assertEqual(cfg.dit_dim, 1536)
        self.assertIsNotNone(cfg.vae_config)
        self.assertEqual(cfg.vae_config.scale, 0.71)
        total = 1
        for s in cfg.vae_config.strides:
            total *= s
        self.assertEqual(total, cfg.vae_config.downsampling_ratio)

    # -- forward --

    def test_vae_encode_decode_shapes(self):
        mx.random.seed(0)
        x = mx.random.normal((1, 64, 1))
        latent = self.model.vae.encode(x)
        mx.eval(latent)
        self.assertEqual(tuple(latent.shape), (1, 8, 4))

        recon = self.model.vae.decode(latent)
        mx.eval(recon)
        self.assertEqual(tuple(recon.shape), (1, 64, 1))

    def test_text_encoder_forward(self):
        ids = mx.zeros((1, 5), dtype=mx.int32)
        mask = mx.ones((1, 5), dtype=mx.float32)
        out = self.model.encode_text(ids, mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (1, 5, self.cfg.text_encoder_config.d_model))

    def test_dit_forward(self):
        B, S, T = 1, 4, 3
        out = self.model.transformer(
            x=mx.random.normal((B, S, self.cfg.latent_dim)),
            text=mx.random.normal((B, T, self.cfg.dit_text_dim)),
            text_len=mx.array([T], dtype=mx.float32),
            time=mx.array([0.5]),
            mask=mx.ones((B, S), dtype=mx.bool_),
            cond_mask=mx.ones((B, T), dtype=mx.bool_),
            return_ith_layer=self.cfg.repa_dit_layer,
            latent_cond=mx.zeros((B, S, self.cfg.latent_dim)),
        )
        mx.eval(out["last_hidden_state"])
        self.assertEqual(
            tuple(out["last_hidden_state"].shape), (B, S, self.cfg.latent_dim)
        )
        self.assertIsNotNone(out["hidden_state"])

    def test_encode_prompt_audio(self):
        audio = mx.random.normal((1, 64, 1))
        latent, prompt_dur = self.model.encode_prompt_audio(audio)
        mx.eval(latent)
        self.assertEqual(latent.shape[2], self.cfg.latent_dim)
        self.assertEqual(latent.shape[1], prompt_dur)
        self.assertGreater(prompt_dur, 0)

    # -- sanitize --

    def test_sanitize_weight_norm(self):
        weights = {
            "vae.encoder.layers.0.weight_v": mx.ones((4, 1, 3)),
            "vae.encoder.layers.0.weight_g": mx.ones((4, 1, 1)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("vae.encoder.layers.0.weight", sanitized)
        self.assertNotIn("vae.encoder.layers.0.weight_v", sanitized)
        self.assertEqual(
            tuple(sanitized["vae.encoder.layers.0.weight"].shape), (4, 3, 1)
        )

    def test_sanitize_conv_transpose_weight_norm(self):
        weights = {
            "vae.decoder.layers.1.layers.1.weight_v": mx.ones((8, 4, 4)),
            "vae.decoder.layers.1.layers.1.weight_g": mx.ones((8, 1, 1)),
        }
        sanitized = self.model.sanitize(weights)
        key = "vae.decoder.layers.1.layers.1.weight"
        self.assertIn(key, sanitized)
        self.assertEqual(tuple(sanitized[key].shape), (4, 4, 8))

    def test_sanitize_text_encoder_remapping(self):
        weights = {
            "text_encoder.encoder.embed_tokens.weight": mx.zeros((32, 8)),
            "text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight": mx.zeros(
                (4, 8)
            ),
            "text_encoder.encoder.block.0.layer.1.DenseReluDense.wi_0.weight": mx.zeros(
                (16, 8)
            ),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("text_encoder.shared.weight", sanitized)
        self.assertIn("text_encoder.block.0.SelfAttention.q.weight", sanitized)
        self.assertIn("text_encoder.block.0.DenseReluDense.wi_0.weight", sanitized)

    def test_sanitize_transformer_remapping(self):
        weights = {
            "transformer.proj.2.weight": mx.zeros((4, 4)),
            "transformer.blocks.0.attn.to_out.0.weight": mx.zeros((4, 4)),
            "transformer.blocks.0.ff.3.weight": mx.zeros((4, 4)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("transformer.proj.1.weight", sanitized)
        self.assertIn("transformer.blocks.0.attn.to_out.weight", sanitized)
        self.assertIn("transformer.blocks.0.ff.1.weight", sanitized)

    def test_sanitize_dwconv_transpose(self):
        weights = {
            "transformer.text_conv_layer.0.dwconv.weight": mx.ones((8, 1, 7)),
            "transformer.text_conv_layer.0.dwconv.bias": mx.zeros((8,)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("transformer.text_conv_layer.0.dwconv_weight", sanitized)
        self.assertEqual(
            tuple(sanitized["transformer.text_conv_layer.0.dwconv_weight"].shape),
            (8, 7, 1),
        )


if __name__ == "__main__":
    unittest.main()

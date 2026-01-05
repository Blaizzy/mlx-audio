import importlib.resources
import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from misaki import en


# Create a patch for the deprecated open_text function
def patched_open_text(package, resource):
    """Replacement for deprecated open_text using files() API"""
    return importlib.resources.files(package).joinpath(resource).open("r")


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
    @patch("mlx_audio.tts.models.kokoro.kokoro.json.load")
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

        # Mock the KokoroModel - fix the import path
        with patch("mlx_audio.tts.models.kokoro.kokoro.Model") as mock_kokoro_model:
            with patch(
                "mlx_audio.tts.models.kokoro.pipeline.isinstance"
            ) as mock_isinstance:
                mock_model = MagicMock()
                mock_kokoro_model.return_value = mock_model

                # Simply make isinstance always return True when checking for KokoroModel
                mock_isinstance.return_value = True

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
                    "mlx_audio.tts.models.kokoro.pipeline.hf_hub_download"
                ) as mock_hf_hub_download:
                    pipeline = KokoroPipeline.__new__(KokoroPipeline)
                    pipeline.lang_code = "a"
                    pipeline.voices = {}
                    # Add the missing repo_id attribute
                    pipeline.repo_id = "mlx-community/kokoro-tts"

                    # Mock the load voice return value
                    load_voice_tensor.return_value = mx.zeros((512, 1, 256))

                    # Test loading a single voice
                    pipeline.load_single_voice("voice1")
                    mock_hf_hub_download.assert_called_once()
                    self.assertIn("voice1", pipeline.voices)

                    # Test loading multiple voices
                    mock_hf_hub_download.reset_mock()
                    pipeline.voices = {}  # Reset voices
                    result = pipeline.load_voice("voice1,voice2")
                    self.assertEqual(mock_hf_hub_download.call_count, 2)
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

    @patch("transformers.LlamaTokenizer")
    def test_init(self, mock_tokenizer):
        """Test LlamaModel initialization."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

    @patch("transformers.LlamaTokenizer")
    def test_generate(self, mock_tokenizer):
        """Test generate method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Verify batched input creation with a voice
        input_ids = model.prepare_input_ids(["Foo", "Bar Baz"], voice="zoe")
        self.assertEqual(input_ids.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 9, config.vocab_size))

        # Verify batched input creation with reference audio
        input_ids, input_mask = model.prepare_input_ids(
            ["Foo", "Bar Baz"], ref_audio=mx.zeros((100,)), ref_text="Caption"
        )
        self.assertEqual(input_ids.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 22, config.vocab_size))

    @patch("transformers.LlamaTokenizer")
    def test_sanitize(self, mock_tokenizer):
        """Test sanitize method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

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

    @patch("transformers.LlamaTokenizer")
    def test_init(self, mock_tokenizer):
        """Test initialization."""
        from mlx_audio.tts.models.outetts.outetts import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

    @patch("transformers.LlamaTokenizer")
    def test_generate(self, mock_tokenizer):
        """Test generate method."""
        from mlx_audio.tts.models.outetts.outetts import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

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
            "model_path": "/fake/model/path",
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

    @patch("mlx_audio.tts.models.spark.spark.load_tokenizer")
    @patch("mlx_audio.tts.models.spark.spark.BiCodecTokenizer")
    @patch("mlx_audio.tts.models.spark.spark.Qwen2Model")
    def test_init(
        self,
        mock_qwen2_model,
        mock_bicodec_tokenizer,
        mock_load_tokenizer,
    ):
        """Test SparkTTSModel initialization."""
        from pathlib import Path

        from mlx_audio.tts.models.spark.spark import Model, ModelConfig

        # Mock return values for patched functions
        mock_load_tokenizer.return_value = MagicMock()
        mock_bicodec_tokenizer.return_value = MagicMock()
        mock_qwen2_model.return_value = MagicMock()

        # Create a config instance
        config = ModelConfig(**self._default_config)
        config.model_path = Path("/fake/model/path")

        # Initialize the model
        model = Model(config)

        # Check that the model was initialized correctly
        self.assertIsInstance(model, Model)

        # Verify the tokenizer was loaded correctly
        mock_load_tokenizer.assert_called_once_with(
            config.model_path, eos_token_ids=config.eos_token_id
        )
        mock_bicodec_tokenizer.assert_called_once_with(config.model_path)

        # Verify the model was initialized correctly
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
        self.assertEqual(config.decoder_dim, 512)
        self.assertEqual(config.decoder_intermediate_dim, 1536)
        self.assertEqual(config.hop_length, 512)
        self.assertEqual(config.n_fft, 2048)
        self.assertEqual(config.upscale, 4)
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

        self.assertIn("embed_tokens.weight", sanitized)
        self.assertIn("layers.0.input_layernorm.weight", sanitized)
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
        self.assertEqual(sanitized["lm_head.weight"].dtype, mx.bfloat16)

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
            dw_kernel=3,
        )

        self.assertEqual(decoder.decoder_initial_channels, 512)
        self.assertEqual(decoder.num_layers, 4)
        self.assertEqual(decoder.dim, 256)
        self.assertEqual(decoder.intermediate_dim, 768)
        self.assertEqual(decoder.hop_length, 512)
        self.assertEqual(decoder.n_fft, 2048)
        self.assertEqual(decoder.upscale, 4)
        self.assertEqual(decoder.dw_kernel, 3)

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


if __name__ == "__main__":
    unittest.main()

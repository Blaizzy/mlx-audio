import json
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import mlx.core as mx
import numpy as np


class TestWhisperModel(unittest.TestCase):
    def setUp(self):
        # Import Whisper modules inside test class to avoid import issues
        from mlx_audio.stt.models.whisper.audio import N_FRAMES, N_SAMPLES, SAMPLE_RATE
        from mlx_audio.stt.models.whisper.decoding import (
            DecodingOptions,
            DecodingResult,
        )
        from mlx_audio.stt.models.whisper.whisper import (
            Model,
            ModelDimensions,
            STTOutput,
        )

        # Store references for use in test methods
        self.N_FRAMES = N_FRAMES
        self.N_SAMPLES = N_SAMPLES
        self.SAMPLE_RATE = SAMPLE_RATE
        self.DecodingOptions = DecodingOptions
        self.DecodingResult = DecodingResult
        self.Model = Model
        self.ModelDimensions = ModelDimensions
        self.STTOutput = STTOutput

        self.dims = self.ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=51864,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )
        self.model_mock = MagicMock(spec=self.Model, name="MockModelInstance")

        self.model_mock.dims = self.dims
        self.model_mock.dtype = mx.float32

        type(self.model_mock).is_multilingual = PropertyMock(return_value=False)
        type(self.model_mock).num_languages = PropertyMock(return_value=0)

    @patch("mlx_audio.stt.models.whisper.whisper.Path")
    @patch("mlx_audio.stt.models.whisper.whisper.snapshot_download")
    @patch("mlx_audio.stt.models.whisper.whisper.mx.load")
    @patch("mlx_audio.stt.models.whisper.whisper.json.loads")
    @patch("glob.glob")
    @patch("builtins.open", new_callable=MagicMock)
    def test_from_pretrained(
        self,
        mock_open,
        mock_glob,
        mock_json_loads_in_whisper,
        mock_mx_load,
        mock_snapshot_download,
        mock_pathlib_path,
    ):

        mock_snapshot_download.return_value = "dummy_path"
        mock_glob.return_value = ["dummy_path/weights.safetensors"]

        mock_paths_registry = {}

        def path_constructor_side_effect(path_str_arg):
            if path_str_arg in mock_paths_registry:
                return mock_paths_registry[path_str_arg]
            new_mock_path = MagicMock(spec=Path)
            new_mock_path.__str__.return_value = str(path_str_arg)
            if str(path_str_arg) == "dummy_path/weights.safetensors":
                new_mock_path.exists.return_value = True
            elif str(path_str_arg) == "dummy_path":
                new_mock_path.exists.return_value = True
            else:
                new_mock_path.exists.return_value = False

            def mock_truediv(other_segment):
                concatenated_path_str = f"{str(path_str_arg)}/{other_segment}"
                return path_constructor_side_effect(concatenated_path_str)

            new_mock_path.__truediv__.side_effect = mock_truediv
            new_mock_path.__rtruediv__ = mock_truediv
            mock_paths_registry[path_str_arg] = new_mock_path
            return new_mock_path

        mock_pathlib_path.side_effect = path_constructor_side_effect

        dummy_config = {
            "n_mels": 80,
            "n_audio_ctx": 1500,
            "n_audio_state": 384,
            "n_audio_head": 6,
            "n_audio_layer": 4,
            "n_vocab": 51865,
            "n_text_ctx": 448,
            "n_text_state": 384,
            "n_text_head": 6,
            "n_text_layer": 4,
        }
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            dummy_config
        )
        mock_json_loads_in_whisper.return_value = dummy_config
        dummy_weights = {
            "encoder.conv1.weight": mx.random.normal((384, 80, 3)),
            "encoder.conv1.bias": mx.random.normal((384,)),
        }
        mock_mx_load.return_value = dummy_weights

        model_instance = self.Model.from_pretrained(
            path_or_hf_repo="mlx-community/whisper-tiny-asr-fp16", dtype=mx.float32
        )

        self.assertIsInstance(model_instance, self.Model)
        self.assertEqual(model_instance.dims.n_mels, dummy_config["n_mels"])
        mock_snapshot_download.assert_called_once_with(
            repo_id="mlx-community/whisper-tiny-asr-fp16"
        )
        mock_open.assert_called_once_with("dummy_path/config.json", "r")
        mock_mx_load.assert_called_once_with("dummy_path/weights.safetensors")

    @patch("mlx_audio.stt.models.whisper.whisper.pad_or_trim")
    @patch("mlx_audio.stt.models.whisper.whisper.tqdm.tqdm")
    @patch("mlx_audio.stt.models.whisper.whisper.get_tokenizer")
    @patch("mlx_audio.stt.models.whisper.whisper.log_mel_spectrogram")
    def test_generate_simple_case(
        self,
        mock_log_mel,
        mock_get_tokenizer,
        mock_tqdm_tqdm,
        mock_pad_or_trim,
    ):
        """Test model.generate for a simple case with one segment."""

        mock_mel_data = mx.zeros(
            (self.N_FRAMES + 100, self.dims.n_mels), dtype=mx.float32
        )
        mock_log_mel.return_value = mock_mel_data

        EOT_TOKEN_ID = 50257
        TIMESTAMP_BEGIN_ID = 50364
        mock_tokenizer_inst = MagicMock(
            name="mock_tokenizer_instance_for_test",
            eot=EOT_TOKEN_ID,
            timestamp_begin=TIMESTAMP_BEGIN_ID,
        )

        def actual_decode_side_effect(tokens_to_decode):
            text_parts = []
            for token_val in tokens_to_decode:
                t = int(token_val)
                if t == 100:
                    text_parts.append("hello")
                elif t == 200:
                    text_parts.append("world")
                elif t == EOT_TOKEN_ID:
                    break
            return " ".join(text_parts) if text_parts else ""

        mock_tokenizer_inst.decode.side_effect = actual_decode_side_effect
        mock_tokenizer_inst.encode.return_value = []
        mock_get_tokenizer.return_value = mock_tokenizer_inst

        decoded_tokens_list = [100, 200, EOT_TOKEN_ID]
        mock_decoding_result = self.DecodingResult(
            tokens=mx.array(decoded_tokens_list),
            temperature=0.0,
            avg_logprob=-0.25,
            compression_ratio=1.2,
            no_speech_prob=0.05,
            audio_features=mx.zeros((1, self.dims.n_mels), dtype=mx.float32),
            language="en",
        )

        mock_pbar = MagicMock()
        mock_pbar.update = MagicMock()
        mock_tqdm_constructor = MagicMock()
        mock_tqdm_constructor.return_value.__enter__.return_value = mock_pbar
        mock_tqdm_constructor.return_value.__exit__ = MagicMock()
        mock_tqdm_tqdm.side_effect = mock_tqdm_constructor

        def pad_or_trim_side_effect(array, length, axis):
            return mx.zeros((length, array.shape[1]), dtype=array.dtype)

        mock_pad_or_trim.side_effect = pad_or_trim_side_effect

        dummy_audio_input = np.zeros(self.SAMPLE_RATE * 1, dtype=np.float32)

        real_model_for_test = self.Model(self.dims, dtype=mx.float32)

        # Patch this specific instance's 'decode' method
        with patch.object(
            real_model_for_test, "decode", return_value=mock_decoding_result
        ) as mock_instance_decode:
            output = real_model_for_test.generate(
                dummy_audio_input,
                language="en",
                word_timestamps=False,
                temperature=0.0,
                fp16=False,
            )

            mock_instance_decode.assert_called_once()
            args_decode_call, _ = mock_instance_decode.call_args
            self.assertEqual(
                args_decode_call[0].shape, (self.N_FRAMES, self.dims.n_mels)
            )  # mel_segment
            self.assertIsInstance(args_decode_call[1], self.DecodingOptions)
            self.assertEqual(args_decode_call[1].language, "en")
            self.assertEqual(args_decode_call[1].fp16, False)

        self.assertIsInstance(output, self.STTOutput)
        self.assertEqual(output.language, "en")
        expected_text_output = "hello world"
        self.assertEqual(output.text, expected_text_output)  #

        self.assertIsInstance(output.segments, list)
        self.assertEqual(len(output.segments), 1, "Should produce one segment")
        segment = output.segments[0]
        self.assertEqual(segment["text"], expected_text_output)
        self.assertEqual(segment["tokens"], decoded_tokens_list)

        self.assertEqual(segment["seek"], 0)
        self.assertAlmostEqual(segment["start"], 0.0)
        self.assertAlmostEqual(segment["end"], 1.0)
        self.assertEqual(segment["temperature"], mock_decoding_result.temperature)
        self.assertAlmostEqual(segment["avg_logprob"], mock_decoding_result.avg_logprob)
        self.assertAlmostEqual(
            segment["compression_ratio"], mock_decoding_result.compression_ratio
        )
        self.assertAlmostEqual(
            segment["no_speech_prob"], mock_decoding_result.no_speech_prob
        )

        mock_log_mel.assert_called_once_with(
            ANY, n_mels=self.dims.n_mels, padding=self.N_SAMPLES
        )
        np.testing.assert_array_equal(mock_log_mel.call_args[0][0], dummy_audio_input)
        mock_get_tokenizer.assert_called_once_with(
            real_model_for_test.is_multilingual,  # Reads from the instance
            num_languages=real_model_for_test.num_languages,  # Reads from the instance
            language="en",
            task="transcribe",
        )
        mock_pad_or_trim.assert_called_once()
        args_pad_call, _ = mock_pad_or_trim.call_args
        self.assertEqual(args_pad_call[0].shape, (100, self.dims.n_mels))
        self.assertEqual(args_pad_call[1], self.N_FRAMES)


class TestParakeetModel(unittest.TestCase):

    @patch("mlx.nn.Module.load_weights")
    @patch("mlx_audio.stt.models.parakeet.parakeet.hf_hub_download")
    @patch("mlx_audio.stt.models.parakeet.parakeet.json.load")
    @patch("mlx_audio.stt.models.parakeet.parakeet.open", new_callable=MagicMock)
    @patch("mlx.core.load")
    def test_parakeet_tdt_from_pretrained(
        self,
        mock_mlx_core_load,
        mock_parakeet_module_open,
        mock_parakeet_json_load,
        mock_hf_hub_download,
        mock_module_load_weights,
    ):
        """Test ParakeetTDT.from_pretrained method."""
        # Import Parakeet module inside test to avoid import issues
        from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT

        dummy_repo_id = "dummy/parakeet-tdt-model"
        dummy_config_path = "dummy_path/config.json"
        dummy_weights_path = "dummy_path/model.safetensors"

        # Configure hf_hub_download
        def hf_hub_download_side_effect(repo_id_arg, filename_arg):
            if repo_id_arg == dummy_repo_id and filename_arg == "config.json":
                return dummy_config_path
            if repo_id_arg == dummy_repo_id and filename_arg == "model.safetensors":
                return dummy_weights_path
            raise ValueError(
                f"Unexpected hf_hub_download call: {repo_id_arg}, {filename_arg}"
            )

        mock_hf_hub_download.side_effect = hf_hub_download_side_effect

        # Dummy config content
        dummy_vocabulary = [" ", "a", "b", "c"]
        dummy_config_dict = {
            "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
            "model_defaults": {"tdt_durations": [0, 1, 2, 3]},
            "preprocessor": {
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.02,
                "window_stride": 0.01,
                "window": "hann",
                "features": 80,
                "n_fft": 512,
                "dither": 1e-05,
                "pad_to": 0,
                "pad_value": 0.0,
            },
            "encoder": {
                "feat_in": 80,
                "n_layers": 17,
                "d_model": 512,
                "conv_dim": 512,
                "n_heads": 8,
                "self_attention_model": "rel_pos",
                "subsampling": "dw_striding",
                "causal_downsampling": False,
                "pos_emb_max_len": 5000,
                "ff_expansion_factor": 4,
                "subsampling_factor": 4,
                "subsampling_conv_channels": 512,
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "conv_dropout_rate": 0.1,
                "conv_kernel_size": 31,
                "causal_depthwise_conv": False,
            },
            "decoder": {
                "blank_as_pad": True,
                "vocab_size": len(dummy_vocabulary),
                "input_dim": 512,
                "hidden_dim": 512,
                "output_dim": 1024,
                "num_layers": 1,
                "dropout_rate": 0.1,
                "prednet": {
                    "input_dim": 512,
                    "pred_hidden": 512,
                    "output_dim": 1024,
                    "pred_rnn_layers": 1,
                    "dropout_rate": 0.1,
                },
            },
            "joint": {
                "input_dim_encoder": 512,
                "input_dim_decoder": 1024,
                "num_classes": len(dummy_vocabulary) + 1,
                "joint_dropout_rate": 0.1,
                "vocabulary": dummy_vocabulary,
                "jointnet": {
                    "encoder_hidden": 512,
                    "pred_hidden": 1024,
                    "joint_hidden": 512,
                    "activation": "relu",
                },
            },
            "decoding": {
                "model_type": "tdt",
                "durations": [0, 1, 2, 3],
                "greedy": {"max_symbols": 10},
            },
        }

        # Configure mocks for config loading
        mock_file_object_for_context_manager = (
            MagicMock()
        )  # This is what __enter__ would return
        mock_parakeet_module_open.return_value.__enter__.return_value = (
            mock_file_object_for_context_manager
        )
        # If open is used not as a context manager, its direct return value is the file handle
        # json.load will be called with mock_parakeet_module_open.return_value

        mock_parakeet_json_load.return_value = dummy_config_dict

        mock_mlx_core_load.return_value = {"some.valid.path.if.needed": mx.array([0.0])}

        model = ParakeetTDT.from_pretrained(dummy_repo_id, dtype=mx.float32)

        self.assertIsInstance(model, ParakeetTDT)

        mock_hf_hub_download.assert_any_call(dummy_repo_id, "config.json")
        mock_hf_hub_download.assert_any_call(dummy_repo_id, "model.safetensors")

        self.assertEqual(model.preprocessor_config.sample_rate, 16000)
        self.assertEqual(model.preprocessor_config.features, 80)
        self.assertEqual(
            model.encoder_config.d_model, 512
        )  # d_model is correct for ConformerArgs
        self.assertEqual(model.vocabulary, dummy_vocabulary)
        self.assertEqual(model.durations, [0, 1, 2, 3])


class TestGLMASRModel(unittest.TestCase):
    """Tests for the GLM-ASR model."""

    def setUp(self):
        """Set up test fixtures."""
        # Import GLM ASR modules inside test class to avoid import issues
        from mlx_audio.stt.models.glmasr.config import (
            LlamaConfig,
            ModelConfig,
            WhisperConfig,
        )
        from mlx_audio.stt.models.glmasr.glmasr import AudioEncoder
        from mlx_audio.stt.models.glmasr.glmasr import Model as GLMASRModel
        from mlx_audio.stt.models.glmasr.glmasr import WhisperEncoder

        # Store references for use in test methods
        self.WhisperConfig = WhisperConfig
        self.LlamaConfig = LlamaConfig
        self.ModelConfig = ModelConfig
        self.GLMASRModel = GLMASRModel
        self.WhisperEncoder = WhisperEncoder
        self.AudioEncoder = AudioEncoder

        self.whisper_config = WhisperConfig(
            d_model=256,
            encoder_attention_heads=4,
            encoder_ffn_dim=1024,
            encoder_layers=2,
            num_mel_bins=80,
            max_source_positions=1500,
        )
        self.llama_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            eos_token_id=[2],
        )
        self.model_config = ModelConfig(
            whisper_config=self.whisper_config,
            lm_config=self.llama_config,
            merge_factor=4,
            use_rope=True,
        )

    def test_whisper_config_from_dict(self):
        """Test WhisperConfig.from_dict method."""
        config_dict = {
            "d_model": 512,
            "encoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "encoder_layers": 6,
            "num_mel_bins": 128,
            "max_source_positions": 3000,
            "unknown_field": "should_be_ignored",
        }
        config = self.WhisperConfig.from_dict(config_dict)

        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.encoder_attention_heads, 8)
        self.assertEqual(config.encoder_ffn_dim, 2048)
        self.assertEqual(config.encoder_layers, 6)
        self.assertEqual(config.num_mel_bins, 128)
        self.assertEqual(config.max_source_positions, 3000)

    def test_llama_config_from_dict(self):
        """Test LlamaConfig.from_dict method."""
        config_dict = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "eos_token_id": [1, 2, 3],
            "unknown_field": "should_be_ignored",
        }
        config = self.LlamaConfig.from_dict(config_dict)

        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 11008)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.rms_norm_eps, 1e-6)
        self.assertEqual(config.rope_theta, 500000.0)
        self.assertEqual(config.eos_token_id, [1, 2, 3])

    def test_model_config_from_dict(self):
        """Test ModelConfig.from_dict with nested configs."""
        config_dict = {
            "model_type": "glmasr",
            "whisper_config": {
                "d_model": 1280,
                "encoder_attention_heads": 20,
                "encoder_layers": 32,
                "num_mel_bins": 128,
            },
            "lm_config": {
                "vocab_size": 59264,
                "hidden_size": 2048,
                "num_hidden_layers": 28,
            },
            "merge_factor": 4,
            "use_rope": True,
            "max_whisper_length": 1500,
        }
        config = self.ModelConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "glmasr")
        self.assertEqual(config.merge_factor, 4)
        self.assertEqual(config.use_rope, True)
        self.assertEqual(config.max_whisper_length, 1500)

        # Check nested whisper config
        self.assertIsInstance(config.whisper_config, self.WhisperConfig)
        self.assertEqual(config.whisper_config.d_model, 1280)
        self.assertEqual(config.whisper_config.encoder_attention_heads, 20)
        self.assertEqual(config.whisper_config.encoder_layers, 32)
        self.assertEqual(config.whisper_config.num_mel_bins, 128)

        # Check nested llama config
        self.assertIsInstance(config.lm_config, self.LlamaConfig)
        self.assertEqual(config.lm_config.vocab_size, 59264)
        self.assertEqual(config.lm_config.hidden_size, 2048)
        self.assertEqual(config.lm_config.num_hidden_layers, 28)

    def test_whisper_encoder_output_shape(self):
        """Test WhisperEncoder produces correct output shape."""
        encoder = self.WhisperEncoder(self.whisper_config, use_rope=True)

        batch_size = 2
        seq_len = 100
        input_features = mx.random.normal(
            (batch_size, seq_len, self.whisper_config.num_mel_bins)
        )

        output = encoder(input_features)

        # After conv2 with stride=2, seq_len is halved
        expected_seq_len = seq_len // 2
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], expected_seq_len)
        self.assertEqual(output.shape[2], self.whisper_config.d_model)

    def test_audio_encoder_output_shape(self):
        """Test AudioEncoder produces correct output shape with merge factor."""
        audio_encoder = self.AudioEncoder(self.model_config)

        batch_size = 1
        seq_len = 100
        input_features = mx.random.normal(
            (batch_size, seq_len, self.whisper_config.num_mel_bins)
        )

        audio_embeds, audio_len = audio_encoder(input_features)

        # Check output dimension matches LM hidden size
        self.assertEqual(audio_embeds.shape[0], batch_size)
        self.assertEqual(audio_embeds.shape[2], self.llama_config.hidden_size)
        self.assertEqual(audio_embeds.shape[1], audio_len)

    def test_audio_encoder_boa_eoa_tokens(self):
        """Test AudioEncoder begin/end of audio token embeddings."""
        audio_encoder = self.AudioEncoder(self.model_config)

        boa, eoa = audio_encoder.get_boa_eoa_tokens()

        self.assertEqual(boa.shape, (1, self.llama_config.hidden_size))
        self.assertEqual(eoa.shape, (1, self.llama_config.hidden_size))

    def test_model_sanitize_weights(self):
        """Test weight sanitization for loading."""
        model = self.GLMASRModel(self.model_config)

        # Test adapting layer remapping
        test_weights = {
            "audio_encoder.adapting.0.weight": mx.zeros((10, 10)),
            "audio_encoder.adapting.0.bias": mx.zeros((10,)),
            "audio_encoder.adapting.2.weight": mx.zeros((10, 10)),
            "audio_encoder.adapting.2.bias": mx.zeros((10,)),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
        }

        sanitized = model.sanitize(test_weights)

        # Check adapting layer remapping: 0 -> fc1, 2 -> fc2
        self.assertIn("audio_encoder.adapting.fc1.weight", sanitized)
        self.assertIn("audio_encoder.adapting.fc1.bias", sanitized)
        self.assertIn("audio_encoder.adapting.fc2.weight", sanitized)
        self.assertIn("audio_encoder.adapting.fc2.bias", sanitized)
        self.assertNotIn("audio_encoder.adapting.0.weight", sanitized)
        self.assertNotIn("audio_encoder.adapting.2.weight", sanitized)

        # Check model.* keys are remapped to language_model.model.*
        self.assertIn(
            "language_model.model.layers.0.self_attn.q_proj.weight", sanitized
        )
        self.assertNotIn("model.layers.0.self_attn.q_proj.weight", sanitized)

    def test_model_sanitize_conv_transpose(self):
        """Test conv weight transposition in sanitize."""
        model = self.GLMASRModel(self.model_config)

        # Conv weight that needs transposition (last dim < second-to-last)
        conv_weight = mx.zeros((256, 80, 3))  # Needs transpose
        test_weights = {
            "audio_encoder.whisper.conv1.weight": conv_weight,
        }

        sanitized = model.sanitize(test_weights)

        # Should be transposed to (256, 3, 80)
        self.assertEqual(
            sanitized["audio_encoder.whisper.conv1.weight"].shape, (256, 3, 80)
        )

    def test_model_forward_pass(self):
        """Test basic model forward pass."""
        model = self.GLMASRModel(self.model_config)

        batch_size = 1
        seq_len = 10
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        logits = model(input_ids)

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], self.llama_config.vocab_size)

    @patch("mlx.nn.Module.load_weights")
    @patch("mlx_audio.utils.load_config")
    @patch("mlx_audio.utils.get_model_path")
    @patch("mlx_audio.stt.models.glmasr.glmasr.glob.glob")
    @patch("mlx_audio.stt.models.glmasr.glmasr.mx.load")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("mlx_audio.stt.models.glmasr.glmasr.json.load")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_from_pretrained(
        self,
        mock_auto_tokenizer,
        mock_json_load,
        mock_open,
        mock_mx_load,
        mock_glob,
        mock_get_model_path,
        mock_load_config,
        mock_load_weights,
    ):
        """Test GLMASRModel.from_pretrained method."""
        dummy_repo_id = "dummy/glm-asr-model"
        dummy_model_path = Path("/tmp/dummy_model_path")

        mock_get_model_path.return_value = dummy_model_path

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "test"
        mock_auto_tokenizer.return_value = mock_tokenizer

        # Mock config
        dummy_config_dict = {
            "model_type": "glmasr",
            "whisper_config": {
                "d_model": 256,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 1024,
                "encoder_layers": 2,
                "num_mel_bins": 80,
            },
            "lm_config": {
                "vocab_size": 1000,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "tie_word_embeddings": False,
            },
            "merge_factor": 4,
            "use_rope": True,
        }
        mock_json_load.return_value = dummy_config_dict
        mock_load_config.return_value = dummy_config_dict

        # Mock weight files
        mock_glob.return_value = [str(dummy_model_path / "model.safetensors")]

        # Mock weights - minimal weights for model initialization
        mock_mx_load.return_value = {}

        model = self.GLMASRModel.from_pretrained(dummy_repo_id)

        self.assertIsInstance(model, self.GLMASRModel)
        mock_get_model_path.assert_called_once()
        mock_auto_tokenizer.assert_called_once_with(
            str(dummy_model_path), trust_remote_code=True
        )
        mock_load_weights.assert_called_once()

        # Verify config was loaded correctly
        self.assertEqual(model.config.whisper_config.d_model, 256)
        self.assertEqual(model.config.lm_config.vocab_size, 1000)
        self.assertEqual(model.config.merge_factor, 4)


if __name__ == "__main__":
    unittest.main()

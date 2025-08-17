import json
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT
from mlx_audio.stt.models.whisper.audio import (
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
)
from mlx_audio.stt.models.whisper.decoding import DecodingOptions, DecodingResult
from mlx_audio.stt.models.whisper.whisper import Model, ModelDimensions, STTOutput


class TestWhisperModel(unittest.TestCase):
    def setUp(self):
        self.dims = ModelDimensions(
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
        self.model_mock = MagicMock(spec=Model, name="MockModelInstance")

        self.model_mock.dims = self.dims
        self.model_mock.dtype = mx.float32

        type(self.model_mock).is_multilingual = PropertyMock(return_value=False)
        type(self.model_mock).num_languages = PropertyMock(return_value=0)

    @patch("mlx_audio.stt.models.whisper.whisper.Path")
    @patch("mlx_audio.stt.models.whisper.whisper.snapshot_download")
    @patch("mlx_audio.stt.models.whisper.whisper.mx.load")
    @patch("mlx_audio.stt.models.whisper.whisper.json.loads")
    @patch("builtins.open", new_callable=MagicMock)
    def test_from_pretrained(
        self,
        mock_open,
        mock_json_loads_in_whisper,
        mock_mx_load,
        mock_snapshot_download,
        mock_pathlib_path,
    ):

        mock_snapshot_download.return_value = "dummy_path"

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

        model_instance = Model.from_pretrained(
            path_or_hf_repo="mlx-community/whisper-tiny", dtype=mx.float32
        )

        self.assertIsInstance(model_instance, Model)
        self.assertEqual(model_instance.dims.n_mels, dummy_config["n_mels"])
        mock_snapshot_download.assert_called_once_with(
            repo_id="mlx-community/whisper-tiny"
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

        mock_mel_data = mx.zeros((N_FRAMES + 100, self.dims.n_mels), dtype=mx.float32)
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
        mock_decoding_result = DecodingResult(
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

        dummy_audio_input = np.zeros(SAMPLE_RATE * 1, dtype=np.float32)

        real_model_for_test = Model(self.dims, dtype=mx.float32)

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
                args_decode_call[0].shape, (N_FRAMES, self.dims.n_mels)
            )  # mel_segment
            self.assertIsInstance(args_decode_call[1], DecodingOptions)
            self.assertEqual(args_decode_call[1].language, "en")
            self.assertEqual(args_decode_call[1].fp16, False)

        self.assertIsInstance(output, STTOutput)
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
            ANY, n_mels=self.dims.n_mels, padding=N_SAMPLES
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
        self.assertEqual(args_pad_call[1], N_FRAMES)


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


if __name__ == "__main__":
    unittest.main()

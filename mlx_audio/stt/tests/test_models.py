import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write


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
    @patch("mlx_audio.stt.models.whisper.whisper.open", new_callable=MagicMock)
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
        mock_open.assert_any_call("dummy_path/config.json", "r")
        self.assertGreaterEqual(mock_open.call_count, 1)
        mock_mx_load.assert_called_once_with("dummy_path/weights.safetensors")

    @patch("mlx_audio.stt.models.whisper.whisper.pad_or_trim")
    @patch("mlx_audio.stt.models.whisper.whisper.tqdm.tqdm")
    @patch("mlx_audio.stt.models.whisper.whisper.log_mel_spectrogram")
    def test_generate_simple_case(
        self,
        mock_log_mel,
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

        # Patch the model's get_tokenizer method and decode method
        with (
            patch.object(
                real_model_for_test, "get_tokenizer", return_value=mock_tokenizer_inst
            ) as mock_get_tokenizer,
            patch.object(
                real_model_for_test, "decode", return_value=mock_decoding_result
            ) as mock_instance_decode,
        ):
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

            mock_get_tokenizer.assert_called_once_with(
                language="en",
                task="transcribe",
            )

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
        mock_pad_or_trim.assert_called_once()
        args_pad_call, _ = mock_pad_or_trim.call_args
        self.assertEqual(args_pad_call[0].shape, (100, self.dims.n_mels))
        self.assertEqual(args_pad_call[1], self.N_FRAMES)


def _cohere_small_config_dict() -> dict[str, Any]:
    return {
        "model_type": "cohere_asr",
        "vocab_size": 64,
        "supported_languages": ["en", "ja"],
        "encoder": {
            "feat_in": 16,
            "feat_out": -1,
            "n_layers": 1,
            "d_model": 32,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "conv_kernel_size": 3,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 8,
            "pos_emb_max_len": 128,
        },
        "head": {"hidden_size": 24, "num_classes": 64, "log_softmax": True},
        "transf_decoder": {
            "config_dict": {
                "hidden_size": 24,
                "inner_size": 48,
                "num_attention_heads": 4,
                "num_layers": 1,
                "max_sequence_length": 128,
            }
        },
        "preprocessor": {
            "sample_rate": 16000,
            "features": 16,
            "n_fft": 512,
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "dither": 1e-5,
            "pad_to": 0,
            "pad_value": 0.0,
            "preemph": 0.97,
            "log": True,
        },
    }


def _cohere_small_config():
    from mlx_audio.stt.models.cohere_asr.config import ModelConfig

    return ModelConfig.from_dict(_cohere_small_config_dict())


def _write_cohere_tokenizer_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    import sentencepiece as spm

    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "hello world\nif not there will be a big crisis\neuropean parliament\n",
        encoding="utf-8",
    )

    special_tokens = [
        "<pad>",
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|startofcontext|>",
        "<|emo:undefined|>",
        "<|en|>",
        "<|ja|>",
        "<|pnc|>",
        "<|nopnc|>",
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
    ]

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(tmp_path / "toy"),
        vocab_size=64,
        model_type="bpe",
        user_defined_symbols=",".join(special_tokens),
        unk_piece="<unk>",
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        hard_vocab_limit=False,
    )

    tokenizer_config = {
        "bos_token": "<|startoftranscript|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "additional_special_tokens": [
            "<|startofcontext|>",
            "<|emo:undefined|>",
            "<|en|>",
            "<|ja|>",
            "<|pnc|>",
            "<|nopnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ],
    }
    special_tokens_map = {
        "bos_token": "<|startoftranscript|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "additional_special_tokens": tokenizer_config["additional_special_tokens"],
    }

    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    special_tokens_map_path = tmp_path / "special_tokens_map.json"
    tokenizer_config_path.write_text(json.dumps(tokenizer_config), encoding="utf-8")
    special_tokens_map_path.write_text(json.dumps(special_tokens_map), encoding="utf-8")

    return (
        tmp_path / "toy.model",
        tokenizer_config_path,
        special_tokens_map_path,
    )


@unittest.skipUnless(
    importlib.util.find_spec("sentencepiece") is not None,
    "sentencepiece is required for quantized Cohere tests",
)
class TestCohereQuantizedModel(unittest.TestCase):
    @staticmethod
    def _small_config():
        return _cohere_small_config()

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._workdirs = []
        tmp_path = Path(cls._tmpdir.name)
        (
            cls.tokenizer_model_path,
            cls.tokenizer_config_path,
            cls.special_tokens_map_path,
        ) = _write_cohere_tokenizer_files(tmp_path)
        cls.audio_fixture_path = tmp_path / "toy.wav"

        sample_rate = 16_000
        duration_seconds = 1.0
        time_axis = np.linspace(
            0.0,
            duration_seconds,
            int(sample_rate * duration_seconds),
            endpoint=False,
            dtype=np.float32,
        )
        waveform = (
            0.25 * np.sin(2 * np.pi * 220.0 * time_axis)
            + 0.15 * np.sin(2 * np.pi * 440.0 * time_axis)
        ).astype(np.float32)
        audio_write(str(cls.audio_fixture_path), waveform, sample_rate, format="wav")

    @classmethod
    def tearDownClass(cls):
        for workdir in cls._workdirs:
            workdir.cleanup()
        cls._tmpdir.cleanup()

    def _small_config_dict(self):
        return _cohere_small_config_dict()

    def _build_quantized_checkpoint(self, bits: int) -> Path:
        from mlx_lm.utils import save_model

        from mlx_audio.convert import convert
        from mlx_audio.stt.models.cohere_asr.cohere_asr import Model, STTOutput

        self.STTOutput = STTOutput
        workdir = tempfile.TemporaryDirectory(prefix=f"cohere-quant-{bits}-")
        self._workdirs.append(workdir)
        workdir_path = Path(workdir.name)
        source_dir = workdir_path / "source"
        output_dir = workdir_path / "output"
        source_dir.mkdir(parents=True, exist_ok=True)

        config_dict = self._small_config_dict()
        config = self._small_config()
        model = Model(config)

        save_model(source_dir, model, donate_model=True)
        (source_dir / "config.json").write_text(
            json.dumps(config_dict), encoding="utf-8"
        )
        for file_path in [
            self.tokenizer_model_path,
            self.tokenizer_config_path,
            self.special_tokens_map_path,
        ]:
            target_name = (
                "tokenizer.model"
                if file_path == self.tokenizer_model_path
                else file_path.name
            )
            (source_dir / target_name).write_bytes(file_path.read_bytes())

        convert(
            hf_path=str(source_dir),
            mlx_path=str(output_dir),
            quantize=True,
            q_group_size=64,
            q_bits=bits,
            model_domain="stt",
        )

        return output_dir

    def _assert_quantized_generation(self, bits: int):
        from mlx_audio.stt import load

        checkpoint_dir = self._build_quantized_checkpoint(bits)
        model = cast(Any, load(str(checkpoint_dir)))
        output = model.generate(
            str(self.audio_fixture_path),
            language="en",
            punctuation=True,
            max_tokens=8,
        )

        self.assertIsInstance(output, self.STTOutput)
        self.assertIsInstance(output.text, str)

    def test_quantized_8bit_generate(self):
        self._assert_quantized_generation(8)

    def test_quantized_4bit_generate(self):
        self._assert_quantized_generation(4)


@unittest.skipUnless(
    importlib.util.find_spec("sentencepiece") is not None,
    "sentencepiece is required for Cohere tokenizer tests",
)
class TestCohereTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)
        (
            cls.model_path,
            cls.tokenizer_config_path,
            cls.special_tokens_map_path,
        ) = _write_cohere_tokenizer_files(tmp_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_prompt_tokens(self):
        from mlx_audio.stt.models.cohere_asr.tokenizer import CohereAsrTokenizer

        tokenizer = CohereAsrTokenizer(
            str(self.model_path),
            str(self.tokenizer_config_path),
            str(self.special_tokens_map_path),
        )
        tokens = tokenizer.build_prompt_tokens("en", punctuation=True)
        self.assertEqual(len(tokens), 9)
        sp = cast(Any, tokenizer.sp)
        self.assertEqual(tokens[0], sp.piece_to_id("<|startofcontext|>"))
        self.assertEqual(tokens[1], tokenizer.bos_token_id)
        self.assertEqual(tokens[-1], sp.piece_to_id("<|nodiarize|>"))


class TestCohereAudioFrontend(unittest.TestCase):
    def test_extract_features_shape_and_lengths(self):
        from mlx_audio.stt.models.cohere_asr.audio import CohereAudioFrontend
        from mlx_audio.stt.models.cohere_asr.config import PreprocessorConfig

        frontend = CohereAudioFrontend(
            PreprocessorConfig(sample_rate=16000, features=16, n_fft=512)
        )
        frontend.fb = mx.random.normal((16, 257))

        waveform = np.random.randn(16000).astype(np.float32)
        features, lengths = frontend([waveform])
        mx.eval(features, lengths)

        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], 16)
        self.assertEqual(lengths.tolist(), [100])


class TestCohereChunking(unittest.TestCase):
    def test_energy_chunking_splits_long_audio(self):
        from mlx_audio.stt.models.cohere_asr.cohere_asr import split_audio_chunks_energy

        waveform = np.ones(16_000 * 8, dtype=np.float32)
        waveform[16_000 * 3 : 16_000 * 3 + 800] = 0.0
        chunks = split_audio_chunks_energy(
            waveform=waveform,
            sample_rate=16_000,
            max_audio_clip_s=4.0,
            overlap_chunk_second=1.0,
            min_energy_window_samples=400,
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], waveform.shape[0])

    def test_join_chunk_texts_language_separator(self):
        from mlx_audio.stt.models.cohere_asr.cohere_asr import join_chunk_texts

        self.assertEqual(
            join_chunk_texts(["hello", "world"], language="en"), "hello world"
        )
        self.assertEqual(join_chunk_texts(["你", "好"], language="zh"), "你好")


class TestCohereSanitize(unittest.TestCase):
    def setUp(self):
        from mlx_audio.stt.models.cohere_asr.cohere_asr import Model

        self.model = Model(_cohere_small_config())

    def test_decoder_key_mapping(self):
        weights = {
            "transf_decoder._embedding.token_embedding.weight": mx.zeros((64, 24))
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("transf_decoder.embedding.token_embedding.weight", sanitized)

    def test_conv_transpose(self):
        weights = {"encoder.pre_encode.conv.0.weight": mx.zeros((8, 1, 3, 3))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(
            sanitized["encoder.pre_encode.conv.0.weight"].shape,
            (8, 3, 3, 1),
        )

    def test_drops_frontend_weights(self):
        weights = {"preprocessor.featurizer.fb": mx.zeros((1, 16, 257))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(sanitized, {})


class TestCohereConfig(unittest.TestCase):
    def test_defaults_from_reference_shape(self):
        config = _cohere_small_config()
        self.assertEqual(config.model_type, "cohere_asr")
        self.assertEqual(config.min_energy_window_samples, 1600)
        self.assertEqual(config.preprocessor.win_length, 400)
        self.assertEqual(config.preprocessor.hop_length, 160)


class TestParakeetModel(unittest.TestCase):
    def _build_parakeet_base_model(self):
        from mlx_audio.stt.models.parakeet.parakeet import Model, PreprocessArgs

        return Model(
            PreprocessArgs(
                sample_rate=16000,
                normalize="per_feature",
                window_size=0.02,
                window_stride=0.01,
                window="hann",
                features=80,
                n_fft=512,
                dither=1e-5,
            )
        )

    def test_generate_stream_uses_streaming_defaults_when_omitted(self):
        model = self._build_parakeet_base_model()
        audio = mx.zeros((16000,))
        model.stream_generate = MagicMock(return_value=iter(()))

        model.generate(audio, stream=True)

        model.stream_generate.assert_called_once_with(
            audio,
            dtype=mx.bfloat16,
            chunk_duration=5.0,
            overlap_duration=1.0,
            verbose=False,
        )

    def test_generate_chunked_raises_when_overlap_is_not_smaller_than_chunk(self):
        model = self._build_parakeet_base_model()

        with self.assertRaisesRegex(ValueError, "must be less than"):
            model.generate(
                mx.zeros((16000 * 10,)),
                chunk_duration=5.0,
                overlap_duration=5.0,
            )

    def test_log_mel_spectrogram_shape_and_params(self):
        """Verify log_mel_spectrogram output shape and NeMo-aligned parameters."""
        from mlx_audio.stt.models.parakeet.audio import (
            PreprocessArgs,
            log_mel_spectrogram,
        )

        args = PreprocessArgs(
            sample_rate=16000,
            normalize="per_feature",
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            features=80,
            n_fft=512,
            dither=0.0,
        )

        duration_s = 0.5
        audio = mx.random.normal((int(16000 * duration_s),))
        mel = log_mel_spectrogram(audio, args)

        # Shape: [1, time_frames, n_mels]
        self.assertEqual(mel.ndim, 3)
        self.assertEqual(mel.shape[0], 1)
        self.assertEqual(mel.shape[2], 80)
        self.assertGreater(mel.shape[1], 0)

        # Output should be normalized (mean ≈ 0 per feature)
        per_feat_mean = np.abs(np.array(mx.mean(mel, axis=1)))
        self.assertTrue(np.all(per_feat_mean < 1.0))

        # Verify configurable log_zero_guard_value default
        self.assertAlmostEqual(args.log_zero_guard_value, 2**-24, places=15)

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

    @patch("mlx_audio.stt.utils.load")
    def test_from_pretrained(
        self,
        mock_stt_load,
    ):
        """Test GLMASRModel.from_pretrained method."""
        dummy_repo_id = "dummy/glm-asr-model"
        loaded_model = MagicMock(spec=self.GLMASRModel)
        mock_stt_load.return_value = loaded_model

        model = self.GLMASRModel.from_pretrained(dummy_repo_id)

        self.assertIs(model, loaded_model)
        mock_stt_load.assert_called_once_with(dummy_repo_id)


class TestQwen3ASRConfig(unittest.TestCase):
    """Tests for Qwen3-ASR configuration classes."""

    def setUp(self):
        from mlx_audio.stt.models.qwen3_asr.config import (
            AudioEncoderConfig,
            ModelConfig,
            TextConfig,
        )
        from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import (
            ForcedAlignerConfig,
        )

        self.AudioEncoderConfig = AudioEncoderConfig
        self.TextConfig = TextConfig
        self.ModelConfig = ModelConfig
        self.ForcedAlignerConfig = ForcedAlignerConfig

    def test_audio_encoder_config_default_values(self):
        config = self.AudioEncoderConfig()
        self.assertEqual(config.num_mel_bins, 128)
        self.assertEqual(config.encoder_layers, 24)
        self.assertEqual(config.d_model, 1024)

    def test_audio_encoder_config_from_dict(self):
        config_dict = {
            "num_mel_bins": 80,
            "encoder_layers": 12,
            "d_model": 512,
            "unknown_key": "should_be_ignored",
        }
        config = self.AudioEncoderConfig.from_dict(config_dict)

        self.assertEqual(config.num_mel_bins, 80)
        self.assertEqual(config.encoder_layers, 12)
        self.assertEqual(config.d_model, 512)

    def test_text_config_default_values(self):
        config = self.TextConfig()
        self.assertEqual(config.vocab_size, 151936)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_hidden_layers, 28)

    def test_text_config_from_dict(self):
        config_dict = {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "extra_field": True,
        }
        config = self.TextConfig.from_dict(config_dict)

        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_hidden_layers, 12)

    def test_model_config_default_nested_configs(self):
        config = self.ModelConfig()
        self.assertIsInstance(config.audio_config, self.AudioEncoderConfig)
        self.assertIsInstance(config.text_config, self.TextConfig)

    def test_model_config_from_dict_with_nested(self):
        config_dict = {
            "model_type": "qwen3_asr",
            "audio_config": {
                "num_mel_bins": 80,
                "encoder_layers": 6,
            },
            "text_config": {
                "vocab_size": 50000,
                "hidden_size": 512,
            },
            "audio_token_id": 12345,
        }
        config = self.ModelConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "qwen3_asr")
        self.assertEqual(config.audio_token_id, 12345)
        self.assertIsInstance(config.audio_config, self.AudioEncoderConfig)
        self.assertEqual(config.audio_config.num_mel_bins, 80)
        self.assertIsInstance(config.text_config, self.TextConfig)
        self.assertEqual(config.text_config.vocab_size, 50000)

    def test_model_config_from_dict_with_thinker_config(self):
        """Test parsing HuggingFace-style config with thinker_config."""
        config_dict = {
            "thinker_config": {
                "model_type": "qwen3_asr",
                "audio_config": {"num_mel_bins": 128},
                "text_config": {"vocab_size": 151936},
                "audio_token_id": 151676,
            }
        }
        config = self.ModelConfig.from_dict(config_dict)

        self.assertIsInstance(config.audio_config, self.AudioEncoderConfig)
        self.assertEqual(config.audio_config.num_mel_bins, 128)
        self.assertEqual(config.audio_token_id, 151676)

    def test_model_config_from_dict_detects_forced_aligner(self):
        """Test that from_dict returns ForcedAlignerConfig for aligner models."""
        config_dict = {
            "thinker_config": {
                "model_type": "qwen3_forced_aligner",
                "audio_config": {"num_mel_bins": 128},
                "text_config": {"vocab_size": 151936},
            }
        }
        config = self.ModelConfig.from_dict(config_dict)

        self.assertIsInstance(config, self.ForcedAlignerConfig)
        self.assertEqual(config.model_type, "qwen3_forced_aligner")

    def test_forced_aligner_config_default_values(self):
        config = self.ForcedAlignerConfig()
        self.assertEqual(config.model_type, "qwen3_forced_aligner")
        self.assertEqual(config.timestamp_token_id, 151705)
        self.assertEqual(config.timestamp_segment_time, 80.0)
        self.assertEqual(config.classify_num, 5000)

    def test_forced_aligner_config_from_dict(self):
        config_dict = {
            "thinker_config": {
                "model_type": "qwen3_forced_aligner",
                "audio_config": {"num_mel_bins": 80},
                "text_config": {"vocab_size": 50000},
                "classify_num": 3000,
            },
            "timestamp_token_id": 12345,
            "timestamp_segment_time": 100.0,
        }
        config = self.ForcedAlignerConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "qwen3_forced_aligner")
        self.assertEqual(config.timestamp_token_id, 12345)
        self.assertEqual(config.timestamp_segment_time, 100.0)
        self.assertEqual(config.classify_num, 3000)


class TestQwen3ASRForceAlignProcessor(unittest.TestCase):
    """Tests for ForceAlignProcessor."""

    def setUp(self):
        from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import (
            ForceAlignProcessor,
        )

        self.processor = ForceAlignProcessor()

    def test_is_kept_char(self):
        self.assertTrue(self.processor.is_kept_char("a"))
        self.assertTrue(self.processor.is_kept_char("Z"))
        self.assertTrue(self.processor.is_kept_char("5"))
        self.assertTrue(self.processor.is_kept_char("'"))
        self.assertFalse(self.processor.is_kept_char(" "))
        self.assertFalse(self.processor.is_kept_char(","))
        self.assertFalse(self.processor.is_kept_char("."))

    def test_clean_token(self):
        self.assertEqual(self.processor.clean_token("hello"), "hello")
        self.assertEqual(self.processor.clean_token("hello!"), "hello")
        self.assertEqual(self.processor.clean_token("it's"), "it's")
        self.assertEqual(self.processor.clean_token("..."), "")

    def test_is_cjk_char(self):
        self.assertTrue(self.processor.is_cjk_char("中"))
        self.assertTrue(self.processor.is_cjk_char("日"))
        self.assertFalse(self.processor.is_cjk_char("a"))
        self.assertFalse(self.processor.is_cjk_char("5"))

    def test_tokenize_chinese_mixed(self):
        tokens = self.processor.tokenize_chinese_mixed("Hello中文World")
        self.assertEqual(tokens, ["Hello", "中", "文", "World"])

    def test_tokenize_space_lang(self):
        tokens = self.processor.tokenize_space_lang("Hello, World!")
        self.assertEqual(tokens, ["Hello", "World"])

        tokens = self.processor.tokenize_space_lang("I have a dream")
        self.assertEqual(tokens, ["I", "have", "a", "dream"])

    def test_encode_timestamp_english(self):
        word_list, input_text = self.processor.encode_timestamp(
            "Hello world", "English"
        )

        self.assertEqual(word_list, ["Hello", "world"])
        self.assertIn("<timestamp>", input_text)
        self.assertIn("<|audio_start|>", input_text)
        self.assertIn("<|audio_end|>", input_text)

    def test_encode_timestamp_chinese(self):
        word_list, input_text = self.processor.encode_timestamp("你好", "Chinese")

        self.assertEqual(word_list, ["你", "好"])
        self.assertIn("<timestamp>", input_text)

    def test_fix_timestamp_monotonic(self):
        """Test that already monotonic timestamps are unchanged."""
        data = np.array([100, 200, 300, 400])
        fixed = self.processor.fix_timestamp(data)
        self.assertEqual(fixed, [100, 200, 300, 400])

    def test_fix_timestamp_non_monotonic(self):
        """Test fixing non-monotonic timestamps."""
        data = np.array([100, 200, 150, 400])  # 150 breaks monotonicity
        fixed = self.processor.fix_timestamp(data)
        # Should fix the anomaly
        self.assertLessEqual(fixed[0], fixed[1])
        self.assertLessEqual(fixed[1], fixed[2])
        self.assertLessEqual(fixed[2], fixed[3])

    def test_fix_timestamp_empty(self):
        data = np.array([])
        fixed = self.processor.fix_timestamp(data)
        self.assertEqual(fixed, [])

    def test_parse_timestamp(self):
        word_list = ["Hello", "world"]
        # 4 timestamps: start1, end1, start2, end2
        timestamp = np.array([0, 500, 500, 1000])

        result = self.processor.parse_timestamp(word_list, timestamp)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Hello")
        self.assertEqual(result[0]["start_time"], 0)
        self.assertEqual(result[0]["end_time"], 500)
        self.assertEqual(result[1]["text"], "world")
        self.assertEqual(result[1]["start_time"], 500)
        self.assertEqual(result[1]["end_time"], 1000)


class TestQwen3ASRForcedAlignResult(unittest.TestCase):
    """Tests for ForcedAlignResult and ForcedAlignItem."""

    def setUp(self):
        from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import (
            ForcedAlignItem,
            ForcedAlignResult,
        )

        self.ForcedAlignItem = ForcedAlignItem
        self.ForcedAlignResult = ForcedAlignResult

    def test_forced_align_item(self):
        item = self.ForcedAlignItem(text="hello", start_time=0.5, end_time=1.0)
        self.assertEqual(item.text, "hello")
        self.assertEqual(item.start_time, 0.5)
        self.assertEqual(item.end_time, 1.0)

    def test_forced_align_result_text_property(self):
        items = [
            self.ForcedAlignItem(text="Hello", start_time=0.0, end_time=0.5),
            self.ForcedAlignItem(text="world", start_time=0.5, end_time=1.0),
        ]
        result = self.ForcedAlignResult(items=items)

        self.assertEqual(result.text, "Hello world")

    def test_forced_align_result_segments_property(self):
        items = [
            self.ForcedAlignItem(text="Hello", start_time=0.0, end_time=0.5),
            self.ForcedAlignItem(text="world", start_time=0.5, end_time=1.0),
        ]
        result = self.ForcedAlignResult(items=items)

        segments = result.segments
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "Hello")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 0.5)

    def test_forced_align_result_iteration(self):
        items = [
            self.ForcedAlignItem(text="a", start_time=0.0, end_time=0.1),
            self.ForcedAlignItem(text="b", start_time=0.1, end_time=0.2),
        ]
        result = self.ForcedAlignResult(items=items)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "a")
        self.assertEqual(result[1].text, "b")

        texts = [item.text for item in result]
        self.assertEqual(texts, ["a", "b"])


class TestQwen3ASRModel(unittest.TestCase):
    """Tests for Qwen3-ASR model components."""

    def setUp(self):
        from mlx_audio.stt.models.qwen3_asr.config import (
            AudioEncoderConfig,
            ModelConfig,
            TextConfig,
        )
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
            AudioAttention,
            AudioEncoder,
            AudioEncoderLayer,
            Model,
            Qwen3ASRModel,
            SinusoidalPositionEmbedding,
            TextModel,
            _get_feat_extract_output_lengths,
            create_additive_causal_mask,
            split_audio_into_chunks,
        )
        from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import (
            ForcedAlignerConfig,
            ForcedAlignerModel,
        )

        self.AudioEncoderConfig = AudioEncoderConfig
        self.TextConfig = TextConfig
        self.ModelConfig = ModelConfig
        self.ForcedAlignerConfig = ForcedAlignerConfig
        self.AudioAttention = AudioAttention
        self.AudioEncoderLayer = AudioEncoderLayer
        self.AudioEncoder = AudioEncoder
        self.TextModel = TextModel
        self.SinusoidalPositionEmbedding = SinusoidalPositionEmbedding
        self.create_additive_causal_mask = create_additive_causal_mask
        self._get_feat_extract_output_lengths = _get_feat_extract_output_lengths
        self.split_audio_into_chunks = split_audio_into_chunks
        self.Qwen3ASRModel = Qwen3ASRModel
        self.ForcedAlignerModel = ForcedAlignerModel
        self.Model = Model

        # Small configs for fast testing
        self.audio_config = AudioEncoderConfig(
            num_mel_bins=80,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
            d_model=64,
            max_source_positions=100,
            n_window=10,
            output_dim=128,
            n_window_infer=80,
            conv_chunksize=50,
            downsample_hidden_size=32,
        )
        self.text_config = TextConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )
        self.model_config = ModelConfig(
            audio_config=self.audio_config,
            text_config=self.text_config,
            audio_token_id=151676,
        )
        self.aligner_config = ForcedAlignerConfig(
            audio_config=self.audio_config,
            text_config=self.text_config,
            audio_token_id=151676,
            timestamp_token_id=151705,
            timestamp_segment_time=80.0,
            classify_num=500,
        )

    def test_sinusoidal_position_embedding_output_shape(self):
        emb = self.SinusoidalPositionEmbedding(length=100, channels=64)
        output = emb(50)
        mx.eval(output)

        self.assertEqual(output.shape, (50, 64))

    def test_sinusoidal_position_embedding_even_channels_required(self):
        with self.assertRaises(ValueError):
            self.SinusoidalPositionEmbedding(length=100, channels=63)

    def test_create_additive_causal_mask_shape(self):
        mask = self.create_additive_causal_mask(10)
        mx.eval(mask)
        self.assertEqual(mask.shape, (10, 10))

    def test_create_additive_causal_mask_is_causal(self):
        mask = self.create_additive_causal_mask(4)
        mx.eval(mask)
        mask_np = np.array(mask)

        # Upper triangle should be large negative (masked)
        # Diagonal and below should be 0 (not masked)
        self.assertEqual(mask_np[0, 0], 0)
        self.assertLess(mask_np[0, 1], -1e8)
        self.assertEqual(mask_np[1, 1], 0)

    def test_get_feat_extract_output_lengths(self):
        input_lengths = mx.array([100, 200, 300])
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
        mx.eval(output_lengths)

        # Output should be shorter than input due to conv downsampling
        self.assertEqual(output_lengths.shape, (3,))
        for i in range(3):
            self.assertLess(int(output_lengths[i]), int(input_lengths[i]))

    def test_split_audio_into_chunks_single(self):
        sr = 16000
        wav = np.zeros(sr * 10)  # 10 seconds
        chunks = self.split_audio_into_chunks(wav, sr, chunk_duration=30.0)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1], 0.0)  # offset

    def test_split_audio_into_chunks_multiple(self):
        sr = 16000
        wav = np.zeros(sr * 100)  # 100 seconds
        chunks = self.split_audio_into_chunks(wav, sr, chunk_duration=30.0)

        self.assertGreater(len(chunks), 1)

        # Verify offsets are increasing
        offsets = [c[1] for c in chunks]
        for i in range(1, len(offsets)):
            self.assertGreater(offsets[i], offsets[i - 1])

    def test_audio_attention_output_shape(self):
        attn = self.AudioAttention(self.audio_config)

        x = mx.random.normal((1, 20, self.audio_config.d_model))
        output = attn(x)
        mx.eval(output)

        self.assertEqual(output.shape, x.shape)

    def test_audio_encoder_layer_output_shape(self):
        layer = self.AudioEncoderLayer(self.audio_config)

        x = mx.random.normal((1, 20, self.audio_config.d_model))
        output = layer(x)
        mx.eval(output)

        self.assertEqual(output.shape, x.shape)

    def test_text_model_forward_with_input_ids(self):
        model = self.TextModel(self.text_config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids=input_ids)
        mx.eval(output)

        self.assertEqual(output.shape, (1, 5, self.text_config.hidden_size))

    def test_text_model_forward_with_embeddings(self):
        model = self.TextModel(self.text_config)

        inputs_embeds = mx.random.normal((1, 5, self.text_config.hidden_size))
        output = model(inputs_embeds=inputs_embeds)
        mx.eval(output)

        self.assertEqual(output.shape, (1, 5, self.text_config.hidden_size))

    def test_qwen3_asr_model_init(self):
        model = self.Qwen3ASRModel(self.model_config)

        self.assertIsInstance(model.audio_tower, self.AudioEncoder)
        self.assertIsInstance(model.model, self.TextModel)

    def test_qwen3_asr_model_forward_logits_shape(self):
        model = self.Qwen3ASRModel(self.model_config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids)
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 5, self.text_config.vocab_size))

    def test_qwen3_asr_model_sample_rate(self):
        model = self.Qwen3ASRModel(self.model_config)
        self.assertEqual(model.sample_rate, 16000)

    def test_qwen3_asr_model_sanitize_removes_thinker_prefix(self):
        weights = {
            "thinker.audio_tower.conv2d1.weight": mx.zeros((32, 3, 3, 1)),
            "thinker.model.embed_tokens.weight": mx.zeros((1000, 64)),
        }
        sanitized = self.Qwen3ASRModel.sanitize(weights)

        self.assertIn("audio_tower.conv2d1.weight", sanitized)
        self.assertIn("model.embed_tokens.weight", sanitized)
        self.assertNotIn("thinker.audio_tower.conv2d1.weight", sanitized)

    def test_qwen3_asr_model_sanitize_transposes_conv_weights(self):
        # PyTorch format: [out, in, h, w] -> MLX format: [out, h, w, in]
        weights = {
            "thinker.audio_tower.conv2d1.weight": mx.zeros((32, 1, 3, 3)),
        }
        sanitized = self.Qwen3ASRModel.sanitize(weights)

        self.assertEqual(sanitized["audio_tower.conv2d1.weight"].shape, (32, 3, 3, 1))

    def test_qwen3_asr_model_sanitize_skips_lm_head(self):
        weights = {
            "lm_head.weight": mx.zeros((1000, 64)),
            "thinker.model.norm.weight": mx.zeros((64,)),
        }
        sanitized = self.Qwen3ASRModel.sanitize(weights)

        self.assertNotIn("lm_head.weight", sanitized)
        self.assertIn("model.norm.weight", sanitized)

    def test_qwen3_asr_model_quant_predicate(self):
        model = self.Qwen3ASRModel(self.model_config)

        # Audio tower should not be quantized
        self.assertFalse(model.model_quant_predicate("audio_tower.conv2d1", None))
        # Text model should be quantized
        self.assertTrue(model.model_quant_predicate("model.layers.0.self_attn", None))

    def test_forced_aligner_model_init(self):
        model = self.ForcedAlignerModel(self.aligner_config)

        self.assertIsInstance(model.audio_tower, self.AudioEncoder)
        self.assertIsInstance(model.model, self.TextModel)
        self.assertIsNotNone(model.lm_head)

    def test_forced_aligner_model_lm_head_output_size(self):
        """Test that lm_head outputs classify_num, not vocab_size."""
        model = self.ForcedAlignerModel(self.aligner_config)

        # lm_head should output classify_num classes
        self.assertEqual(
            model.lm_head.weight.shape[0], self.aligner_config.classify_num
        )

    def test_forced_aligner_model_forward_logits_shape(self):
        model = self.ForcedAlignerModel(self.aligner_config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids)
        mx.eval(logits)

        # Output should be [batch, seq_len, classify_num]
        self.assertEqual(logits.shape, (1, 5, self.aligner_config.classify_num))

    def test_forced_aligner_model_sanitize_keeps_lm_head(self):
        """Test that ForcedAligner sanitize keeps lm_head (unlike ASR)."""
        weights = {
            "thinker.lm_head.weight": mx.zeros((500, 64)),
            "thinker.model.norm.weight": mx.zeros((64,)),
        }
        sanitized = self.ForcedAlignerModel.sanitize(weights)

        self.assertIn("lm_head.weight", sanitized)
        self.assertIn("model.norm.weight", sanitized)

    def test_model_wrapper_creates_asr_model_for_asr_config(self):
        model = self.Model(self.model_config)

        self.assertIsInstance(model._model, self.Qwen3ASRModel)

    def test_model_wrapper_creates_aligner_model_for_aligner_config(self):
        model = self.Model(self.aligner_config)

        self.assertIsInstance(model._model, self.ForcedAlignerModel)

    def test_model_wrapper_delegation(self):
        model = self.Model(self.model_config)

        # Test that attributes are delegated
        self.assertEqual(model.sample_rate, 16000)
        self.assertEqual(len(model.layers), self.text_config.num_hidden_layers)

    def test_model_wrapper_call_delegation(self):
        model = self.Model(self.model_config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids)
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 5, self.text_config.vocab_size))

    def test_model_wrapper_is_forced_aligner_weights(self):
        # Small lm_head output (with thinker prefix) = ForcedAligner
        self.assertTrue(
            self.Model._is_forced_aligner_weights(
                {
                    "thinker.lm_head.weight": mx.zeros((5000, 64)),
                }
            )
        )

        # Small lm_head output (without prefix, converted models) = ForcedAligner
        self.assertTrue(
            self.Model._is_forced_aligner_weights(
                {
                    "lm_head.weight": mx.zeros((5000, 64)),
                }
            )
        )

        # Large lm_head output = ASR (vocab_size)
        self.assertFalse(
            self.Model._is_forced_aligner_weights(
                {
                    "thinker.lm_head.weight": mx.zeros((151936, 64)),
                }
            )
        )
        self.assertFalse(
            self.Model._is_forced_aligner_weights(
                {
                    "lm_head.weight": mx.zeros((151936, 64)),
                }
            )
        )

        # No lm_head = not aligner
        self.assertFalse(
            self.Model._is_forced_aligner_weights(
                {
                    "thinker.model.norm.weight": mx.zeros((64,)),
                }
            )
        )

    def test_model_wrapper_sanitize_uses_correct_method(self):
        # For ASR weights (large lm_head)
        asr_weights = {
            "thinker.lm_head.weight": mx.zeros((151936, 64)),
            "lm_head.weight": mx.zeros((151936, 64)),
        }
        sanitized = self.Model.sanitize(asr_weights)
        # ASR sanitize should skip lm_head
        self.assertNotIn("lm_head.weight", sanitized)

        # For ForcedAligner weights (small lm_head)
        aligner_weights = {
            "thinker.lm_head.weight": mx.zeros((5000, 64)),
        }
        sanitized = self.Model.sanitize(aligner_weights)
        # ForcedAligner sanitize should keep lm_head
        self.assertIn("lm_head.weight", sanitized)


if __name__ == "__main__":
    unittest.main()

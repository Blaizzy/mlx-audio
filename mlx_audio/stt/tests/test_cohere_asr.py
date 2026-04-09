import importlib.util
import json
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.cohere_asr.audio import CohereAudioFrontend
from mlx_audio.stt.models.cohere_asr.cohere_asr import (
    Model,
    join_chunk_texts,
    split_audio_chunks_energy,
)
from mlx_audio.stt.models.cohere_asr.config import ModelConfig, PreprocessorConfig


def _small_config():
    return ModelConfig.from_dict(
        {
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
    )


@unittest.skipUnless(
    importlib.util.find_spec("sentencepiece") is not None,
    "sentencepiece is required for tokenizer tests",
)
class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sentencepiece as spm

        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls._tmpdir.name

        corpus_path = f"{tmp_path}/corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("hello world\n")
            f.write("if not there will be a big crisis\n")
            f.write("european parliament\n")

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
            input=corpus_path,
            model_prefix=f"{tmp_path}/toy",
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

        with open(f"{tmp_path}/tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f)
        with open(f"{tmp_path}/special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f)

        cls.model_path = f"{tmp_path}/toy.model"
        cls.tokenizer_config_path = f"{tmp_path}/tokenizer_config.json"
        cls.special_tokens_map_path = f"{tmp_path}/special_tokens_map.json"

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_prompt_tokens(self):
        from mlx_audio.stt.models.cohere_asr.tokenizer import CohereAsrTokenizer

        tokenizer = CohereAsrTokenizer(
            self.model_path,
            self.tokenizer_config_path,
            self.special_tokens_map_path,
        )
        tokens = tokenizer.build_prompt_tokens("en", punctuation=True)
        self.assertEqual(len(tokens), 9)
        self.assertEqual(tokens[0], tokenizer.sp.piece_to_id("<|startofcontext|>"))
        self.assertEqual(tokens[1], tokenizer.bos_token_id)
        self.assertEqual(tokens[-1], tokenizer.sp.piece_to_id("<|nodiarize|>"))


class TestAudioFrontend(unittest.TestCase):
    def test_extract_features_shape_and_lengths(self):
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


class TestChunking(unittest.TestCase):
    def test_energy_chunking_splits_long_audio(self):
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
        self.assertEqual(
            join_chunk_texts(["hello", "world"], language="en"), "hello world"
        )
        self.assertEqual(join_chunk_texts(["你", "好"], language="zh"), "你好")


class TestSanitize(unittest.TestCase):
    def setUp(self):
        self.model = Model(_small_config())

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


class TestConfig(unittest.TestCase):
    def test_defaults_from_reference_shape(self):
        config = _small_config()
        self.assertEqual(config.model_type, "cohere_asr")
        self.assertEqual(config.min_energy_window_samples, 1600)
        self.assertEqual(config.preprocessor.win_length, 400)
        self.assertEqual(config.preprocessor.hop_length, 160)

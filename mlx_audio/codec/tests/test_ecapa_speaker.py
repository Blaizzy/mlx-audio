import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.codec import EcapaSpeakerEmbeddingExtractor
from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import (
    AttentiveStatisticsPooling,
    TDNNBlock,
)
from mlx_audio.codec.models.ecapa_tdnn.speaker import speaker_log_mel


class TestSpeakerFrontend(unittest.TestCase):
    def test_speechbrain_frontend_conventions(self):
        # Independent NumPy STFT reference catches window periodicity, center
        # padding, power-vs-magnitude, mel bandwidths and dB normalization.
        audio = np.random.default_rng(42).normal(0, 0.1, 4800).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(
            np.pad(audio, (200, 200)), 400
        )[::160]
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 400)
        spectrum = np.fft.rfft(frames * window, axis=-1)
        hz = 700 * (
            10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 82) / 2595) - 1
        )
        slopes = (np.linspace(0, 8000, 201)[:, None] - hz[None, 1:-1]) / np.diff(hz)[
            None, :-1
        ]
        filters = np.maximum(0, np.minimum(slopes + 1, 1 - slopes))
        expected = 10 * np.log10(np.maximum(np.abs(spectrum) ** 2 @ filters, 1e-10))
        expected = np.maximum(expected, expected.max() - 80)
        actual = np.array(speaker_log_mel(mx.array(audio)))
        self.assertEqual(actual.shape, (31, 80))
        # GPU mel matrix multiplication has small float32 rounding differences.
        np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=0)

    def test_silence_is_finite(self):
        actual = np.array(speaker_log_mel(mx.zeros((1600,))))
        np.testing.assert_allclose(actual, -100, atol=1e-5)


class TestSpeakerBackboneOptions(unittest.TestCase):
    def test_reflection_padding_matches_numpy(self):
        block = TDNNBlock(2, 3, kernel_size=3, dilation=2, padding_mode="reflect")
        block.eval()
        x = mx.arange(22, dtype=mx.float32).reshape(1, 11, 2)
        expected_input = mx.array(
            np.pad(np.array(x), ((0, 0), (2, 2), (0, 0)), mode="reflect")
        )
        expected = block.norm(mx.maximum(block.conv(expected_input), 0))
        np.testing.assert_allclose(
            np.array(block(x)), np.array(expected), atol=0, rtol=0
        )

    def test_reflection_rejects_too_few_frames(self):
        block = TDNNBlock(2, 3, kernel_size=3, dilation=4, padding_mode="reflect")
        with self.assertRaisesRegex(ValueError, "requires more than"):
            block(mx.zeros((1, 4, 2)))

    def test_centered_statistics_preserve_small_variance(self):
        pool = AttentiveStatisticsPooling(
            2, attention_channels=2, centered_statistics=True
        )
        pool.eval()
        pool.conv.weight = mx.zeros_like(pool.conv.weight)
        pool.conv.bias = mx.zeros_like(pool.conv.bias)
        x = np.array(
            [[[10000, 3], [10001, 3], [10002, 3], [10003, 3]]], dtype=np.float32
        )
        actual = np.array(pool(mx.array(x)))
        expected = np.concatenate(
            [x.mean(axis=1), np.maximum(x.std(axis=1), 1e-6)], axis=-1
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class TestSpeakerExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = object.__new__(EcapaSpeakerEmbeddingExtractor)
        self.extractor.model = MagicMock(
            return_value=mx.arange(1, 193, dtype=mx.float32)[None]
        )

    def test_normalized_192_dimension_output(self):
        output = self.extractor.encode(np.zeros(3200, dtype=np.float32))
        self.assertEqual(output.shape, (192,))
        self.assertAlmostEqual(mx.linalg.norm(output).item(), 1, places=6)
        features = self.extractor.model.call_args.args[0]
        self.assertEqual(features.shape, (1, 21, 80))
        np.testing.assert_allclose(np.array(features.mean(axis=1)), 0, atol=1e-5)

    def test_rejects_invalid_audio_before_model(self):
        cases = [
            np.zeros(1599),
            np.zeros((1, 3200)),
            np.full(3200, np.nan),
            np.full(3200, np.inf),
        ]
        for audio in cases:
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                self.extractor.encode(audio)
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            self.extractor.encode(np.zeros(3200), sample_rate=8000)
        self.extractor.model.assert_not_called()

    def test_loading_is_strict_and_downloads_only_weights(self):
        fake_model = MagicMock()
        fake_model.parameters.return_value = {}

        def initialize(extractor):
            extractor.model = fake_model

        weights = {
            "embedding_model.blocks.block0.conv.weight": mx.ones((1,)),
            "embedding_model.blocks.block1.tdnn1.conv.weight": mx.ones((1,)),
            "embedding_model.fc.bias": mx.ones((1,)),
        }
        with (
            patch.object(EcapaSpeakerEmbeddingExtractor, "__init__", initialize),
            patch(
                "huggingface_hub.hf_hub_download", return_value="/tmp/model.safetensors"
            ) as download,
            patch("mlx.core.load", return_value=weights),
        ):
            EcapaSpeakerEmbeddingExtractor.from_pretrained(
                "owner/speaker-model", revision="pinned-commit"
            )
        download.assert_called_once_with(
            "owner/speaker-model", "model.safetensors", revision="pinned-commit"
        )
        args = fake_model.load_weights.call_args
        self.assertTrue(args.kwargs["strict"])
        self.assertEqual(
            set(dict(args.args[0])),
            {
                "block0.conv.weight",
                "block1.tdnn1.conv.weight",
                "blocks.0.tdnn1.conv.weight",
                "fc.bias",
            },
        )

    def test_missing_absolute_path_does_not_download(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("huggingface_hub.hf_hub_download") as download,
        ):
            with self.assertRaises(FileNotFoundError):
                EcapaSpeakerEmbeddingExtractor.from_pretrained(
                    Path(directory) / "missing.safetensors"
                )
            download.assert_not_called()


@unittest.skipUnless(
    os.environ.get("MLX_AUDIO_TEST_ECAPA_SPEAKER") == "1",
    "Set MLX_AUDIO_TEST_ECAPA_SPEAKER=1 for pretrained SpeechBrain parity (requires torch and speechbrain)",
)
class TestPretrainedSpeechBrainParity(unittest.TestCase):
    def test_voxceleb_checkpoint_and_public_example(self):
        import torch
        from huggingface_hub import hf_hub_download
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        from mlx_audio.utils import load_audio

        source = "speechbrain/spkrec-ecapa-voxceleb"
        revision = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
        weights = hf_hub_download(source, "embedding_model.ckpt", revision=revision)
        example = hf_hub_download(source, "example1.wav", revision=revision)
        audio = np.array(load_audio(example, sample_rate=16000))
        reference = (
            ECAPA_TDNN(
                input_size=80,
                channels=[1024, 1024, 1024, 1024, 3072],
                kernel_sizes=[5, 3, 3, 3, 1],
                dilations=[1, 2, 3, 4, 1],
                attention_channels=128,
                lin_neurons=192,
            )
            .cpu()
            .float()
            .eval()
        )
        reference.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
        frontend = Fbank(n_mels=80).eval()
        normalization = InputNormalization(norm_type="sentence", std_norm=False).eval()
        with torch.inference_mode():
            lengths = torch.ones(1)
            features = frontend(torch.from_numpy(audio)[None])
            expected = (
                reference(normalization(features, lengths), lengths).numpy().reshape(-1)
            )
            expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(
            np.array(speaker_log_mel(mx.array(audio))),
            features.numpy()[0],
            atol=1e-2,
            rtol=0,
        )
        extractor = EcapaSpeakerEmbeddingExtractor.from_pretrained(
            revision="e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77"
        )
        actual = np.array(extractor.encode(audio))
        self.assertEqual(actual.shape, (192,))
        self.assertGreater(float(expected @ actual), 0.99999)
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()

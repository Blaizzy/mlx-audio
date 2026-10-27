import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.lingv3_tts.speaker import (
    SPEAKER_FILE,
    SPEAKER_REPO,
    SPEAKER_REVISION,
    SpeakerEncoder,
    resolve_speaker_weights,
)


class TestMingSpeaker(unittest.TestCase):
    def test_local_native_weights_do_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / SPEAKER_FILE
            path.write_bytes(b"fixture")
            with patch("huggingface_hub.hf_hub_download") as download:
                self.assertEqual(resolve_speaker_weights(path), path)
            download.assert_not_called()

    def test_missing_weights_fetch_only_pinned_native_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "huggingface_hub.hf_hub_download",
                return_value="/tmp/native.safetensors",
            ) as download:
                path = resolve_speaker_weights(Path(directory) / SPEAKER_FILE)
            self.assertEqual(path, Path("/tmp/native.safetensors"))
            download.assert_called_once_with(
                SPEAKER_REPO, SPEAKER_FILE, revision=SPEAKER_REVISION
            )

    def test_frontend_returns_native_192_value_embedding(self):
        encoder = SpeakerEncoder.__new__(SpeakerEncoder)
        encoder.model = MagicMock(return_value=mx.arange(192)[None].astype(mx.float32))
        audio = np.sin(np.arange(1600, dtype=np.float32) * 0.1)
        embedding = encoder(audio, sample_rate=16000)
        self.assertEqual(embedding.shape, (1, 192))
        features = encoder.model.call_args.args[0]
        self.assertEqual(features.shape, (1, 8, 80))
        self.assertTrue(
            bool(mx.allclose(features.mean(1), mx.zeros((1, 80)), atol=4e-6))
        )

    def test_invalid_references_and_nonfinite_embeddings_fail(self):
        encoder = SpeakerEncoder.__new__(SpeakerEncoder)
        encoder.model = lambda features: mx.full((1, 192), float("nan"))
        for waveform in (np.zeros(719), np.full(1600, np.nan)):
            with self.assertRaises(ValueError):
                encoder(waveform, sample_rate=16000)
        with self.assertRaises(FloatingPointError):
            encoder(np.ones(1600), sample_rate=16000)

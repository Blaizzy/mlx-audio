from unittest import mock

import mlx.nn as nn

from mlx_audio.tts.models.spark import audio_tokenizer
from mlx_audio.tts.models.spark.audio_tokenizer import BiCodecTokenizer


class _Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.BatchNorm(4)
        self.config = mock.Mock()


def test_loaded_models_are_in_eval_mode(tmp_path):
    with (
        mock.patch.object(audio_tokenizer, "load_config", return_value={}),
        mock.patch.object(
            audio_tokenizer.BiCodec, "load_from_checkpoint", return_value=_Module()
        ),
        mock.patch.object(audio_tokenizer.Wav2Vec2FeatureExtractor, "from_pretrained"),
        mock.patch.object(
            audio_tokenizer.Wav2Vec2Model, "from_pretrained", return_value=_Module()
        ),
    ):
        tokenizer = BiCodecTokenizer(tmp_path)

    for model in (tokenizer.model, tokenizer.feature_extractor):
        assert not model.training
        assert not model.dropout.training
        assert not model.norm.training

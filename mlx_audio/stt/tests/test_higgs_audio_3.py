"""Tests for the Higgs-Audio v3 STT MLX model.

The lightweight tests build a tiny randomly-initialized model and exercise the
config, audio encoder/projector shapes, embedding merge, and weight-sanitize
logic without any download. The integration test is gated behind the
``requires_weights`` marker and an env var pointing at the model directory.
"""

import os

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.base import BaseModelArgs
from mlx_audio.stt.models.higgs_audio_3 import AudioEncoderConfig, Model, ModelConfig
from mlx_audio.stt.models.higgs_audio_3.audio import AudioFeatureExtractor
from mlx_audio.stt.models.higgs_audio_3.higgs_audio_3 import (
    HiggsAudioEncoder,
    HiggsAudioFeatureProjector,
)


def _tiny_config() -> dict:
    return {
        "model_type": "higgs_audio_3",
        "projector_temporal_downsample": 2,
        "projector_type": "mlp",
        "audio_num_codebooks": 1,
        "audio_codebook_size": 1,
        "audio_in_token_idx": 40,
        "audio_encoder_config": {
            "num_mel_bins": 32,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "d_model": 16,
            "max_source_positions": 200,
        },
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    }


def _tiny_model() -> Model:
    model = ModelConfig.from_dict(_tiny_config())
    m = Model(model)
    mx.eval(m.parameters())
    return m


def test_config_parses_nested():
    cfg = ModelConfig.from_dict(_tiny_config())
    assert isinstance(cfg, BaseModelArgs)
    assert isinstance(cfg.audio_encoder_config, AudioEncoderConfig)
    assert cfg.audio_encoder_config.d_model == 16
    assert cfg.text_config.num_hidden_layers == 2
    assert cfg.projector_temporal_downsample == 2
    assert cfg.audio_in_token_idx == 40


def test_feature_extractor_returns_mel_array():
    fx = AudioFeatureExtractor(num_mel_bins=128)
    wav = mx.zeros(16000)
    mel = fx(wav)
    assert isinstance(mel, mx.array)
    assert mel.shape[0] == 1
    assert mel.shape[1] == 128


def test_encoder_halves_time_dim():
    cfg = ModelConfig.from_dict(_tiny_config())
    enc = HiggsAudioEncoder(cfg.audio_encoder_config)
    mx.eval(enc.parameters())
    mel = mx.zeros((1, cfg.audio_encoder_config.num_mel_bins, 80))
    out = enc(mel)
    # conv2 stride 2 then avg-pool stride 2 => time roughly /4
    assert out.shape[0] == 1
    assert out.shape[2] == cfg.audio_encoder_config.d_model
    assert out.shape[1] == 80 // 4


def test_projector_downsamples_and_projects():
    cfg = ModelConfig.from_dict(_tiny_config())
    proj = HiggsAudioFeatureProjector(cfg)
    mx.eval(proj.parameters())
    x = mx.zeros((1, 20, cfg.audio_encoder_config.d_model))
    out = proj(x)
    assert out.shape[0] == 1
    assert out.shape[2] == cfg.text_config.hidden_size
    # stride-2 temporal conv: ceil(20 / 2) == 10
    assert out.shape[1] == 10


def test_forward_logits_shape():
    m = _tiny_model()
    ids = mx.array([[1, 2, 3, 4, 5]])
    logits = m(ids)
    assert logits.shape == (1, 5, m.config.text_config.vocab_size)


def test_get_input_embeddings_merges_audio():
    m = _tiny_model()
    m.config.vad_cut = False
    m._tokenizer = _StubTokenizer(m.config.audio_in_token_idx)
    wav = np.zeros(16000, dtype=np.float32)
    ids, embeds, plen = m.get_input_embeddings(wav)
    # one <|AUDIO|> placeholder expands into many audio rows
    assert embeds.shape[1] > len(ids)
    assert embeds.shape[2] == m.config.text_config.hidden_size
    assert plen == embeds.shape[1]


def test_vad_chunk_ranges_fallback_when_no_backend():
    from mlx_audio.stt.models.higgs_audio_3.vad import vad_chunk_ranges

    wav = np.zeros(10 * 16000, dtype=np.float32)
    ranges = vad_chunk_ranges(wav, chunk_samples=4 * 16000, backend=None)
    assert ranges == [(0, 64000), (64000, 128000), (128000, 160000)]


def test_vad_chunk_ranges_respects_cuts():
    from mlx_audio.stt.models.higgs_audio_3 import vad as vad_mod

    class _StubBackend:
        def speech_ranges(self, wav):
            return [(16000, 48000), (96000, 144000)]

    wav = np.zeros(10 * 16000, dtype=np.float32)
    chunk = 4 * 16000

    merged = vad_mod.vad_chunk_ranges(
        wav, chunk, backend=_StubBackend(), split_vads=False
    )
    # start of first cut clamped to 0, last cut extended to the full length
    assert merged[0][0] == 0
    assert merged[-1][1] == 10 * 16000
    assert all(e - s <= chunk for s, e in merged)

    split = vad_mod.vad_chunk_ranges(
        wav, chunk, backend=_StubBackend(), split_vads=True
    )
    # split_vads keeps only the detected speech ranges (sub-chunked)
    assert split[0][0] == 16000
    assert split[-1][1] == 144000


def test_sanitize_remaps_and_transposes():
    cfg = ModelConfig.from_dict(_tiny_config())
    d = cfg.audio_encoder_config.d_model
    h = cfg.text_config.hidden_size
    v = cfg.text_config.vocab_size
    weights = {
        "embed_tokens.weight": mx.zeros((v, h)),
        "norm.weight": mx.zeros((h,)),
        "layers.0.self_attn.q_proj.weight": mx.zeros((16, h)),
        "audio_tower.conv1.weight": mx.zeros(
            (d, cfg.audio_encoder_config.num_mel_bins, 3)
        ),
        "audio_encoder_proj.temporal.weight": mx.zeros((d, 1, 3)),
        "audio_decoder_proj.text_lm_head.weight": mx.zeros((v, h)),
        "audio_decoder_proj.audio_lm_head.weight": mx.zeros((3, h)),
        "audio_codebook_embeddings.weight": mx.zeros((3, h)),
    }
    san = Model.sanitize(dict(weights))
    assert "audio_decoder_proj.audio_lm_head.weight" not in san
    assert "audio_codebook_embeddings.weight" not in san
    assert "lm_head.weight" in san
    assert "model.embed_tokens.weight" in san
    assert "model.norm.weight" in san
    assert "model.layers.0.self_attn.q_proj.weight" in san
    # Conv1d weights transposed (out, in, k) -> (out, k, in)
    assert san["audio_tower.conv1.weight"].shape == (
        d,
        3,
        cfg.audio_encoder_config.num_mel_bins,
    )
    assert san["audio_encoder_proj.temporal.weight"].shape == (d, 3, 1)


def test_sanitize_covers_all_model_keys():
    m = _tiny_model()
    model_keys = {k for k, _ in tree_flatten(m.parameters())}
    weights = {}
    for k in model_keys:
        if k == "lm_head.weight":
            weights["audio_decoder_proj.text_lm_head.weight"] = mx.zeros(
                m.lm_head.weight.shape
            )
            continue
        shape = dict(tree_flatten(m.parameters()))[k].shape
        src = k
        if k.startswith("model.embed_tokens"):
            src = "embed_tokens.weight"
        elif k.startswith("model.norm"):
            src = "norm.weight"
        elif k.startswith("model.layers"):
            src = k[len("model.") :]
        if "audio_tower.conv" in k and len(shape) == 3:
            shape = (shape[0], shape[2], shape[1])
        if "audio_encoder_proj.temporal" in k and len(shape) == 3:
            shape = (shape[0], shape[2], shape[1])
        weights[src] = mx.zeros(shape)
    san = Model.sanitize(weights)
    assert set(san.keys()) == model_keys


class _StubTokenizer:
    def __init__(self, audio_id):
        self._audio_id = audio_id
        self._counter = 1

    def encode(self, text, add_special_tokens=False):
        if "<|AUDIO|>" in text:
            return [self._audio_id]
        n = max(1, len(text.split()))
        return list(range(1, 1 + n))


@pytest.mark.requires_weights
def test_real_model_transcribes():
    path = os.environ.get("HIGGS_AUDIO_3_PATH")
    if not path:
        pytest.skip("set HIGGS_AUDIO_3_PATH to the higgs-audio-v3-stt model directory")
    audio = os.environ.get("HIGGS_AUDIO_3_TEST_AUDIO")
    if not audio:
        pytest.skip("set HIGGS_AUDIO_3_TEST_AUDIO to a wav/flac file")

    from mlx_audio.stt import load

    model = load(path)
    result = model.generate(audio, max_tokens=256)
    assert result.text.strip()

"""Tests for the Nemotron 3.5 ASR MLX model.

The lightweight tests build a tiny randomly-initialized model and exercise the
config/shape/mask/tokenizer logic without any download. The integration test is
gated behind the ``requires_weights`` marker and an env var pointing at a
converted MLX model directory.
"""

import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.nemotron_asr import Model, ModelConfig
from mlx_audio.stt.models.nemotron_asr import tokenizer as tok
from mlx_audio.stt.models.nemotron_asr.audio import (
    iter_log_mel_spectrogram,
    log_mel_spectrogram,
)
from mlx_audio.stt.models.nemotron_asr.conformer import create_chunked_limited_mask


def _tiny_config() -> dict:
    vocab = ["<unk>", "<en-US>", "▁hello", "▁world", "!", "a", "b", "c"]
    return {
        "model_type": "nemotron_asr",
        "preprocessor": {"features": 80, "n_fft": 512, "normalize": "NA"},
        "encoder": {
            "feat_in": 80,
            "n_layers": 2,
            "d_model": 32,
            "n_heads": 2,
            "ff_expansion_factor": 2,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 8,
            "conv_kernel_size": 9,
            "causal_downsampling": True,
            "conv_context_size": "causal",
            "conv_norm_type": "layer_norm",
            "att_context_style": "chunked_limited",
            "att_context_size": [[56, 13]],
            "pos_emb_max_len": 500,
            "use_bias": False,
        },
        "prompt": {
            "num_prompts": 4,
            "prompt_hidden": 16,
            "prompt_dictionary": {"en-US": 0, "auto": 1},
        },
        "decoder": {
            "pred_hidden": 16,
            "pred_rnn_layers": 2,
            "vocab_size": len(vocab),
            "blank_as_pad": True,
        },
        "joint": {
            "joint_hidden": 16,
            "activation": "relu",
            "encoder_hidden": 32,
            "pred_hidden": 16,
            "num_classes": len(vocab),
        },
        "vocabulary": vocab,
        "default_language": "auto",
        "default_att_context_size": [56, 13],
        "max_symbols": 5,
    }


def _build_tiny() -> Model:
    model = Model(ModelConfig.from_dict(_tiny_config()))
    mx.eval(model.parameters())
    model.eval()
    return model


def test_config_roundtrip():
    cfg = ModelConfig.from_dict(_tiny_config()).config
    assert cfg.encoder.n_layers == 2
    assert cfg.encoder.causal_downsampling is True
    assert cfg.prompt.num_prompts == 4
    assert cfg.decoder.vocab_size == len(cfg.vocabulary)
    assert cfg.prompt.prompt_dictionary["en-US"] == 0


def test_chunked_limited_mask():
    # left=2, right=1 -> chunk_size=2, left_chunks=2//2=1: a frame sees its chunk + 1 prior chunk.
    mask = create_chunked_limited_mask(6, left_context=2, right_context=1)
    visible = np.array(mask[0, 0]) == 0.0
    # frame 0 (chunk 0): only chunk 0 -> frames 0,1
    assert visible[0].tolist() == [True, True, False, False, False, False]
    # frame 2 (chunk 1): chunks 0 and 1 -> frames 0..3
    assert visible[2].tolist() == [True, True, True, True, False, False]
    # frame 5 (chunk 2): chunks 1 and 2 -> frames 2..5 (chunk 0 now out of left window)
    assert visible[5].tolist() == [False, False, True, True, True, True]


def test_chunked_log_mel_matches_full():
    args = ModelConfig.from_dict(_tiny_config()).config.preprocessor
    audio = mx.array(
        (np.random.randn(args.sample_rate * 2 + 123) * 0.1).astype(np.float32)
    )

    full = log_mel_spectrogram(audio, args)
    chunked = mx.concatenate(
        list(iter_log_mel_spectrogram(audio, args, chunk_frames=37)),
        axis=1,
    )
    np.testing.assert_allclose(np.array(chunked), np.array(full), rtol=1e-3, atol=1e-3)


def test_encoder_and_prompt_shapes():
    model = _build_tiny()
    d_model = model.encoder_config.d_model
    # ~1s of fake mel: (1, T, 80)
    mel = mx.array(np.random.randn(1, 200, 80).astype(np.float32))
    enc, lengths = model.encoder(mel, att_context_size=[56, 13])
    assert enc.shape[0] == 1 and enc.shape[2] == d_model
    # subsampling factor 8 -> roughly T/8 frames
    assert enc.shape[1] == int(lengths[0])
    assert abs(enc.shape[1] - math.ceil(200 / 8)) <= 2

    prompted = model.apply_prompt(enc, "en-US")
    assert prompted.shape == enc.shape  # projected back to d_model


def test_decode_runs_and_is_clean():
    model = _build_tiny()
    mel = mx.array(np.random.randn(1, 120, 80).astype(np.float32))
    result = model.decode(mel, language="auto")
    assert hasattr(result, "text")
    # special tokens (e.g. <en-US>, <unk>) never leak into the decoded text.
    assert "<" not in result.text


def _total_tokens(result) -> int:
    return sum(len(sentence.tokens) for sentence in result.sentences)


def test_stream_generate_runs_and_is_clean():
    model = _build_tiny()
    sr = model.preprocessor_config.sample_rate
    # ~2.5s of fake waveform -> several native chunks.
    audio = mx.array((np.random.randn(int(2.5 * sr)) * 0.1).astype(np.float32))

    results = list(model.stream_generate(audio, language="auto"))
    assert len(results) >= 1
    # cumulative: the hypothesis only grows, so token count never decreases.
    counts = [_total_tokens(r) for r in results]
    assert counts == sorted(counts)
    # special tokens (e.g. <en-US>, <unk>) never leak into any chunk's text.
    assert all("<" not in r.text for r in results)


def test_stream_matches_offline():
    # Cache-aware streaming is frame-identical to the offline chunked_limited
    # encoder at the native chunk size, so the greedy decode must be identical.
    model = _build_tiny()
    sr = model.preprocessor_config.sample_rate
    audio = mx.array((np.random.randn(int(2.5 * sr)) * 0.1).astype(np.float32))

    offline = model.generate(audio, language="auto")
    streamed = list(model.stream_generate(audio, language="auto"))[-1]
    assert streamed.text == offline.text


def test_tokenizer_decode_and_lang_tags():
    vocab = ["<unk>", "<en-US>", "▁hello", "▁world", "!", "<"]
    assert tok.is_lang_tag("<en-US>") and not tok.is_lang_tag("<")
    assert tok.is_special_token(1, vocab) and not tok.is_special_token(2, vocab)
    # ids: <en-US>, ▁hello, ▁world, !  -> "hello world!" with lang tag stripped
    assert tok.decode([1, 2, 3, 4], vocab) == " hello world!"
    assert (
        tok.decode([1, 2, 3, 4], vocab, strip_lang_tags=False) == "<en-US> hello world!"
    )
    assert tok.detected_language([1, 2, 3], vocab) == "en-US"


@pytest.mark.requires_weights
def test_real_model_transcribes():
    path = os.environ.get("NEMOTRON_ASR_MLX_PATH")
    if not path:
        pytest.skip("set NEMOTRON_ASR_MLX_PATH to a converted MLX model directory")
    from mlx_audio.stt import load

    model = load(path)
    audio = os.environ.get("NEMOTRON_ASR_TEST_AUDIO")
    if not audio:
        pytest.skip("set NEMOTRON_ASR_TEST_AUDIO to a wav file")
    result = model.generate(audio, language="auto")
    assert result.text.strip()

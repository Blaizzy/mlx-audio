"""Tests for Nemotron Unified (buffered streaming) support in ``nemotron_asr``.

Tiny randomly initialized models cover the converter config, the
``chunked_limited_with_rc`` mask, padding isolation, the per-feature mel and the
buffered segmentation. Real weights are checked against NeMo's tokens in
``test_nemotron_unified_nemo_parity.py``.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.nemotron_asr import Model, ModelConfig
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram
from mlx_audio.stt.models.nemotron_asr.buffered import (
    BufferedContext,
    BufferedEncoderState,
)
from mlx_audio.stt.models.nemotron_asr.conformer import create_chunked_rc_visibility
from mlx_audio.stt.models.nemotron_asr.convert import _build_config
from mlx_audio.stt.tests.test_nemotron_asr import _tiny_config


def _unified_config() -> dict:
    config = _tiny_config()
    config["preprocessor"].update(normalize="per_feature", pad_mode="constant")
    config["encoder"].update(
        causal_downsampling=False,
        conv_context_size=[4, 4],
        conv_norm_type="batch_norm",
        att_context_style="chunked_limited_with_rc",
        att_context_size=[[-1, -1, -1]],
        use_bias=True,
        xscaling=True,
    )
    config["default_att_context_size"] = [4, 1, 2]
    return config


def _unified_model() -> Model:
    model = Model(ModelConfig.from_dict(_unified_config()))
    mx.eval(model.parameters())
    model.eval()
    return model


def test_converter_maps_unified_yaml():
    nemo_cfg = {
        "preprocessor": {"normalize": "per_feature"},
        "encoder": {
            "feat_in": 128,
            "n_layers": 24,
            "d_model": 1024,
            "n_heads": 8,
            "ff_expansion_factor": 4,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 256,
            "conv_kernel_size": 9,
            "causal_downsampling": False,
            "conv_context_size": None,
            "conv_norm_type": "batch_norm",
            "att_context_size": [-1, -1],
            "att_context_style": "chunked_limited_with_rc",
            "xscaling": True,
        },
        "decoder": {
            "prednet": {"pred_hidden": 640, "pred_rnn_layers": 2},
            "vocab_size": 4,
        },
        "joint": {
            "jointnet": {
                "joint_hidden": 640,
                "encoder_hidden": 1024,
                "pred_hidden": 640,
            },
            "num_classes": 4,
            "vocabulary": ["<unk>", "a", "b", "c"],
        },
        "model_defaults": {
            "enc_hidden": 1024,
            "prompt_dictionary": {"en": 0, "auto": 127},
        },
    }
    config = _build_config(nemo_cfg)
    assert config["encoder"]["conv_context_size"] == [4, 4]
    assert config["encoder"]["att_context_size"] == [[-1, -1, -1]]
    assert config["encoder"]["use_bias"] is True
    assert config["preprocessor"]["pad_mode"] == "constant"
    assert config["default_att_context_size"] == [70, 1, 3]
    assert Model(ModelConfig.from_dict(config)).buffered


def test_chunked_rc_visibility():
    # left 2, chunk 2, right 1: frame i sees [chunk_start - 2, chunk_start + 2].
    visible = np.array(create_chunked_rc_visibility(7, 2, 2, 1)).astype(int)
    assert visible[0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert visible[3].tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert visible[6].tolist() == [0, 0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("context", [[-1, -1, -1], [4, 1, 2]])
def test_encoder_ignores_padded_frames(context):
    model = _unified_model()
    rng = np.random.default_rng(3)
    mel = mx.array(rng.normal(size=(1, 80, 80)).astype(np.float32))
    noise = mx.array(rng.normal(size=(1, 24, 80)).astype(np.float32))
    lengths = mx.array([80])
    a, la = model.encoder(mel, lengths=lengths, att_context_size=context)
    b, lb = model.encoder(
        mx.concatenate([mel, noise], axis=1), lengths=lengths, att_context_size=context
    )
    valid = int(la[0])
    assert valid == int(lb[0]) == 10
    np.testing.assert_allclose(
        np.array(a[:, :valid]), np.array(b[:, :valid]), rtol=1e-4, atol=1e-4
    )


def test_float16_mask_stays_finite_when_a_chunk_is_all_padding():
    # A left-0 layout leaves a padded chunk with every key masked. A mask value that
    # overflows to -inf in float16 turns such rows NaN, and the NaN reaches valid
    # frames through later layers (0 x NaN).
    model = _unified_model()
    model.set_dtype(mx.float16)
    rng = np.random.default_rng(3)
    mel = mx.array(rng.normal(size=(1, 80, 80)).astype(np.float16))
    noise = mx.array(rng.normal(size=(1, 24, 80)).astype(np.float16))
    out, lengths = model.encoder(
        mx.concatenate([mel, noise], axis=1),
        lengths=mx.array([80]),
        att_context_size=[0, 2, 0],
    )
    assert int(lengths[0]) == 10
    assert np.isfinite(np.array(out[:, :10].astype(mx.float32))).all()


def test_per_feature_mel_uses_valid_frames():
    args = ModelConfig.from_dict(_unified_config()).config.preprocessor
    rng = np.random.default_rng(1)
    audio = mx.array(rng.normal(0, 0.1, 40 * args.hop_length + 7).astype(np.float32))
    mel = np.array(log_mel_spectrogram(audio, args)[0])
    assert mel.shape[0] == 41  # centered STFT: samples // hop + 1 frames
    assert np.all(mel[40] == 0)
    np.testing.assert_allclose(mel[:40].mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(mel[:40].std(axis=0, ddof=1), 1, atol=1e-3)


def _nemo_windows(total, left, chunk, right):
    """Buffer windows of NeMo's streaming loop (ContextSize.add_frames_get_removed_)."""
    windows = []
    ctx = [0, 0, 0]  # left, chunk, right samples in the buffer
    buffer_end = 0
    left_sample, right_sample = 0, min(chunk + right, total)
    while left_sample < total:
        added = min(right_sample, total) - left_sample
        is_last = right_sample >= total
        ctx[0] += ctx[1]
        ctx[1] = 0
        ctx[2] += added
        if is_last:
            ctx[1], ctx[2] = ctx[2], 0
        elif ctx[2] > chunk:
            ctx[1], ctx[2] = chunk, ctx[2] - chunk
        ctx[0] -= max(sum(ctx) - (left + chunk + right), 0)
        buffer_end += added
        windows.append((buffer_end - sum(ctx), buffer_end, ctx[0], is_last))
        left_sample = right_sample
        right_sample = min(right_sample + chunk, total)
    return windows


@pytest.mark.parametrize("feed", [97, 512, 4000, 10**6])
@pytest.mark.parametrize(
    "frames_and_extra", [(0, 1), (1, 0), (3, 0), (5, 0), (6, 0), (9, 300), (31, 7)]
)
def test_buffered_windows_match_nemo(feed, frames_and_extra):
    model = _unified_model()
    context = BufferedContext(4, 1, 2)
    state = BufferedEncoderState(model, context, "en-US")
    fs = state.frame_samples
    total = frames_and_extra[0] * fs + frames_and_extra[1]
    seen = []

    def record(starts, length, left, *, last):
        seen.extend((s, s + length, left, last) for s in starts)
        return [mx.zeros((1, 1, 32))] * len(starts)

    state._encode = record
    audio = np.zeros(total, dtype=np.float32)
    for start in range(0, total, feed):
        state.push(audio[start : start + feed])
    state.flush()
    expected = [
        (start, end, left // fs, last)
        for start, end, left, last in _nemo_windows(total, 4 * fs, fs, 2 * fs)
    ]
    assert seen == expected


def _session_text(model, audio, feed):
    session = model.create_streaming_session(language="en-US")
    parts = []
    for start in range(0, audio.size, feed):
        session.feed(audio[start : start + feed])
        parts.extend(session.step(max_decode_tokens=8))
    session.close()
    while not session.done:
        parts.extend(session.step(max_decode_tokens=8))
    return "".join(parts)


@pytest.mark.parametrize("frames_and_extra", [(9, 37), (12, 0), (2, 1)])
def test_session_matches_stream_generate(frames_and_extra):
    model = _unified_model()
    fs = model.encoder_config.subsampling_factor * model.preprocessor_config.hop_length
    total = frames_and_extra[0] * fs + frames_and_extra[1]
    audio = np.random.default_rng(5).normal(0, 0.05, total).astype(np.float32)
    whole = list(model.stream_generate(mx.array(audio), language="en-US"))[-1].text
    for feed in (61, 5 * fs):
        assert _session_text(model, audio, feed) == whole.strip()


@pytest.mark.parametrize(
    ("delay_ms", "chunk_right"),
    [
        (80, [1, 1]),
        (160, [1, 1]),
        (200, [1, 2]),
        (320, [1, 3]),
        (400, [1, 4]),
        (480, [1, 5]),
        (560, [2, 5]),
        (1040, [2, 11]),
        (1120, [7, 7]),
        (1200, [7, 8]),
        (2080, [13, 13]),
        (5000, [13, 13]),
    ],
)
def test_session_maps_transcription_delay(delay_ms, chunk_right):
    model = _unified_model()
    session = model.create_streaming_session(transcription_delay_ms=delay_ms)
    assert session.att_context_size == [4] + chunk_right


@pytest.mark.parametrize("samples", [1, 80, 159, 160, 161])
def test_session_stops_cleanly_on_tiny_input(samples):
    model = _unified_model()
    audio = np.random.default_rng(2).normal(0, 0.05, samples).astype(np.float32)
    session = model.create_streaming_session(language="en-US")
    session.feed(audio)
    session.close()
    parts = []
    for _ in range(100):
        parts.extend(session.step())
        if session.done:
            break
    assert session.done
    results = list(model.stream_generate(mx.array(audio), language="en-US"))
    assert "".join(parts) == (results[-1].text.strip() if results else "")


def test_language_resolution_follows_nemo():
    model = _unified_model()
    model.prompt_dictionary = {"en-US": 0, "en-GB": 1, "fr-FR": 2, "de": 3, "auto": 4}
    assert model._resolve_prompt_index(None) == 4  # the default, auto
    assert model._resolve_prompt_index("de") == 3
    assert model._resolve_prompt_index("fr") == 2  # unique locale for a bare code
    assert model._resolve_prompt_index("FR") == 2
    with pytest.raises(ValueError, match="ambiguous"):
        model._resolve_prompt_index("en")
    with pytest.raises(ValueError, match="not supported"):
        model._resolve_prompt_index("fr-CH")


def test_session_takes_an_explicit_context():
    model = _unified_model()
    session = model.create_streaming_session(att_context_size=[2, 2, 1])
    assert session.att_context_size == [2, 2, 1]
    audio = np.random.default_rng(9).normal(0, 0.05, 20 * 1280 + 70).astype(np.float32)
    assert _session_text_with(model, audio, 61, [2, 2, 1]) == _session_text_with(
        model, audio, 5000, [2, 2, 1]
    )


def _session_text_with(model, audio, feed, context):
    session = model.create_streaming_session(language="en-US", att_context_size=context)
    parts = []
    for start in range(0, audio.size, feed):
        session.feed(audio[start : start + feed])
        parts.extend(session.step(max_decode_tokens=8))
    session.close()
    while not session.done:
        parts.extend(session.step(max_decode_tokens=8))
    return "".join(parts)

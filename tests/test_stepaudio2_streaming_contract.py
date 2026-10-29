"""CPU-only StepAudio2 sequence tests; no MLX/Torch/network/model imports."""

import builtins
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HELPER = (
    Path(__file__).resolve().parents[1]
    / "mlx_audio/codec/models/stepaudio2/streaming.py"
)
spec = importlib.util.spec_from_file_location("stepaudio2_streaming_contract", HELPER)
codec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = codec
spec.loader.exec_module(codec)
functions = streaming = codec


class MX:
    float32 = np.float32
    int32 = np.int32
    linalg = np.linalg
    array = staticmethod(np.array)
    zeros = staticmethod(lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype))
    zeros_like = staticmethod(np.zeros_like)
    full = staticmethod(np.full)
    pad = staticmethod(np.pad)
    transpose = staticmethod(np.transpose)
    tile = staticmethod(np.tile)
    stack = staticmethod(np.stack)
    repeat = staticmethod(np.repeat)
    concatenate = staticmethod(np.concatenate)
    split = staticmethod(np.split)
    swapaxes = staticmethod(np.swapaxes)
    expand_dims = staticmethod(np.expand_dims)
    broadcast_to = staticmethod(np.broadcast_to)
    linspace = staticmethod(np.linspace)
    cos = staticmethod(np.cos)
    maximum = staticmethod(np.maximum)
    all = staticmethod(np.all)
    isfinite = staticmethod(np.isfinite)

    @staticmethod
    def eval(*_):
        pass

    @staticmethod
    def softmax(x, axis):
        x = x - x.max(axis=axis, keepdims=True)
        x = np.exp(x)
        return x / x.sum(axis=axis, keepdims=True)


def identity(x):
    return x


class Conv:
    def __init__(self, width):
        self.width = width

    def __call__(self, x):
        return np.stack(
            [
                x[:, i : i + self.width, :].mean(axis=1)
                for i in range(x.shape[1] - self.width + 1)
            ],
            axis=1,
        )


class FakePosition:
    def _extend_pe(self, length):
        self.length = length

    def position_encoding(self, size):
        assert size == self.length
        return np.zeros((1, 2 * size - 1, 2))


class FakeEmbed:
    def __init__(self):
        self.pos_enc = FakePosition()

    def __call__(self, x, _):
        return x, None, None


class FakeLayer:
    def __call__(self, x, mask, pos, att_cache):
        assert mask is None
        values = x[:, None, :, :]
        cache = np.concatenate([values, values], axis=-1)
        if att_cache is not None:
            cache = np.concatenate([att_cache, cache], axis=2)
        assert pos.shape[1] == 2 * cache.shape[2] - 1
        return x, mask, cache, None


def encoder():
    return SimpleNamespace(
        embed=FakeEmbed(),
        up_embed=FakeEmbed(),
        encoders=[FakeLayer()],
        up_encoders=[FakeLayer()],
        pre_lookahead_layer=SimpleNamespace(
            pre_lookahead_len=3, conv1=Conv(4), conv2=Conv(3)
        ),
        up_layer=SimpleNamespace(stride=2, conv=Conv(5)),
        normalize_before=True,
        after_norm=identity,
    )


NN = SimpleNamespace(leaky_relu=lambda x: np.where(x >= 0, x, x * 0.01))


def test_encoder_split_preserves_lookahead_cnn_history_and_complete_final_flush():
    tokens = np.arange(16, dtype=np.float32).reshape(1, 8, 2)
    whole, _, whole_att = functions.encoder_chunk(
        encoder(), tokens, True, None, None, MX, NN
    )
    enc = encoder()
    first, cnn, att = functions.encoder_chunk(
        enc, tokens[:, :7], False, None, None, MX, NN
    )
    saved_cnn, saved_att = cnn.copy(), att.copy()
    final, _, final_att = functions.encoder_chunk(
        enc, tokens[:, 4:], True, cnn, att, MX, NN
    )
    assert first.shape == (1, 8, 2)
    assert final.shape == (1, 8, 2)
    assert cnn.shape == (1, 2, 6)
    assert final_att.shape == whole_att.shape == (2, 1, 1, 16, 4)
    np.testing.assert_allclose(np.concatenate([first, final], axis=1), whole)
    np.testing.assert_array_equal(cnn, saved_cnn)
    np.testing.assert_array_equal(att, saved_att)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda e: setattr(e.up_layer, "stride", 3), "stride 2"),
        (
            lambda e: setattr(e.pre_lookahead_layer, "pre_lookahead_len", 2),
            "lookahead 3",
        ),
    ],
)
def test_encoder_rejects_unreviewed_architecture(mutation, message):
    enc = encoder()
    mutation(enc)
    with pytest.raises(ValueError, match=message):
        functions.encoder_chunk(enc, np.zeros((1, 7, 2)), False, None, None, MX, NN)


@pytest.mark.parametrize(
    "cnn,att,message",
    [
        (np.zeros((1, 2, 5)), None, "six"),
        (None, np.zeros((2, 1, 1, 3, 4)), "even"),
    ],
)
def test_encoder_rejects_malformed_history(cnn, att, message):
    with pytest.raises(ValueError, match=message):
        functions.encoder_chunk(encoder(), np.zeros((1, 7, 2)), False, cnn, att, MX, NN)


def prepared_reference():
    codes = np.arange(8, dtype=np.int32)[None, :]
    return (
        codes,
        np.array([8], np.int32),
        np.ones((1, 192), np.float32),
        np.zeros((1, 16, 80), np.float32),
        np.array([16], np.int32),
    )


class FakeHiFT:
    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, mel, source):
        self.calls.append((mel.copy(), source.copy()))
        wave = np.repeat(mel.mean(axis=1), 480, axis=1).astype(np.float32)
        if self.fail:
            wave[0, -1] = np.nan
        return wave, wave[:, None, :].copy()


@pytest.fixture
def networks(monkeypatch):
    """Real extracted Conformer/flow/vocoder paths with a bounded fake solver.

    The solver exposes persistent per-step cache progression; this fixture does
    not pretend to validate trained DiT/HiFT numerics or voice quality.
    """
    flow = SimpleNamespace(
        up_rate=2,
        encoder=encoder(),
        encoder_proj=identity,
        input_embedding=lambda t: np.repeat(
            t[:, :, None].astype(np.float32) * 1e-5, 80, 2
        ),
        spk_embed_affine_layer=lambda s: s[:, :80],
        decoder=object(),
    )
    calls = []

    def solver(decoder, mu, speaker, cond, steps, cnn, att, mx, attention_mode="eager"):
        assert steps == 10
        previous = 0 if att is None else att.shape[4]
        calls.append((previous, mu.shape[2], attention_mode))
        new_att = np.full((10, 1, 2, 1, previous + mu.shape[2], 2), 0.1, np.float32)
        new_cnn = np.full((10, 1, 2, 4, 2), 0.2, np.float32)
        return (
            (mu + cond + speaker[:, :, None] * 0.01).astype(np.float32),
            new_cnn,
            new_att,
        )

    monkeypatch.setattr(functions, "cfm_chunk", solver)
    return flow, FakeHiFT(), calls


def caches_equal(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])


def state_copy(state):
    return (
        {name: value.copy() for name, value in state.flow_cache.items()},
        {name: value.copy() for name, value in state.vocoder_cache.items()},
    )


def decode(networks, state, codes, last=False):
    flow, hift, _ = networks
    return codec.decode_chunk(
        flow,
        hift,
        codes,
        state,
        prepared_reference()[2],
        codec.hamming_window(MX),
        last_chunk=last,
        mx=MX,
        nn=NN,
    )


@pytest.mark.parametrize("tail", [3, 17, 25])
def test_first_28_consumes_25_continuation_and_complete_remaining_final(networks, tail):
    flow, hift, calls = networks
    base = codec.prepare_stream(flow, prepared_reference(), mx=MX, nn=NN)
    state = codec.reset_stream(base, mx=MX)
    base_snapshot = state_copy(base)
    generated = list(range(100, 150))
    first_codes = [4218] * 3 + generated[:25]
    first = decode(networks, state, first_codes)
    second_codes = first_codes[-3:] + generated[25:]
    second = decode(networks, first.state, second_codes)
    remaining = second_codes[-3:] + list(range(200, 200 + tail - 3))
    final = decode(networks, second.state, remaining, True)
    assert first.consumed_codes == second.consumed_codes == 25
    assert final.consumed_codes == tail
    assert first.waveform.shape == second.waveform.shape == (1, 24000)
    assert final.waveform.shape == (1, (8 + tail * 2) * 480)
    assert np.count_nonzero(first.waveform[:, :3840]) == 0
    assert first.state.flow_cache["conformer_att_cache"].shape[3] == 16 + 50
    assert second.state.flow_cache["conformer_att_cache"].shape[3] == 16 + 100
    assert final.state.flow_cache["conformer_att_cache"].shape[3] == 16 + 100
    assert [entry[:2] for entry in calls] == [
        (0, 16),
        (16, 50),
        (66, 50),
        (116, tail * 2),
    ]
    assert len(hift.calls) == 3
    assert [mel.shape[2] for mel, _ in hift.calls] == [50, 58, 8 + tail * 2]
    assert [source.shape[2] for _, source in hift.calls] == [0, 3840, 3840]
    assert final.state.finalized and final.state.consumed_codes == 50 + tail
    caches_equal(base.flow_cache, base_snapshot[0])
    caches_equal(base.vocoder_cache, base_snapshot[1])
    with pytest.raises(codec.CodecError, match="open stream"):
        decode(networks, final.state, [1, 2, 3], True)


def test_independent_interleaved_streams_and_reset_reproduce_with_fresh_leaves(
    networks,
):
    flow, _, _ = networks
    base = codec.prepare_stream(flow, prepared_reference(), mx=MX, nn=NN)
    a, b = codec.reset_stream(base, mx=MX), codec.reset_stream(base, mx=MX)
    for name in base.flow_cache:
        assert not np.shares_memory(a.flow_cache[name], b.flow_cache[name])
        assert not np.shares_memory(a.flow_cache[name], base.flow_cache[name])
    a_codes, b_codes = [4218] * 3 + list(range(25)), [4218] * 3 + list(range(50, 75))
    a1, b1 = decode(networks, a, a_codes), decode(networks, b, b_codes)
    a2 = decode(networks, a1.state, a_codes[-3:], True)
    b2 = decode(networks, b1.state, b_codes[-3:], True)
    assert not np.array_equal(a1.waveform, b1.waveform)
    again = codec.reset_stream(base, mx=MX)
    again1 = decode(networks, again, a_codes)
    again2 = decode(networks, again1.state, a_codes[-3:], True)
    np.testing.assert_array_equal(a1.waveform, again1.waveform)
    np.testing.assert_array_equal(a2.waveform, again2.waveform)
    assert b2.state.finalized and base.chunks == 0
    with pytest.raises(codec.CodecError, match="unconsumed"):
        codec.reset_stream(a1.state, mx=MX)


def test_failed_vocoder_leaves_original_flow_and_overlap_state_untouched(networks):
    flow, hift, _ = networks
    base = codec.prepare_stream(flow, prepared_reference(), mx=MX, nn=NN)
    first = decode(networks, base, [4218] * 3 + list(range(25)))
    saved = state_copy(first.state)
    hift.fail = True
    with pytest.raises(codec.CodecError, match="non-finite"):
        decode(networks, first.state, [1, 2, 3], True)
    caches_equal(first.state.flow_cache, saved[0])
    caches_equal(first.state.vocoder_cache, saved[1])
    hift.fail = False
    assert decode(networks, first.state, [1, 2, 3], True).state.finalized


@pytest.mark.parametrize(
    "codes,last",
    [
        ([], False),
        ([True, 1, 2, 3], False),
        ([6561, 1, 2, 3], False),
        ([-1], True),
        ([1.0], True),
        ([1, 2, 3], False),
        ([1, 2, 3, 4], 1),
    ],
)
def test_invalid_codes_or_final_marker_fail_before_network_calls(networks, codes, last):
    flow, hift, calls = networks
    base = codec.prepare_stream(flow, prepared_reference(), mx=MX, nn=NN)
    before = len(calls)
    with pytest.raises(codec.CodecError):
        decode(networks, base, codes, last)
    assert len(calls) == before and hift.calls == []


@pytest.mark.parametrize(
    "slot,replacement",
    [
        (0, np.zeros((1, 8), np.float32)),
        (0, np.full((1, 8), 6561, np.int32)),
        (1, np.array([7], np.int32)),
        (1, np.array([8], np.int64)),
        (2, np.ones((1, 191), np.float32)),
        (2, np.full((1, 192), np.nan, np.float32)),
        (3, np.zeros((1, 16, 79), np.float32)),
        (3, np.zeros((1, 16, 80), np.float16)),
        (4, np.array([15], np.int32)),
    ],
)
def test_reference_layout_alignment_dtype_and_nonfinite_refused(slot, replacement):
    reference = list(prepared_reference())
    reference[slot] = replacement
    with pytest.raises(codec.CodecError):
        codec.validate_reference(reference, MX)


def test_prepared_offline_prompt_uses_aligned_shape_without_mutating_legacy_metadata(
    networks,
):
    reference = prepared_reference()
    prompt = dict(
        zip(
            (
                "prompt_token",
                "prompt_token_len",
                "embedding",
                "prompt_feat",
                "prompt_feat_len",
            ),
            reference,
        )
    )
    prompt["prompt_feat_len"] = np.array([17], np.int32)
    converted = codec.reference_from_prompt(prompt, mx=MX)
    assert converted[4].item() == 16
    assert prompt["prompt_feat_len"].item() == 17
    assert all(converted[i] is reference[i] for i in range(4))
    base = codec.prepare_stream(networks[0], prompt, mx=MX, nn=NN)
    assert base.prompt_frames == 16


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(extra=1),
        lambda p: p.pop("embedding"),
        lambda p: p.update(prompt_feat_len=np.array([0], np.int32)),
        lambda p: p.update(prompt_feat_len=np.array([16], np.int64)),
    ],
)
def test_offline_prompt_adapter_refuses_malformed_schema_and_lengths(mutation):
    prompt = dict(
        zip(
            (
                "prompt_token",
                "prompt_token_len",
                "embedding",
                "prompt_feat",
                "prompt_feat_len",
            ),
            prepared_reference(),
        )
    )
    mutation(prompt)
    with pytest.raises(codec.CodecError):
        codec.reference_from_prompt(prompt, mx=MX)


def test_contract_module_loading_does_not_import_device_runtime(monkeypatch):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "mlx_audio", "mlx_lm", "torch", "torchaudio"}:
            pytest.fail(f"native runtime imported: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    fresh_spec = importlib.util.spec_from_file_location(
        "stepaudio2_streaming_import_contract", HELPER
    )
    fresh = importlib.util.module_from_spec(fresh_spec)
    monkeypatch.setitem(sys.modules, fresh_spec.name, fresh)
    fresh_spec.loader.exec_module(fresh)
    assert fresh.FLOW_STEPS == 10


def test_native_pcm_clips_amplitude_and_refuses_nonfinite():
    raw = codec.pcm16_bytes(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], np.float32))
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype="<i2"), [-32767, -32767, 0, 32767, 32767]
    )
    with pytest.raises(codec.CodecError, match="finite mono"):
        codec.pcm16_bytes(np.array([[np.nan]], np.float32))


def test_explicit_eager_attention_keeps_current_before_history_without_mutating_layer():
    layer = SimpleNamespace(
        to_heads=lambda x: x[:, None, :, :],
        to_q=identity,
        to_k=identity,
        to_v=identity,
        q_norm=None,
        scale=1.0,
        inner_dim=2,
        proj=identity,
        _venus_attention_mode="unsupported-private-setting",
    )
    x = np.array([[[1.0, 2.0]]], np.float32)
    cache = np.array([[[[3.0, 4.0, 5.0, 6.0]]]], np.float32)
    saved = cache.copy()
    out, new = functions._dit_attention(layer, x, cache, MX, "eager")
    np.testing.assert_array_equal(
        new[0, 0], [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0]]
    )
    expected = MX.softmax(np.array([[[[5.0, 11.0]]]], np.float32), -1) @ np.array(
        [[[[1.0, 2.0], [5.0, 6.0]]]], np.float32
    )
    np.testing.assert_allclose(out, expected.reshape(1, 1, 2))
    np.testing.assert_array_equal(cache, saved)
    assert layer._venus_attention_mode == "unsupported-private-setting"


@pytest.mark.parametrize("history", [0, 5])
def test_explicit_sdpa_preserves_projection_normalization_scale_and_layout(history):
    class RecordedMX(MX):
        gpu = object()

    calls = []

    def attention(q, k, v, *, scale, mask, stream):
        calls.append((q.copy(), k.copy(), v.copy(), scale, mask, stream))
        return MX.softmax((q @ k.swapaxes(-1, -2)) * scale, -1) @ v

    RecordedMX.fast = SimpleNamespace(scaled_dot_product_attention=attention)
    layer = SimpleNamespace(
        num_heads=8,
        head_dim=64,
        inner_dim=512,
        scale=0.125,
        to_heads=lambda x: x.reshape(x.shape[0], x.shape[1], 8, 64).transpose(
            0, 2, 1, 3
        ),
        to_q=lambda x: x * 0.25,
        to_k=lambda x: x * 0.5,
        to_v=lambda x: x + 1,
        q_norm=lambda x: x * 0.75,
        k_norm=lambda x: x * 0.625,
        proj=lambda x: x * 2,
    )
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 3, 512)).astype(np.float32)
    cache = (
        None
        if not history
        else rng.normal(size=(2, 8, history, 128)).astype(np.float32)
    )
    expected, expected_cache = functions._dit_attention(layer, x, cache, MX, "eager")
    out, new = functions._dit_attention(layer, x, cache, RecordedMX, "sdpa")
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(new, expected_cache)
    q, k, v, scale, mask, stream = calls[0]
    assert q.shape == (2, 8, 3, 64) and k.shape == v.shape == (2, 8, 3 + history, 64)
    assert scale == 0.125 and mask is None and stream is RecordedMX.gpu
    if history:
        np.testing.assert_array_equal(k[:, :, 3:], cache[..., :64])
        np.testing.assert_array_equal(v[:, :, 3:], cache[..., 64:])


@pytest.mark.parametrize("mode", [None, "auto", "cuda"])
def test_invalid_attention_never_projects_or_falls_back(mode):
    with pytest.raises(ValueError, match="eager or sdpa"):
        functions._dit_attention(SimpleNamespace(), np.zeros((1, 1, 2)), None, MX, mode)


def test_ten_step_flow_uses_persistent_offset_noise_and_signed_cfg_without_reseeding(
    monkeypatch,
):
    calls = []

    def estimator(model, x, mu, t, spks, cond, cnn, att, mx, attention_mode="eager"):
        # Conditional current sequence followed by unconditional zeros is the
        # native CFG contract; no random noise is requested inside the steps.
        np.testing.assert_array_equal(mu[1], np.zeros_like(mu[1]))
        np.testing.assert_array_equal(spks[1], np.zeros_like(spks[1]))
        np.testing.assert_array_equal(cond[1], np.zeros_like(cond[1]))
        old = 0 if att is None else att.shape[3]
        calls.append((t.copy(), attention_mode, old))
        velocity = np.concatenate(
            [np.full_like(x[:1], 3), np.full_like(x[:1], -1)], axis=0
        )
        return (
            velocity,
            np.zeros((1, 2, 4, 2), np.float32),
            np.zeros((1, 2, 1, old + x.shape[2], 2), np.float32),
        )

    monkeypatch.setattr(functions, "dit_chunk", estimator)
    noise = np.arange(40, dtype=np.float32).reshape(1, 2, 20)
    saved = noise.copy()
    decoder = SimpleNamespace(
        rand_noise=noise, estimator=object(), inference_cfg_rate=0.7
    )
    mu, spks, cond = (
        np.ones((1, 2, 4), np.float32),
        np.ones((1, 2), np.float32),
        np.ones((1, 2, 4), np.float32),
    )
    first, cnn, att = functions.cfm_chunk(
        decoder, mu, spks, cond, 10, None, None, MX, "sdpa"
    )
    second, _, second_att = functions.cfm_chunk(
        decoder, mu, spks, cond, 10, cnn, att, MX, "sdpa"
    )
    np.testing.assert_allclose(first, noise[:, :, :4] + 5.8, rtol=1e-6)
    np.testing.assert_allclose(second, noise[:, :, 4:8] + 5.8, rtol=1e-6)
    assert len(calls) == 20 and all(mode == "sdpa" for _, mode, _ in calls)
    assert [offset for _, _, offset in calls] == [0] * 10 + [4] * 10
    assert calls[0][0][0] == calls[10][0][0] == 0
    assert att.shape[4] == 4 and second_att.shape[4] == 8
    assert decoder.rand_noise is noise
    np.testing.assert_array_equal(noise, saved)
    with pytest.raises(ValueError, match="ten flow"):
        functions.cfm_chunk(decoder, mu, spks, cond, 9, None, None, MX)
    too_long = np.ones((1, 2, 18), np.float32)
    with pytest.raises(ValueError, match="exhausted"):
        functions.cfm_chunk(decoder, too_long, spks, too_long, 10, cnn, att, MX)
    assert len(calls) == 20


def test_flow_crop_retains_native_first_prompt_last_100_on_distinct_axes():
    cache = {
        "estimator_att_cache": np.arange(160, dtype=np.float32).reshape(
            1, 1, 1, 1, 160, 1
        ),
        "conformer_att_cache": np.arange(160, dtype=np.float32).reshape(
            1, 1, 1, 160, 1
        ),
        "conformer_cnn_cache": np.ones((1, 2, 6), np.float32),
        "estimator_cnn_cache": np.ones((10, 1, 1, 4, 2), np.float32),
    }
    saved = {key: value.copy() for key, value in cache.items()}
    cropped = codec.crop_flow_cache(cache, 16, MX)
    expected = np.concatenate([np.arange(16), np.arange(60, 160)])
    np.testing.assert_array_equal(cropped["estimator_att_cache"].ravel(), expected)
    np.testing.assert_array_equal(cropped["conformer_att_cache"].ravel(), expected)
    assert cropped["conformer_cnn_cache"] is cache["conformer_cnn_cache"]
    caches_equal(cache, saved)

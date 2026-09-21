"""Qwen3-ASR live-input contracts, without downloaded weights."""

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.registry import classify_model, model_type_from_config
from mlx_audio.stt.models.qwen3_asr import ModelConfig
from mlx_audio.stt.models.qwen3_asr import Qwen3ASRModel as Model
from mlx_audio.stt.models.qwen3_asr.config import AudioEncoderConfig, TextConfig
from mlx_audio.utils import base_load_model


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode("utf-8", errors="replace")


def model():
    config = ModelConfig(
        audio_config=AudioEncoderConfig(
            num_mel_bins=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            output_dim=32,
            downsample_hidden_size=8,
        ),
        text_config=TextConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
        ),
        support_languages=["English", "Chinese", "Russian"],
    )
    result = Model(config)
    result._tokenizer = CharacterTokenizer()
    result._feature_extractor = object()
    result._eos_token_ids = lambda: {999}

    # Use the real session and tiny model with deterministic decoder output.
    def decode(audio, prefix, *, language, context, max_tokens):
        for token in result._tokenizer.encode("abc")[:max_tokens]:
            yield token
        yield 999

    result._streaming_decode = decode
    return result


def scripted_model(outputs):
    result = model()
    result.calls = []
    iterator = iter(outputs)

    def decode(audio, prefix, *, language, context, max_tokens):
        result.calls.append((audio.copy(), prefix, language, context, max_tokens))
        yield from result._tokenizer.encode(next(iterator))[:max_tokens]
        yield 999

    result._streaming_decode = decode
    return result


def drain(session, budget=1):
    text = []
    for _ in range(2000):
        text.extend(session.step(max_decode_tokens=budget))
        if session.done:
            assert session.step() == []
            return "".join(text)
    pytest.fail("session did not finish")


@pytest.mark.parametrize("name", ["Confucius4-R2T2", "Qwen3-ASR-1.7B", "renamed"])
def test_shared_loader_and_converter(tmp_path, name):
    from mlx_audio.convert import Domain, get_model_type
    from mlx_audio.stt.utils import MODEL_REMAPPING

    path = tmp_path / name
    path.mkdir()
    config = {"model_type": "qwen3_asr"}
    assert model_type_from_config(config) == "qwen3_asr"
    assert classify_model("qwen3_asr", name) == "stt"
    assert get_model_type(config, path, Domain.STT) == "qwen3_asr"
    tiny = model()
    with (
        patch("mlx_audio.utils.load_config", return_value=config),
        patch("mlx_audio.utils.load_weights", return_value={}),
        patch("mlx_audio.stt.models.qwen3_asr.Model", return_value=tiny) as factory,
    ):
        factory.post_load_hook.return_value = tiny
        assert base_load_model(path, "stt", MODEL_REMAPPING, lazy=True) is tiny
    assert factory.call_args.args[0].model_type == "qwen3_asr"


def test_wrapper_exposes_session_only_for_asr():
    from mlx_audio.stt.models.qwen3_asr import ForcedAlignerConfig
    from mlx_audio.stt.models.qwen3_asr import Model as Wrapper

    config = model().config
    asr = Wrapper(config)
    assert callable(asr.create_streaming_session)
    aligner = Wrapper(
        ForcedAlignerConfig(
            audio_config=config.audio_config,
            text_config=config.text_config,
            classify_num=16,
        )
    )
    assert not hasattr(aligner, "create_streaming_session")


@pytest.mark.parametrize("tail", [0, 17, 1279])
def test_committed_prefix_and_final_tail(tail):
    m = scripted_model(["aX", "bcX", "d."])
    session = m.create_streaming_session(
        language="English", context="names", chunk_size_sec=0.32, lookahead_sec=0
    )
    session.feed(np.zeros(5120))
    assert session.step(max_decode_tokens=1) == []
    assert session.step(max_decode_tokens=1) == []
    assert session.step(max_decode_tokens=1) == ["a"]
    session.feed(np.ones(5120))
    assert session.step(max_decode_tokens=4) == ["bc"]
    session.feed(np.full(tail, 2.0))
    session.close()
    session.close()
    assert drain(session) == "d."
    assert session.text == "abcd."
    assert [call[1] for call in m.calls] == ["", "a", "abc"]
    assert m.calls[-1][0].size == 10240 + tail
    assert m.calls[-1][3] == "names"
    with pytest.raises(RuntimeError):
        session.feed(np.zeros(1))


def test_backlogged_input_drains_one_chunk_at_a_time():
    m = scripted_model(["ab", "bc", "cd."])
    session = m.create_streaming_session(language="English", lookahead_sec=0)
    session.feed(np.arange(3 * 2560, dtype=np.float32))
    session.close()
    assert drain(session) == "abcd."
    assert [call[0].size for call in m.calls] == [2560, 5120, 7680]


def test_arbitrary_feed_sizes_and_initial_lookahead():
    m = scripted_model(["abcX", "d."])
    session = m.create_streaming_session(language="English")
    original = np.zeros(5119, dtype=np.float32)
    session.feed(original)
    original[:] = 1
    assert session.step() == []
    assert m.calls == []
    session.feed(np.zeros(1))
    assert session.step() == ["abc"]
    assert not m.calls[0][0].any()  # producer cannot mutate queued PCM
    session.close()
    assert drain(session) == "d."


def test_unfinished_unicode_is_not_committed():
    m = scripted_model(["你", "你", "你好"])
    session = m.create_streaming_session(
        language="Chinese", chunk_size_sec=0.32, lookahead_sec=0
    )
    session.feed(np.zeros(5120))
    assert session.step() == []  # held final UTF-8 byte makes the char incomplete
    session.feed(np.zeros(5120))
    assert session.step(max_decode_tokens=8) == []
    session.close()
    assert drain(session) == "你好"
    assert "\ufffd" not in session.text


def test_auto_language_hides_metadata_and_stop_marker():
    m = scripted_model(["language English<asr_text>Hello|ignored"])
    session = m.create_streaming_session()
    session.feed(np.zeros(100))
    session.close()
    assert drain(session) == "Hello"


def test_empty_silence_cancel_and_independent_sessions():
    m = scripted_model(["", ""])
    empty = m.create_streaming_session()
    empty.close()
    assert drain(empty) == ""
    assert not m.calls
    first = m.create_streaming_session(language="English")
    other = m.create_streaming_session(language="English")
    first.feed(np.zeros(5120))
    assert first.step() == []
    assert not first.done
    first.close()
    assert drain(first) == ""
    assert other.text == ""
    other.feed(np.zeros(100))
    other.cancel()
    assert other.done and other._queued == 0
    assert other.step() == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 1},
        {"chunk_size_sec": 0},
        {"chunk_size_sec": float("nan")},
        {"chunk_size_sec": 3},
        {"lookahead_sec": -1},
        {"lookahead_sec": float("inf")},
        {"unfixed_token_num": -1},
        {"unfixed_token_num": 0.5},
        {"max_audio_seconds": 0},
        {"max_audio_seconds": 0.1},
        {"language": "Unknown"},
    ],
)
def test_invalid_session_options(kwargs):
    with pytest.raises(ValueError):
        model().create_streaming_session(**kwargs)


def test_input_limits_and_budget_validation():
    session = model().create_streaming_session(max_audio_seconds=1)
    for samples in [np.zeros((2, 2)), np.array([np.nan]), np.array([np.inf])]:
        with pytest.raises(ValueError):
            session.feed(samples)
    session.feed(np.zeros(16000))
    with pytest.raises(BufferError):
        session.feed(np.zeros(1))
    for budget in [0, -1, 1.5]:
        with pytest.raises(ValueError):
            session.step(max_decode_tokens=budget)


def test_decode_budget_preserves_generator_state():
    m = scripted_model(["abcdef"])
    session = m.create_streaming_session(language="English")
    session.feed(np.zeros(1))
    session.close()
    for _ in range(6):
        assert session.step(max_decode_tokens=1) == []
        assert not session.done
    assert session.step(max_decode_tokens=1) == ["abcdef"]
    assert session.done and len(m.calls) == 1


def test_network_decode_uses_audio_and_text_prefix():
    m = model()
    # Exercise the production generator with controlled token generation.
    del m._streaming_decode
    m._preprocess_audio = lambda audio: (mx.zeros((1, 16, 8)), None, 1)
    m._build_prompt = lambda *args: mx.array([[1, 2]])
    m.get_audio_features = lambda *args: mx.zeros((1, 32))
    m._build_inputs_embeds = lambda ids, features: mx.zeros((1, ids.shape[1], 32))
    with patch("mlx_audio.lm.generate.generate_step") as generate:
        generate.return_value = iter([(65, None), (999, None), (66, None)])
        actual = list(
            m._streaming_decode(
                np.zeros(100), "hi", language="English", context="", max_tokens=4
            )
        )
    assert actual == [65, 999]
    assert generate.call_args.kwargs["prompt"].tolist() == [1, 2, ord("h"), ord("i")]
    assert generate.call_args.kwargs["input_embeddings"].shape == (4, 32)


def test_short_final_audio_has_a_frontend_frame():
    m = model()
    del m._streaming_decode
    seen = []

    def preprocess(audio):
        seen.append(audio)
        return mx.zeros((1, 16, 8)), None, 1

    m._preprocess_audio = preprocess
    m._build_prompt = lambda *args: mx.array([[1, 2]])
    m.get_audio_features = lambda *args: mx.zeros((1, 32))
    m._build_inputs_embeds = lambda ids, features: mx.zeros((1, ids.shape[1], 32))
    with patch(
        "mlx_audio.lm.generate.generate_step",
        return_value=iter([(999, None)]),
    ):
        assert list(
            m._streaming_decode(
                np.array([0.5]), "", language="English", context="", max_tokens=1
            )
        ) == [999]
    assert seen[0].shape == (1280,)
    assert seen[0][0] == 0.5
    assert not seen[0][1:].any()


def test_feed_and_close_from_producer_thread():
    from threading import Event, Thread

    m = model()
    session = m.create_streaming_session(language="English", lookahead_sec=0)
    first_ready = Event()
    continue_feed = Event()

    def producer():
        session.feed(np.zeros(2560))
        first_ready.set()
        continue_feed.wait(5)
        session.feed(np.zeros(17))
        session.close()

    thread = Thread(target=producer)
    thread.start()
    assert first_ready.wait(5)
    first = session.step(max_decode_tokens=4)
    continue_feed.set()
    thread.join(5)
    assert not thread.is_alive()
    assert "".join(first) + drain(session) == "aabc"


def test_larger_unfixed_window_still_commits_before_close():
    m = scripted_model(["abcdefghijklmnop", "ijklmnop"])
    session = m.create_streaming_session(language="English", unfixed_token_num=8)
    session.feed(np.zeros(5120))
    deltas = []
    for _ in range(4):
        deltas.extend(session.step(max_decode_tokens=4))
    assert deltas == ["abcdefgh"]
    assert not session.done
    session.close()
    assert drain(session) == "ijklmnop"

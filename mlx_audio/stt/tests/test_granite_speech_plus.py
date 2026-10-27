"""Weight-free tests for granite-speech-4.1-2b-plus support.

The plus checkpoint concatenates intermediate Conformer layer outputs onto the
final encoder output (``cat_hidden_layers``), expects a system turn plus a
space between the audio placeholder and the instruction, and emits
``[Speaker N]:`` / ``[T:N]`` tags that are parsed into the repo's ``segments``
schema.
"""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_audio.lm.models.rope_utils import YarnRoPE
from mlx_audio.lm.sample_utils import make_repetition_penalty
from mlx_audio.stt.generate import save_as_json
from mlx_audio.stt.models.granite_speech.config import (
    EncoderConfig,
    ModelConfig,
    ProjectorConfig,
    TextConfig,
)
from mlx_audio.stt.models.granite_speech.granite_speech import (
    PLUS_SYSTEM_PROMPT,
    TASK_PROMPTS,
    ConformerAttention,
    CTCEncoder,
    EncoderProjector,
    Model,
    StructuredTranscriptError,
    UnsupportedTranscriptionTask,
    _make_repetition_penalty,
    _parse_saa,
    _parse_timestamps,
    _resolve_prompt,
)

HIDDEN_DIM = 8


def _tiny_encoder_config(**overrides):
    params = dict(
        input_dim=4,
        num_layers=2,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dim_head=4,
        output_dim=4,
        context_size=8,
        max_pos_emb=16,
    )
    params.update(overrides)
    return EncoderConfig(**params)


@pytest.mark.parametrize(
    ("cat_hidden_layers", "width"), [(None, 1), ([], 1), ([1], 2), ([0, 1], 3)]
)
def test_encoder_output_width(cat_hidden_layers, width):
    encoder = CTCEncoder(_tiny_encoder_config(cat_hidden_layers=cat_hidden_layers))
    assert encoder(mx.zeros((1, 8, 4))).shape == (1, 8, width * HIDDEN_DIM)


def test_encoder_exports_match_reference():
    # Layer 2 is the mid layer of four: HF adds the CTC term in place, so its
    # exported state carries the injection while layer 1 is exported raw.
    encoder = CTCEncoder(
        _tiny_encoder_config(num_layers=4, cat_hidden_layers=[0, 1, 2])
    )
    x = mx.random.normal((1, 8, 4), key=mx.random.key(0))

    def layer(idx, hidden):
        return encoder.layers[idx - 1](hidden, attention_dists=encoder._attention_dists)

    h0 = encoder.input_linear(x)
    h1 = layer(1, h0)
    h2 = layer(2, h1)
    h2 = h2 + encoder.out_mid(mx.softmax(encoder.out(h2), axis=-1))
    h4 = layer(4, layer(3, h2))

    expected = mx.concatenate([h0, h1, h2, h4], axis=-1)
    assert mx.allclose(encoder(x), expected, atol=1e-5).item()


def test_projector_consumes_concatenated_features():
    config = ModelConfig(
        encoder_config=_tiny_encoder_config(cat_hidden_layers=[1]),
        projector_config=ProjectorConfig(
            hidden_size=HIDDEN_DIM,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            encoder_hidden_size=2 * HIDDEN_DIM,
        ),
        text_config=TextConfig(hidden_size=12),
    )
    out = EncoderProjector(config)(
        CTCEncoder(config.encoder_config)(mx.zeros((1, 8, 4)))
    )
    assert out.shape == (1, config.window_size // config.downsample_rate, 12)


def _tiny_plus_model(**text_config):
    return Model(
        ModelConfig(
            model_type="granite_speech_plus",
            encoder_config=_tiny_encoder_config(cat_hidden_layers=[1]),
            projector_config=ProjectorConfig(
                hidden_size=HIDDEN_DIM,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                encoder_hidden_size=2 * HIDDEN_DIM,
            ),
            text_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "tie_word_embeddings": True,
                **text_config,
            },
        )
    )


@pytest.mark.parametrize(
    ("rope_parameters", "rope_class"),
    [
        # October 2026 checkpoint (v2610).
        (
            {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
                "rope_theta": 10000,
            },
            YarnRoPE,
        ),
        # April 2026 checkpoint (v2604).
        ({"rope_type": "default", "rope_theta": 10000}, nn.RoPE),
    ],
)
def test_decoder_rope_follows_rope_parameters(rope_parameters, rope_class):
    model = _tiny_plus_model(rope_parameters=rope_parameters)
    assert type(model.language_model.model.layers[0].self_attn.rope) is rope_class


def test_sanitize_unnests_v2610_decoder_keys():
    model = _tiny_plus_model()
    native = dict(tree_flatten(model.parameters()))
    nested = {
        k.replace("language_model.model.", "language_model.model.model.", 1): v
        for k, v in native.items()
    }

    sanitized = Model.sanitize(nested)

    assert sanitized.keys() == native.keys()
    assert Model.sanitize(sanitized).keys() == native.keys()
    model.load_weights(list(sanitized.items()), strict=True)


def test_non_aligned_attention_preserves_bfloat16():
    config = _tiny_encoder_config()
    attention = ConformerAttention(config)
    attention.set_dtype(mx.bfloat16)
    x = mx.zeros((1, config.context_size + 1, config.hidden_dim), dtype=mx.bfloat16)

    output = attention(x, CTCEncoder(config)._attention_dists)

    assert output.dtype == mx.bfloat16


@pytest.mark.parametrize(
    ("model_type", "encoder_config", "expected"),
    [
        ("granite_speech_plus", None, True),
        # mlx_audio.convert rewrites model_type; cat_hidden_layers survives.
        ("granite_speech", {"cat_hidden_layers": [3]}, True),
        ("granite_speech", None, False),
    ],
)
def test_is_plus(model_type, encoder_config, expected):
    config = ModelConfig(model_type=model_type, encoder_config=encoder_config)
    assert Model.is_plus.fget(SimpleNamespace(config=config)) is expected


@pytest.mark.parametrize(
    ("is_plus", "encoder_dtype", "expected_dtype"),
    [
        (True, mx.bfloat16, mx.bfloat16),
        # 4.0/4.1 keep float32 activations regardless of the weight dtype.
        (False, mx.bfloat16, mx.float32),
    ],
)
def test_audio_feature_dtype(is_plus, encoder_dtype, expected_dtype):
    class IdentityEncoder:
        input_linear = SimpleNamespace(weight=mx.zeros((1,), dtype=encoder_dtype))

        def __call__(self, features):
            return features

    stub_model = SimpleNamespace(
        encoder=IdentityEncoder(), projector=lambda x: x, is_plus=is_plus
    )

    output = Model.get_audio_features(stub_model, mx.zeros((1, 2, 4)))

    assert output.dtype == expected_dtype


class StubTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in chat)

    def encode(self, text):
        self.rendered = text
        return [0]


@pytest.mark.parametrize(
    ("is_plus", "system_prompt", "expected"),
    [
        (True, None, "<user><|audio|><|audio|> do it</user>"),
        (False, None, "<user><|audio|><|audio|>  do it</user>"),
        (True, "sys", "<system>sys</system><user><|audio|><|audio|> do it</user>"),
    ],
)
def test_build_prompt(is_plus, system_prompt, expected):
    tokenizer = StubTokenizer()
    stub_model = SimpleNamespace(_tokenizer=tokenizer, is_plus=is_plus)
    Model._build_prompt(stub_model, 2, "  do it", system_prompt=system_prompt)
    assert tokenizer.rendered == expected


def _stream_kwargs(is_plus=True, **generate_kwargs):
    """Return what generate() forwards to _stream_generate."""
    forwarded = {}

    def stream_generate(audio, **kwargs):
        forwarded.update(kwargs)
        return iter(())

    stub_model = SimpleNamespace(
        is_plus=is_plus,
        config=SimpleNamespace(model_type="granite_speech"),
        _stream_generate=stream_generate,
    )
    assert (
        list(Model.generate(stub_model, mx.zeros((1,)), stream=True, **generate_kwargs))
        == []
    )
    return forwarded


@pytest.mark.parametrize(
    ("is_plus", "expected"), [(True, PLUS_SYSTEM_PROMPT), (False, None)]
)
def test_system_turn_defaults_for_plus_only(is_plus, expected):
    assert _stream_kwargs(is_plus)["system_prompt"] == expected


def test_word_timestamps_selects_timestamp_task_on_plus_only():
    assert (
        _stream_kwargs(True, word_timestamps=True)["prompt"]
        == TASK_PROMPTS["timestamps"]
    )
    assert _stream_kwargs(False, word_timestamps=True)["prompt"] == TASK_PROMPTS["asr"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("task", ["saa", "timestamps"])
def test_base_checkpoint_rejects_rich_tasks(task, stream):
    base = SimpleNamespace(
        is_plus=False, config=SimpleNamespace(model_type="granite_speech")
    )
    with pytest.raises(UnsupportedTranscriptionTask, match="requires.*Plus"):
        Model.generate(base, mx.zeros((1,)), task=task, stream=stream)


def test_rich_task_prompts_are_fixed():
    assert _resolve_prompt("saa", None, "en") == TASK_PROMPTS["saa"]
    with pytest.raises(ValueError, match="prompt cannot override"):
        _resolve_prompt("timestamps", "custom", None)
    with pytest.raises(ValueError, match="Unknown task"):
        _resolve_prompt("diarize", None, None)


def test_parse_saa():
    segments = _parse_saa(
        "[Speaker 1]: Hello how are you [Speaker 2]: I'm fine [Speaker 1]: Great"
    )
    assert segments == [
        {"speaker_id": 1, "text": "Hello how are you"},
        {"speaker_id": 2, "text": "I'm fine"},
        {"speaker_id": 1, "text": "Great"},
    ]


def test_parse_timestamps_rollover_and_silence():
    (segment,) = _parse_timestamps(
        "hello [T:995] world [T:012] _ [T:100] again [T:150] more [T:1234]"
    )
    words = segment["words"]
    assert [w["word"] for w in words] == ["hello", "world", "again", "more"]
    assert [w["end"] for w in words] == pytest.approx([9.95, 10.12, 11.50, 12.34])
    # the dropped silence still advances the next word's start
    assert words[2]["start"] == pytest.approx(11.00)
    assert (segment["start"], segment["text"]) == (0.0, "hello world again more")


@pytest.mark.parametrize(
    ("parser", "text"),
    [
        (_parse_saa, "plain transcription without tags"),
        (_parse_saa, "intro [Speaker 1]: tagged turn"),
        (_parse_saa, "[Speaker 2]: out of order"),
        (_parse_saa, "[Speaker 1]:"),
        (_parse_timestamps, "plain transcription without tags"),
        (_parse_timestamps, "hello world [T:100]"),
        (_parse_timestamps, "hello [T:50] trailing words"),
        (_parse_timestamps, "first [T:1234] second [T:1200]"),
    ],
)
def test_parsers_reject_malformed_output(parser, text):
    with pytest.raises(StructuredTranscriptError) as exc_info:
        parser(text)
    assert exc_info.value.raw_text == text


@pytest.mark.parametrize(
    ("task", "pieces", "expected_segments"),
    [
        ("asr", ["hello ", "world"], []),
        ("saa", ["[Speaker 1]: ", "hello"], [{"speaker_id": 1, "text": "hello"}]),
        (
            "timestamps",
            ["hello ", "[T:50]"],
            [
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 0.5,
                    "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                }
            ],
        ),
    ],
)
def test_generate_returns_parsed_segments(monkeypatch, task, pieces, expected_segments):
    import mlx_audio.lm.generate as lm_generate

    tokenizer = SimpleNamespace(
        eos_token_id=99, decode=lambda ids, **kwargs: "".join(pieces[i] for i in ids)
    )
    monkeypatch.setattr(
        lm_generate,
        "generate_step",
        lambda **kwargs: iter([(0, None), (1, None), (99, None)]),
    )
    stub_model = SimpleNamespace(
        is_plus=True,
        config=SimpleNamespace(model_type="granite_speech_plus"),
        _tokenizer=tokenizer,
        _load_audio=lambda audio: audio,
        _extract_features=lambda audio: (mx.zeros((1, 1, 160)), 1),
        get_audio_features=lambda features: mx.zeros((1, 1, 4)),
        _build_prompt=lambda *args, **kwargs: mx.array([0]),
        _build_inputs_embeds=lambda *args, **kwargs: mx.zeros((1, 1, 4)),
    )

    result = Model.generate(stub_model, mx.zeros((1,)), task=task)

    assert result.text == "".join(pieces)
    assert result.segments == expected_segments


def _log_softmax(logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def test_repetition_penalty_scales_log_probs_of_seen_tokens():
    logits = mx.array([[4.0, 2.0, 0.0, -1.0]])
    (processor,) = _make_repetition_penalty(2.0)

    out = processor(mx.array([0, 2, 2]), logits)

    expected = _log_softmax(logits) * mx.array([[2.0, 1.0, 2.0, 1.0]])
    assert mx.allclose(out, expected).item()


def test_repetition_penalty_keeps_confident_repeat():
    # The same penalty on raw logits flips a confident repeat to the runner-up.
    logits = mx.array([[8.0, 5.0, 0.0]])
    seen = mx.array([0])

    raw = make_repetition_penalty(2.0, context_size=100)
    (log_prob,) = _make_repetition_penalty(2.0)

    assert mx.argmax(raw(seen, mx.array(logits)), axis=-1).item() == 1
    assert mx.argmax(log_prob(seen, logits), axis=-1).item() == 0


def test_repetition_penalty_context_size_limits_window():
    logits = mx.array([[1.0, 1.0, 1.0]])
    (processor,) = _make_repetition_penalty(2.0, context_size=1)

    out = processor(mx.array([0, 1, 2]), logits)

    expected = _log_softmax(logits) * mx.array([[1.0, 1.0, 2.0]])
    assert mx.allclose(out, expected).item()


def test_repetition_penalty_disabled_and_validated():
    assert _make_repetition_penalty(None) == []
    for penalty in (0, -1.0, "2"):
        with pytest.raises(ValueError):
            _make_repetition_penalty(penalty)


@pytest.mark.parametrize("stream", [False, True])
def test_generate_applies_repetition_penalty_over_prompt(monkeypatch, stream):
    import mlx_audio.lm.generate as lm_generate

    captured = {}

    def fake_generate_step(**kwargs):
        captured.update(kwargs)
        return iter([(99, None)])

    monkeypatch.setattr(lm_generate, "generate_step", fake_generate_step)
    stub_model = SimpleNamespace(
        is_plus=True,
        config=SimpleNamespace(model_type="granite_speech_plus"),
        _tokenizer=SimpleNamespace(eos_token_id=99, decode=lambda ids, **kwargs: ""),
        _load_audio=lambda audio: audio,
        _extract_features=lambda audio: (mx.zeros((1, 1, 160)), 1),
        get_audio_features=lambda features: mx.zeros((1, 1, 4)),
        _build_prompt=lambda *args, **kwargs: mx.array([0, 1]),
        _build_inputs_embeds=lambda *args, **kwargs: mx.zeros((1, 2, 4)),
    )

    if stream:
        list(Model._stream_generate(stub_model, mx.zeros((1,)), repetition_penalty=2.0))
    else:
        Model.generate(stub_model, mx.zeros((1,)), repetition_penalty=2.0)

    (processor,) = captured["logits_processors"]
    logits = mx.array([[1.0, 1.0, 1.0]])
    out = processor(captured["prompt"], logits)
    expected = _log_softmax(logits) * mx.array([[2.0, 2.0, 1.0]])
    assert mx.allclose(out, expected).item()


def test_json_preserves_untimed_speaker_segments(tmp_path):
    text = "[Speaker 1]: hi [Speaker 2]: hey"
    output = SimpleNamespace(text=text, segments=_parse_saa(text))

    save_as_json(output, str(tmp_path / "speakers"))
    saved = json.loads((tmp_path / "speakers.json").read_text())
    assert saved["segments"] == [
        {"text": "hi", "speaker_id": 1},
        {"text": "hey", "speaker_id": 2},
    ]

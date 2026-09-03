"""Focused tests for VibeVoice-ASR-Streaming without checkpoint downloads."""

import inspect
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.vibevoice_asr import DETECTION_HINTS, Model, ModelConfig


class _FakeTokenizer:
    eos_token_id = 5
    unk_token_id = None
    chat_template = None

    _ids = {
        "<|object_ref_start|>": 1,
        "<|object_ref_end|>": 2,
        "<|box_start|>": 3,
        "<|text_chunk_end|>": 4,
    }

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token)

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return [6, 7]

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        pieces = {8: "hello", 9: " world"}
        return "".join(pieces.get(int(token), "") for token in tokens)


class _FakeLanguageModel(nn.Module):
    """Small LM that emits ``hello world`` and then the chunk delimiter."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = [None]
        self.phase = 0

    def __call__(self, inputs=None, cache=None, input_embeddings=None):
        del inputs
        batch, length, _ = input_embeddings.shape
        if cache is not None:
            keys = mx.zeros((batch, 1, length, 1))
            cache[0].update_and_fetch(keys, keys)

        if length > 1:
            token = 8
            self.phase = 1
        elif self.phase == 1:
            token = 9
            self.phase = 2
        elif self.phase == 2:
            token = 4
            self.phase = 3
        else:
            token = 0
            self.phase = 0

        row = mx.full((16,), -100.0).at[token].add(200.0)
        return mx.broadcast_to(row, (batch, length, 16))


def _streaming_model() -> Model:
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.language_model = _FakeLanguageModel()
    model.tokenizer = _FakeTokenizer()
    model.sample_rate = 8
    model.speech_tok_compress_ratio = 4
    model.normalize_audio = False
    model.chunk_frames = 2
    model.lookahead_frames = 1
    model._speech_start_id = 1
    model._speech_end_id = 2
    model._speech_pad_id = 3
    model._text_chunk_end_id = 4
    return model


def test_streaming_config_fields_are_parsed():
    config = ModelConfig.from_dict(
        {
            "target_sample_rate": 16_000,
            "speech_tok_compress_ratio": 2_000,
            "normalize_audio": False,
            "chunk_frames": 22,
            "lookahead_frames": 4,
        }
    )
    assert config.sample_rate == 16_000
    assert config.speech_tok_compress_ratio == 2_000
    assert config.normalize_audio is False
    assert config.chunk_frames == 22
    assert config.lookahead_frames == 4


def test_converter_detection_includes_streaming_architecture():
    assert "vibevoice" in DETECTION_HINTS["model_type_aliases"]
    assert "VibeVoiceForASRStreamingTraining" in DETECTION_HINTS["architectures"]


def test_post_load_reads_streaming_processor_metadata(tmp_path, monkeypatch):
    processor_config = {
        "target_sample_rate": 24_000,
        "speech_tok_compress_ratio": 3_200,
        "normalize_audio": False,
        "chunk_frames": 22,
        "lookahead_frames": 4,
    }
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(processor_config), encoding="utf-8"
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )

    model = _streaming_model()
    model.chunk_frames = None
    model.lookahead_frames = None
    Model.post_load_hook(model, tmp_path)

    assert model.is_streaming_model
    assert model.streaming_chunk_samples == 70_400
    assert model.streaming_window_samples == 83_200
    assert model.normalize_audio is False


def test_streaming_windows_keep_lookahead_and_pad_final_chunk():
    model = _streaming_model()
    audio = mx.array(np.arange(17, dtype=np.float32))[None, :]
    chunks = list(model._iter_streaming_audio_chunks(audio))

    assert [(index, total) for index, total, _ in chunks] == [
        (0, 3),
        (1, 3),
        (2, 3),
    ]
    assert [chunk.shape[-1] for _, _, chunk in chunks] == [12, 12, 12]
    np.testing.assert_array_equal(np.array(chunks[0][2][0]), np.arange(12))
    np.testing.assert_array_equal(np.array(chunks[1][2][0, :9]), np.arange(8, 17))
    np.testing.assert_array_equal(np.array(chunks[2][2][0, :1]), np.array([16]))
    np.testing.assert_array_equal(np.array(chunks[2][2][0, 1:]), np.zeros(11))


def test_streaming_step_reuses_cache_and_appends_delimiter():
    model = _streaming_model()
    state = model.init_streaming_state(context_info="VibeVoice")
    initial_offset = state["cache"][0].offset

    text, state = model.streaming_generate_step(
        mx.zeros((1, 3, 4)), state, max_new_tokens=8
    )
    first_offset = state["cache"][0].offset
    second_text, state = model.streaming_generate_step(
        mx.zeros((1, 3, 4)), state, max_new_tokens=8
    )

    assert text == "hello world"
    assert second_text == "hello world"
    assert initial_offset == 2
    # speech start + 3 audio frames + speech end + 2 text tokens + delimiter
    assert first_offset - initial_offset == 8
    assert state["cache"][0].offset - first_offset == 8
    assert state["generation_tokens"] == 4


def test_streaming_generate_emits_one_result_per_trained_chunk():
    model = _streaming_model()
    model.encode_speech = lambda audio, verbose=False: mx.zeros(
        (1, audio.shape[-1] // model.speech_tok_compress_ratio, 4)
    )

    chunks = list(
        model.streaming_generate(
            np.zeros(17, dtype=np.float32), max_new_tokens_per_chunk=8
        )
    )

    assert chunks == [
        (0, 3, "hello world"),
        (1, 3, "hello world"),
        (2, 3, "hello world"),
    ]


def test_generate_uses_streaming_protocol_for_streaming_checkpoint():
    model = _streaming_model()
    model.encode_speech = lambda audio, verbose=False: mx.zeros(
        (1, audio.shape[-1] // model.speech_tok_compress_ratio, 4)
    )

    result = model.generate(np.zeros(17, dtype=np.float32), max_tokens_per_chunk=8)

    assert result.text == "hello worldhello worldhello world"
    assert len(result.segments) == 3
    assert result.prompt_tokens == 2
    assert result.generation_tokens == 6
    assert result.total_tokens == 8


def test_generic_cli_chunk_default_cannot_override_trained_geometry():
    # The shared CLI has a 30-second `chunk_duration` default for offline ASR.
    # VibeVoice streaming must continue to use the checkpoint's 22-frame chunk.
    assert "chunk_duration" not in inspect.signature(Model.generate).parameters
    assert "streaming_chunk_duration" in inspect.signature(Model.generate).parameters


def test_streaming_checkpoint_sanitizes_existing_weight_layout():
    weights = {
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 4)),
        "model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight": mx.zeros(
            (2, 1, 3)
        ),
        "model.acoustic_tokenizer.decoder.head.conv.conv.weight": mx.zeros((1, 1, 3)),
    }

    sanitized = Model.sanitize(weights)

    assert "language_model.model.layers.0.self_attn.q_proj.weight" in sanitized
    conv = sanitized["acoustic_tokenizer.encoder.downsample_layers.0.conv.weight"]
    assert conv.shape == (2, 3, 1)
    assert not any("decoder" in key for key in sanitized)

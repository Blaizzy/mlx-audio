import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx


class _FakeTokenizer:
    sot = 1
    eot = 99
    sot_sequence = (1, 2)
    sot_sequence_including_notimestamps = (1, 2, 3)
    no_speech = None

    def decode(self, tokens):
        return f"token-{tokens[0]}" if tokens else ""


class _FakeModel:
    def __init__(self):
        self.dims = SimpleNamespace(
            n_audio_ctx=3,
            n_audio_state=4,
            n_text_ctx=8,
            n_vocab=120,
        )
        self.tokenizer = _FakeTokenizer()

    def get_tokenizer(self, language, task):
        return self.tokenizer


def _load_whisper_decoding(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    models_dir = repo_root / "mlx_audio" / "stt" / "models"
    whisper_dir = models_dir / "whisper"

    models_pkg = types.ModuleType("mlx_audio.stt.models")
    models_pkg.__path__ = [str(models_dir)]
    whisper_pkg = types.ModuleType("mlx_audio.stt.models.whisper")
    whisper_pkg.__path__ = [str(whisper_dir)]

    monkeypatch.setitem(sys.modules, "mlx_audio.stt.models", models_pkg)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.models.whisper", whisper_pkg)

    module_name = "mlx_audio.stt.models.whisper.decoding"
    spec = importlib.util.spec_from_file_location(
        module_name, whisper_dir / "decoding.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_best_of_expands_tokens_and_audio_features(monkeypatch):
    decoding = _load_whisper_decoding(monkeypatch)

    mel = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)
    best_of = 3

    def fake_main_loop(self, audio_features, tokens):
        assert tokens.shape == (6, 3)
        assert tokens.tolist() == [list(self.initial_tokens)] * 6

        expected_audio_features = mx.repeat(mel, best_of, axis=0)
        assert audio_features.shape == expected_audio_features.shape
        assert audio_features.tolist() == expected_audio_features.tolist()

        sampled_tokens = mx.array([[10], [11], [12], [13], [14], [15]])
        tokens = mx.concatenate([tokens, sampled_tokens], axis=-1)
        sum_logprobs = mx.array([0.0, 3.0, 1.0, 0.5, 2.0, 4.0])
        no_speech_probs = mx.arange(tokens.shape[0], dtype=mx.float32)
        return tokens, sum_logprobs, no_speech_probs

    monkeypatch.setattr(decoding.DecodingTask, "_main_loop", fake_main_loop)

    options = decoding.DecodingOptions(
        language="en",
        temperature=0.7,
        best_of=best_of,
        sample_len=1,
        suppress_blank=False,
        suppress_tokens=(),
        without_timestamps=True,
        fp16=False,
    )

    results = decoding.DecodingTask(_FakeModel(), options).run(mel)

    assert [result.tokens for result in results] == [[11], [15]]
    assert [result.text for result in results] == ["token-11", "token-15"]
    assert [result.no_speech_prob for result in results] == [0.0, 3.0]
    assert [result.audio_features.tolist() for result in results] == mel.tolist()


def test_greedy_decoder_logprobs_broadcast_across_batch(monkeypatch):
    decoding = _load_whisper_decoding(monkeypatch)

    decoder = decoding.GreedyDecoder(temperature=0.0, eot=99)
    tokens = mx.array([[1, 2], [1, 2], [1, 2]])
    logits = mx.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 1.0, 0.0],
            [0.0, 5.0, 1.0],
        ]
    )
    sum_logprobs = mx.zeros(3)

    tokens, completed, sum_logprobs = decoder.update(tokens, logits, sum_logprobs)

    assert tokens.tolist() == [[1, 2, 2], [1, 2, 0], [1, 2, 1]]
    assert completed.item() is False
    assert sum_logprobs.shape == (3,)


def test_timestamp_rules_logprobs_broadcast_across_batch(monkeypatch):
    decoding = _load_whisper_decoding(monkeypatch)
    tokenizer = SimpleNamespace(timestamp_begin=4, no_timestamps=None)
    logit_filter = decoding.ApplyTimestampRules(
        tokenizer,
        sample_begin=2,
        max_initial_timestamp_index=None,
    )

    logits = mx.zeros((3, 8))
    tokens = mx.array([[1, 2], [1, 2], [1, 2]])

    filtered = logit_filter.apply(logits, tokens)

    assert filtered.shape == logits.shape

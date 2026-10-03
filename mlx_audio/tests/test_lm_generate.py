import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_audio.lm.generate import StreamingDetokenizer, generate_step, stream_generate

VOCAB = 17
EOS = 5


class _Cache:
    state = []


class CycleModel(nn.Module):
    """Emits token (t + 1) % VOCAB, so a prompt of EOS-1 produces EOS first."""

    layers = [object()]

    def make_cache(self):
        return [_Cache()]

    def __call__(self, tokens, cache=None, input_embeddings=None):
        del cache, input_embeddings
        return mx.eye(VOCAB)[(tokens + 1) % VOCAB]


class Tok:
    eos_token_ids = {EOS}
    eos_token_id = EOS
    bos_token = None
    clean_up_tokenization_spaces = False

    def encode(self, text, **kwargs):
        return [1]

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


def responses(prompt, **kwargs):
    return list(stream_generate(CycleModel(), Tok(), mx.array(prompt), **kwargs))


class IntEosTok(Tok):
    """transformers>=5 exposes eos_token_ids as a single int, not a collection."""

    eos_token_ids = EOS


class NoEosTok(Tok):
    eos_token_ids = None
    eos_token_id = None


def test_final_response_carries_eos_token_and_stop_reason():
    out = responses([EOS - 1], max_tokens=10)
    assert out, "expected at least the terminal response"
    assert out[-1].finish_reason == "stop"
    assert out[-1].token == EOS


def test_first_token_eos_still_yields_a_final_response():
    """A caller reading finish_reason off the last response must get one."""
    out = responses([EOS - 1], max_tokens=10)
    assert len(out) == 1
    assert out[-1].finish_reason == "stop"


def test_length_finish_reason_when_eos_never_reached():
    out = responses([EOS + 1], max_tokens=3)
    assert out[-1].finish_reason == "length"
    assert out[-1].token != EOS


def test_eos_token_is_not_a_duplicate_of_the_previous_token():
    """Re-emitting the last audio code instead of EOS shifts codec framing."""
    out = responses([EOS - 3], max_tokens=10)
    assert out[-1].token == EOS
    if len(out) > 1:
        assert out[-1].token != out[-2].token


@pytest.mark.parametrize("max_tokens", [1, 2, 5])
def test_generate_step_respects_max_tokens(max_tokens):
    toks = [
        int(t)
        for t, _ in generate_step(mx.array([1, 2]), CycleModel(), max_tokens=max_tokens)
    ]
    assert len(toks) == max_tokens


def test_generate_step_negative_max_tokens_is_unbounded():
    stream = generate_step(mx.array([1, 2]), CycleModel(), max_tokens=-1)
    produced = [int(pair[0]) for pair, _ in zip(stream, range(40))]
    assert len(produced) == 40


@pytest.mark.parametrize(
    ("pieces", "tokens", "expected"),
    [
        (
            {1: b"hello ", 2: b"\xe4", 3: b"\xb8", 4: b"\x96"},
            [1, 2, 3, 4],
            "hello \u4e16",
        ),
        (
            {1: b"hi", 2: b"\n\xe4\xb8", 3: b"\x96"},
            [1, 2, 3],
            "hi\n\u4e16",
        ),
    ],
)
def test_streaming_detokenizer_buffers_split_utf8_codepoints(pieces, tokens, expected):
    class ByteTokenizer:
        clean_up_tokenization_spaces = False

        def decode(self, token_ids, **kwargs):
            del kwargs
            encoded = b"".join(pieces[token_id] for token_id in token_ids)
            return encoded.decode("utf-8", errors="replace")

    detokenizer = StreamingDetokenizer(ByteTokenizer())
    deltas = []
    for token in tokens:
        detokenizer.add_token(token)
        deltas.append(detokenizer.last_segment)
    detokenizer.finalize()
    deltas.append(detokenizer.last_segment)

    assert "".join(deltas) == expected
    assert "\ufffd" not in "".join(deltas)
    assert detokenizer.text == expected


def test_streaming_detokenizer_last_segment_decodes_once():
    class CountingTokenizer:
        clean_up_tokenization_spaces = False

        def __init__(self):
            self.decode_calls = 0

        def decode(self, token_ids, **kwargs):
            del kwargs
            self.decode_calls += 1
            return "".join(str(token_id) for token_id in token_ids)

    tokenizer = CountingTokenizer()
    detokenizer = StreamingDetokenizer(tokenizer)
    detokenizer.add_token(1)

    assert detokenizer.last_segment == "1"
    assert tokenizer.decode_calls == 1


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["▁hello", "<0x0A>", "▁world"], "hello\n world"),
        (["▁hello", "<0x0A>", "▁", "▁ind", "ented"], "hello\n  indented"),
        (
            ["▁hello", "▁", "<0xF0>", "<0x9F>", "<0xAB>", "<0xA0>"],
            "hello 🫠",
        ),
        (
            ["▁hello", "<0x0A>", "▁", "<0xF0>", "<0x9F>", "<0xAB>", "<0xA0>"],
            "hello\n 🫠",
        ),
        (
            ["x"] * 63 + ["<0xF0>", "<0x9F>", "<0xAB>", "<0xA0>"],
            "x" * 63 + "🫠",
        ),
        (["▁hello", "<0x0A>"] + ["▁world"] * 100, "hello\n" + " world" * 100),
    ],
)
def test_backendless_sentencepiece_preserves_context_and_split_bytes(pieces, expected):
    from tokenizers import decoders

    # SentencePiece strips the initial space and renders each incomplete byte
    # as U+FFFD. Exercise those decoder semantics without a downloaded model.
    decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )

    class Tokenizer:
        def decode(self, token_ids):
            return decoder.decode([pieces[token_id] for token_id in token_ids])

    tokenizer = Tokenizer()
    token_ids = list(range(len(pieces)))
    assert tokenizer.decode(token_ids) == expected
    detokenizer = StreamingDetokenizer(tokenizer)
    text = ""
    for token_id in token_ids:
        detokenizer.add_token(token_id)
        segment = detokenizer.last_segment
        assert "\ufffd" not in segment
        text += segment
        assert expected.startswith(text)
    detokenizer.finalize()
    text += detokenizer.last_segment

    assert text == expected
    assert detokenizer.text == expected


def test_stream_generate_accepts_scalar_eos_token_ids():
    """A plain transformers tokenizer reports eos_token_ids as an int."""
    out = list(
        stream_generate(CycleModel(), IntEosTok(), mx.array([EOS - 1]), max_tokens=8)
    )
    assert out[-1].token == EOS
    assert out[-1].finish_reason == "stop"


def test_stream_generate_without_any_eos_runs_to_max_tokens():
    out = list(
        stream_generate(CycleModel(), NoEosTok(), mx.array([EOS - 1]), max_tokens=4)
    )
    assert len(out) == 4
    assert out[-1].finish_reason == "length"


def test_fast_streaming_detokenizer_is_linear_and_matches_batch_decode():
    from tokenizers import Tokenizer, models

    backend = Tokenizer(models.WordLevel({"世": 0, "[EOS]": 1}, unk_token="[EOS]"))
    backend.add_special_tokens(["[EOS]"])

    class FastTokenizer:
        backend_tokenizer = backend
        clean_up_tokenization_spaces = False

        def __init__(self):
            self.decode_calls = 0

        def decode(self, token_ids, **kwargs):
            self.decode_calls += 1
            return backend.decode(token_ids, **kwargs)

    tokenizer = FastTokenizer()
    token_ids = [0, 1] * 500
    detokenizer = StreamingDetokenizer(tokenizer, skip_special_tokens=True)
    deltas = []
    for token_id in token_ids:
        detokenizer.add_token(token_id)
        deltas.append(detokenizer.last_segment)
    detokenizer.finalize()
    deltas.append(detokenizer.last_segment)

    expected = backend.decode(token_ids, skip_special_tokens=True)
    assert "".join(deltas) == expected
    assert detokenizer.text == expected
    assert tokenizer.decode_calls == 0

    detokenizer.reset()
    assert detokenizer.text == ""
    detokenizer.add_token(0)
    assert detokenizer.last_segment == "世"


def test_backendless_fallback_bounds_total_decode_work():
    class CountingTokenizer:
        clean_up_tokenization_spaces = False

        def __init__(self):
            self.decoded_ids = 0
            self.max_decode_size = 0

        def decode(self, token_ids, **kwargs):
            del kwargs
            self.decoded_ids += len(token_ids)
            self.max_decode_size = max(self.max_decode_size, len(token_ids))
            return "".join("x" for _ in token_ids)

    tokenizer = CountingTokenizer()
    detokenizer = StreamingDetokenizer(tokenizer)
    deltas = []
    for token_id in range(1_000):
        detokenizer.add_token(token_id)
        deltas.append(detokenizer.last_segment)
    detokenizer.finalize()
    deltas.append(detokenizer.last_segment)

    assert "".join(deltas) == "x" * 1_000
    assert tokenizer.max_decode_size <= 65
    assert tokenizer.decoded_ids < 70_000


def test_backendless_fallback_keeps_unsplittable_output_pending():
    class LeadingSpaceTokenizer:
        clean_up_tokenization_spaces = False

        def __init__(self):
            self.decoded_ids = 0

        def decode(self, token_ids, **kwargs):
            del kwargs
            self.decoded_ids += len(token_ids)
            # Like SentencePiece, drop the leading space of every decoded
            # sequence, so no split point reproduces the joint decode.
            return "".join(f" w{token_id}" for token_id in token_ids).lstrip()

    tokenizer = LeadingSpaceTokenizer()
    detokenizer = StreamingDetokenizer(tokenizer)
    token_ids = list(range(300))
    deltas = []
    for token_id in token_ids:
        detokenizer.add_token(token_id)
        deltas.append(detokenizer.last_segment)
    detokenizer.finalize()
    deltas.append(detokenizer.last_segment)

    assert "".join(deltas) == " ".join(f"w{token_id}" for token_id in token_ids)
    # Split-point searches back off as pending tokens double instead of
    # repeating on every token.
    assert tokenizer.decoded_ids < 250_000

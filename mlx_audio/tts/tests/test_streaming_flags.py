"""Streaming chunk flags, checked through each model's public ``generate``.

With ``stream=True`` every result is a streaming chunk and only the last result
of the call is final; without it neither flag is set. Token streams and codecs
are scripted, so no checkpoints are needed.
"""

import importlib
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

TINY_LM = dict(
    model_type="llama",
    hidden_size=8,
    num_hidden_layers=1,
    intermediate_size=16,
    num_attention_heads=2,
    rms_norm_eps=1e-5,
    vocab_size=16,
    head_dim=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)


def token_interval(tokens, tokens_per_second):
    """streaming_interval in seconds that yields every ``tokens`` tokens."""
    return (tokens + 0.5) / tokens_per_second


def scripted_stream_generate(tokens):
    def stream_generate(*args, **kwargs):
        for token in tokens:
            yield SimpleNamespace(token=token)

    return stream_generate


def fake_snac_decode(code_list):
    # Two samples per 7-code frame, so any split decodes to the same audio.
    codes = mx.array([int(c) for c in code_list], dtype=mx.float32)
    return codes.reshape(-1, 7)[:, :2].reshape(1, -1)


@contextmanager
def snac_lm(module_path, start_of_speech, end_of_speech, audio_offset):
    module = importlib.import_module(module_path)
    with patch("transformers.AutoTokenizer"):
        model = module.Model(module.ModelConfig(**TINY_LM))
    model.prepare_input_ids = lambda *args, **kwargs: mx.array([[1, 2, 3]])
    # Four 7-code frames between the speech markers.
    codes = [audio_offset + code for code in range(28)]
    tokens = [start_of_speech, *codes, end_of_speech]
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(module, "stream_generate", scripted_stream_generate(tokens))
        )
        stack.enter_context(
            patch.object(module, "decode_audio_from_codes", fake_snac_decode)
        )
        if hasattr(module, "decode_audio_stream"):
            stack.enter_context(
                patch.object(
                    module,
                    "decode_audio_stream",
                    lambda codes, prev, context_frames: (fake_snac_decode(codes), prev),
                )
            )
        yield model.generate


def llama():
    return snac_lm("mlx_audio.tts.models.llama.llama", 128257, 128258, 128266)


def qwen3():
    from mlx_audio.tts.models.qwen3 import qwen3

    return snac_lm(
        "mlx_audio.tts.models.qwen3.qwen3",
        qwen3.START_OF_SPEECH,
        qwen3.END_OF_SPEECH,
        qwen3.AUDIO_TOKENS_START,
    )


@contextmanager
def outetts():
    from mlx_audio.tts.models.outetts import outetts as module

    class PromptProcessor:
        def __init__(self, tokenizer):
            pass

        def get_completion_prompt(self, prompt, speaker):
            return prompt

        def extract_audio_from_tokens(self, tokens):
            return [list(tokens), list(tokens)]

    class Codec:
        def decode(self, codes):
            # Two samples per token, so any split decodes to the same audio.
            return mx.repeat(codes[0, 0].astype(mx.float32), 2)[None, None, :]

    class AudioProcessor:
        audio_codec = Codec()

        def load_speaker(self, voice):
            return {}

    with patch.object(module, "AutoTokenizer"):
        model = module.Model(module.ModelConfig(**TINY_LM))
    model.tokenizer = MagicMock()
    model.tokenizer.encode.return_value = mx.array([[1, 2, 3]])
    with (
        patch.object(
            module, "stream_generate", scripted_stream_generate(range(10, 40))
        ),
        patch.object(module, "PromptProcessor", PromptProcessor),
        patch.object(module, "AudioProcessor", AudioProcessor),
    ):
        yield model.generate


@contextmanager
def pocket_tts():
    from mlx_audio.tts.models.pocket_tts import Model
    from mlx_audio.tts.tests.test_pocket_tts import flat_config

    words = ["[UNK]", "Hello", "world", ".", "!", "?", "test", "Another"]
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save(str(Path(tmp) / "tokenizer.json"))
        (Path(tmp) / "voices").mkdir()
        mx.save_safetensors(
            str(Path(tmp) / "voices" / "Freya.safetensors"),
            {"emb": mx.ones((1, 2, 8))},
        )
        config = flat_config()
        config["model_path"] = tmp
        config["mimi"]["encoder_enabled"] = False
        yield Model(config).generate


@contextmanager
def sesame():
    from mlx_audio.tts.models.sesame import sesame as module

    class SesameModel:
        def __init__(self, config):
            self.args = SimpleNamespace(audio_num_codebooks=3)

        def setup_caches(self, max_batch_size):
            pass

        def reset_caches(self):
            self.frames = 0

        def generate_frame(self, tokens, tokens_mask, input_pos, sampler):
            # Nine audio frames, then the all-zero end-of-speech frame.
            self.frames += 1
            value = self.frames if self.frames <= 9 else 0
            return mx.full((1, 3), value, dtype=mx.int32)

    class StreamingDecoder:
        def __init__(self, mimi):
            pass

        def reset(self):
            pass

        def decode_frames(self, tokens):
            # Four samples per frame, so any split decodes to the same audio.
            return mx.repeat(tokens[0, 0].astype(mx.float32), 4)[None, None, :]

    tokenizer = MagicMock()
    tokenizer.encode.return_value = mx.array([[7, 8]])
    mimi = MagicMock(cfg=SimpleNamespace(sample_rate=24000))
    with (
        patch.object(module, "SesameModel", SesameModel),
        patch.object(module, "load_llama3_tokenizer", return_value=tokenizer),
        patch.object(module.Mimi, "from_pretrained", return_value=mimi),
        patch.object(module, "MimiStreamingDecoder", StreamingDecoder),
        patch.object(module, "load_watermarker", side_effect=RuntimeError),
    ):
        model = module.Model(
            {
                "audio_num_codebooks": 3,
                "use_default_voice_prompt": False,
                "voice_match": False,
            }
        )
    yield model.generate


# Each model runs once with audio left after its last partial chunk and once
# where the partial chunks already carry all audio. The random tiny pocket_tts
# model has no fixed length, so its leftover case streams a single chunk.
CASES = [
    pytest.param(llama, "one", token_interval(8, 137.5), True, id="llama"),
    pytest.param(llama, "one", token_interval(15, 137.5), False, id="llama-drained"),
    pytest.param(llama, "one\ntwo", token_interval(8, 137.5), True, id="llama-two"),
    pytest.param(qwen3, "one", token_interval(8, 137.5), True, id="qwen3"),
    pytest.param(qwen3, "one", token_interval(15, 137.5), False, id="qwen3-drained"),
    pytest.param(qwen3, "one\ntwo", token_interval(8, 137.5), True, id="qwen3-two"),
    pytest.param(outetts, "Hi.", token_interval(7, 137.5), True, id="outetts"),
    pytest.param(
        outetts, "Hi.", token_interval(10, 137.5), False, id="outetts-drained"
    ),
    pytest.param(
        outetts,
        f"{' '.join(['one'] * 20)}. {' '.join(['two'] * 20)}.",
        token_interval(7, 137.5),
        True,
        id="outetts-two",
    ),
    pytest.param(pocket_tts, "Hello world.", 100.0, True, id="pocket_tts"),
    pytest.param(pocket_tts, "Hello world.", 0.5, False, id="pocket_tts-drained"),
    pytest.param(
        pocket_tts, "Hello world. Another test.", 0.5, False, id="pocket_tts-two"
    ),
    pytest.param(sesame, "one", token_interval(2, 12.5), True, id="sesame"),
    pytest.param(sesame, "one", token_interval(3, 12.5), False, id="sesame-drained"),
    pytest.param(sesame, "one\ntwo", token_interval(2, 12.5), True, id="sesame-two"),
]


def joined_audio(results):
    return np.concatenate([np.array(r.audio) for r in results])


@pytest.mark.parametrize("build, text, interval, final_has_audio", CASES)
def test_only_the_last_streamed_chunk_is_final(build, text, interval, final_has_audio):
    with build() as generate:
        mx.random.seed(0)
        streamed = list(generate(text, stream=True, streaming_interval=interval))
        mx.random.seed(0)
        whole = list(generate(text))

    assert all(r.is_streaming_chunk for r in streamed)
    assert [r.is_final_chunk for r in streamed] == [False] * (len(streamed) - 1) + [
        True
    ]
    assert all(r.samples > 0 for r in streamed[:-1])
    assert (streamed[-1].samples > 0) == final_has_audio
    assert streamed[-1].sample_rate == streamed[0].sample_rate
    assert not any(r.is_streaming_chunk or r.is_final_chunk for r in whole)
    np.testing.assert_array_equal(joined_audio(streamed), joined_audio(whole))

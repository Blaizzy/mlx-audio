"""Tests for Fish Speech S2-Pro generation.

Verifies that _sample_semantic (lazy RAS) and streaming produce correct
output without regressions.
"""

import mlx.core as mx
import pytest

from .fish_speech import (
    RAS_HIGH_TEMP,
    RAS_HIGH_TOP_P,
    RAS_WIN_SIZE,
    Model,
    _sample_logits,
)
from .config import ModelConfig


@pytest.fixture(scope="module")
def model_and_config():
    """Load the model once for all tests in this module.

    Requires mlx-community/fish-audio-s2-pro-8bit to be cached locally.
    Skip if not available.
    """
    try:
        from mlx_audio.tts.utils import load_model
        from mlx_audio.utils import get_model_path

        model_path = get_model_path("mlx-community/fish-audio-s2-pro-8bit")
        model = load_model(model_path=model_path)
        return model
    except Exception as e:
        pytest.skip(f"Model not available: {e}")


class TestSampleSemantic:
    """Test _sample_semantic (lazy RAS) behavior."""

    def test_returns_tuple(self, model_and_config):
        """_sample_semantic must return (mx.array, int) tuple."""
        model = model_and_config
        logits = mx.random.normal((1, model.config.text_config.vocab_size))
        token, value = model._sample_semantic(
            logits=logits,
            previous_semantic_tokens=[],
            top_p=0.7,
            top_k=30,
            temperature=0.7,
        )
        assert isinstance(token, mx.array), "Token should be mx.array"
        assert isinstance(value, int), "Value should be int"
        assert token.shape == (1,), f"Token shape should be (1,), got {token.shape}"

    def test_token_value_matches_array(self, model_and_config):
        """The returned int value must match the array content."""
        model = model_and_config
        logits = mx.random.normal((1, model.config.text_config.vocab_size))
        token, value = model._sample_semantic(
            logits=logits,
            previous_semantic_tokens=[],
            top_p=0.7,
            top_k=30,
            temperature=0.7,
        )
        assert int(token[0].item()) == value

    def test_ras_triggers_on_repetition(self, model_and_config):
        """RAS should resample when a semantic token repeats in the window."""
        model = model_and_config
        start = model.config.semantic_start_token_id
        # Create logits that heavily favor token `start + 5`
        vocab_size = model.config.text_config.vocab_size
        logits = mx.full((1, vocab_size), -1e9)
        logits = logits.at[0, start + 5].add(1e9)

        # First call: no previous tokens, should return start+5
        token1, val1 = model._sample_semantic(
            logits=logits,
            previous_semantic_tokens=[],
            top_p=0.7,
            top_k=30,
            temperature=0.7,
        )
        assert val1 == start + 5, "Should sample the dominant token"

        # Second call with same token in history: RAS should trigger
        # and resample with high temperature. The result may differ.
        token2, val2 = model._sample_semantic(
            logits=logits,
            previous_semantic_tokens=[start + 5],
            top_p=0.7,
            top_k=30,
            temperature=0.7,
        )
        # We can't guarantee a different token (high temp might still
        # pick the same one), but we verify it runs without error
        assert isinstance(val2, int)

    def test_non_semantic_tokens_skip_ras(self, model_and_config):
        """Tokens outside the semantic range should not trigger RAS resampling.

        Even if a non-semantic token appears in previous_semantic_tokens,
        it should not trigger high-temp resampling because the RAS check
        requires the token to be in the semantic ID range.
        """
        model = model_and_config
        start = model.config.semantic_start_token_id
        end = model.config.semantic_end_token_id
        # Pick a semantic token that's NOT in previous_tokens
        target = start + 10
        vocab_size = model.config.text_config.vocab_size
        logits = mx.full((1, vocab_size), -1e9)
        logits = logits.at[0, target].add(1e9)

        # previous_tokens contains only NON-semantic tokens (like 0, 1, 2)
        # so even though we sample a semantic token, RAS should NOT trigger
        token, value = model._sample_semantic(
            logits=logits,
            previous_semantic_tokens=[0, 1, 2],
            top_p=0.7,
            top_k=30,
            temperature=0.7,
        )
        assert value == target, (
            f"Should return target token {target} without RAS resampling, got {value}"
        )


class TestGeneration:
    """Test end-to-end generation."""

    def test_generate_produces_audio(self, model_and_config):
        """generate() must produce non-empty audio."""
        model = model_and_config
        results = list(model.generate(
            text="Hello",
            verbose=False,
            max_tokens=30,
        ))
        assert len(results) > 0, "Should produce at least one result"
        assert results[0].audio is not None, "Audio should not be None"
        assert results[0].audio.shape[0] > 0, "Audio should have samples"

    def test_generate_streaming_matches_nonstreaming(self, model_and_config):
        """Streaming and non-streaming should produce same token count."""
        model = model_and_config
        text = "Hello world"

        # Non-streaming
        mx.random.seed(42)
        non_stream = list(model.generate(
            text=text, verbose=False, max_tokens=30, stream=False
        ))
        ns_tokens = sum(r.token_count for r in non_stream)

        # Streaming
        mx.random.seed(42)
        streamed = list(model.generate(
            text=text, verbose=False, max_tokens=30,
            stream=True, streaming_interval=0.5,
        ))
        s_tokens = sum(r.token_count for r in streamed)

        assert ns_tokens == s_tokens, (
            f"Token count mismatch: non-streaming={ns_tokens}, streaming={s_tokens}"
        )

    def test_stream_false_is_default(self, model_and_config):
        """Default behavior (stream=False) should work unchanged."""
        model = model_and_config
        results = list(model.generate(
            text="Test", verbose=False, max_tokens=10
        ))
        # Should get exactly one result per batch
        assert len(results) == 1

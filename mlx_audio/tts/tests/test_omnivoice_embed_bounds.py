"""OmniVoice must never look an id up in a table that is too small for it.

MLX does not bounds-check gathers, so an out-of-range id reads past the table
(garbage values, or a segfault when the read lands on an unmapped page).
Positions use only one of the two tables: text ids can exceed the audio vocab,
and the audio mask id can exceed a small text vocab. CPU-only, no weights.
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
from mlx_audio.tts.models.omnivoice.omnivoice import Model

TEXT_VOCAB = 16
AUDIO_VOCAB = 8
MASK_ID = AUDIO_VOCAB - 1
CODEBOOKS = 2


_PREVIOUS_DEVICE = None


def setUpModule():
    global _PREVIOUS_DEVICE
    _PREVIOUS_DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    if _PREVIOUS_DEVICE is not None:
        mx.set_default_device(_PREVIOUS_DEVICE)


class _Spy(nn.Module):
    """Wraps an embedding and records the largest id it was asked to look up."""

    def __init__(self, inner: nn.Embedding):
        super().__init__()
        self.inner = inner
        self.num_embeddings = inner.weight.shape[0]
        self.max_id = -1

    def __call__(self, ids: mx.array) -> mx.array:
        self.max_id = max(self.max_id, int(mx.max(ids).item()))
        return self.inner(ids)


def _tiny_model() -> Model:
    cfg = OmniVoiceConfig(
        audio_vocab_size=AUDIO_VOCAB,
        audio_mask_id=MASK_ID,
        num_audio_codebook=CODEBOOKS,
        audio_codebook_weights=[1] * CODEBOOKS,
        llm_config={
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "vocab_size": TEXT_VOCAB,
            "head_dim": 4,
        },
    )
    return Model(cfg)


def _inputs():
    # 4 text positions (ids up to TEXT_VOCAB - 1, past the audio table) then
    # 3 audio positions (ids up to the mask id, which is the audio table's last row).
    text = mx.array([[3, 15, 9, 12]], dtype=mx.int32)
    audio = mx.array([[0, 5, MASK_ID]], dtype=mx.int32)
    ids = mx.concatenate(
        [
            mx.broadcast_to(text[:, :, None], (1, 4, CODEBOOKS)),
            mx.broadcast_to(audio[:, :, None], (1, 3, CODEBOOKS)),
        ],
        axis=1,
    )
    mask = mx.array([[False] * 4 + [True] * 3])
    return ids, mask


class TestOmniVoiceEmbedBounds(unittest.TestCase):
    def test_lookups_stay_inside_their_tables(self):
        model = _tiny_model()
        ids, mask = _inputs()
        model.backbone.embed_tokens = _Spy(model.backbone.embed_tokens)
        model.audio_embeddings = [_Spy(e) for e in model.audio_embeddings]

        model._prepare_embed_inputs(ids, mask)

        for spy in [model.backbone.embed_tokens, *model.audio_embeddings]:
            self.assertLess(
                spy.max_id,
                spy.num_embeddings,
                f"looked up id {spy.max_id} in a table of {spy.num_embeddings} rows",
            )

    def test_valid_ids_give_the_same_embeddings(self):
        model = _tiny_model()
        ids, mask = _inputs()
        out = model._prepare_embed_inputs(ids, mask)

        text_w = model.backbone.embed_tokens.weight
        expected_text = text_w[mx.array([3, 15, 9, 12])]
        expected_audio = sum(
            model.audio_embeddings[i].weight[mx.array([0, 5, MASK_ID])]
            for i in range(CODEBOOKS)
        )
        expected = mx.concatenate([expected_text, expected_audio], axis=0)[None]
        self.assertTrue(mx.allclose(out, expected).item())


if __name__ == "__main__":
    unittest.main()

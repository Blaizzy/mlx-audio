import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.moss_tts.config import ModelConfig
from mlx_audio.tts.models.moss_tts.delay_model import MossTTSDelayModel
from mlx_audio.tts.models.moss_tts.inference_utils import (
    DelaySchedulerState,
    build_delay_audio_sampling_mask,
    build_delay_forced_text_tokens,
    update_delay_scheduler_state,
)
from mlx_audio.tts.models.moss_tts.model import Model
from mlx_audio.tts.models.moss_tts.processor import AUDIO_PLACEHOLDER, MossTTSProcessor


def _tiny_delay_config_dict():
    return {
        "model_type": "moss_tts_delay",
        "language_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "tie_word_embeddings": True,
            "attention_bias": False,
        },
        "n_vq": 2,
        "audio_vocab_size": 16,
        "audio_user_slot_token_id": 40,
        "audio_assistant_gen_slot_token_id": 41,
        "audio_assistant_delay_slot_token_id": 42,
        "audio_start_token_id": 43,
        "audio_end_token_id": 44,
        "pad_token_id": 0,
        "im_start_token_id": 1,
        "im_end_token_id": 2,
        "audio_pad_code": 16,
        "sampling_rate": 24000,
        "additional_mlp_ffn_hidden_size": 24,
    }


class _DummyTokenizer:
    def __init__(self, special_id_to_token):
        self.special_id_to_token = dict(special_id_to_token)
        self.special_token_to_id = {v: k for k, v in special_id_to_token.items()}

    def convert_ids_to_tokens(self, token_id):
        return self.special_id_to_token.get(int(token_id), chr(97 + int(token_id) % 26))

    def encode(self, text):
        output = []
        for char in text:
            if char in self.special_token_to_id:
                output.append(self.special_token_to_id[char])
            else:
                output.append((ord(char) % 20) + 3)
        return output

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        joined = "".join(m["content"] for m in messages)
        if add_generation_prompt:
            joined = joined + "<assistant>"
        return joined

    def decode(self, token_ids):
        chars = [self.convert_ids_to_tokens(token_id) for token_id in token_ids]
        return "".join(chars)


class _DummyAudioTokenizer:
    def batch_encode(self, wav_list, num_quantizers=None):
        n_q = int(num_quantizers or 2)
        batch = len(wav_list)
        codes = mx.zeros((n_q, batch, 2), dtype=mx.int32)
        for b in range(batch):
            for q in range(n_q):
                codes[q, b, 0] = q + 1
                codes[q, b, 1] = q + 2
        lengths = mx.full((batch,), 2, dtype=mx.int32)
        return SimpleNamespace(audio_codes=codes, audio_codes_lengths=lengths)

    def decode(self, audio_codes, return_dict=True, chunk_duration=8.0, num_quantizers=None):
        if audio_codes.ndim == 2:
            steps = int(audio_codes.shape[1])
        else:
            steps = int(audio_codes.shape[-1])
        samples = steps * 10
        audio = mx.linspace(0, 1, samples, dtype=mx.float32)[None, None, :]
        lengths = mx.array([samples], dtype=mx.int32)
        return SimpleNamespace(audio=audio, audio_lengths=lengths)


def _build_dummy_processor(config: ModelConfig) -> MossTTSProcessor:
    token_map = {
        config.audio_user_slot_token_id: "§",
        config.audio_assistant_gen_slot_token_id: "¤",
        config.audio_assistant_delay_slot_token_id: "¦",
        config.audio_start_token_id: "†",
        config.audio_end_token_id: "‡",
    }
    tokenizer = _DummyTokenizer(token_map)
    return MossTTSProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=_DummyAudioTokenizer(),
        model_config=config,
    )


class TestMossTTSDelayRuntime(unittest.TestCase):
    def test_delay_model_shape_contracts(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        model = MossTTSDelayModel(config)
        input_ids = mx.zeros((1, 4, config.channels), dtype=mx.int32)
        hidden = model(input_ids, cache=model.make_cache(), n_vq_for_inference=config.n_vq)
        self.assertEqual(hidden.shape, (1, 4, config.hidden_size))

        logits = model.compute_next_logits(hidden[:, -1, :], n_vq_for_inference=config.n_vq)
        self.assertEqual(len(logits), config.channels)
        self.assertEqual(logits[0].shape, (1, config.vocab_size))
        self.assertEqual(logits[1].shape, (1, config.audio_vocab_size + 1))

    def test_scheduler_transitions_cover_sample_delay_and_flush(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        state = DelaySchedulerState(
            is_stopping=mx.array([False], dtype=mx.bool_),
            is_audio=mx.array([True], dtype=mx.bool_),
            audio_lengths=mx.array([4], dtype=mx.int64),
            delayed_lengths=mx.array([-1], dtype=mx.int64),
        )

        next_text, sampling_mask, forcing_audio_eos = build_delay_forced_text_tokens(
            state, config, config.n_vq
        )
        self.assertTrue(bool(sampling_mask[0]))
        self.assertFalse(bool(forcing_audio_eos[0]))

        state = update_delay_scheduler_state(
            state,
            next_text_token=mx.array(
                [config.audio_assistant_delay_slot_token_id], dtype=mx.int32
            ),
            config=config,
            n_vq=config.n_vq,
            forcing_audio_eos=mx.array([False], dtype=mx.bool_),
        )
        self.assertEqual(int(state.delayed_lengths[0]), 1)

        next_text, _, forcing_audio_eos = build_delay_forced_text_tokens(
            state, config, config.n_vq
        )
        self.assertEqual(int(next_text[0]), config.audio_assistant_delay_slot_token_id)
        self.assertFalse(bool(forcing_audio_eos[0]))

        flush_state = DelaySchedulerState(
            is_stopping=mx.array([False], dtype=mx.bool_),
            is_audio=mx.array([True], dtype=mx.bool_),
            audio_lengths=mx.array([4], dtype=mx.int64),
            delayed_lengths=mx.array([config.n_vq], dtype=mx.int64),
        )
        next_text, _, forcing_audio_eos = build_delay_forced_text_tokens(
            flush_state, config, config.n_vq
        )
        self.assertEqual(int(next_text[0]), config.audio_end_token_id)
        self.assertTrue(bool(forcing_audio_eos[0]))

        audio_mask = build_delay_audio_sampling_mask(flush_state, n_vq=config.n_vq)
        self.assertEqual(audio_mask.shape, (1, config.n_vq))

    def test_processor_delay_pattern_round_trip(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)
        codes = mx.array([[1, 2], [3, 4], [5, 6]], dtype=mx.int32)

        delayed = processor.apply_delay_pattern(codes)
        restored = processor.apply_de_delay_pattern(delayed)
        np.testing.assert_array_equal(np.array(restored), np.array(codes))

    def test_processor_delay_segment_parsing(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)
        codes = mx.array([[7, 8], [9, 10]], dtype=mx.int32)
        delayed = processor.apply_delay_pattern(codes)
        segments = processor.parse_generated_assistant_segments(delayed, delayed=True)
        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(np.array(segments[0]), np.array(codes))

        assistant = processor.build_assistant_message([codes], content=AUDIO_PLACEHOLDER)
        user = processor.build_user_message(text="hello", input_type="text")
        packed = processor.prepare_generation_inputs(
            [assistant, user], n_vq=config.n_vq, apply_chat_template=False
        )
        self.assertEqual(packed["input_ids"].shape[0], 1)
        self.assertEqual(packed["input_ids"].shape[2], 1 + config.n_vq)

    def test_delay_sanitize_remaps_expected_prefixes(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        model = Model(config)
        weights = {
            "language_model.layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
            "language_model.embed_tokens.weight": mx.ones((4, 4)),
            "emb_ext.0.weight": mx.ones((4, 4)),
            "lm_heads.0.weight": mx.ones((4, 4)),
            "local_transformer.layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("model.backbone.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertIn("model.text_embedding.weight", sanitized)
        self.assertIn("model.emb_ext.0.weight", sanitized)
        self.assertIn("model.lm_heads.0.weight", sanitized)
        self.assertNotIn(
            "local_transformer.layers.0.self_attn.q_proj.weight", sanitized
        )

    def test_delay_generate_smoke_with_stubbed_logits(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        model = Model(config)
        model.processor = _build_dummy_processor(config)
        model.tokenizer = model.processor.tokenizer

        step_counter = {"step": 0}

        def fake_logits(*args, **kwargs):
            step = step_counter["step"]
            step_counter["step"] += 1

            text = np.full((1, config.vocab_size), -1e9, dtype=np.float32)
            if step == 0:
                text[0, config.audio_assistant_gen_slot_token_id] = 0.0
            elif step == 1:
                text[0, config.audio_assistant_delay_slot_token_id] = 0.0
            else:
                text[0, config.audio_assistant_gen_slot_token_id] = 0.0

            audio_vocab = config.audio_vocab_size + 1
            audio0 = np.full((1, audio_vocab), -1e9, dtype=np.float32)
            audio1 = np.full((1, audio_vocab), -1e9, dtype=np.float32)
            audio0[0, 1] = 0.0
            audio1[0, 2] = 0.0
            return [mx.array(text), mx.array(audio0), mx.array(audio1)]

        with patch.object(model.model, "compute_next_logits", side_effect=fake_logits):
            results = list(
                model.generate(
                    text="hello",
                    max_tokens=8,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    input_type="text",
                )
            )

        self.assertGreaterEqual(len(results), 1)
        self.assertGreater(results[-1].samples, 0)


if __name__ == "__main__":
    unittest.main()

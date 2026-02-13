import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.moss_tts.config import ModelConfig
from mlx_audio.tts.models.moss_tts.local_model import MossTTSLocalModel
from mlx_audio.tts.models.moss_tts.model import Model
from mlx_audio.tts.models.moss_tts.processor import MossTTSProcessor
from mlx_audio.tts.models.moss_tts.sampling import resolve_channel_sampling_configs


def _tiny_local_config_dict():
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
        "local_ffn_hidden_size": 24,
        "local_hidden_size": 12,
        "local_num_layers": 2,
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


class TestMossTTSLocalRuntime(unittest.TestCase):
    def test_model_config_detects_local_vs_delay(self):
        local = ModelConfig.from_dict(_tiny_local_config_dict())
        self.assertTrue(local.is_local_variant)

        delay_payload = _tiny_local_config_dict()
        delay_payload.pop("local_num_layers")
        delay_payload.pop("local_hidden_size")
        delay_payload.pop("local_ffn_hidden_size")
        delay = ModelConfig.from_dict(delay_payload)
        self.assertFalse(delay.is_local_variant)

    def test_local_model_shape_contracts(self):
        config = ModelConfig.from_dict(_tiny_local_config_dict())
        model = MossTTSLocalModel(config)

        input_ids = mx.zeros((1, 4, config.channels), dtype=mx.int32)
        hidden = model(input_ids, cache=model.make_cache(), n_vq_for_inference=config.n_vq)
        self.assertEqual(hidden.shape, (1, 4, config.hidden_size))

        sampling = resolve_channel_sampling_configs(
            config.channels,
            default_temperature=1.0,
            default_top_p=1.0,
            default_top_k=0,
            default_repetition_penalty=1.0,
            do_samples=[False] * config.channels,
        )
        next_tokens = model.sample_next_channels(
            hidden[:, -1, :],
            input_ids,
            sampling,
            n_vq_for_inference=config.n_vq,
        )
        self.assertEqual(next_tokens.shape, (1, config.channels))

    def test_local_sanitize_remaps_expected_prefixes(self):
        config = ModelConfig.from_dict(_tiny_local_config_dict())
        model = Model(config)

        weights = {
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
            "model.embedding_list.0.weight": mx.ones((8, 8)),
            "local_transformer.layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
            "speech_embedding_to_local_mlp.gate_proj.weight": mx.ones((4, 4)),
            "local_to_speech_embedding_mlps.0.gate_proj.weight": mx.ones((4, 4)),
            "layer_norm_before_lm_heads.0.weight": mx.ones((4,)),
            "lm_heads.0.weight": mx.ones((4, 4)),
            "model.language_model.embed_tokens.weight": mx.ones((4, 4)),
            "foo.num_batches_tracked": mx.array(0),
        }

        sanitized = model.sanitize(weights)
        self.assertIn("model.backbone.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertIn("model.embedding_list.0.weight", sanitized)
        self.assertIn("model.local_transformer.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertIn("model.speech_embedding_to_local_mlp.gate_proj.weight", sanitized)
        self.assertIn(
            "model.local_to_speech_embedding_mlps.0.gate_proj.weight", sanitized
        )
        self.assertIn("model.layer_norm_before_lm_heads.0.weight", sanitized)
        self.assertIn("model.lm_heads.0.weight", sanitized)
        self.assertNotIn("model.backbone.embed_tokens.weight", sanitized)
        self.assertNotIn("foo.num_batches_tracked", sanitized)

    def test_quant_predicate_blocks_sensitive_modules(self):
        config = ModelConfig.from_dict(_tiny_local_config_dict())
        model = Model(config)
        self.assertFalse(model.model_quant_predicate("model.embedding_list.0", nn.Embedding(2, 2)))
        self.assertFalse(model.model_quant_predicate("model.lm_heads.0", nn.Linear(2, 2)))
        self.assertFalse(
            model.model_quant_predicate(
                "model.layer_norm_before_lm_heads.0",
                nn.RMSNorm(2),
            )
        )
        self.assertTrue(model.model_quant_predicate("model.backbone.layers.0", nn.Linear(2, 2)))

    def test_processor_user_message_and_input_type_contract(self):
        config = ModelConfig.from_dict(_tiny_local_config_dict())
        processor = _build_dummy_processor(config)

        msg = processor.build_user_message(
            text="ni3 hao3",
            reference=[mx.zeros((240,))],
            instruction="warm",
            tokens=32,
            language="zh",
            input_type="pinyin",
        )
        packed = processor.prepare_generation_inputs(msg, n_vq=config.n_vq, apply_chat_template=False)
        self.assertEqual(packed["input_ids"].shape[0], 1)
        self.assertEqual(packed["input_ids"].shape[2], 1 + config.n_vq)
        self.assertEqual(int(packed["input_ids"][0, -1, 0]), config.audio_start_token_id)

        with self.assertRaises(ValueError):
            processor.build_user_message(text="hello", input_type="kana")

    def test_generate_streaming_smoke_with_stubbed_channel_sampler(self):
        config = ModelConfig.from_dict(_tiny_local_config_dict())
        model = Model(config)
        model.processor = _build_dummy_processor(config)
        model.tokenizer = model.processor.tokenizer

        generated = {"count": 0}

        def fake_sample_next_channels(*args, **kwargs):
            generated["count"] += 1
            if generated["count"] < 4:
                text_token = config.audio_assistant_gen_slot_token_id
            else:
                text_token = config.audio_end_token_id
            return mx.array(
                [[text_token, 1, 2]],
                dtype=mx.int32,
            )

        with patch.object(model.model, "sample_next_channels", side_effect=fake_sample_next_channels):
            results = list(
                model.generate(
                    text="hello",
                    stream=True,
                    streaming_interval=0.05,
                    max_tokens=6,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    input_type="text",
                )
            )

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(results[-1].is_final_chunk)
        self.assertTrue(any(r.samples > 0 for r in results))

    def test_request_resolution_allows_instruct_with_ref_text(self):
        config = ModelConfig.from_dict(_tiny_local_config_dict())
        model = Model(config)

        request = model._resolve_generation_request(
            text="hello",
            ref_audio="ref.wav",
            ref_text="greetings",
            instruct="warm tone",
        )
        self.assertEqual(request.reference, ["ref.wav"])
        self.assertIn("warm tone", request.instruction)
        self.assertIn("Reference transcript: greetings", request.instruction)


if __name__ == "__main__":
    unittest.main()

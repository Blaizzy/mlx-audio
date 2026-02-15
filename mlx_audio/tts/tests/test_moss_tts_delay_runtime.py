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

    def test_processor_user_message_normalize_flag(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        raw = processor.build_user_message(
            text='Line1\n[drop]{meta}"Line2"',
            instruction="Calm\n[style]{voice}",
            input_type="text",
            normalize=False,
        )
        normalized = processor.build_user_message(
            text='Line1\n[drop]{meta}"Line2"',
            instruction="Calm\n[style]{voice}",
            input_type="text",
            normalize=True,
        )

        self.assertIn("[drop]", raw["content"])
        self.assertIn("{voice}", raw["content"])
        self.assertIn("Line1 Line2", normalized["content"])
        self.assertIn("- Instruction:\nCalm", normalized["content"])
        self.assertNotIn("[drop]", normalized["content"])
        self.assertNotIn("{voice}", normalized["content"])

    def test_processor_continuation_mode_uses_assistant_suffix(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)
        codes = mx.array([[3, 4], [5, 6]], dtype=mx.int32)

        conversation = [
            processor.build_user_message(text="[S1] Prompt", reference=[codes], input_type="text"),
            processor.build_assistant_message(audio_codes_list=[codes]),
        ]
        packed = processor.prepare_generation_inputs(
            conversation,
            n_vq=config.n_vq,
            apply_chat_template=False,
            mode="continuation",
        )

        self.assertEqual(packed["input_ids"].shape[0], 1)
        self.assertEqual(packed["input_ids"].shape[2], 1 + config.n_vq)
        self.assertNotEqual(
            int(packed["input_ids"][0, -1, 0]),
            config.audio_start_token_id,
        )

    def test_processor_builds_ttsd_continuation_messages_from_speaker_schema(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)
        ref_wave_1 = mx.zeros((80,), dtype=mx.float32)
        ref_wave_3 = mx.ones((80,), dtype=mx.float32)

        messages = processor.build_ttsd_continuation_messages(
            dialogue_text="[S1] Continue the discussion. [S3] Add a counterpoint.",
            speakers=[
                {"speaker_id": 1, "ref_audio": ref_wave_1, "ref_text": "Prompt one."},
                {"speaker_id": 3, "ref_audio": ref_wave_3, "text": "Prompt three."},
            ],
            input_type="text",
        )

        self.assertEqual(len(messages), 3)
        user_message = messages[0]
        assistant_message = messages[1]
        generation_user_message = messages[2]
        self.assertIn("[S1]:", user_message["content"])
        self.assertIn("[S2]: None", user_message["content"])
        self.assertIn("[S3]:", user_message["content"])
        self.assertIn("[S1] Prompt one.", user_message["content"])
        self.assertIn("[S3] Prompt three.", user_message["content"])
        self.assertEqual(len(assistant_message["audio_references"]), 1)
        self.assertEqual(tuple(assistant_message["audio_references"][0].shape), (2, 2))
        self.assertEqual(generation_user_message["role"], "user")
        self.assertIn("[S1] Continue the discussion.", generation_user_message["content"])

    def test_ttsd_speaker_id_normalization_supports_zero_based_schema(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)
        ref_wave_0 = mx.zeros((80,), dtype=mx.float32)
        ref_wave_1 = mx.ones((80,), dtype=mx.float32)

        messages = processor.build_ttsd_continuation_messages(
            dialogue_text="[S1] Continue. [S2] Reply.",
            speakers=[
                {"speaker_id": 0, "ref_audio": ref_wave_0, "ref_text": "Speaker zero."},
                {"speaker_id": 1, "ref_audio": ref_wave_1, "ref_text": "Speaker one."},
            ],
            input_type="text",
        )

        self.assertEqual(len(messages), 3)
        user_message = messages[0]
        assistant_message = messages[1]
        self.assertIn("[S1]:", user_message["content"])
        self.assertIn("[S2]:", user_message["content"])
        self.assertNotIn("[S0]:", user_message["content"])
        self.assertIn("[S1] Speaker zero.", user_message["content"])
        self.assertIn("[S2] Speaker one.", user_message["content"])
        self.assertEqual(tuple(assistant_message["audio_references"][0].shape), (2, 2))

    def test_ttsd_assistant_prompt_codes_encode_concatenated_waveform_once(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        class _ConcatTracingTokenizer:
            def __init__(self):
                self.received_shapes: list[tuple[int, ...]] = []

            def batch_encode(self, wav_list, num_quantizers=None):
                self.received_shapes = [tuple(wav.shape) for wav in wav_list]
                n_q = int(num_quantizers or 2)
                length = int(wav_list[0].shape[0])
                codes = np.zeros((n_q, 1, length), dtype=np.int32)
                for q_idx in range(n_q):
                    codes[q_idx, 0, :] = q_idx + 1
                lengths = np.array([length], dtype=np.int32)
                return SimpleNamespace(
                    audio_codes=mx.array(codes, dtype=mx.int32),
                    audio_codes_lengths=mx.array(lengths, dtype=mx.int32),
                )

        tracing_tokenizer = _ConcatTracingTokenizer()
        processor.audio_tokenizer = tracing_tokenizer

        ref_wave_1 = mx.zeros((80,), dtype=mx.float32)
        ref_wave_2 = mx.ones((120,), dtype=mx.float32)
        messages = processor.build_ttsd_continuation_messages(
            dialogue_text="[S1] Continue. [S2] Reply.",
            speakers=[
                {"speaker_id": 1, "ref_audio": ref_wave_1, "ref_text": "One."},
                {"speaker_id": 2, "ref_audio": ref_wave_2, "ref_text": "Two."},
            ],
            input_type="text",
        )

        self.assertEqual(tracing_tokenizer.received_shapes, [(200,)])
        assistant_codes = messages[1]["audio_references"][0]
        self.assertEqual(tuple(assistant_codes.shape), (200, 2))

    def test_ttsd_assistant_prompt_codes_preserve_preencoded_reference_support(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        time_major = mx.array(
            [
                [1, 11],
                [2, 12],
                [3, 13],
            ],
            dtype=mx.int32,
        )
        codebook_major = time_major.transpose(1, 0)
        messages = processor.build_ttsd_continuation_messages(
            dialogue_text="[S1] Continue. [S2] Reply.",
            speakers=[
                {"speaker_id": 1, "ref_audio": time_major, "ref_text": "One."},
                {"speaker_id": 2, "ref_audio": codebook_major, "ref_text": "Two."},
            ],
            input_type="text",
        )

        assistant_codes = messages[1]["audio_references"][0]
        self.assertEqual(tuple(assistant_codes.shape), (6, 2))

    def test_ttsd_continuation_preserves_multilingual_text_and_language_hint(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)
        ref_wave_1 = mx.zeros((80,), dtype=mx.float32)
        ref_wave_2 = mx.ones((80,), dtype=mx.float32)

        messages = processor.build_ttsd_continuation_messages(
            dialogue_text=(
                "[S1] \u4f60\u597d\uff0c\u4eca\u5929\u600e\u4e48\u6837\uff1f "
                "[S2] \u0645\u0645\u062a\u0627\u0632\u060c let's continue."
            ),
            speakers=[
                {
                    "speaker_id": 1,
                    "ref_audio": ref_wave_1,
                    "ref_text": "\u4f60\u597d\u3002",
                },
                {
                    "speaker_id": 2,
                    "ref_audio": ref_wave_2,
                    "ref_text": "\u0645\u0631\u062d\u0628\u0627.",
                },
            ],
            language="zh-CN",
            input_type="text",
        )

        self.assertEqual(len(messages), 3)
        user_message = messages[0]
        generation_user_message = messages[2]
        self.assertIn("Language:\nzh-CN", user_message["content"])
        self.assertIn("[S1] \u4f60\u597d\u3002", user_message["content"])
        self.assertIn("[S2] \u0645\u0631\u062d\u0628\u0627.", user_message["content"])
        self.assertIn(
            "\u4f60\u597d\uff0c\u4eca\u5929\u600e\u4e48\u6837\uff1f",
            generation_user_message["content"],
        )
        self.assertIn("\u0645\u0645\u062a\u0627\u0632", generation_user_message["content"])

    def test_encode_audios_from_reference_accepts_preencoded_layouts(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        # Canonical `(T, NQ)` with extra quantizers.
        time_major = mx.array(
            [
                [1, 11, 101, 111],
                [2, 12, 102, 112],
                [3, 13, 103, 113],
            ],
            dtype=mx.int32,
        )
        # Canonical tokenizer output `(NQ, T)` with the same values.
        codebook_major = time_major.transpose(1, 0)

        normalized = processor.encode_audios_from_reference(
            [time_major, codebook_major],
            n_vq=2,
        )

        expected = mx.array(
            [
                [1, 11],
                [2, 12],
                [3, 13],
            ],
            dtype=mx.int32,
        )
        self.assertEqual(len(normalized), 2)
        np.testing.assert_array_equal(np.array(normalized[0]), np.array(expected))
        np.testing.assert_array_equal(np.array(normalized[1]), np.array(expected))

    def test_encode_audios_from_reference_preserves_time_axis_for_2d_waveforms(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        class _ShapeTracingAudioTokenizer:
            def __init__(self):
                self.received_shapes: list[tuple[int, ...]] = []

            def batch_encode(self, wav_list, num_quantizers=None):
                self.received_shapes = [tuple(wav.shape) for wav in wav_list]
                n_q = int(num_quantizers or 2)
                batch = len(wav_list)
                max_steps = max(int(wav.shape[0]) for wav in wav_list)
                codes = np.zeros((n_q, batch, max_steps), dtype=np.int32)
                lengths = np.zeros((batch,), dtype=np.int32)
                for batch_idx, wav in enumerate(wav_list):
                    length = int(wav.shape[0])
                    lengths[batch_idx] = length
                    for q_idx in range(n_q):
                        codes[q_idx, batch_idx, :length] = q_idx + 1
                return SimpleNamespace(
                    audio_codes=mx.array(codes, dtype=mx.int32),
                    audio_codes_lengths=mx.array(lengths, dtype=mx.int32),
                )

        shape_tracing_tokenizer = _ShapeTracingAudioTokenizer()
        processor.audio_tokenizer = shape_tracing_tokenizer

        t = 240
        mono = mx.arange(t, dtype=mx.float32)
        references = [
            mono[None, :],  # (1, T)
            mx.stack([mono, mono + 1], axis=0),  # (2, T)
            mono[:, None],  # (T, 1)
            mx.stack([mono, mono + 1], axis=1),  # (T, 2)
        ]

        normalized = processor.encode_audios_from_reference(references, n_vq=2)

        self.assertEqual(
            shape_tracing_tokenizer.received_shapes,
            [(t,), (t,), (t,), (t,)],
        )
        self.assertEqual(len(normalized), 4)
        for codes in normalized:
            self.assertEqual(tuple(codes.shape), (t, 2))

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

    def test_delay_variant_rejects_n_vq_for_inference_override(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        model = Model(config)
        model.processor = _build_dummy_processor(config)
        model.tokenizer = model.processor.tokenizer

        with self.assertRaises(ValueError):
            list(
                model.generate(
                    text="hello",
                    n_vq_for_inference=1,
                    max_tokens=2,
                    input_type="text",
                )
            )

    def test_delay_variant_rejects_realtime_preset(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        model = Model(config)
        model.processor = _build_dummy_processor(config)
        model.tokenizer = model.processor.tokenizer

        with self.assertRaisesRegex(ValueError, "not valid for runtime"):
            list(
                model.generate(
                    text="hello",
                    preset="realtime",
                    max_tokens=2,
                    input_type="text",
                )
            )

    def test_voice_generator_defaults_normalize_inputs_with_override(self):
        payload = _tiny_delay_config_dict()
        payload["gen_token_id"] = 151656
        payload["audio_ch0_vocab_size"] = 1024
        config = ModelConfig.from_dict(payload)
        model = Model(config)

        self.assertTrue(model._resolve_normalize_inputs(None))
        self.assertFalse(model._resolve_normalize_inputs(False))
        self.assertTrue(model._resolve_normalize_inputs(True))

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
                text[0, config.audio_start_token_id] = 0.0
            elif step == 1:
                text[0, config.audio_assistant_gen_slot_token_id] = 0.0
            elif step == 2:
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

    def test_delay_generate_with_dialogue_speakers_structured_schema(self):
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
                    text="[S1] Continue. [S2] Respond.",
                    dialogue_speakers=[
                        {"speaker_id": 1, "ref_text": "Intro A."},
                        {"speaker_id": 2, "ref_text": "Intro B."},
                    ],
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

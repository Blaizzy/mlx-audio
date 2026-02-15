import time
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Optional
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.tts.models.moss_tts.sampling import (
    apply_repetition_penalty,
    resolve_channel_sampling_configs,
)
from mlx_audio.tts.models.moss_tts_realtime.config import ModelConfig
from mlx_audio.tts.models.moss_tts_realtime.inference import (
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    RealtimeSession,
    RealtimeTextDeltaBridge,
)
from mlx_audio.tts.models.moss_tts_realtime.model import Model, MossTTSRealtimeCore
from mlx_audio.tts.models.moss_tts_realtime.processor import MossTTSRealtimeProcessor
from mlx_audio.tts.models.moss_tts_realtime.request import RealtimeNormalizedRequest


def _tiny_realtime_config_dict() -> dict[str, Any]:
    return {
        "model_type": "moss_tts_realtime",
        "rvq": 2,
        "audio_vocab_size": 32,
        "audio_pad_token": 31,
        "audio_bos_token": 30,
        "audio_eos_token": 29,
        "reference_audio_pad": 28,
        "text_pad": 27,
        "sampling_rate": 24000,
        "max_context_tokens": 16,
        "language_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 48,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 96,
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
        },
        "local_config": {
            "model_type": "moss_tts_realtime_local_transformer",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 48,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 33,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "rvq": 2,
            "audio_vocab_size": 32,
            "audio_pad_token": 31,
        },
    }


class _TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [3 + (ord(char) % 21) for char in str(text)]


class _PromptParityTokenizer:
    _SPECIAL = {
        "<|audio_pad|>": 28,
        "<|text_pad|>": 27,
        "<|im_start|>": 40,
        "<|im_end|>": 41,
    }

    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str):
        return self._SPECIAL.get(token, self.unk_token_id)

    def __call__(self, text: str, add_special_tokens: bool = True):
        del add_special_tokens
        return {"input_ids": self.encode(text)}

    def encode(self, text: str, add_special_tokens: bool = True):
        del add_special_tokens
        input_text = str(text)
        token_ids: list[int] = []
        idx = 0
        while idx < len(input_text):
            matched = None
            for token_text in sorted(self._SPECIAL.keys(), key=len, reverse=True):
                if input_text.startswith(token_text, idx):
                    matched = token_text
                    break
            if matched is not None:
                token_ids.append(int(self._SPECIAL[matched]))
                idx += len(matched)
                continue
            token_ids.append(60 + (ord(input_text[idx]) % 31))
            idx += 1
        return token_ids


class _FakeRealtimeModel:
    def __init__(self, config: ModelConfig, *, eos_after: int = 5):
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.eos_after = int(max(1, eos_after))
        self.sample_calls = 0
        self.make_cache_calls = 0
        self.model_call_shapes: list[tuple[int, int, int]] = []
        self.repetition_windows_seen: list[Optional[int]] = []

    def make_cache(self):
        self.make_cache_calls += 1
        return [SimpleNamespace(offset=0)]

    def __call__(self, input_ids: mx.array, cache=None):
        if input_ids.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {input_ids.shape}")
        self.model_call_shapes.append(tuple(int(dim) for dim in input_ids.shape))
        if cache:
            cache[0].offset = int(getattr(cache[0], "offset", 0)) + int(input_ids.shape[1])
        batch = int(input_ids.shape[0])
        time_steps = int(input_ids.shape[1])
        return mx.zeros((batch, time_steps, self.hidden_size), dtype=mx.float32)

    def sample_next_audio_tokens(
        self,
        global_hidden_state: mx.array,
        *,
        generated_history: Optional[mx.array],
        channel_sampling,
        repetition_window: Optional[int] = None,
    ) -> mx.array:
        del generated_history, channel_sampling
        self.sample_calls += 1
        self.repetition_windows_seen.append(repetition_window)
        batch_size = int(global_hidden_state.shape[0])

        max_non_eos = max(1, int(self.config.audio_eos_token) - 1)
        token_value = 1 + ((self.sample_calls - 1) % max_non_eos)
        tokens = np.full((batch_size, self.config.rvq), token_value, dtype=np.int32)
        if self.sample_calls >= self.eos_after:
            tokens[:, 0] = int(self.config.audio_eos_token)
        return mx.array(tokens, dtype=mx.int32)


@dataclass
class _DecodeCall:
    frames: int
    chunk_duration: Optional[float]
    decode_kwargs: dict[str, Any]


class _FakeRealtimeProcessor:
    def __init__(self, config: ModelConfig):
        self.model_config = config
        self.tokenizer = _TinyTokenizer()
        self.decode_calls: list[_DecodeCall] = []

    def build_turn_input_ids(
        self,
        *,
        user_text: str,
        user_audio_tokens: Optional[mx.array] = None,
        include_system_prompt: bool = True,
        voice_prompt_tokens: Optional[mx.array] = None,
    ) -> mx.array:
        del user_text, include_system_prompt, voice_prompt_tokens
        channels = self.model_config.channels
        rows = mx.full((1, channels), self.model_config.audio_pad_token, dtype=mx.int32)
        rows[:, 0] = int(self.model_config.text_pad)
        if user_audio_tokens is not None and int(user_audio_tokens.shape[0]) > 0:
            prompt = mx.full(
                (int(user_audio_tokens.shape[0]), channels),
                self.model_config.audio_pad_token,
                dtype=mx.int32,
            )
            prompt[:, 0] = int(self.model_config.reference_audio_pad)
            prompt[:, 1:] = user_audio_tokens[:, : self.model_config.rvq]
            rows = mx.concatenate([rows, prompt], axis=0)
        return rows[None, :, :]

    def decode_audio_codes(
        self,
        audio_codes: mx.array,
        *,
        chunk_duration: Optional[float],
        decode_kwargs: Optional[dict[str, Any]] = None,
    ) -> mx.array:
        kwargs = {} if decode_kwargs is None else dict(decode_kwargs)
        frames = int(audio_codes.shape[0])
        self.decode_calls.append(
            _DecodeCall(
                frames=frames,
                chunk_duration=chunk_duration,
                decode_kwargs=kwargs,
            )
        )
        if frames <= 0:
            return mx.zeros((0,), dtype=mx.float32)
        samples = frames * 6
        waveform = np.linspace(0.0, 1.0, samples, endpoint=False, dtype=np.float32)
        return mx.array(waveform, dtype=mx.float32)


def _build_prompt_parity_processor(config: ModelConfig) -> MossTTSRealtimeProcessor:
    return MossTTSRealtimeProcessor(
        tokenizer=_PromptParityTokenizer(),
        audio_tokenizer=None,
        model_config=config,
    )


class _RecordingSession:
    def __init__(self, tokenizer):
        self.processor = SimpleNamespace(tokenizer=tokenizer)
        self.recorded_token_ids: list[int] = []
        self.end_calls = 0
        self.drain_calls = 0

    def push_text_tokens(self, token_ids: Iterable[int]):
        self.recorded_token_ids.extend(int(token) for token in token_ids)
        return []

    def end_text(self):
        self.end_calls += 1
        return []

    def drain(self, *, max_steps: Optional[int] = None):
        del max_steps
        self.drain_calls += 1
        return []


class TestMossTTSRealtimeModelContracts(unittest.TestCase):
    def test_core_forward_logit_and_sampling_shape_contracts(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        core = MossTTSRealtimeCore(config)
        cache = core.make_cache()

        input_ids = mx.zeros((1, 4, config.channels), dtype=mx.int32)
        hidden = core(input_ids, cache=cache)
        self.assertEqual(tuple(hidden.shape), (1, 4, config.hidden_size))

        # Warm-cache call should preserve shape contracts.
        hidden_warm = core(input_ids[:, :1, :], cache=cache)
        self.assertEqual(tuple(hidden_warm.shape), (1, 1, config.hidden_size))

        forced_local_tokens = mx.zeros((1, config.rvq), dtype=mx.int32)
        logits = core.compute_next_audio_logits(
            hidden[:, -1, :],
            local_input_ids=forced_local_tokens,
        )
        self.assertEqual(len(logits), config.rvq)
        for channel_logits in logits:
            self.assertEqual(tuple(channel_logits.shape), (1, config.audio_vocab_size))

        # Pad token must be blocked from sampling.
        first_channel_logits = np.array(logits[0])[0]
        self.assertTrue(np.isneginf(first_channel_logits[config.audio_pad_token]))

        sampling_cfg = resolve_channel_sampling_configs(
            config.rvq,
            default_temperature=1.0,
            default_top_p=1.0,
            default_top_k=0,
            default_repetition_penalty=1.0,
            do_samples=[False] * config.rvq,
        )
        sampled = core.sample_next_audio_tokens(
            hidden[:, -1, :],
            generated_history=None,
            channel_sampling=sampling_cfg,
        )
        self.assertEqual(tuple(sampled.shape), (1, config.rvq))

    def test_realtime_model_sanitize_and_quant_guard_contracts(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        model = Model(config)
        parameter_names = {name for name, _ in tree_flatten(model.parameters())}

        weights = {
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
            "model.language_model.embed_tokens.1.weight": mx.ones((4, 4)),
            "model.language_model.embed_tokens.weight": mx.ones((4, 4)),
            "language_model.embed_tokens.2.weight": mx.ones((4, 4)),
            "language_model.embed_tokens.weight": mx.ones((4, 4)),
            "model.embed_tokens.0.weight": mx.ones((4, 4)),
            "model.embed_tokens.weight": mx.ones((4, 4)),
            "embed_tokens.weight": mx.ones((4, 4)),
            "local_transformer.model.layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
            "local_transformer.model.embed_tokens.0.weight": mx.ones((4, 4)),
            "model.local_transformer.model.layers.0.self_attn.k_proj.weight": mx.ones((4, 4)),
            "model.local_transformer.model.embed_tokens.1.weight": mx.ones((4, 4)),
            "local_transformer.local_lm_heads.0.weight": mx.ones((4, 4)),
            "model.local_transformer.local_lm_heads.1.weight": mx.ones((4, 4)),
            "local_transformer.layer_norm_before_lm_heads.0.weight": mx.ones((4,)),
            "model.local_transformer.layer_norm_before_lm_heads.1.weight": mx.ones((4,)),
            "local_transformer.layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
            "speech_embedding_to_local_mlp.gate_proj.weight": mx.ones((4, 4)),
            "local_to_speech_embedding_mlps.0.gate_proj.weight": mx.ones((4, 4)),
            "layer_norm_before_lm_heads.0.weight": mx.ones((4,)),
            "lm_heads.0.weight": mx.ones((4, 4)),
            "foo.num_batches_tracked": mx.array(0),
        }

        sanitized = model.sanitize(weights)
        self.assertIn("model.backbone.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertIn("model.embedding_list.1.weight", sanitized)
        self.assertIn("model.embedding_list.2.weight", sanitized)
        self.assertIn("model.embedding_list.0.weight", sanitized)
        self.assertIn("model.local_transformer.layers.0.self_attn.k_proj.weight", sanitized)
        self.assertIn("model.local_transformer.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertIn("model.lm_heads.0.weight", sanitized)
        self.assertIn("model.lm_heads.1.weight", sanitized)
        self.assertIn("model.layer_norm_before_lm_heads.1.weight", sanitized)
        self.assertIn("model.speech_embedding_to_local_mlp.gate_proj.weight", sanitized)
        self.assertIn(
            "model.local_to_speech_embedding_mlps.0.gate_proj.weight",
            sanitized,
        )
        self.assertIn("model.layer_norm_before_lm_heads.0.weight", sanitized)
        self.assertIn("model.lm_heads.0.weight", sanitized)
        self.assertNotIn("foo.num_batches_tracked", sanitized)
        self.assertNotIn("model.embedding_list.weight", sanitized)
        self.assertNotIn("model.local_transformer.embed_tokens.0.weight", sanitized)
        self.assertNotIn("model.local_transformer.embed_tokens.1.weight", sanitized)
        self.assertTrue(all(key in parameter_names for key in sanitized))

        self.assertFalse(
            model.model_quant_predicate("model.embedding_list.0", nn.Embedding(2, 2))
        )
        self.assertFalse(model.model_quant_predicate("model.lm_heads.0", nn.Linear(2, 2)))
        self.assertFalse(
            model.model_quant_predicate("model.layer_norm_before_lm_heads.0", nn.RMSNorm(2))
        )
        self.assertTrue(model.model_quant_predicate("model.backbone.layers.0", nn.Linear(2, 2)))

    def test_realtime_model_applies_preset_defaults(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        model = Model(config)

        class _PresetProcessor:
            def __init__(self):
                self.audio_tokenizer = object()

            @staticmethod
            def tokens_from_text(text: str, add_special_tokens: bool = False):
                del text, add_special_tokens
                return [5, 6]

            @staticmethod
            def make_text_prefix(token_ids):
                return list(token_ids)

            @staticmethod
            def encode_prompt_audio(_):
                return mx.array([[1, 2]], dtype=mx.int32)

        captured = {}

        class _PresetSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            @staticmethod
            def reset_turn(
                user_text: str,
                user_audio_tokens: Optional[mx.array] = None,
                include_system_prompt: bool = True,
                reset_cache: bool = True,
            ):
                del user_text, user_audio_tokens, include_system_prompt, reset_cache

            @staticmethod
            def push_text_tokens(token_ids):
                del token_ids
                return []

            @staticmethod
            def end_text():
                return []

            @staticmethod
            def drain(*, max_steps: Optional[int] = None):
                del max_steps
                return [mx.zeros((8,), dtype=mx.float32)]

            @staticmethod
            def close():
                return None

        model.processor = _PresetProcessor()
        model.tokenizer = _TinyTokenizer()

        with patch(
            "mlx_audio.tts.models.moss_tts_realtime.model.MossTTSRealtimeInference",
            return_value=SimpleNamespace(),
        ), patch(
            "mlx_audio.tts.models.moss_tts_realtime.model.RealtimeSession",
            side_effect=lambda **kwargs: _PresetSession(**kwargs),
        ):
            results = list(
                model.generate(
                    text="hello",
                    preset="realtime",
                    max_tokens=4,
                )
            )

        self.assertEqual(captured["temperature"], 0.8)
        self.assertEqual(captured["top_p"], 0.6)
        self.assertEqual(captured["top_k"], 30)
        self.assertEqual(captured["repetition_penalty"], 1.1)
        self.assertEqual(captured["repetition_window"], 50)
        self.assertEqual(len(results), 1)

    def test_realtime_model_rejects_non_realtime_preset(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        model = Model(config)
        model.processor = SimpleNamespace(audio_tokenizer=object())
        model.tokenizer = _TinyTokenizer()

        with self.assertRaisesRegex(ValueError, "not valid for runtime"):
            list(
                model.generate(
                    text="hello",
                    preset="moss_tts",
                    max_tokens=4,
                )
            )


class TestMossTTSRealtimeInferencerTransitions(unittest.TestCase):
    def _build_inferencer(
        self,
        *,
        eos_after: int = 4,
        max_context_tokens: Optional[int] = None,
    ):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        model = _FakeRealtimeModel(config, eos_after=eos_after)
        inferencer = MossTTSRealtimeInference(
            model=model,
            tokenizer=_TinyTokenizer(),
            config=config,
            max_length=12,
            max_context_tokens=max_context_tokens,
        )
        return config, model, inferencer

    def test_step_requires_prefill(self):
        _, _, inferencer = self._build_inferencer()
        with self.assertRaisesRegex(ValueError, "prefill\\(\\) must be called"):
            inferencer.step(None)

    def test_prefill_step_finish_and_reset_paths(self):
        config, model, inferencer = self._build_inferencer(eos_after=3)
        turn = mx.full((2, config.channels), config.audio_pad_token, dtype=mx.int32)
        turn[:, 0] = config.text_pad

        first = inferencer.prefill(
            input_ids=turn,
            text_prefix_ids=[9, 10, 11],
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        self.assertEqual(tuple(first.shape), (1, config.rvq))
        self.assertTrue(inferencer.has_prefilled)
        self.assertEqual(inferencer.step_idx, 1)

        second = inferencer.step(12, do_sample=False, top_k=0, top_p=1.0)
        self.assertEqual(tuple(second.shape), (1, config.rvq))
        self.assertEqual(inferencer.step_idx, 2)

        produced = inferencer.finish(max_steps=8, do_sample=False, top_k=0, top_p=1.0)
        self.assertGreaterEqual(len(produced), 1)
        self.assertTrue(inferencer.is_finished)

        cache_ref = inferencer.cache
        inferencer.reset_generation_state(keep_cache=True)
        self.assertIs(inferencer.cache, cache_ref)
        self.assertEqual(inferencer.step_idx, 0)
        self.assertFalse(inferencer.has_prefilled)

        inferencer.reset_generation_state(keep_cache=False)
        self.assertIsNone(inferencer.cache)
        self.assertEqual(inferencer.generated_tokens, [])
        self.assertIsNone(inferencer.last_audio_tokens)

        # Ensure we exercised both cold and warm calls.
        self.assertGreaterEqual(model.make_cache_calls, 1)
        self.assertGreaterEqual(len(model.model_call_shapes), 2)

    def test_reset_turn_keep_vs_drop_cache(self):
        config, _, inferencer = self._build_inferencer(eos_after=5)
        turn = mx.full((2, config.channels), config.audio_pad_token, dtype=mx.int32)
        turn[:, 0] = config.text_pad
        inferencer.prefill(input_ids=turn, text_prefix_ids=[1, 2], do_sample=False, top_k=0)
        self.assertIsNotNone(inferencer.cache)
        cache_ref = inferencer.cache

        inferencer.reset_turn(
            user_text="hi",
            user_audio_tokens=None,
            include_system_prompt=False,
            reset_cache=False,
        )
        self.assertIs(inferencer.cache, cache_ref)
        self.assertFalse(inferencer.has_prefilled)

        inferencer.reset_turn(
            user_text="again",
            user_audio_tokens=None,
            include_system_prompt=False,
            reset_cache=True,
        )
        self.assertIsNone(inferencer.cache)

    def test_prefill_broadcasts_flat_prefix_ids_for_batched_inputs(self):
        config, _, inferencer = self._build_inferencer(eos_after=6)
        turn_a = mx.full((2, config.channels), config.audio_pad_token, dtype=mx.int32)
        turn_b = mx.full((3, config.channels), config.audio_pad_token, dtype=mx.int32)
        turn_a[:, 0] = config.text_pad
        turn_b[:, 0] = config.text_pad

        first = inferencer.prefill(
            input_ids=[turn_a, turn_b],
            text_prefix_ids=[9, 10, 11],
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        self.assertEqual(tuple(first.shape), (2, config.rvq))

        second = inferencer.step(
            [12, 13],
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        self.assertEqual(tuple(second.shape), (2, config.rvq))

    def test_cache_cap_rebuilds_cache_when_capacity_exceeded(self):
        config, model, inferencer = self._build_inferencer(
            eos_after=10,
            max_context_tokens=3,
        )
        turn = mx.full((2, config.channels), config.audio_pad_token, dtype=mx.int32)
        turn[:, 0] = config.text_pad
        inferencer.prefill(input_ids=turn, text_prefix_ids=[1, 2], do_sample=False, top_k=0)
        self.assertGreaterEqual(model.make_cache_calls, 2)

    def test_repetition_window_flows_through_prefill_step_finish(self):
        config, model, inferencer = self._build_inferencer(eos_after=9)
        turn = mx.full((2, config.channels), config.audio_pad_token, dtype=mx.int32)
        turn[:, 0] = config.text_pad

        inferencer.prefill(
            input_ids=turn,
            text_prefix_ids=[1, 2, 3],
            do_sample=False,
            top_k=0,
            repetition_window=7,
        )
        inferencer.step(
            4,
            do_sample=False,
            top_k=0,
            repetition_window=5,
        )
        inferencer.finish(
            max_steps=1,
            do_sample=False,
            top_k=0,
            repetition_window=3,
        )

        self.assertEqual(model.repetition_windows_seen[:3], [7, 5, 3])


class TestMossTTSRealtimeSamplingContracts(unittest.TestCase):
    def test_repetition_window_changes_penalized_history_set(self):
        logits = mx.array([[10.0, 9.0, 8.0, 7.0]], dtype=mx.float32)
        history = mx.array([[1, 2, 2, 3]], dtype=mx.int32)

        unbounded = apply_repetition_penalty(
            logits,
            history,
            penalty=2.0,
            repetition_window=None,
        )
        windowed = apply_repetition_penalty(
            logits,
            history,
            penalty=2.0,
            repetition_window=1,
        )

        unbounded_np = np.array(unbounded)
        windowed_np = np.array(windowed)
        self.assertEqual(float(unbounded_np[0, 1]), 4.5)
        self.assertEqual(float(windowed_np[0, 1]), 9.0)
        self.assertEqual(float(unbounded_np[0, 3]), 3.5)
        self.assertEqual(float(windowed_np[0, 3]), 3.5)


class TestMossTTSRealtimeSessionLifecycle(unittest.TestCase):
    def _build_session(self, *, eos_after: int = 5, max_context_tokens: int = 16):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        fake_model = _FakeRealtimeModel(config, eos_after=eos_after)
        inferencer = MossTTSRealtimeInference(
            model=fake_model,
            tokenizer=_TinyTokenizer(),
            config=config,
            max_length=20,
            max_context_tokens=max_context_tokens,
        )
        processor = _FakeRealtimeProcessor(config)
        session = RealtimeSession(
            inferencer=inferencer,
            processor=processor,
            chunk_frames=2,
            overlap_frames=0,
            max_pending_frames=16,
            prefill_text_len=2,
            text_buffer_size=6,
            min_text_chunk_chars=2,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        return config, fake_model, inferencer, processor, session

    def test_session_requires_reset_turn_before_ingest(self):
        _, _, _, _, session = self._build_session()
        with self.assertRaisesRegex(RuntimeError, "reset_turn\\(\\) must be called"):
            session.push_text_tokens([1, 2])

    def test_active_turn_requires_drain_before_reset_or_next_reset_turn(self):
        _, _, _, _, session = self._build_session(eos_after=8)
        session.reset_turn("hello", include_system_prompt=False, reset_cache=False)
        session.push_text_tokens([1, 2])

        with self.assertRaisesRegex(RuntimeError, "drain\\(\\) must be called before reset_turn"):
            session.reset_turn("next", include_system_prompt=False, reset_cache=False)

        with self.assertRaisesRegex(RuntimeError, "end_text\\(\\) \\+ drain\\(\\) are required"):
            session.reset(reset_cache=False)

    def test_end_text_and_drain_lifecycle_then_close(self):
        _, _, inferencer, _, session = self._build_session(eos_after=5)
        session.reset_turn("hello", include_system_prompt=False, reset_cache=False)

        chunks_from_push = session.push_text_tokens([3, 4, 5])
        chunks_from_end = session.end_text()
        chunks_from_drain = session.drain(max_steps=6)
        chunks = [*chunks_from_push, *chunks_from_end, *chunks_from_drain]

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(int(chunk.shape[0]) > 0 for chunk in chunks))
        self.assertTrue(inferencer.is_finished)

        session.reset(reset_cache=False)
        session.close()
        session.close()  # idempotent close
        with self.assertRaisesRegex(RuntimeError, "RealtimeSession is closed"):
            session.reset_turn("after close", include_system_prompt=False)

    def test_close_drains_active_turn_without_leaking_state(self):
        _, _, inferencer, _, session = self._build_session(eos_after=4)
        session.reset_turn("hello", include_system_prompt=False, reset_cache=False)
        session.push_text_tokens([1, 2, 3])
        session.close()

        self.assertTrue(inferencer.cache is None or inferencer.step_idx == 0)
        self.assertFalse(inferencer.has_prefilled)


class TestMossTTSRealtimeBridgeParity(unittest.TestCase):
    def test_delta_bridge_pushes_same_tokens_as_direct_tokenization(self):
        tokenizer = _TinyTokenizer()
        recording_session = _RecordingSession(tokenizer)
        bridge = RealtimeTextDeltaBridge(recording_session, hold_back=1)

        full_text = "Hello realtime bridge"
        bridge.push_text_delta("Hello ")
        bridge.push_text_delta("realtime")
        bridge.push_text_delta(" bridge")
        bridge.end_text()
        bridge.drain(max_steps=4)

        expected = tokenizer.encode(full_text, add_special_tokens=False)
        self.assertEqual(recording_session.recorded_token_ids, expected)
        self.assertEqual(recording_session.end_calls, 1)
        self.assertEqual(recording_session.drain_calls, 1)

    def test_bridge_and_direct_token_paths_produce_equal_chunk_lengths(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        text = "equal parity"
        token_ids = _TinyTokenizer().encode(text, add_special_tokens=False)

        def run_direct():
            model = _FakeRealtimeModel(config, eos_after=5)
            inferencer = MossTTSRealtimeInference(
                model=model,
                tokenizer=_TinyTokenizer(),
                config=config,
                max_length=20,
            )
            processor = _FakeRealtimeProcessor(config)
            session = RealtimeSession(
                inferencer=inferencer,
                processor=processor,
                chunk_frames=2,
                overlap_frames=0,
                max_pending_frames=16,
                prefill_text_len=2,
                do_sample=False,
                top_k=0,
                top_p=1.0,
            )
            session.reset_turn("", include_system_prompt=False, reset_cache=False)
            chunks = []
            chunks.extend(session.push_text_tokens(token_ids))
            chunks.extend(session.end_text())
            chunks.extend(session.drain(max_steps=8))
            session.close()
            return [int(chunk.shape[0]) for chunk in chunks]

        def run_bridge():
            model = _FakeRealtimeModel(config, eos_after=5)
            inferencer = MossTTSRealtimeInference(
                model=model,
                tokenizer=_TinyTokenizer(),
                config=config,
                max_length=20,
            )
            processor = _FakeRealtimeProcessor(config)
            session = RealtimeSession(
                inferencer=inferencer,
                processor=processor,
                chunk_frames=2,
                overlap_frames=0,
                max_pending_frames=16,
                prefill_text_len=2,
                do_sample=False,
                top_k=0,
                top_p=1.0,
            )
            session.reset_turn("", include_system_prompt=False, reset_cache=False)
            bridge = RealtimeTextDeltaBridge(session, hold_back=0)
            chunks = []
            chunks.extend(bridge.push_text_delta("equal "))
            chunks.extend(bridge.push_text_delta("parity"))
            chunks.extend(bridge.end_text())
            chunks.extend(bridge.drain(max_steps=8))
            session.close()
            return [int(chunk.shape[0]) for chunk in chunks]

        self.assertEqual(run_bridge(), run_direct())

    def test_delta_bridge_handles_multilingual_unicode_deltas(self):
        tokenizer = _TinyTokenizer()
        recording_session = _RecordingSession(tokenizer)
        bridge = RealtimeTextDeltaBridge(recording_session, hold_back=0)

        full_text = (
            "\u4f60\u597d "
            "\u0645\u0631\u062d\u0628\u0627 "
            "hello"
        )
        bridge.push_text_delta("\u4f60\u597d ")
        bridge.push_text_delta("\u0645\u0631\u062d\u0628\u0627 ")
        bridge.push_text_delta("hello")
        bridge.end_text()
        bridge.drain(max_steps=4)

        expected = tokenizer.encode(full_text, add_special_tokens=False)
        self.assertEqual(recording_session.recorded_token_ids, expected)


class TestMossTTSRealtimePromptPackingParity(unittest.TestCase):
    def _build_config(self, *, delay_tokens_len: int = 3) -> ModelConfig:
        config_dict = _tiny_realtime_config_dict()
        config_dict["delay_tokens_len"] = int(delay_tokens_len)
        return ModelConfig.from_dict(config_dict)

    def test_make_ensemble_fills_audio_pad_placeholder_rows(self):
        config = self._build_config()
        processor = _build_prompt_parity_processor(config)
        voice_prompt_tokens = mx.array([[11, 12], [13, 14]], dtype=mx.int32)
        normalized_voice_tokens = processor.normalize_audio_prompt_tokens(
            voice_prompt_tokens
        )

        system_prompt = processor.make_ensemble(voice_prompt_tokens)
        system_prompt_np = np.array(system_prompt)
        placeholder_rows = np.where(system_prompt_np[:, 0] == processor.audio_pad_token_id)[0]

        self.assertEqual(len(placeholder_rows), 2)
        np.testing.assert_array_equal(
            system_prompt_np[placeholder_rows, 1:],
            np.array(normalized_voice_tokens),
        )

    def test_make_user_prompt_places_bos_eos_and_assistant_boundary(self):
        config = self._build_config(delay_tokens_len=3)
        processor = _build_prompt_parity_processor(config)
        user_audio_tokens = mx.array([[3, 4], [5, 6]], dtype=mx.int32)
        normalized_user_audio_tokens = processor.normalize_audio_prompt_tokens(
            user_audio_tokens
        )

        long_prompt = processor.make_user_prompt("long text branch", user_audio_tokens)
        short_prompt = processor.make_user_prompt("x", user_audio_tokens)

        assistant_boundary = processor.tokens_from_text(
            "<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        boundary_len = len(assistant_boundary)

        for packed in (long_prompt, short_prompt):
            packed_np = np.array(packed)
            bos_rows = np.where(packed_np[:, 1] == int(config.audio_bos_token))[0]
            eos_rows = np.where(packed_np[:, 1] == int(config.audio_eos_token))[0]
            self.assertEqual(len(bos_rows), 1)
            self.assertEqual(len(eos_rows), 1)
            bos_idx = int(bos_rows[0])
            eos_idx = int(eos_rows[0])
            self.assertLess(bos_idx, eos_idx)
            np.testing.assert_array_equal(
                packed_np[bos_idx + 1 : eos_idx, 1:],
                np.array(normalized_user_audio_tokens),
            )
            self.assertEqual(
                packed_np.shape[1],
                config.channels,
            )
            self.assertGreaterEqual(packed_np.shape[0], boundary_len)
            np.testing.assert_array_equal(
                packed_np[-boundary_len:, 0],
                np.array(assistant_boundary, dtype=np.int32),
            )

    def test_reset_turn_build_and_input_override_share_same_shape_and_voice_separation(self):
        config = self._build_config()
        processor = _build_prompt_parity_processor(config)
        inferencer = MossTTSRealtimeInference(
            model=_FakeRealtimeModel(config, eos_after=7),
            tokenizer=processor.tokenizer,
            config=config,
            max_length=10,
        )
        session = RealtimeSession(
            inferencer=inferencer,
            processor=processor,
            do_sample=False,
            top_k=0,
            top_p=1.0,
        )

        voice_prompt_tokens = mx.array([[21, 22], [23, 24]], dtype=mx.int32)
        user_audio_tokens = mx.array([[7, 8], [9, 10]], dtype=mx.int32)
        normalized_voice_tokens = processor.normalize_audio_prompt_tokens(
            voice_prompt_tokens
        )
        session.set_voice_prompt_tokens(voice_prompt_tokens)

        built = processor.build_turn_input_ids(
            user_text="hello",
            user_audio_tokens=user_audio_tokens,
            include_system_prompt=True,
            voice_prompt_tokens=normalized_voice_tokens,
        )
        session.reset_turn(
            "hello",
            user_audio_tokens=user_audio_tokens,
            include_system_prompt=True,
            reset_cache=False,
        )
        np.testing.assert_array_equal(np.array(session._turn_input_ids), np.array(built))

        session.reset(reset_cache=False)
        session.reset_turn(
            None,
            input_ids=built,
            include_system_prompt=False,
            reset_cache=False,
        )
        self.assertEqual(tuple(session._turn_input_ids.shape), tuple(built.shape))

        packed_np = np.array(built[0])
        system_prompt = np.array(processor.make_ensemble(normalized_voice_tokens))
        user_prompt = np.array(processor.make_user_prompt("hello", user_audio_tokens))
        np.testing.assert_array_equal(packed_np[: system_prompt.shape[0]], system_prompt)
        np.testing.assert_array_equal(packed_np[system_prompt.shape[0] :], user_prompt)

        session.clear_voice_prompt_tokens()
        session.close()


class TestMossTTSRealtimeDecodeFlowControl(unittest.TestCase):
    def test_decode_control_defaults_and_override(self):
        request = RealtimeNormalizedRequest.from_generate_kwargs(text="hello")
        self.assertEqual(request.chunk_frames, 40)
        self.assertEqual(request.overlap_frames, 4)
        self.assertEqual(request.max_pending_frames, 4096)
        self.assertAlmostEqual(request.decode_chunk_duration, 0.32)
        self.assertEqual(request.repetition_window, 50)

        disabled_window = RealtimeNormalizedRequest.from_generate_kwargs(
            text="hello",
            repetition_window=0,
        )
        self.assertIsNone(disabled_window.repetition_window)

        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        processor = _FakeRealtimeProcessor(config)
        decoder = AudioStreamDecoder(
            processor=processor,
            chunk_frames=3,
            overlap_frames=0,
            decode_kwargs={"chunk_duration": 0.25, "mode": "synthetic"},
            max_pending_frames=8,
        )

        tokens = mx.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
            ],
            dtype=mx.int32,
        )
        decoder.push_tokens(tokens)
        chunks = decoder.audio_chunks()
        tail = decoder.flush()
        if tail is not None:
            chunks.append(tail)

        chunk_lengths = [int(chunk.shape[0]) for chunk in chunks]
        self.assertEqual(chunk_lengths, [18, 12])

        self.assertEqual(len(processor.decode_calls), 2)
        self.assertEqual(processor.decode_calls[0].chunk_duration, 0.25)
        self.assertEqual(processor.decode_calls[1].chunk_duration, 0.25)
        self.assertEqual(processor.decode_calls[0].decode_kwargs, {"mode": "synthetic"})
        self.assertEqual(processor.decode_calls[1].decode_kwargs, {"mode": "synthetic"})

        cumulative = np.cumsum(chunk_lengths).tolist()
        self.assertEqual(cumulative, sorted(cumulative))

    def test_overlap_crossfade_non_final_decode_path_is_mlx_safe(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        processor = _FakeRealtimeProcessor(config)
        decoder = AudioStreamDecoder(
            processor=processor,
            chunk_frames=3,
            overlap_frames=1,
            max_pending_frames=8,
        )

        decoder.push_tokens(
            mx.array(
                [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                ],
                dtype=mx.int32,
            )
        )
        chunks = decoder.audio_chunks()
        tail = decoder.flush()

        self.assertEqual([int(chunk.shape[0]) for chunk in chunks], [18])
        self.assertIsNotNone(tail)
        if tail is not None:
            self.assertEqual(int(tail.shape[0]), 12)
        emitted_samples = int(sum(int(chunk.shape[0]) for chunk in chunks)) + int(
            0 if tail is None else int(tail.shape[0])
        )
        decoded_samples = int(sum(call.frames * 6 for call in processor.decode_calls))
        self.assertEqual(emitted_samples, decoded_samples)
        self.assertIsNone(decoder._previous_tail)

    def test_decoder_rejects_invalid_or_overflow_paths(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        processor = _FakeRealtimeProcessor(config)

        with self.assertRaisesRegex(ValueError, "overlap_frames must be smaller"):
            AudioStreamDecoder(
                processor=processor,
                chunk_frames=4,
                overlap_frames=4,
            )

        with self.assertRaisesRegex(ValueError, "overlap_frames must be smaller"):
            RealtimeNormalizedRequest.from_generate_kwargs(
                text="x",
                chunk_frames=2,
                overlap_frames=2,
            )

        decoder = AudioStreamDecoder(
            processor=processor,
            chunk_frames=2,
            overlap_frames=0,
            max_pending_frames=2,
        )
        decoder.push_tokens(mx.array([[1, 2], [3, 4]], dtype=mx.int32))
        with self.assertRaisesRegex(RuntimeError, "buffer overflow"):
            decoder.push_tokens(mx.array([[5, 6]], dtype=mx.int32))


class TestMossTTSRealtimeSyntheticSoak(unittest.TestCase):
    def test_multi_turn_stubbed_soak_keeps_state_bounded(self):
        config = ModelConfig.from_dict(_tiny_realtime_config_dict())
        # Keep EOS far enough away that each turn yields decodeable frames.
        fake_model = _FakeRealtimeModel(config, eos_after=10_000)
        inferencer = MossTTSRealtimeInference(
            model=fake_model,
            tokenizer=_TinyTokenizer(),
            config=config,
            max_length=10,
            max_context_tokens=8,
        )
        processor = _FakeRealtimeProcessor(config)
        session = RealtimeSession(
            inferencer=inferencer,
            processor=processor,
            chunk_frames=2,
            overlap_frames=0,
            max_pending_frames=16,
            prefill_text_len=2,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )

        peak_pending_frames = 0
        max_cache_offset = 0
        latencies_ms: list[float] = []
        for turn_idx in range(20):
            turn_start = time.perf_counter()
            session.reset_turn(
                f"turn {turn_idx}",
                include_system_prompt=(turn_idx == 0),
                reset_cache=False,
            )
            chunks = []
            chunks.extend(session.push_text_tokens([7, 8, 9]))
            chunks.extend(session.end_text())
            chunks.extend(session.drain(max_steps=6))

            self.assertGreater(len(chunks), 0)
            self.assertTrue(all(int(chunk.shape[0]) > 0 for chunk in chunks))

            peak_pending_frames = max(peak_pending_frames, session.decoder.pending_frames)
            if inferencer.cache:
                max_cache_offset = max(
                    max_cache_offset,
                    int(getattr(inferencer.cache[0], "offset", 0)),
                )
            latencies_ms.append((time.perf_counter() - turn_start) * 1000.0)

        session.close()

        self.assertLessEqual(peak_pending_frames, 16)
        self.assertLessEqual(max_cache_offset, inferencer.max_context_tokens)
        self.assertGreaterEqual(fake_model.make_cache_calls, 2)
        self.assertLess(np.mean(latencies_ms), 200.0)


if __name__ == "__main__":
    unittest.main()

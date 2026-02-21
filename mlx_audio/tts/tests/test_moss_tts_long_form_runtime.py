import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_audio.tts.models.moss_tts.config import ModelConfig
from mlx_audio.tts.models.moss_tts.long_form import (
    ContinuityConfig,
    ContinuityState,
    SegmentPlannerConfig,
    advance_continuity_state,
    compose_segment_text,
    compute_prefix_audio_sample_cap,
    evaluate_segment_boundary,
    extract_prefix_audio_tail,
    plan_text_segments,
    trim_prefix_text_window,
)
from mlx_audio.tts.models.moss_tts.model import Model
from mlx_audio.tts.models.moss_tts.processor import MossTTSProcessor


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
        del tokenize
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

    def decode(
        self, audio_codes, return_dict=True, chunk_duration=8.0, num_quantizers=None
    ):
        del return_dict, chunk_duration, num_quantizers
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


class TestMossTTSLongFormPlanner(unittest.TestCase):
    def test_segment_planner_is_deterministic_for_boundary_cases(self):
        cfg = SegmentPlannerConfig(min_chars=48, target_chars=72, max_chars=96)
        source_text = (
            "第一段：混合语言句子。English sentence follows! "
            "Then we continue with punctuation-heavy phrasing; still deterministic?\n\n"
            "第二段：继续测试分段逻辑。Another sentence arrives, with commas, pauses, and stops. "
            "Finally we close this paragraph with one more sentence."
        )

        first = plan_text_segments(source_text, config=cfg)
        second = plan_text_segments(source_text, config=cfg)

        self.assertEqual([seg.text for seg in first], [seg.text for seg in second])
        self.assertGreater(len(first), 1)
        self.assertTrue(all(seg.char_count <= cfg.max_chars for seg in first))
        self.assertTrue(any(seg.text.endswith(("。", "!", "?", ".")) for seg in first))

    def test_segment_planner_splits_long_no_punctuation_text(self):
        cfg = SegmentPlannerConfig(min_chars=24, target_chars=32, max_chars=40)
        source_text = " ".join(["token"] * 80)

        segments = plan_text_segments(source_text, config=cfg)

        self.assertGreater(len(segments), 2)
        self.assertTrue(all(seg.char_count <= cfg.max_chars for seg in segments))

    def test_segment_planner_respects_single_segment_short_input(self):
        cfg = SegmentPlannerConfig(min_chars=40, target_chars=60, max_chars=80)
        segments = plan_text_segments("Short prompt.", config=cfg)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "Short prompt.")


class TestMossTTSLongFormContinuity(unittest.TestCase):
    def test_prefix_audio_sample_cap_prefers_tighter_limit(self):
        cfg = ContinuityConfig(
            prefix_audio_seconds=5.0,
            prefix_audio_max_tokens=25,
            prefix_text_max_chars=0,
        )
        cap = compute_prefix_audio_sample_cap(sample_rate=100, config=cfg)
        self.assertEqual(cap, 200)

        audio = mx.arange(1000, dtype=mx.float32)
        tail = extract_prefix_audio_tail(audio, sample_cap=cap)
        self.assertIsNotNone(tail)
        self.assertEqual(int(tail.shape[0]), 200)
        self.assertEqual(float(tail[0]), 800.0)

    def test_prefix_text_trimming_is_deterministic(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        one = trim_prefix_text_window(text, max_chars=18)
        two = trim_prefix_text_window(text, max_chars=18)

        self.assertEqual(one, two)
        self.assertIsNotNone(one)
        self.assertLessEqual(len(one), 18)

    def test_advance_continuity_state_updates_audio_and_text_bounds(self):
        state = ContinuityState(prefix_audio=None, prefix_text="alpha beta")
        cfg = ContinuityConfig(
            prefix_audio_seconds=1.0,
            prefix_audio_max_tokens=12,
            prefix_text_max_chars=16,
        )

        next_state = advance_continuity_state(
            previous_state=state,
            segment_audio=mx.arange(400, dtype=mx.float32),
            segment_text="gamma delta epsilon",
            sample_rate=100,
            config=cfg,
        )

        self.assertIsNotNone(next_state.prefix_audio)
        self.assertLessEqual(int(next_state.prefix_audio.shape[0]), 96)
        self.assertIsNotNone(next_state.prefix_text)
        self.assertLessEqual(len(next_state.prefix_text), 16)
        self.assertIn(
            "delta",
            compose_segment_text(
                segment_text="tail", prefix_text=next_state.prefix_text
            ),
        )


class TestMossTTSLongFormBoundaryQuality(unittest.TestCase):
    def test_boundary_heuristic_flags_large_discontinuity(self):
        metric = evaluate_segment_boundary(
            left_audio=mx.ones((400,), dtype=mx.float32),
            right_audio=mx.full((400,), -1.0, dtype=mx.float32),
            left_segment_idx=0,
            right_segment_idx=1,
        )
        self.assertIsNotNone(metric)
        self.assertTrue(metric.flagged)
        self.assertGreater(metric.normalized_jump, 1.5)

    def test_boundary_heuristic_keeps_smooth_transition_clean(self):
        left = mx.linspace(0.0, 1.0, 500, dtype=mx.float32)
        right = mx.linspace(1.0, 2.0, 500, dtype=mx.float32)
        metric = evaluate_segment_boundary(
            left_audio=left,
            right_audio=right,
            left_segment_idx=0,
            right_segment_idx=1,
        )
        self.assertIsNotNone(metric)
        self.assertFalse(metric.flagged)


class TestMossTTSLongFormOrchestrator(unittest.TestCase):
    def _build_model(self) -> Model:
        config = ModelConfig.from_dict(_tiny_local_config_dict())
        model = Model(config)
        model.processor = _build_dummy_processor(config)
        model.tokenizer = model.processor.tokenizer
        return model

    def _make_segment_result(
        self,
        model: Model,
        samples: int,
        token_count: int,
        *,
        start_value: float = 0.0,
        end_value: float = 1.0,
    ):
        audio = mx.linspace(start_value, end_value, samples, dtype=mx.float32)
        return model._build_generation_result(
            audio,
            start_time=time.perf_counter() - 0.01,
            token_count=token_count,
            segment_idx=0,
        )

    def test_long_form_emits_aggregated_result_and_metrics(self):
        model = self._build_model()
        long_text = " ".join(["sentence boundary."] * 140)

        captured_requests = []

        def fake_segment(*, request, **kwargs):
            del kwargs
            idx = len(captured_requests)
            captured_requests.append(request)
            return self._make_segment_result(
                model, samples=120 + idx * 10, token_count=40
            )

        with patch.object(
            model, "_generate_single_segment_result", side_effect=fake_segment
        ):
            results = list(
                model.generate(
                    text=long_text,
                    long_form=True,
                    long_form_prefix_text_chars=32,
                    max_tokens=120,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    input_type="text",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertGreater(len(model._last_long_form_segment_metrics), 1)
        self.assertEqual(
            results[0].prompt.get("segments"),
            len(model._last_long_form_segment_metrics),
        )
        self.assertEqual(
            len(captured_requests), len(model._last_long_form_segment_metrics)
        )
        self.assertEqual(
            len(model._last_long_form_boundary_metrics),
            max(0, len(model._last_long_form_segment_metrics) - 1),
        )

        self.assertIsNone(captured_requests[0].reference)
        self.assertIsNotNone(captured_requests[1].reference)
        self.assertIsInstance(captured_requests[1].reference[-1], mx.array)
        self.assertIn("\n", captured_requests[1].text)
        self.assertEqual(
            model._last_long_form_segment_metrics[0].boundary_note, "segment-start"
        )

    def test_long_form_retries_segment_deterministically(self):
        model = self._build_model()
        long_text = " ".join(["retry boundary."] * 120)
        call_state = {"calls": 0}

        def flaky_segment(*, request, **kwargs):
            del request, kwargs
            call_state["calls"] += 1
            if call_state["calls"] == 1:
                raise RuntimeError("synthetic first-attempt failure")
            return self._make_segment_result(model, samples=100, token_count=24)

        with patch.object(
            model, "_generate_single_segment_result", side_effect=flaky_segment
        ):
            results = list(
                model.generate(
                    text=long_text,
                    long_form=True,
                    long_form_retry_attempts=1,
                    max_tokens=80,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    input_type="text",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertGreater(
            call_state["calls"], len(model._last_long_form_segment_metrics)
        )
        self.assertEqual(model._last_long_form_segment_metrics[0].retry_count, 1)

    def test_long_form_streaming_emits_monotonic_chunks_without_silent_drop(self):
        model = self._build_model()
        long_text = " ".join(["stream boundary."] * 120)

        call_idx = {"value": 0}

        def fake_segment(*, request, **kwargs):
            del request, kwargs
            idx = call_idx["value"]
            call_idx["value"] += 1
            return self._make_segment_result(
                model,
                samples=100 + idx * 20,
                token_count=30 + idx,
                start_value=float(idx),
                end_value=float(idx) + 1.0,
            )

        with patch.object(
            model, "_generate_single_segment_result", side_effect=fake_segment
        ):
            streamed = list(
                model.generate(
                    text=long_text,
                    long_form=True,
                    stream=True,
                    max_tokens=120,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    input_type="text",
                )
            )

        self.assertGreater(len(streamed), 1)
        self.assertTrue(all(result.is_streaming_chunk for result in streamed))
        self.assertTrue(streamed[-1].is_final_chunk)
        self.assertTrue(all(int(result.samples) > 0 for result in streamed))
        self.assertEqual(len(streamed), len(model._last_long_form_segment_metrics))

        cumulative = [int(result.prompt["cumulative_samples"]) for result in streamed]
        self.assertEqual(cumulative, sorted(cumulative))
        self.assertEqual(
            cumulative[-1], sum(int(result.samples) for result in streamed)
        )

    def test_long_form_non_stream_metrics_are_monotonic_and_non_silent(self):
        model = self._build_model()
        long_text = " ".join(["monotonic boundary."] * 120)

        call_idx = {"value": 0}

        def fake_segment(*, request, **kwargs):
            del request, kwargs
            idx = call_idx["value"]
            call_idx["value"] += 1
            return self._make_segment_result(
                model,
                samples=90 + idx * 15,
                token_count=22,
                start_value=float(idx),
                end_value=float(idx) + 0.5,
            )

        with patch.object(
            model, "_generate_single_segment_result", side_effect=fake_segment
        ):
            results = list(
                model.generate(
                    text=long_text,
                    long_form=True,
                    max_tokens=90,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    input_type="text",
                )
            )

        self.assertEqual(len(results), 1)
        segment_samples = [
            int(metric.emitted_samples)
            for metric in model._last_long_form_segment_metrics
        ]
        self.assertTrue(all(samples > 0 for samples in segment_samples))

        cumulative = []
        running = 0
        for samples in segment_samples:
            running += samples
            cumulative.append(running)
        self.assertEqual(cumulative, sorted(cumulative))
        self.assertEqual(int(results[0].prompt["cumulative_samples"]), cumulative[-1])

    def test_long_form_calls_cache_boundaries_each_segment_attempt(self):
        model = self._build_model()
        long_text = " ".join(["cache boundary."] * 100)

        with patch.object(model, "_generate_single_segment_result") as segment_mock:
            segment_mock.side_effect = lambda **kwargs: self._make_segment_result(
                model, samples=90, token_count=20
            )
            with patch(
                "mlx_audio.tts.models.moss_tts.model.mx.clear_cache"
            ) as clear_cache_mock:
                list(
                    model.generate(
                        text=long_text,
                        long_form=True,
                        max_tokens=80,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        repetition_penalty=1.0,
                        input_type="text",
                    )
                )

        segment_count = len(model._last_long_form_segment_metrics)
        self.assertGreater(segment_count, 1)
        self.assertGreaterEqual(clear_cache_mock.call_count, segment_count * 2)


if __name__ == "__main__":
    unittest.main()

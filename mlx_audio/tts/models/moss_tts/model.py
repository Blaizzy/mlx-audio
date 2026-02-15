"""MLX runtime model for MOSS-TTS Local + Delay variants."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.tts.models.base import GenerationResult

from .config import ModelConfig
from .delay_model import MossTTSDelayModel
from .inference_utils import (
    DelaySchedulerState,
    build_delay_audio_sampling_mask,
    build_delay_forced_text_tokens,
    initialize_delay_scheduler_state,
    update_delay_scheduler_state,
)
from .local_model import MossTTSLocalModel
from .long_form import (
    BoundarySeamMetric,
    ContinuityConfig,
    ContinuityState,
    LongFormRuntimeConfig,
    LongFormSegmentMetric,
    SegmentPlannerConfig,
    advance_continuity_state,
    compose_segment_text,
    evaluate_segment_boundary,
    merge_reference_with_prefix_audio,
    plan_text_segments,
)
from .processor import MossTTSProcessor, VALID_INPUT_TYPES
from .request import MossNormalizedRequest
from .sampling import resolve_channel_sampling_configs, sample_channel_token


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _indices_from_mask(mask: mx.array) -> mx.array:
    mask_np = np.array(mask).astype(bool)
    indices = np.nonzero(mask_np)[0].astype(np.int32)
    return mx.array(indices, dtype=mx.int32)


def _mask_any(mask: mx.array) -> bool:
    return bool(np.any(np.array(mask).astype(bool)))


class Model(nn.Module):
    """Unified MOSS-TTS runtime model (Local + Delay)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self._sample_rate = config.sampling_rate

        self.is_local_variant = config.is_local_variant
        if self.is_local_variant:
            self.model = MossTTSLocalModel(config)
        else:
            self.model = MossTTSDelayModel(config)

        self.tokenizer = None
        self.audio_tokenizer: Optional[MossAudioTokenizer] = None
        self.processor: Optional[MossTTSProcessor] = None
        self.generation_config: Dict[str, Union[int, float, bool]] = {}
        self._last_long_form_segment_metrics: List[LongFormSegmentMetric] = []
        self._last_long_form_boundary_metrics: List[BoundarySeamMetric] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def layers(self):
        return self.model.backbone.layers

    def make_cache(self):
        return self.model.make_cache()

    def model_quant_predicate(self, path: str, module) -> bool:
        if isinstance(module, nn.Embedding):
            return False
        skip_patterns = [
            "audio_tokenizer",
            "quantizer",
            "codebook",
            "model.lm_heads",
        ]
        if self.is_local_variant:
            skip_patterns.extend(
                [
                    "model.layer_norm_before_lm_heads",
                    "model.embedding_list",
                ]
            )
        else:
            skip_patterns.extend(
                [
                    "model.emb_ext",
                    "model.text_embedding",
                ]
            )
        return not any(pattern in path for pattern in skip_patterns)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        if self.is_local_variant:
            return self._sanitize_local(weights)
        return self._sanitize_delay(weights)

    def _sanitize_local(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            if key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "model.backbone.", 1)
            elif key.startswith("local_transformer."):
                new_key = key.replace("local_transformer.", "model.local_transformer.", 1)
            elif key.startswith("speech_embedding_to_local_mlp."):
                new_key = key.replace(
                    "speech_embedding_to_local_mlp.",
                    "model.speech_embedding_to_local_mlp.",
                    1,
                )
            elif key.startswith("local_to_speech_embedding_mlps."):
                new_key = key.replace(
                    "local_to_speech_embedding_mlps.",
                    "model.local_to_speech_embedding_mlps.",
                    1,
                )
            elif key.startswith("layer_norm_before_lm_heads."):
                new_key = key.replace(
                    "layer_norm_before_lm_heads.",
                    "model.layer_norm_before_lm_heads.",
                    1,
                )
            elif key.startswith("lm_heads."):
                new_key = key.replace("lm_heads.", "model.lm_heads.", 1)
            else:
                new_key = key

            # Local variant never consumes `language_model.embed_tokens`.
            if new_key == "model.backbone.embed_tokens.weight":
                continue

            sanitized[new_key] = value
        return sanitized

    def _sanitize_delay(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            if key in {
                "language_model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
            }:
                new_key = "model.text_embedding.weight"
            elif key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "model.backbone.", 1)
            elif key.startswith("language_model."):
                new_key = key.replace("language_model.", "model.backbone.", 1)
            elif key.startswith("model.emb_ext."):
                new_key = key
            elif key.startswith("emb_ext."):
                new_key = key.replace("emb_ext.", "model.emb_ext.", 1)
            elif key.startswith("model.lm_heads."):
                new_key = key
            elif key.startswith("lm_heads."):
                new_key = key.replace("lm_heads.", "model.lm_heads.", 1)
            elif key.startswith("local_transformer.") or key.startswith(
                "local_to_speech_embedding_mlps."
            ):
                continue
            elif key.startswith("layer_norm_before_lm_heads."):
                continue
            else:
                new_key = key

            sanitized[new_key] = value
        return sanitized

    def _build_generation_result(
        self,
        audio: mx.array,
        *,
        start_time: float,
        token_count: int,
        segment_idx: int,
        is_streaming_chunk: bool = False,
        is_final_chunk: bool = False,
    ) -> GenerationResult:
        samples = int(audio.shape[0])
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        audio_duration_seconds = samples / self.sample_rate
        rtf = audio_duration_seconds / elapsed

        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=_format_duration(audio_duration_seconds),
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": token_count / elapsed,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples / elapsed,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

    def _decode_local_audio_rows(self, audio_rows: List[mx.array]) -> mx.array:
        if self.processor is None:
            raise RuntimeError("processor is not initialized")
        if len(audio_rows) == 0:
            raise RuntimeError("no audio rows available to decode")
        audio_codes = mx.stack(audio_rows, axis=0).astype(mx.int32)
        audio = self.processor.decode_audio_codes(audio_codes, chunk_duration=8.0)
        mx.eval(audio)
        return audio

    def _decode_delay_audio_rows(self, delay_rows: List[mx.array]) -> mx.array:
        if self.processor is None:
            raise RuntimeError("processor is not initialized")
        if len(delay_rows) < self.config.n_vq:
            return mx.zeros((0,), dtype=mx.float32)
        delayed_codes = mx.stack(delay_rows, axis=0).astype(mx.int32)
        complete_codes = self.processor.extract_complete_delay_rows(delayed_codes)
        if complete_codes.shape[0] == 0:
            return mx.zeros((0,), dtype=mx.float32)
        audio = self.processor.decode_audio_codes(complete_codes, chunk_duration=8.0)
        mx.eval(audio)
        return audio

    def _resolve_generation_request(
        self,
        *,
        text: Optional[str],
        ref_audio: Optional[Union[str, mx.array]],
        ref_text: Optional[str],
        instruct: Optional[str],
        **kwargs,
    ) -> MossNormalizedRequest:
        instruction = kwargs.pop("instruction", None)
        if instruction is not None and instruct is not None and instruction != instruct:
            raise ValueError(
                "Both instruction and instruct were provided with different values"
            )

        if ref_text is not None:
            transcript_context = f"Reference transcript: {ref_text}"
            base_instruction = instruction if instruction is not None else instruct
            instruction = (
                transcript_context
                if base_instruction is None
                else f"{base_instruction}\n{transcript_context}"
            )
            instruct = None

        return MossNormalizedRequest.from_generate_kwargs(
            text=text,
            reference=kwargs.pop("reference", None),
            instruction=instruction,
            instruct=instruct,
            ref_audio=ref_audio,
            tokens=kwargs.pop("tokens", None),
            duration_s=kwargs.pop("duration_s", None),
            seconds=kwargs.pop("seconds", None),
            quality=kwargs.pop("quality", None),
            sound_event=kwargs.pop("sound_event", None),
            ambient_sound=kwargs.pop("ambient_sound", None),
            language=kwargs.pop("language", None),
        )

    def _resolve_normalize_inputs(self, value: Optional[bool]) -> bool:
        if value is not None:
            return bool(value)
        # VoiceGenerator checkpoints expose this pair while other MOSS variants do not.
        return bool(
            self.config.gen_token_id is not None
            and self.config.audio_ch0_vocab_size is not None
        )

    def _resolve_local_n_vq_for_inference(self, value: Optional[int]) -> int:
        if value is None:
            return self.config.n_vq
        resolved = int(value)
        if not (1 <= resolved <= self.config.n_vq):
            raise ValueError(
                f"n_vq_for_inference must be in [1, {self.config.n_vq}] "
                f"for this checkpoint, got {resolved}"
            )
        return resolved

    def _ensure_processor_ready(self):
        if self.processor is None or self.tokenizer is None:
            raise RuntimeError(
                "MOSS processor is not initialized. Ensure tokenizer + codec are "
                "available (post_load_hook)."
            )
        if self.processor.audio_tokenizer is None:
            raise RuntimeError(
                "MOSS audio tokenizer is not available. Provide a checkpoint with "
                "embedded codec files or ensure "
                "`OpenMOSS-Team/MOSS-Audio-Tokenizer` is reachable."
            )

    def _resolve_effective_max_tokens(
        self,
        *,
        max_tokens: int,
        request: MossNormalizedRequest,
    ) -> int:
        effective_max_tokens = int(max_tokens)
        if request.tokens is not None:
            effective_max_tokens = min(effective_max_tokens, int(request.tokens))

        if request.reference:
            text_token_count = 0
            if self.tokenizer is not None and request.text:
                text_token_count = len(self.tokenizer.encode(request.text))
            reference_safe_cap = max(75, text_token_count * 6)
            effective_max_tokens = min(effective_max_tokens, reference_safe_cap)

        return max(effective_max_tokens, 1)

    def _build_prompt_inputs(
        self,
        *,
        request: MossNormalizedRequest,
        input_type: str,
        normalize_inputs: bool,
    ) -> mx.array:
        if self.processor is None:
            raise RuntimeError("processor is not initialized")
        user_message = self.processor.build_user_message(
            **request.to_user_message_kwargs(),
            input_type=input_type,
            normalize=normalize_inputs,
        )
        batch = self.processor.prepare_generation_inputs(
            user_message,
            n_vq=self.config.n_vq,
            apply_chat_template=True,
            normalize_inputs=normalize_inputs,
        )
        return batch["input_ids"].astype(mx.int32)

    def _build_prompt_inputs_from_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        mode: str,
        normalize_inputs: bool,
    ) -> mx.array:
        if self.processor is None:
            raise RuntimeError("processor is not initialized")
        batch = self.processor.prepare_generation_inputs(
            list(messages),
            n_vq=self.config.n_vq,
            apply_chat_template=True,
            mode=mode,
            normalize_inputs=normalize_inputs,
        )
        return batch["input_ids"].astype(mx.int32)

    def _build_ttsd_continuation_messages(
        self,
        *,
        request: MossNormalizedRequest,
        dialogue_speakers: Sequence[Mapping[str, Any]],
        input_type: str,
        normalize_inputs: bool,
    ) -> List[Dict[str, Any]]:
        if self.processor is None:
            raise RuntimeError("processor is not initialized")
        if (
            isinstance(dialogue_speakers, (str, bytes))
            or not isinstance(dialogue_speakers, Sequence)
            or len(dialogue_speakers) == 0
        ):
            raise ValueError(
                "dialogue_speakers must be a non-empty sequence of speaker mappings"
            )
        for entry in dialogue_speakers:
            if not isinstance(entry, Mapping):
                raise ValueError("Each dialogue_speakers entry must be a mapping")
        if not request.text:
            raise ValueError("dialogue_speakers requires a non-empty `text` prompt")

        messages = self.processor.build_ttsd_continuation_messages(
            dialogue_text=request.text,
            speakers=dialogue_speakers,
            instruction=request.instruction,
            tokens=request.tokens,
            quality=request.quality,
            sound_event=request.sound_event,
            ambient_sound=request.ambient_sound,
            language=request.language,
            input_type=input_type,
            n_vq=self.config.n_vq,
            normalize_inputs=normalize_inputs,
        )
        return messages

    def _resolve_long_form_runtime_config(
        self,
        kwargs: Dict[str, Any],
    ) -> tuple[bool, LongFormRuntimeConfig]:
        long_form_enabled = bool(kwargs.pop("long_form", False))
        if not long_form_enabled:
            return False, LongFormRuntimeConfig()
        planner = SegmentPlannerConfig(
            min_chars=int(kwargs.pop("long_form_min_chars", 160)),
            target_chars=int(kwargs.pop("long_form_target_chars", 320)),
            max_chars=int(kwargs.pop("long_form_max_chars", 520)),
        )
        continuity = ContinuityConfig(
            prefix_audio_seconds=float(kwargs.pop("long_form_prefix_audio_seconds", 2.0)),
            prefix_audio_max_tokens=int(kwargs.pop("long_form_prefix_audio_max_tokens", 25)),
            prefix_text_max_chars=int(kwargs.pop("long_form_prefix_text_chars", 0)),
        )
        runtime = LongFormRuntimeConfig(
            planner=planner,
            continuity=continuity,
            retry_attempts=int(kwargs.pop("long_form_retry_attempts", 0)),
        )
        return long_form_enabled, runtime

    def _generate_single_segment_result(
        self,
        *,
        request: MossNormalizedRequest,
        input_type: str,
        normalize_inputs: bool,
        sampling_cfg,
        effective_max_tokens: int,
        n_vq_for_inference: int,
    ) -> GenerationResult:
        input_ids = self._build_prompt_inputs(
            request=request,
            input_type=input_type,
            normalize_inputs=normalize_inputs,
        )

        if self.is_local_variant:
            segment_results = list(
                self._generate_local(
                    input_ids=input_ids,
                    sampling_cfg=sampling_cfg,
                    effective_max_tokens=effective_max_tokens,
                    n_vq_for_inference=n_vq_for_inference,
                    stream=False,
                    streaming_interval=2.0,
                )
            )
        else:
            segment_results = list(
                self._generate_delay(
                    input_ids=input_ids,
                    sampling_cfg=sampling_cfg,
                    effective_max_tokens=effective_max_tokens,
                    stream=False,
                    streaming_interval=2.0,
                )
            )

        if len(segment_results) != 1:
            raise RuntimeError(
                "Long-form segment execution expected exactly one non-streaming "
                f"result, got {len(segment_results)}"
            )
        return segment_results[0]

    def _generate_long_form(
        self,
        *,
        request: MossNormalizedRequest,
        input_type: str,
        normalize_inputs: bool,
        sampling_cfg,
        effective_max_tokens: int,
        n_vq_for_inference: int,
        stream: bool,
        runtime_config: LongFormRuntimeConfig,
    ) -> Generator[GenerationResult, None, None]:
        if request.text is None or not str(request.text).strip():
            raise ValueError("long_form generation requires a non-empty `text` prompt")

        segment_plan = plan_text_segments(request.text, config=runtime_config.planner)
        if not segment_plan:
            raise ValueError("long_form planner produced no segments from input text")

        self._last_long_form_segment_metrics = []
        self._last_long_form_boundary_metrics = []
        continuity_state = ContinuityState()
        base_reference = list(request.reference) if request.reference is not None else None
        merged_audio_segments: List[mx.array] = []
        total_generated_tokens = 0
        total_emitted_samples = 0
        metrics: List[LongFormSegmentMetric] = []
        boundary_metrics: List[BoundarySeamMetric] = []
        previous_segment_audio: Optional[mx.array] = None
        overall_start = time.perf_counter()
        segment_count = len(segment_plan)

        for planned_segment in segment_plan:
            retries_used = 0
            segment_result: Optional[GenerationResult] = None
            segment_failure: Optional[Exception] = None

            for attempt in range(int(runtime_config.retry_attempts) + 1):
                retries_used = attempt
                segment_text = compose_segment_text(
                    segment_text=planned_segment.text,
                    prefix_text=continuity_state.prefix_text,
                )
                segment_reference = merge_reference_with_prefix_audio(
                    base_reference=base_reference,
                    prefix_audio=continuity_state.prefix_audio,
                )
                segment_request = replace(
                    request,
                    text=segment_text,
                    reference=segment_reference,
                )
                try:
                    segment_result = self._generate_single_segment_result(
                        request=segment_request,
                        input_type=input_type,
                        normalize_inputs=normalize_inputs,
                        sampling_cfg=sampling_cfg,
                        effective_max_tokens=effective_max_tokens,
                        n_vq_for_inference=n_vq_for_inference,
                    )
                    break
                except Exception as exc:
                    segment_failure = exc
                    if attempt >= int(runtime_config.retry_attempts):
                        break
                finally:
                    mx.clear_cache()

            if segment_result is None:
                raise RuntimeError(
                    "Long-form segment execution failed at "
                    f"segment_idx={planned_segment.segment_idx} after "
                    f"{int(runtime_config.retry_attempts) + 1} attempt(s)"
                ) from segment_failure

            mx.eval(segment_result.audio)
            if int(segment_result.samples) <= 0:
                raise RuntimeError(
                    "Long-form segment execution produced zero samples at "
                    f"segment_idx={planned_segment.segment_idx}"
                )
            if not stream:
                merged_audio_segments.append(segment_result.audio)
            total_generated_tokens += int(segment_result.token_count)
            total_emitted_samples += int(segment_result.samples)

            boundary_note = "segment-start"
            boundary_metric = evaluate_segment_boundary(
                left_audio=previous_segment_audio,
                right_audio=segment_result.audio,
                left_segment_idx=max(0, planned_segment.segment_idx - 1),
                right_segment_idx=planned_segment.segment_idx,
            )
            if boundary_metric is not None:
                boundary_metrics.append(boundary_metric)
                boundary_note = (
                    "discontinuity-flagged"
                    if boundary_metric.flagged
                    else "seam-clean"
                )
            previous_segment_audio = segment_result.audio

            continuity_state = advance_continuity_state(
                previous_state=continuity_state,
                segment_audio=segment_result.audio,
                segment_text=planned_segment.text,
                sample_rate=self.sample_rate,
                config=runtime_config.continuity,
            )
            prefix_audio_samples = (
                int(continuity_state.prefix_audio.shape[0])
                if continuity_state.prefix_audio is not None
                else 0
            )
            prefix_text_chars = (
                len(continuity_state.prefix_text)
                if continuity_state.prefix_text is not None
                else 0
            )
            metrics.append(
                LongFormSegmentMetric(
                    segment_idx=planned_segment.segment_idx,
                    segment_chars=planned_segment.char_count,
                    start_offset=planned_segment.start_offset,
                    end_offset=planned_segment.end_offset,
                    retry_count=retries_used,
                    prompt_tokens_generated=int(segment_result.token_count),
                    emitted_samples=int(segment_result.samples),
                    emitted_seconds=float(segment_result.samples) / float(self.sample_rate),
                    segment_latency_seconds=float(segment_result.processing_time_seconds),
                    segment_peak_memory_gb=float(segment_result.peak_memory_usage),
                    total_peak_memory_gb=float(mx.get_peak_memory() / 1e9),
                    prefix_audio_samples=prefix_audio_samples,
                    prefix_text_chars=prefix_text_chars,
                    boundary_note=boundary_note,
                )
            )

            if stream:
                segment_elapsed = max(float(segment_result.processing_time_seconds), 1e-6)
                chunk_result = self._build_generation_result(
                    segment_result.audio,
                    start_time=time.perf_counter() - segment_elapsed,
                    token_count=int(segment_result.token_count),
                    segment_idx=planned_segment.segment_idx,
                    is_streaming_chunk=True,
                    is_final_chunk=planned_segment.segment_idx == (segment_count - 1),
                )
                chunk_result.prompt["segments"] = segment_count
                chunk_result.prompt["cumulative_samples"] = total_emitted_samples
                chunk_result.prompt["segment_idx"] = planned_segment.segment_idx
                yield chunk_result
            mx.clear_cache()

        self._last_long_form_segment_metrics = metrics
        self._last_long_form_boundary_metrics = boundary_metrics
        if stream:
            if not metrics:
                raise RuntimeError("long_form generation produced no segment audio")
            return
        if not merged_audio_segments:
            raise RuntimeError("long_form generation produced no segment audio")

        merged_audio = (
            merged_audio_segments[0]
            if len(merged_audio_segments) == 1
            else mx.concatenate(merged_audio_segments, axis=0)
        )
        mx.eval(merged_audio)
        result = self._build_generation_result(
            merged_audio,
            start_time=overall_start,
            token_count=total_generated_tokens,
            segment_idx=0,
        )
        result.prompt["segments"] = segment_count
        result.prompt["cumulative_samples"] = total_emitted_samples
        yield result
        mx.clear_cache()

    def _generate_local(
        self,
        *,
        input_ids: mx.array,
        sampling_cfg,
        effective_max_tokens: int,
        n_vq_for_inference: int,
        stream: bool,
        streaming_interval: float,
    ) -> Generator[GenerationResult, None, None]:
        cache = self.model.make_cache()
        hidden_states = self.model(
            input_ids, cache=cache, n_vq_for_inference=n_vq_for_inference
        )
        global_hidden = hidden_states[:, -1, :]

        chunk_rows = max(1, int(streaming_interval * 12.5))
        chunk_start_time = time.perf_counter()
        last_yielded_samples = 0
        generated_row_count = 0
        audio_rows: List[mx.array] = []

        for _ in range(effective_max_tokens):
            next_tokens = self.model.sample_next_channels(
                global_hidden,
                input_ids,
                sampling_cfg,
                n_vq_for_inference=n_vq_for_inference,
            )
            input_ids = mx.concatenate([input_ids, next_tokens[:, None, :]], axis=1)
            generated_row_count += 1

            text_token = int(next_tokens[0, 0])
            if text_token == self.config.audio_assistant_gen_slot_token_id:
                audio_rows.append(next_tokens[0, 1 : 1 + n_vq_for_inference])

            if text_token == self.config.audio_end_token_id:
                break

            hidden_states = self.model(
                next_tokens[:, None, :],
                cache=cache,
                n_vq_for_inference=n_vq_for_inference,
            )
            global_hidden = hidden_states[:, -1, :]

            if stream and len(audio_rows) > 0 and len(audio_rows) % chunk_rows == 0:
                decoded = self._decode_local_audio_rows(audio_rows)
                if decoded.shape[0] > last_yielded_samples:
                    new_audio = decoded[last_yielded_samples:]
                    last_yielded_samples = int(decoded.shape[0])
                    yield self._build_generation_result(
                        new_audio,
                        start_time=chunk_start_time,
                        token_count=generated_row_count,
                        segment_idx=0,
                        is_streaming_chunk=True,
                        is_final_chunk=False,
                    )
                    chunk_start_time = time.perf_counter()
                mx.clear_cache()

        if len(audio_rows) == 0:
            raise RuntimeError(
                "No audio rows were generated. Check prompt formatting/sampling "
                "configuration for this checkpoint."
            )

        decoded = self._decode_local_audio_rows(audio_rows)
        if stream:
            if decoded.shape[0] > last_yielded_samples:
                final_chunk = decoded[last_yielded_samples:]
            else:
                final_chunk = mx.zeros((0,), dtype=mx.float32)
            yield self._build_generation_result(
                final_chunk,
                start_time=chunk_start_time,
                token_count=generated_row_count,
                segment_idx=0,
                is_streaming_chunk=True,
                is_final_chunk=True,
            )
        else:
            yield self._build_generation_result(
                decoded,
                start_time=chunk_start_time,
                token_count=generated_row_count,
                segment_idx=0,
            )
        mx.clear_cache()

    def _apply_delay_text_constraints(
        self,
        *,
        text_logits: mx.array,
        scheduler_state: DelaySchedulerState,
        step: int,
    ) -> mx.array:
        logits_np = np.array(text_logits, copy=True)
        is_audio_np = np.array(scheduler_state.is_audio).astype(bool)

        disallow_non_audio = np.array(
            [
                self.config.pad_token_id,
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
                self.config.audio_end_token_id,
            ],
            dtype=np.int64,
        )
        allow_audio = np.array(
            [
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
            ],
            dtype=np.int64,
        )
        for row_idx in range(logits_np.shape[0]):
            if is_audio_np[row_idx]:
                row = logits_np[row_idx]
                original = row.copy()
                row[:] = -np.inf
                row[allow_audio] = original[allow_audio]
            else:
                logits_np[row_idx, disallow_non_audio] = -np.inf

        if step == 0:
            logits_np[:, self.config.audio_assistant_delay_slot_token_id] = -np.inf
        if step <= self.config.n_vq:
            logits_np[:, self.config.im_end_token_id] = -np.inf
        return mx.array(logits_np, dtype=text_logits.dtype)

    def _generate_delay(
        self,
        *,
        input_ids: mx.array,
        sampling_cfg,
        effective_max_tokens: int,
        stream: bool,
        streaming_interval: float,
    ) -> Generator[GenerationResult, None, None]:
        cache = self.model.make_cache()
        hidden_states = self.model(input_ids, cache=cache, n_vq_for_inference=self.config.n_vq)
        global_hidden = hidden_states[:, -1, :]
        scheduler_state = initialize_delay_scheduler_state(input_ids, self.config)

        chunk_rows = max(1, int(streaming_interval * 12.5))
        chunk_start_time = time.perf_counter()
        last_yielded_samples = 0
        generated_row_count = 0
        delay_rows: List[mx.array] = []

        for step in range(effective_max_tokens):
            logits_per_channel = self.model.compute_next_logits(
                global_hidden,
                n_vq_for_inference=self.config.n_vq,
            )
            next_text_token, text_sampling_mask, forcing_audio_eos = (
                build_delay_forced_text_tokens(scheduler_state, self.config, self.config.n_vq)
            )

            text_logits = self._apply_delay_text_constraints(
                text_logits=logits_per_channel[0].astype(mx.float32),
                scheduler_state=scheduler_state,
                step=step,
            )
            if _mask_any(text_sampling_mask):
                sampling_indices = _indices_from_mask(text_sampling_mask)
                sampled_text = sample_channel_token(
                    text_logits[sampling_indices],
                    sampling_cfg[0],
                    previous_tokens=input_ids[sampling_indices, :, 0],
                )
                next_text_np = np.array(next_text_token)
                idx_np = np.array(sampling_indices)
                next_text_np[idx_np] = np.array(sampled_text)
                next_text_token = mx.array(next_text_np, dtype=mx.int32)

            next_audio_np = np.full(
                (int(input_ids.shape[0]), self.config.n_vq),
                self.config.audio_pad_code,
                dtype=np.int32,
            )
            audio_sampling_mask = build_delay_audio_sampling_mask(
                scheduler_state,
                n_vq=self.config.n_vq,
            )
            for channel_idx in range(self.config.n_vq):
                mask_column = audio_sampling_mask[:, channel_idx]
                if not _mask_any(mask_column):
                    continue
                row_indices = _indices_from_mask(mask_column)
                sampled_audio = sample_channel_token(
                    logits_per_channel[channel_idx + 1][row_indices],
                    sampling_cfg[channel_idx + 1],
                    previous_tokens=input_ids[row_indices, :, channel_idx + 1],
                )
                next_audio_np[np.array(row_indices), channel_idx] = np.array(sampled_audio)

            next_audio_tokens = mx.array(next_audio_np, dtype=mx.int32)
            next_tokens = mx.concatenate([next_text_token[:, None], next_audio_tokens], axis=1)
            input_ids = mx.concatenate([input_ids, next_tokens[:, None, :]], axis=1)
            generated_row_count += 1

            scheduler_state = update_delay_scheduler_state(
                scheduler_state,
                next_text_token=next_text_token,
                config=self.config,
                n_vq=self.config.n_vq,
                forcing_audio_eos=forcing_audio_eos,
            )

            text_token = int(next_text_token[0])
            if text_token in {
                self.config.audio_start_token_id,
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
            }:
                delay_rows.append(next_audio_tokens[0, : self.config.n_vq])

            if stream and len(delay_rows) >= self.config.n_vq and len(delay_rows) % chunk_rows == 0:
                decoded = self._decode_delay_audio_rows(delay_rows)
                if decoded.shape[0] > last_yielded_samples:
                    new_audio = decoded[last_yielded_samples:]
                    last_yielded_samples = int(decoded.shape[0])
                    yield self._build_generation_result(
                        new_audio,
                        start_time=chunk_start_time,
                        token_count=generated_row_count,
                        segment_idx=0,
                        is_streaming_chunk=True,
                        is_final_chunk=False,
                    )
                    chunk_start_time = time.perf_counter()
                mx.clear_cache()

            if text_token in {self.config.audio_end_token_id, self.config.im_end_token_id}:
                break

            hidden_states = self.model(
                next_tokens[:, None, :],
                cache=cache,
                n_vq_for_inference=self.config.n_vq,
            )
            global_hidden = hidden_states[:, -1, :]

        if len(delay_rows) == 0:
            raise RuntimeError(
                "No audio rows were generated. Check prompt formatting/sampling "
                "configuration for this checkpoint."
            )

        decoded = self._decode_delay_audio_rows(delay_rows)
        if decoded.shape[0] == 0:
            raise RuntimeError(
                "No audio generated after delay de-patterning. Increase token budget "
                "or verify prompt/reference inputs."
            )

        if stream:
            if decoded.shape[0] > last_yielded_samples:
                final_chunk = decoded[last_yielded_samples:]
            else:
                final_chunk = mx.zeros((0,), dtype=mx.float32)
            yield self._build_generation_result(
                final_chunk,
                start_time=chunk_start_time,
                token_count=generated_row_count,
                segment_idx=0,
                is_streaming_chunk=True,
                is_final_chunk=True,
            )
        else:
            yield self._build_generation_result(
                decoded,
                start_time=chunk_start_time,
                token_count=generated_row_count,
                segment_idx=0,
            )
        mx.clear_cache()

    def generate(
        self,
        text: Optional[str] = None,
        voice: Optional[str] = None,  # Unused, retained for generate() compatibility
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        max_tokens: int = 1200,
        verbose: bool = False,
        ref_audio: Optional[Union[str, mx.array]] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        del voice, verbose
        self._ensure_processor_ready()

        conversation = kwargs.pop("conversation", None)
        dialogue_speakers = kwargs.pop("dialogue_speakers", None)
        if conversation is not None and dialogue_speakers is not None:
            raise ValueError("Provide either `conversation` or `dialogue_speakers`, not both")
        long_form_enabled, long_form_runtime = self._resolve_long_form_runtime_config(kwargs)
        if not long_form_enabled:
            self._last_long_form_segment_metrics = []
            self._last_long_form_boundary_metrics = []

        request = self._resolve_generation_request(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            **kwargs,
        )
        normalize_inputs = self._resolve_normalize_inputs(
            kwargs.pop("normalize_inputs", None)
        )
        requested_n_vq_for_inference = kwargs.pop("n_vq_for_inference", None)
        if self.is_local_variant:
            local_n_vq_for_inference = self._resolve_local_n_vq_for_inference(
                requested_n_vq_for_inference
            )
        else:
            if requested_n_vq_for_inference is not None:
                raise ValueError(
                    "n_vq_for_inference is only supported for MOSS-TTS Local checkpoints"
                )
            local_n_vq_for_inference = self.config.n_vq
        input_type = kwargs.pop("input_type", "text")
        if input_type not in VALID_INPUT_TYPES:
            raise ValueError(
                f"Unsupported input_type '{input_type}'. "
                f"Expected one of {sorted(VALID_INPUT_TYPES)}"
            )
        if long_form_enabled:
            if conversation is not None or dialogue_speakers is not None:
                raise ValueError(
                    "long_form cannot be combined with conversation/dialogue_speakers"
                )
            if request.text is None or not str(request.text).strip():
                raise ValueError("long_form requires non-empty text input")
        input_ids: Optional[mx.array] = None
        if not long_form_enabled:
            if conversation is not None:
                if not isinstance(conversation, list):
                    raise ValueError("conversation must be a list of message dicts")
                if request.reference is not None or request.instruction is not None or request.text:
                    raise ValueError(
                        "When `conversation` is provided, do not also provide text/ref/instruct fields"
                    )
                input_ids = self._build_prompt_inputs_from_messages(
                    messages=conversation,
                    mode="continuation",
                    normalize_inputs=normalize_inputs,
                )
            elif dialogue_speakers is not None:
                if request.reference is not None:
                    raise ValueError(
                        "dialogue_speakers cannot be combined with ref_audio/reference"
                    )
                continuation_messages = self._build_ttsd_continuation_messages(
                    request=request,
                    dialogue_speakers=dialogue_speakers,
                    input_type=input_type,
                    normalize_inputs=normalize_inputs,
                )
                continuation_mode = (
                    "continuation"
                    if continuation_messages and continuation_messages[-1]["role"] == "assistant"
                    else "generation"
                )
                input_ids = self._build_prompt_inputs_from_messages(
                    messages=continuation_messages,
                    mode=continuation_mode,
                    normalize_inputs=normalize_inputs,
                )
            else:
                input_ids = self._build_prompt_inputs(
                    request=request,
                    input_type=input_type,
                    normalize_inputs=normalize_inputs,
                )
        do_samples = kwargs.get("do_samples", None)
        layers = kwargs.get("layers", None)
        sampling_cfg = resolve_channel_sampling_configs(
            self.config.channels,
            default_temperature=temperature,
            default_top_p=top_p,
            default_top_k=top_k,
            default_repetition_penalty=repetition_penalty,
            do_samples=do_samples,
            layers=layers,
        )
        effective_max_tokens = self._resolve_effective_max_tokens(
            max_tokens=max_tokens,
            request=request,
        )
        if long_form_enabled:
            yield from self._generate_long_form(
                request=request,
                input_type=input_type,
                normalize_inputs=normalize_inputs,
                sampling_cfg=sampling_cfg,
                effective_max_tokens=effective_max_tokens,
                n_vq_for_inference=local_n_vq_for_inference,
                stream=stream,
                runtime_config=long_form_runtime,
            )
            return

        if input_ids is None:
            raise RuntimeError("input_ids were not prepared for non-long-form generation")

        if self.is_local_variant:
            yield from self._generate_local(
                input_ids=input_ids,
                sampling_cfg=sampling_cfg,
                effective_max_tokens=effective_max_tokens,
                n_vq_for_inference=local_n_vq_for_inference,
                stream=stream,
                streaming_interval=streaming_interval,
            )
        else:
            yield from self._generate_delay(
                input_ids=input_ids,
                sampling_cfg=sampling_cfg,
                effective_max_tokens=effective_max_tokens,
                stream=stream,
                streaming_interval=streaming_interval,
            )

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - import failure is environment specific
            raise RuntimeError("transformers is required for MOSS-TTS tokenizer") from exc

        model.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        codec_path_candidates = [
            model_path / "audio_tokenizer",
            model_path / "moss_audio_tokenizer",
            model_path / "codec",
        ]
        for candidate in codec_path_candidates:
            if candidate.exists():
                model.audio_tokenizer = MossAudioTokenizer.from_pretrained(candidate)
                break

        if model.audio_tokenizer is None:
            try:
                model.audio_tokenizer = MossAudioTokenizer.from_pretrained(
                    "OpenMOSS-Team/MOSS-Audio-Tokenizer"
                )
            except Exception:
                model.audio_tokenizer = None

        model.processor = MossTTSProcessor(
            tokenizer=model.tokenizer,
            audio_tokenizer=model.audio_tokenizer,
            model_config=model.config,
        )

        generation_config_path = model_path / "generation_config.json"
        if generation_config_path.exists():
            model.generation_config = json.loads(
                generation_config_path.read_text(encoding="utf-8")
            )

        return model


__all__ = ["Model", "ModelConfig"]

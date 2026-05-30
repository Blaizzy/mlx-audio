# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache

from mlx_audio.tts.continuous import TTSBatchEvent, TTSBatchItem, TTSBatchOptions


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class _ActiveRequest:
    sequence_id: int
    text: str
    voice: Optional[str]
    instruct: Optional[str]
    cache: List[Any]
    input_embeds: mx.array
    trailing_text_hidden: mx.array
    tts_pad_embed: mx.array
    generated_codes: List[mx.array] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    trailing_idx: int = 0


class Qwen3TTSBatchSession:
    """Step-wise non-streaming Qwen3 TTS batch session.

    The session keeps per-request state and advances active requests in one
    batched talker step at a time. New pending requests are admitted at step
    boundaries.
    """

    def __init__(self, model, options: TTSBatchOptions):
        if model.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")

        self.model = model
        self.options = options
        self.config = model.config.talker_config
        self.eos_token_id = self.config.codec_eos_token_id
        self.suppress_tokens = model._suppress_codec_tokens(self.eos_token_id)

        self._pending: List[TTSBatchItem] = []
        self._active: List[_ActiveRequest] = []
        self._start_time = time.time()
        self._step_count = 0

    @property
    def idle(self) -> bool:
        return not self._pending and not self._active

    @property
    def available_slots(self) -> int:
        return max(0, self.options.max_batch_size - len(self._active))

    def add(self, items: list[TTSBatchItem]) -> None:
        self._pending.extend(items)

    def cancel(self, sequence_id: int) -> None:
        self._pending = [
            item for item in self._pending if item.sequence_id != sequence_id
        ]
        self._active = [
            state for state in self._active if state.sequence_id != sequence_id
        ]

    def step(self) -> list[TTSBatchEvent]:
        events: list[TTSBatchEvent] = []
        if self._active:
            events.extend(self._advance_active())

        if self.available_slots > 0 and self._pending:
            events.extend(self._admit_pending())

        self._step_count += 1
        if self._step_count > 0 and self._step_count % 50 == 0:
            mx.clear_cache()

        return events

    def _take_pending_batch(self) -> list[TTSBatchItem]:
        batch_size = min(self.available_slots, len(self._pending))
        batch = self._pending[:batch_size]
        del self._pending[:batch_size]
        return batch

    def _admit_pending(self) -> list[TTSBatchEvent]:
        pending = self._take_pending_batch()
        if not pending:
            return []

        if self.options.max_tokens <= 0:
            return [self._empty_event(item.sequence_id) for item in pending]

        batch_inputs = self.model._prepare_batch_inputs(
            [item.text for item in pending],
            language=self.options.lang_code,
            speakers=[item.voice for item in pending],
            instructs=[item.instruct for item in pending],
            return_metadata=True,
        )

        cache = self._make_batch_cache(batch_inputs.left_padding)
        logits, hidden = self.model.talker(
            batch_inputs.input_embeds,
            cache=cache,
            attention_mask=batch_inputs.attention_mask,
        )
        next_token_batch, code_tokens, all_codes = self._sample_code_tokens(
            logits,
            hidden,
            generated_tokens_per_seq=[[] for _ in pending],
        )
        finished = next_token_batch[:, 0] == self.eos_token_id
        next_input_embeds = self._next_input_embeds_for_batch(
            batch_inputs.trailing_text_hidden,
            batch_inputs.tts_pad_embed,
            mx.zeros((len(pending), 1), dtype=mx.int32),
            code_tokens,
        )

        caches_by_request = self._extract_batch_cache_rows(cache, len(pending))
        finished_cpu = finished.tolist()
        token_ids_cpu = next_token_batch[:, 0].tolist()

        events: list[TTSBatchEvent] = []
        for index, item in enumerate(pending):
            trailing_len = batch_inputs.trailing_lens[index]
            state = _ActiveRequest(
                sequence_id=item.sequence_id,
                text=item.text,
                voice=item.voice,
                instruct=item.instruct,
                cache=caches_by_request[index],
                input_embeds=next_input_embeds[index : index + 1],
                trailing_text_hidden=batch_inputs.trailing_text_hidden[
                    index : index + 1, :trailing_len, :
                ],
                tts_pad_embed=batch_inputs.tts_pad_embed,
                trailing_idx=1,
            )

            if not finished_cpu[index]:
                state.generated_token_ids.append(token_ids_cpu[index])
                state.generated_codes.append(all_codes[index : index + 1])

            if (
                finished_cpu[index]
                or len(state.generated_codes) >= self.options.max_tokens
            ):
                events.append(self._decode_state(state))
            else:
                self._active.append(state)

        return events

    def _advance_active(self) -> list[TTSBatchEvent]:
        states = list(self._active)
        batch_size = len(states)
        cache = self._merge_state_caches(states)
        input_embeds = mx.concatenate([state.input_embeds for state in states], axis=0)
        attention_mask = self._attention_mask_for_states(states)

        logits, hidden = self.model.talker(
            input_embeds,
            cache=cache,
            attention_mask=attention_mask,
        )
        next_token_batch, code_tokens, all_codes = self._sample_code_tokens(
            logits,
            hidden,
            generated_tokens_per_seq=[state.generated_token_ids for state in states],
        )
        finished = next_token_batch[:, 0] == self.eos_token_id
        next_input_embeds = self._next_input_embeds_for_states(states, code_tokens)

        caches_by_state = self._extract_batch_cache_rows(cache, batch_size)
        finished_cpu = finished.tolist()
        token_ids_cpu = next_token_batch[:, 0].tolist()

        events: list[TTSBatchEvent] = []
        still_active: list[_ActiveRequest] = []
        for index, state in enumerate(states):
            state.cache = caches_by_state[index]
            state.input_embeds = next_input_embeds[index : index + 1]

            if not finished_cpu[index]:
                state.generated_token_ids.append(token_ids_cpu[index])
                state.generated_codes.append(all_codes[index : index + 1])
                state.trailing_idx += 1

            if (
                finished_cpu[index]
                or len(state.generated_codes) >= self.options.max_tokens
            ):
                events.append(self._decode_state(state))
            else:
                still_active.append(state)

        self._active = still_active
        return events

    def _sample_code_tokens(
        self,
        logits: mx.array,
        hidden: mx.array,
        *,
        generated_tokens_per_seq: list[list[int]],
    ) -> Tuple[mx.array, list[mx.array], mx.array]:
        next_token_batch = self.model._sample_token_batch(
            logits,
            temperature=self.options.temperature,
            top_k=self.options.top_k,
            top_p=self.options.top_p,
            repetition_penalty=self.options.repetition_penalty,
            generated_tokens_per_seq=generated_tokens_per_seq,
            suppress_tokens=self.suppress_tokens,
            eos_token_id=self.eos_token_id,
        )

        code_tokens, all_codes = self.model._predict_code_tokens(
            next_token_batch,
            hidden,
            temperature=self.options.temperature,
            top_k=self.options.top_k,
            top_p=self.options.top_p,
        )
        return next_token_batch, code_tokens, all_codes

    def _next_input_embeds_for_batch(
        self,
        trailing_text_hidden: mx.array,
        tts_pad_embed: mx.array,
        trailing_indices: mx.array,
        code_tokens: list[mx.array],
    ) -> mx.array:
        return self.model._next_batch_input_embeds(
            trailing_text_hidden,
            tts_pad_embed,
            trailing_indices,
            code_tokens,
        )

    def _next_input_embeds_for_states(
        self,
        states: list[_ActiveRequest],
        code_tokens: list[mx.array],
    ) -> mx.array:
        text_embeds = []
        for state in states:
            if state.trailing_idx < state.trailing_text_hidden.shape[1]:
                text_embeds.append(
                    state.trailing_text_hidden[
                        :, state.trailing_idx : state.trailing_idx + 1, :
                    ]
                )
            else:
                text_embeds.append(state.tts_pad_embed)
        return mx.concatenate(
            text_embeds, axis=0
        ) + self.model._codec_embeds_for_tokens(code_tokens)

    def _attention_mask_for_states(self, states: list[_ActiveRequest]) -> mx.array:
        cache_lengths = [
            (
                state.cache[0].offset
                if state.cache and state.cache[0].keys is not None
                else 0
            )
            for state in states
        ]
        max_cache_len = max(cache_lengths) if cache_lengths else 0
        mask_rows = []
        for cache_len in cache_lengths:
            left_pad = max_cache_len - cache_len
            mask_rows.append(
                mx.concatenate(
                    [
                        mx.zeros((1, left_pad)),
                        mx.ones((1, cache_len + 1)),
                    ],
                    axis=1,
                )
            )
        return mx.concatenate(mask_rows, axis=0)

    def _make_batch_cache(self, left_padding: list[int]) -> list[BatchKVCache]:
        return [
            BatchKVCache(left_padding) for _ in range(self.config.num_hidden_layers)
        ]

    def _merge_state_caches(self, states: list[_ActiveRequest]) -> list[BatchKVCache]:
        return [
            KVCache.merge([state.cache[layer_idx] for state in states])
            for layer_idx in range(len(states[0].cache))
        ]

    def _extract_batch_cache_rows(
        self,
        cache: list[Any],
        batch_size: int,
    ) -> list[list[Any]]:
        rows: list[list[Any]] = [[] for _ in range(batch_size)]
        for layer_cache in cache:
            for index in range(batch_size):
                rows[index].append(layer_cache.extract(index))
        return rows

    def _decode_state(self, state: _ActiveRequest) -> TTSBatchEvent:
        if not state.generated_codes:
            return self._empty_event(state.sequence_id)

        audio = self.model._decode_generated_codes(state.generated_codes)
        duration_seconds = audio.shape[0] / self.model.sample_rate
        return TTSBatchEvent(
            sequence_id=state.sequence_id,
            audio=audio,
            sample_rate=self.model.sample_rate,
            samples=audio.shape[0],
            token_count=len(state.generated_token_ids),
            done=True,
            metadata={
                "audio_duration": _format_duration(duration_seconds),
                "processing_time_seconds": time.time() - self._start_time,
                "peak_memory_usage": mx.get_peak_memory() / 1e9,
            },
        )

    def _empty_event(self, sequence_id: int) -> TTSBatchEvent:
        audio = mx.zeros((0,), dtype=mx.float32)
        return TTSBatchEvent(
            sequence_id=sequence_id,
            audio=audio,
            sample_rate=self.model.sample_rate,
            samples=0,
            token_count=0,
            done=True,
            metadata={
                "audio_duration": _format_duration(0.0),
                "processing_time_seconds": time.time() - self._start_time,
                "peak_memory_usage": mx.get_peak_memory() / 1e9,
            },
        )

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

from mlx_audio.tts.continuous import TTSBatchEvent, TTSBatchItem, TTSBatchOptions

from .generation import HiggsSamplerState


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class _ActiveRequest:
    sequence_id: int
    text: str
    sampler: HiggsSamplerState
    delayed_rows: list[mx.array] = field(default_factory=list)


class HiggsAudioV3BatchSession:
    """Step-wise non-streaming Higgs Audio v3 batch session."""

    def __init__(self, model, options: TTSBatchOptions):
        self.model = model
        self.options = options
        self._pending: list[TTSBatchItem] = []
        self._active: list[_ActiveRequest] = []
        self._cache: Optional[list[BatchKVCache]] = None
        self._last_hidden_batch: Optional[mx.array] = None
        self._start_time = time.perf_counter()
        self._step_count = 0

    @property
    def idle(self) -> bool:
        return not self._pending and not self._active

    @property
    def available_slots(self) -> int:
        return max(0, int(self.options.max_batch_size) - len(self._active))

    def add(self, items: list[TTSBatchItem]) -> None:
        self._pending.extend(items)

    def cancel(self, sequence_id: int) -> None:
        self._pending = [
            item for item in self._pending if item.sequence_id != sequence_id
        ]
        keep_positions = [
            index
            for index, state in enumerate(self._active)
            if state.sequence_id != sequence_id
        ]
        if len(keep_positions) == len(self._active):
            return

        self._active = [self._active[index] for index in keep_positions]
        if not self._active:
            self._cache = None
            self._last_hidden_batch = None
            return

        keep = mx.array(keep_positions, dtype=mx.int32)
        if self._cache is not None:
            for layer_cache in self._cache:
                layer_cache.filter(keep)
        if self._last_hidden_batch is not None:
            self._last_hidden_batch = self._last_hidden_batch[keep]
            mx.eval(self._last_hidden_batch)

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

        self._validate_items(pending)
        references_by_sequence = self.model._normalize_batch_references(
            batch_size=len(pending),
            ref_audios=[item.ref_audio for item in pending],
            ref_texts=[item.ref_text for item in pending],
        )

        prompt_embeddings = []
        prompt_lengths = []
        for item, references in zip(pending, references_by_sequence):
            prompt_embeds, _prompt_tokens = self.model._build_prompt_embeddings(
                item.text,
                references,
            )
            mx.eval(prompt_embeds)
            prompt_embeddings.append(prompt_embeds)
            prompt_lengths.append(int(prompt_embeds.shape[1]))

        max_prompt_length = max(prompt_lengths)
        left_padding = [max_prompt_length - length for length in prompt_lengths]
        padded_prompt_embeddings = []
        for prompt_embeds, pad in zip(prompt_embeddings, left_padding):
            if pad:
                prompt_embeds = mx.pad(prompt_embeds, [(0, 0), (pad, 0), (0, 0)])
            padded_prompt_embeddings.append(prompt_embeds)

        prompt_batch = mx.concatenate(padded_prompt_embeddings, axis=0)
        cache = self._make_batch_cache(left_padding)
        dummy = mx.zeros((len(pending), max_prompt_length), dtype=mx.int32)
        hidden = self.model.backbone(dummy, cache=cache, input_embeddings=prompt_batch)
        last_hidden_batch = hidden[:, -1, :]
        mx.eval(last_hidden_batch)

        new_states = [
            _ActiveRequest(
                sequence_id=item.sequence_id,
                text=item.text,
                sampler=HiggsSamplerState(
                    num_codebooks=self.model.config.audio_num_codebooks
                ),
            )
            for item in pending
        ]

        if self._cache is None:
            self._cache = cache
            self._last_hidden_batch = last_hidden_batch
        else:
            self._eval_cache(self._cache)
            self._eval_cache(cache)
            active_rows = self._extract_batch_cache_rows(
                self._cache,
                len(self._active),
            )
            new_rows = self._extract_batch_cache_rows(cache, len(pending))
            self._cache = [
                BatchKVCache.merge(
                    [
                        *(row[layer_index] for row in active_rows),
                        *(row[layer_index] for row in new_rows),
                    ]
                )
                for layer_index in range(len(self._cache))
            ]
            self._last_hidden_batch = mx.concatenate(
                [self._last_hidden_batch, last_hidden_batch],
                axis=0,
            )
            mx.eval(self._last_hidden_batch)

        self._active.extend(new_states)
        return []

    def _advance_active(self) -> list[TTSBatchEvent]:
        if self._cache is None or self._last_hidden_batch is None:
            return []

        states = list(self._active)
        batch_size = len(states)
        logits_batch = self.model._audio_logits(self._last_hidden_batch)
        sampled_rows = self.model._step_batch_sampler(
            logits_batch,
            [state.sampler for state in states],
            temperature=float(self.options.temperature),
            top_p=self.options.top_p,
            top_k=self.options.top_k,
        )

        keep_positions = []
        next_codes = []
        events: list[TTSBatchEvent] = []
        still_active: list[_ActiveRequest] = []
        for index, (state, codes) in enumerate(zip(states, sampled_rows)):
            state.delayed_rows.append(codes)
            if (
                state.sampler.generation_done
                or len(state.delayed_rows) >= self.options.max_tokens
            ):
                events.append(self._decode_state(state))
            else:
                keep_positions.append(index)
                next_codes.append(codes)
                still_active.append(state)

        if not still_active:
            self._active = []
            self._cache = None
            self._last_hidden_batch = None
            return events

        if len(still_active) != batch_size:
            keep = mx.array(keep_positions, dtype=mx.int32)
            for layer_cache in self._cache:
                layer_cache.filter(keep)

        codes_batch = mx.stack(next_codes, axis=0)
        next_embed = self.model._embed_audio_codes(codes_batch)[:, None, :]
        decode_dummy = mx.zeros((len(still_active), 1), dtype=mx.int32)
        hidden = self.model.backbone(
            decode_dummy,
            cache=self._cache,
            input_embeddings=next_embed,
        )
        self._last_hidden_batch = hidden[:, -1, :]
        self._active = still_active
        return events

    def _validate_items(self, items: list[TTSBatchItem]) -> None:
        for item in items:
            if item.voice is not None:
                raise ValueError(
                    "Higgs Audio v3 continuous batching does not support voices"
                )
            if item.instruct is not None:
                raise ValueError(
                    "Higgs Audio v3 continuous batching does not support instructs"
                )
            if item.gender not in (None, "male"):
                raise ValueError(
                    "Higgs Audio v3 continuous batching does not support gender"
                )
            if item.speed not in (None, 1.0) or item.pitch not in (None, 1.0):
                raise ValueError(
                    "Higgs Audio v3 continuous batching does not support speed or pitch"
                )

    def _make_batch_cache(self, left_padding: list[int]) -> list[BatchKVCache]:
        return [BatchKVCache(left_padding) for _ in self.model.layers]

    def _eval_cache(self, cache: list[BatchKVCache]) -> None:
        arrays = []
        for layer_cache in cache:
            if layer_cache.keys is not None:
                arrays.extend([layer_cache.keys, layer_cache.values])
            arrays.extend([layer_cache.offset, layer_cache.left_padding])
        if arrays:
            mx.eval(*arrays)

    def _extract_batch_cache_rows(
        self,
        cache: list[BatchKVCache],
        batch_size: int,
    ) -> list[list[object]]:
        rows: list[list[object]] = [[] for _ in range(batch_size)]
        for layer_cache in cache:
            for index in range(batch_size):
                rows[index].append(layer_cache.extract(index))
        return rows

    def _decode_state(self, state: _ActiveRequest) -> TTSBatchEvent:
        audio = self.model._decode_audio(state.delayed_rows)
        mx.eval(audio)
        audio = self.model._apply_fades(
            audio,
            fade_in_ms=30.0,
            fade_out_ms=15.0,
        )
        mx.eval(audio)
        samples = int(audio.shape[0])
        duration_s = samples / self.model.sample_rate if self.model.sample_rate else 0.0
        return TTSBatchEvent(
            sequence_id=state.sequence_id,
            audio=audio,
            sample_rate=self.model.sample_rate,
            samples=samples,
            token_count=len(state.delayed_rows),
            done=True,
            metadata={
                "audio_duration": _format_duration(duration_s),
                "processing_time_seconds": time.perf_counter() - self._start_time,
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
                "processing_time_seconds": time.perf_counter() - self._start_time,
                "peak_memory_usage": mx.get_peak_memory() / 1e9,
            },
        )

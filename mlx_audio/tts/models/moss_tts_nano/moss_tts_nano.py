from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.base import GenerationResult

from .config import ModelConfig
from .gpt2 import GPT2Model
from .sampling import sample_assistant_text_token, sample_next_token
from .text import (
    TextTokenizer,
    build_assistant_prompt_prefix,
    build_prompt_token_ids,
    build_user_prompt_after_reference,
    build_user_prompt_prefix,
    encode_text,
    lightweight_normalize_text,
    load_tokenizer,
    split_text_into_best_sentences,
)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.transformer = GPT2Model(config.gpt2_config, use_token_embedding=True)
        self.audio_embeddings = [
            nn.Embedding(codebook_size, config.gpt2_config.n_embd)
            for codebook_size in config.audio_codebook_sizes
        ]
        self.local_transformer = GPT2Model(
            config.local_gpt2_config(),
            use_token_embedding=False,
        )
        self.tokenizer: TextTokenizer | None = None
        self.audio_tokenizer = None

    @property
    def sample_rate(self) -> int:
        return int(self.config.audio_tokenizer_sample_rate)

    @property
    def model_type(self) -> str:
        return self.config.model_type

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        tokenizer_path = Path(model_path) / "tokenizer.model"
        if tokenizer_path.exists():
            model.tokenizer = load_tokenizer(model_path)
        return model

    def model_quant_predicate(self, path: str, module) -> bool:
        del module
        # Keep small/discrete embeddings and the future codec in full precision by default.
        skip_prefixes = ("audio_embeddings", "audio_tokenizer")
        return not any(path.startswith(prefix) for prefix in skip_prefixes)

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}
        for key, value in weights.items():
            if key == "text_lm_head.weight":
                continue
            if key.startswith("audio_lm_heads."):
                continue
            if key == "local_transformer.wte.weight":
                continue
            if key.startswith("transformer.wpe.") and self.transformer.wpe is None:
                continue
            if (
                key.startswith("local_transformer.wpe.")
                and self.local_transformer.wpe is None
            ):
                continue
            sanitized[key] = value
        return sanitized

    def _ensure_audio_tokenizer(
        self,
        *,
        device: str = "cpu",
        source: str | None = None,
    ):
        if self.audio_tokenizer is None:
            from .audio_tokenizer import MLXMossAudioTokenizer

            self.audio_tokenizer = MLXMossAudioTokenizer.from_model_dir(
                self.config.model_path,
                fallback_source=source
                or self.config.audio_tokenizer_pretrained_name_or_path,
                device=device,
            )
        return self.audio_tokenizer

    def encode_reference_audio(
        self,
        ref_audio,
        *,
        sample_rate: int | None = None,
        num_quantizers: int | None = None,
        device: str = "cpu",
        source: str | None = None,
    ) -> mx.array:
        tokenizer = self._ensure_audio_tokenizer(device=device, source=source)
        return tokenizer.encode_audio(
            ref_audio,
            sample_rate=sample_rate,
            num_quantizers=num_quantizers or self.config.n_vq,
        )

    def decode_audio_token_ids(
        self,
        audio_token_ids: mx.array,
        *,
        num_quantizers: int | None = None,
        device: str = "cpu",
        source: str | None = None,
    ) -> mx.array:
        tokenizer = self._ensure_audio_tokenizer(device=device, source=source)
        return tokenizer.decode_audio_codes(
            audio_token_ids,
            num_quantizers=num_quantizers or self.config.n_vq,
        )

    def _build_generation_result(
        self,
        *,
        audio: mx.array,
        started_at: float,
        token_count: int,
        prompt_token_count: int,
    ) -> GenerationResult:
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        samples = int(audio.shape[0]) if audio.ndim > 0 else 0
        audio_duration_seconds = samples / float(self.sample_rate)
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = (
            f"{int(audio_duration_seconds // 3600):02d}:"
            f"{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"
        )
        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=(audio_duration_seconds / elapsed if elapsed > 0 else 0.0),
            prompt={
                "tokens": prompt_token_count,
                "tokens-per-sec": round(prompt_token_count / elapsed, 2),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2),
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _text_lm_head(self, hidden_states: mx.array) -> mx.array:
        assert self.transformer.wte is not None
        return hidden_states @ self.transformer.wte.weight.T

    def _audio_lm_head(self, hidden_states: mx.array, channel_index: int) -> mx.array:
        return hidden_states @ self.audio_embeddings[channel_index].weight.T

    def _build_inputs_embeds(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                "Expected input_ids shape "
                f"[batch, seq, {self.config.n_vq + 1}], got {input_ids.shape}"
            )
        assert self.transformer.wte is not None
        text_ids = input_ids[..., 0]
        inputs_embeds = self.transformer.wte(text_ids)

        for channel_index, embedding in enumerate(self.audio_embeddings):
            channel_ids = input_ids[..., channel_index + 1]
            valid_mask = channel_ids != self.config.audio_pad_token_id
            safe_ids = mx.where(valid_mask, channel_ids, 0)
            audio_embeds = embedding(safe_ids)
            inputs_embeds = inputs_embeds + audio_embeds * valid_mask[..., None]
        return inputs_embeds

    def _build_text_rows(self, token_ids: Sequence[int]) -> mx.array:
        if len(token_ids) == 0:
            return mx.zeros((0, self.config.n_vq + 1), dtype=mx.int32)
        rows = mx.full(
            (len(token_ids), self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=mx.int32,
        )
        text_ids = mx.array([int(token_id) for token_id in token_ids], dtype=mx.int32)
        rows[:, 0] = text_ids
        return rows

    def _build_audio_prefix_rows(
        self,
        prompt_audio_codes: mx.array,
        slot_token_id: int,
    ) -> mx.array:
        if prompt_audio_codes.ndim != 2:
            raise ValueError(
                "prompt_audio_codes must have shape [frames, n_vq], "
                f"got {prompt_audio_codes.shape}"
            )
        rows = mx.full(
            (prompt_audio_codes.shape[0], self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=mx.int32,
        )
        rows[:, 0] = int(slot_token_id)
        copy_channels = min(prompt_audio_codes.shape[1], self.config.n_vq)
        rows[:, 1 : 1 + copy_channels] = prompt_audio_codes[:, :copy_channels].astype(
            mx.int32
        )
        return rows

    def build_inference_input_ids(
        self,
        *,
        text: str,
        tokenizer: TextTokenizer,
        mode: str = "voice_clone",
        prompt_text: str | None = None,
        prompt_audio_codes: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        normalized_mode = str(mode or "voice_clone").strip().lower()
        if normalized_mode not in {"voice_clone", "continuation"}:
            raise ValueError("mode must be either 'voice_clone' or 'continuation'")

        if normalized_mode == "voice_clone":
            if prompt_audio_codes is None:
                raise ValueError("voice_clone mode requires prompt_audio_codes")
            if prompt_text is not None:
                raise ValueError("voice_clone mode does not accept prompt_text")
            text_token_ids = encode_text(tokenizer, text)
            prefix_token_ids = build_user_prompt_prefix(tokenizer, self.config) + [
                self.config.audio_start_token_id
            ]
            suffix_token_ids = (
                [self.config.audio_end_token_id]
                + build_user_prompt_after_reference(tokenizer)
                + text_token_ids
                + build_assistant_prompt_prefix(tokenizer, self.config)
                + [self.config.audio_start_token_id]
            )
            sections = [
                self._build_text_rows(prefix_token_ids),
                self._build_audio_prefix_rows(
                    prompt_audio_codes,
                    slot_token_id=self.config.audio_user_slot_token_id,
                ),
                self._build_text_rows(suffix_token_ids),
            ]
        else:
            if (prompt_text is None) != (prompt_audio_codes is None):
                raise ValueError(
                    "continuation mode accepts target text only, or both "
                    "prompt_text and prompt_audio_codes"
                )
            effective_text = text if prompt_text is None else prompt_text + text
            prompt_token_ids = build_prompt_token_ids(
                tokenizer,
                self.config,
                encode_text(tokenizer, effective_text),
            )
            sections = [
                self._build_text_rows(prompt_token_ids),
                self._build_text_rows([self.config.audio_start_token_id]),
            ]
            if prompt_audio_codes is not None:
                sections.append(
                    self._build_audio_prefix_rows(
                        prompt_audio_codes,
                        slot_token_id=self.config.audio_assistant_slot_token_id,
                    )
                )

        input_ids = mx.concatenate(sections, axis=0)[None, :, :]
        attention_mask = mx.ones(input_ids.shape[:2], dtype=mx.bool_)
        return input_ids, attention_mask

    def _left_pad_inference_batch(
        self,
        input_id_batches: list[mx.array],
        attention_mask_batches: list[mx.array],
    ) -> tuple[mx.array, mx.array]:
        if not input_id_batches:
            raise ValueError("input_id_batches must not be empty")
        max_seq_len = max(int(batch.shape[1]) for batch in input_id_batches)
        row_width = self.config.n_vq + 1
        padded_ids = []
        padded_masks = []
        for input_ids, attention_mask in zip(input_id_batches, attention_mask_batches):
            seq_len = int(input_ids.shape[1])
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                pad_rows = mx.full(
                    (1, pad_len, row_width),
                    self.config.audio_pad_token_id,
                    dtype=mx.int32,
                )
                pad_rows[:, :, 0] = self.config.pad_token_id
                input_ids = mx.concatenate([pad_rows, input_ids], axis=1)
                attention_mask = mx.concatenate(
                    [mx.zeros((1, pad_len), dtype=mx.bool_), attention_mask],
                    axis=1,
                )
            padded_ids.append(input_ids)
            padded_masks.append(attention_mask)
        return mx.concatenate(padded_ids, axis=0), mx.concatenate(padded_masks, axis=0)

    def _resolve_nq(self, nq: int | None) -> int:
        if nq is None:
            return self.config.n_vq
        resolved_nq = int(nq)
        if resolved_nq < 1 or resolved_nq > self.config.n_vq:
            raise ValueError(
                f"nq must be in [1, {self.config.n_vq}], got {resolved_nq}"
            )
        return resolved_nq

    def generate_audio_token_ids(
        self,
        *,
        prompt_input_ids: mx.array,
        attention_mask: mx.array | None = None,
        nq: int | None = None,
        max_new_frames: int = 375,
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        use_kv_cache: bool = True,
    ) -> mx.array:
        if prompt_input_ids.ndim == 2:
            prompt_input_ids = prompt_input_ids[None, :, :]
        if prompt_input_ids.ndim != 3:
            raise ValueError(
                f"Expected prompt_input_ids with 3 dims, got {prompt_input_ids.shape}"
            )
        if prompt_input_ids.shape[0] != 1:
            raise NotImplementedError(
                "Batched MOSS-TTS-Nano token generation is not implemented yet."
            )
        if attention_mask is None:
            attention_mask = mx.ones(prompt_input_ids.shape[:2], dtype=mx.bool_)

        effective_nq = self._resolve_nq(nq)
        cache = self.transformer.make_cache() if use_kv_cache else None
        current_model_input_ids = prompt_input_ids
        current_attention_mask = attention_mask.astype(mx.bool_)
        generated_frames: list[mx.array] = []

        for _step_index in range(int(max_new_frames)):
            global_inputs_embeds = self._build_inputs_embeds(current_model_input_ids)
            global_outputs = self.transformer(
                inputs_embeds=global_inputs_embeds,
                attention_mask=current_attention_mask,
                cache=cache,
            )
            global_hidden = global_outputs[:, -1, :]

            local_inputs_embeds = global_hidden[:, None, :]
            local_outputs = self.local_transformer(inputs_embeds=local_inputs_embeds)
            local_hidden = local_outputs[:, -1, :]
            text_logits = self._text_lm_head(local_hidden)
            next_text_token = sample_assistant_text_token(
                text_logits,
                audio_assistant_slot_token_id=self.config.audio_assistant_slot_token_id,
                audio_end_token_id=self.config.audio_end_token_id,
                do_sample=do_sample,
                temperature=text_temperature,
                top_k=text_top_k,
                top_p=text_top_p,
            )
            mx.eval(next_text_token)
            if int(next_text_token.item()) != self.config.audio_assistant_slot_token_id:
                break

            assert self.transformer.wte is not None
            current_local_input = self.transformer.wte(next_text_token)
            frame_tokens: list[mx.array] = []
            history = mx.stack(generated_frames, axis=1) if generated_frames else None
            for channel_index in range(effective_nq):
                local_inputs_embeds = mx.concatenate(
                    [local_inputs_embeds, current_local_input[:, None, :]],
                    axis=1,
                )
                local_outputs = self.local_transformer(
                    inputs_embeds=local_inputs_embeds
                )
                local_hidden = local_outputs[:, -1, :]
                channel_logits = self._audio_lm_head(local_hidden, channel_index)
                previous_tokens = (
                    None if history is None else history[:, :, channel_index]
                )
                channel_token = sample_next_token(
                    channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    previous_token_ids=previous_tokens,
                    repetition_penalty=audio_repetition_penalty,
                )
                frame_tokens.append(channel_token)
                current_local_input = self.audio_embeddings[channel_index](
                    channel_token
                )

            frame = mx.stack(frame_tokens, axis=-1)
            if effective_nq < self.config.n_vq:
                pad = mx.full(
                    (frame.shape[0], self.config.n_vq - effective_nq),
                    self.config.audio_pad_token_id,
                    dtype=mx.int32,
                )
                frame = mx.concatenate([frame, pad], axis=-1)
            generated_frames.append(frame)

            text_column = mx.full(
                (frame.shape[0], 1, 1),
                self.config.audio_assistant_slot_token_id,
                dtype=mx.int32,
            )
            next_row = mx.concatenate([text_column, frame[:, None, :]], axis=-1)
            current_model_input_ids = next_row
            current_attention_mask = mx.concatenate(
                [current_attention_mask, mx.ones((frame.shape[0], 1), dtype=mx.bool_)],
                axis=1,
            )
            mx.eval(frame)

            if not use_kv_cache:
                prompt_input_ids = mx.concatenate([prompt_input_ids, next_row], axis=1)
                current_model_input_ids = prompt_input_ids

        if not generated_frames:
            return mx.zeros((1, 0, self.config.n_vq), dtype=mx.int32)
        return mx.stack(generated_frames, axis=1).astype(mx.int32)

    def generate(
        self,
        text: str,
        ref_audio=None,
        ref_text: str | None = None,
        prompt_audio_codes: mx.array | None = None,
        mode: str = "voice_clone",
        stream: bool = False,
        max_tokens: int = 375,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        if stream:
            raise NotImplementedError("MOSS-TTS-Nano streaming is not implemented yet.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        if prompt_audio_codes is None:
            if ref_audio is not None:
                prompt_audio_codes = self.encode_reference_audio(
                    ref_audio,
                    sample_rate=kwargs.get("ref_audio_sample_rate"),
                    num_quantizers=self.config.n_vq,
                    device=str(kwargs.get("audio_tokenizer_device", "cpu")),
                    source=kwargs.get("audio_tokenizer_source"),
                )
            elif str(mode or "voice_clone").strip().lower() == "voice_clone":
                raise ValueError(
                    "voice_clone generation requires ref_audio or prompt_audio_codes."
                )
        elif not isinstance(prompt_audio_codes, mx.array):
            prompt_audio_codes = mx.array(prompt_audio_codes, dtype=mx.int32)

        started_at = time.perf_counter()
        normalized_text = lightweight_normalize_text(text)
        chunks = split_text_into_best_sentences(
            self.tokenizer,
            normalized_text,
            max_tokens=int(kwargs.get("voice_clone_max_text_tokens", 75)),
        )
        all_audio_tokens = []
        prompt_token_count = 0
        normalized_mode = str(mode or "voice_clone").strip().lower()
        for chunk in chunks:
            input_ids, attention_mask = self.build_inference_input_ids(
                text=chunk,
                tokenizer=self.tokenizer,
                mode=normalized_mode,
                prompt_text=ref_text if normalized_mode == "continuation" else None,
                prompt_audio_codes=prompt_audio_codes,
            )
            prompt_token_count += int(input_ids.shape[1])
            audio_tokens = self.generate_audio_token_ids(
                prompt_input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_frames=int(max_tokens),
                do_sample=bool(kwargs.get("do_sample", True)),
                text_temperature=float(kwargs.get("text_temperature", 1.0)),
                text_top_p=float(kwargs.get("text_top_p", 1.0)),
                text_top_k=int(kwargs.get("text_top_k", 50)),
                audio_temperature=float(
                    kwargs.get("audio_temperature", kwargs.get("temperature", 0.8))
                ),
                audio_top_p=float(kwargs.get("audio_top_p", kwargs.get("top_p", 0.95))),
                audio_top_k=int(kwargs.get("audio_top_k", kwargs.get("top_k", 25))),
                audio_repetition_penalty=float(
                    kwargs.get(
                        "audio_repetition_penalty",
                        kwargs.get("repetition_penalty", 1.2),
                    )
                ),
            )
            all_audio_tokens.append(audio_tokens)

        audio_token_ids = (
            mx.concatenate(all_audio_tokens, axis=1)
            if all_audio_tokens
            else mx.zeros((1, 0, self.config.n_vq), dtype=mx.int32)
        )
        audio = self.decode_audio_token_ids(
            audio_token_ids,
            num_quantizers=self.config.n_vq,
            device=str(kwargs.get("audio_tokenizer_device", "cpu")),
            source=kwargs.get("audio_tokenizer_source"),
        )
        yield self._build_generation_result(
            audio=audio,
            started_at=started_at,
            token_count=int(audio_token_ids.shape[1]),
            prompt_token_count=prompt_token_count,
        )

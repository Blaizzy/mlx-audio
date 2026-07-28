from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import BatchKVCache, make_prompt_cache
from mlx_lm.models.qwen3 import Qwen3Model

from ..base import BatchGenerationResult, GenerationResult
from .config import HiggsAudioV3Config
from .generation import (
    HiggsSamplerState,
    apply_delay_pattern,
    reverse_delay_pattern,
    sample_batch,
    step,
)
from .prompt import AUDIO_PLACEHOLDER_ID, HiggsAudioV3PromptBuilder, ReferenceCodes

ModelConfig = HiggsAudioV3Config


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class Model(nn.Module):
    def __init__(self, config: HiggsAudioV3Config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.backbone = Qwen3Model(config.text_config.to_qwen3_args())
        self.multimodal_embedding = nn.Embedding(
            config.audio_num_codebooks * config.audio_codebook_size,
            config.text_config.hidden_size,
        )
        self._tokenizer = None
        self._prompt_builder: Optional[HiggsAudioV3PromptBuilder] = None
        self._codec = None

    @property
    def layers(self):
        return self.backbone.layers

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def codec(self):
        return self._codec

    def __call__(
        self,
        input_ids: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        return self.backbone(input_ids, cache=cache, input_embeddings=input_embeddings)

    def model_quant_predicate(self, name: str, module: nn.Module) -> bool:
        if name.startswith("multimodal_embedding"):
            return False
        return isinstance(module, (nn.Linear, nn.Embedding))

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ....codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        raw = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
        )
        model._tokenizer = tokenizer
        model._prompt_builder = HiggsAudioV3PromptBuilder(tokenizer)

        audio_tokenizer_dir = Path(model_path) / "audio_tokenizer"
        if audio_tokenizer_dir.exists():
            model._codec = HiggsAudioTokenizer.from_pretrained(str(model_path))
        else:
            model._codec = HiggsAudioTokenizer.from_higgs_tts_checkpoint(
                str(model_path)
            )
        return model

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        result: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith("tied.embedding.text_embedding."):
                result[
                    "backbone.embed_tokens."
                    + key[len("tied.embedding.text_embedding.") :]
                ] = value
            elif key.startswith("body.layers."):
                result["backbone.layers." + key[len("body.layers.") :]] = value
            elif key.startswith("body.norm."):
                result["backbone.norm." + key[len("body.norm.") :]] = value
            elif key.startswith("tied.embedding.modality_embeddings.0.embedding."):
                result[
                    "multimodal_embedding."
                    + key[len("tied.embedding.modality_embeddings.0.embedding.") :]
                ] = value
            elif key.startswith("tied.embedding.modality_embeddings.0.model."):
                continue
            elif key.startswith("tied.head."):
                continue
            else:
                result[key] = value
        return result

    def _embed_audio_codes(self, codes: mx.array) -> mx.array:
        if codes.ndim == 1:
            codes = codes[None, :]
        if codes.ndim != 2 or codes.shape[1] != self.config.audio_num_codebooks:
            raise ValueError(
                f"audio codes must be [T, {self.config.audio_num_codebooks}], "
                f"got {codes.shape}"
            )
        offsets = (
            mx.arange(self.config.audio_num_codebooks, dtype=mx.int32)
            * self.config.audio_codebook_size
        )
        fused_ids = codes.astype(mx.int32) + offsets
        return mx.sum(self.multimodal_embedding(fused_ids), axis=-2)

    def _audio_logits(self, hidden: mx.array) -> mx.array:
        flat = self.multimodal_embedding.as_linear(hidden)
        return flat.reshape(
            *hidden.shape[:-1],
            self.config.audio_num_codebooks,
            self.config.audio_codebook_size,
        )

    def _text_embeddings(self, token_ids: list[int]) -> mx.array:
        if not token_ids:
            return mx.zeros((0, self.config.text_config.hidden_size))
        ids = mx.array([token_ids], dtype=mx.int32)
        return self.backbone.embed_tokens(ids)[0]

    def _build_prompt_embeddings(
        self,
        text: str,
        references: list[ReferenceCodes],
    ) -> tuple[mx.array, int]:
        if self._prompt_builder is None:
            raise RuntimeError(
                "Tokenizer missing. Load via mlx_audio.tts.utils.load()."
            )

        prompt = self._prompt_builder.build_prompt(text, references=references)
        pieces = []
        cursor = 0
        for start, delayed_codes in prompt.audio_segments:
            pieces.append(self._text_embeddings(prompt.token_ids[cursor:start]))
            pieces.append(self._embed_audio_codes(delayed_codes))
            cursor = start + int(delayed_codes.shape[0])

        tail_ids = prompt.token_ids[cursor:]
        if any(token_id == AUDIO_PLACEHOLDER_ID for token_id in tail_ids):
            raise ValueError("Internal prompt error: unresolved audio placeholder")
        pieces.append(self._text_embeddings(tail_ids))

        full = mx.concatenate([piece for piece in pieces if piece.shape[0] > 0], axis=0)
        return full[None], len(prompt.token_ids)

    def _normalize_audio(self, audio: Any) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            from mlx_audio.utils import load_audio

            audio = load_audio(str(audio), sample_rate=self.sample_rate)
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] <= 2:
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=-1)
        return arr.reshape(-1).astype(np.float32, copy=False)

    def _prepare_reference_waveform(self, audio: Any) -> np.ndarray:
        audio_np = self._normalize_audio(audio)
        if audio_np.shape[0] < self.sample_rate:
            audio_np = np.pad(audio_np, (0, self.sample_rate - audio_np.shape[0]))
        return np.ascontiguousarray(audio_np, dtype=np.float32)

    def encode_reference_audio(self, audio: Any) -> mx.array:
        """Encode reference audio into delayed Higgs v3 reference codes.

        The returned array can be reused in ``generate(..., ref_audio_codes=...)``
        or as ``{"codes": encoded, "text": ...}`` inside ``references``.
        """
        if self._codec is None:
            raise RuntimeError("Codec missing. Load via mlx_audio.tts.utils.load().")

        audio_np = self._prepare_reference_waveform(audio)
        waveform = mx.array(audio_np).reshape(1, -1, 1)
        codes = self._codec.encode(waveform)[0].astype(mx.int32)
        delayed = apply_delay_pattern(
            codes,
            boc_id=self.config.audio_boc_token_id,
            eoc_id=self.config.audio_eoc_token_id,
        )
        mx.eval(delayed)
        return delayed

    def _encode_reference_audio(self, audio: Any) -> mx.array:
        return self.encode_reference_audio(audio)

    def _normalize_reference_codes(self, codes: Any) -> mx.array:
        normalized = mx.array(codes, dtype=mx.int32)

        if normalized.ndim != 2:
            raise ValueError(
                "reference audio codes must be a 2D array with shape "
                f"[T, {self.config.audio_num_codebooks}], got {normalized.shape}"
            )
        if normalized.shape[1] != self.config.audio_num_codebooks:
            raise ValueError(
                "reference audio codes must have "
                f"{self.config.audio_num_codebooks} codebooks, got "
                f"{normalized.shape[1]}"
            )
        return normalized

    def _normalize_references(
        self,
        ref_audio=None,
        ref_text=None,
        references=None,
        ref_audios=None,
        ref_texts=None,
        ref_audio_codes=None,
        ref_audio_codes_list=None,
    ) -> list[ReferenceCodes]:
        audios = _as_list(ref_audios if ref_audios is not None else ref_audio)
        if ref_audio_codes_list is not None:
            code_values = _as_list(ref_audio_codes_list)
        elif ref_audio_codes is not None:
            code_values = [ref_audio_codes]
        else:
            code_values = []
        texts = _as_list(ref_texts if ref_texts is not None else ref_text)

        if audios and code_values:
            raise ValueError("Use either ref_audio or ref_audio_codes, not both")

        refs = []
        reference_inputs: list[tuple[str, Any]] = []
        reference_inputs.extend(("codes", codes) for codes in code_values)
        reference_inputs.extend(("audio", audio) for audio in audios)

        if texts and len(texts) != len(reference_inputs):
            raise ValueError(
                "ref_text must have the same length as ref_audio/ref_audio_codes"
            )
        if not texts:
            texts = [None] * len(reference_inputs)

        for (kind, value), text in zip(reference_inputs, texts):
            if kind == "codes":
                refs.append(
                    ReferenceCodes(
                        codes=self._normalize_reference_codes(value),
                        text=str(text) if text is not None else None,
                    )
                )
            else:
                refs.append(
                    ReferenceCodes(
                        codes=self.encode_reference_audio(value),
                        text=str(text) if text is not None else None,
                    )
                )

        if references:
            for item in references:
                if isinstance(item, ReferenceCodes):
                    refs.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("ref_text")
                codes = None
                for key in (
                    "codes",
                    "audio_codes",
                    "ref_audio_codes",
                    "reference_codes",
                ):
                    if key in item and item[key] is not None:
                        codes = item[key]
                        break
                if codes is not None:
                    refs.append(
                        ReferenceCodes(
                            codes=self._normalize_reference_codes(codes),
                            text=str(text) if text is not None else None,
                        )
                    )
                    continue
                audio = None
                for key in ("audio", "audio_path", "path", "ref_audio"):
                    if key in item and item[key] is not None:
                        audio = item[key]
                        break
                if audio is None:
                    continue
                refs.append(
                    ReferenceCodes(
                        codes=self.encode_reference_audio(audio),
                        text=str(text) if text is not None else None,
                    )
                )
        return refs

    def _decode_audio(self, delayed_rows: list[mx.array]) -> mx.array:
        if self._codec is None:
            raise RuntimeError("Codec missing. Load via mlx_audio.tts.utils.load().")
        if not delayed_rows:
            return mx.zeros((0,), dtype=mx.float32)
        delayed = mx.stack(delayed_rows, axis=0).astype(mx.int32)
        if delayed.shape[0] < self.config.audio_num_codebooks:
            return mx.zeros((0,), dtype=mx.float32)
        raw_codes = reverse_delay_pattern(delayed)
        audio = self._codec.decode(raw_codes)
        return audio.astype(mx.float32).reshape(-1)

    def _apply_fades(
        self,
        audio: mx.array,
        *,
        fade_in_ms: float,
        fade_out_ms: float,
    ) -> mx.array:
        audio_np = np.array(audio).astype(np.float32, copy=False)
        n_in = int(fade_in_ms * self.sample_rate / 1000.0)
        n_out = int(fade_out_ms * self.sample_rate / 1000.0)
        if n_in > 0 and audio_np.size > n_in:
            audio_np[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        if n_out > 0 and audio_np.size > n_out:
            audio_np[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
        return mx.array(audio_np)

    @staticmethod
    def _normalize_batch_arg(name: str, value: Any, batch_size: int) -> list[Any]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(
                    f"{name} length ({len(value)}) must match texts length "
                    f"({batch_size})"
                )
            return list(value)
        return [value] * batch_size

    @staticmethod
    def _all_equal_hashable(values: list[Any]) -> bool:
        if not values:
            return True
        first = values[0]
        if not isinstance(first, (str, Path, int, float, bool, type(None))):
            return False
        return all(value == first for value in values[1:])

    def _normalize_batch_references(
        self,
        *,
        batch_size: int,
        ref_audio=None,
        ref_text=None,
        references=None,
        ref_audios=None,
        ref_texts=None,
        ref_audio_codes=None,
        ref_audio_codes_list=None,
    ) -> list[list[ReferenceCodes]]:
        ref_audio_items = self._normalize_batch_arg(
            "ref_audios",
            ref_audios if ref_audios is not None else ref_audio,
            batch_size,
        )
        ref_text_items = self._normalize_batch_arg(
            "ref_texts",
            ref_texts if ref_texts is not None else ref_text,
            batch_size,
        )

        has_explicit_shared_ref = ref_audios is None and ref_texts is None
        has_equal_per_item_refs = self._all_equal_hashable(
            ref_audio_items
        ) and self._all_equal_hashable(ref_text_items)
        if references is None and (has_explicit_shared_ref or has_equal_per_item_refs):
            shared_refs = self._normalize_references(
                ref_audio=ref_audio_items[0],
                ref_text=ref_text_items[0],
                ref_audio_codes=ref_audio_codes,
                ref_audio_codes_list=ref_audio_codes_list,
            )
            return [shared_refs] * batch_size

        if ref_audio_codes is not None or ref_audio_codes_list is not None:
            shared_code_refs = self._normalize_references(
                ref_audio_codes=ref_audio_codes,
                ref_audio_codes_list=ref_audio_codes_list,
            )
            return [
                [
                    ReferenceCodes(
                        codes=reference.codes,
                        text=(
                            str(ref_text_items[index])
                            if ref_text_items[index] is not None
                            else reference.text
                        ),
                    )
                    for reference in shared_code_refs
                ]
                for index in range(batch_size)
            ]

        return [
            self._normalize_references(
                ref_audio=ref_audio_items[index],
                ref_text=ref_text_items[index],
                references=references,
            )
            for index in range(batch_size)
        ]

    def _step_batch_sampler(
        self,
        logits_batch: mx.array,
        states: list[HiggsSamplerState],
        *,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> list[mx.array]:
        sampled = sample_batch(
            logits_batch,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        first_codes = sampled[:, 0].tolist()
        codebook_positions = mx.arange(self.config.audio_num_codebooks, dtype=mx.int32)

        rows = []
        for index, state in enumerate(states):
            codes = sampled[index]
            if state.delay_count < state.num_codebooks:
                next_codebook = state.delay_count + 1
                if next_codebook < state.num_codebooks:
                    tail_mask = codebook_positions >= next_codebook
                    codes = mx.where(
                        tail_mask,
                        mx.array(self.config.audio_boc_token_id, dtype=mx.int32),
                        codes,
                    )
                state.delay_count += 1
            elif state.eoc_countdown is not None:
                state.eoc_countdown -= 1
                if state.eoc_countdown <= 0:
                    state.generation_done = True
            elif int(first_codes[index]) == self.config.audio_eoc_token_id:
                if state.num_codebooks <= 2:
                    state.generation_done = True
                else:
                    state.eoc_countdown = state.num_codebooks - 2

            if not state.generation_done:
                state.last_codes = codes
            rows.append(codes)
        return rows

    def _resolve_generation_limit(
        self,
        *,
        max_new_tokens: Optional[int],
        max_new_frames: Optional[int],
        max_tokens: Optional[int],
    ) -> int:
        limit = max_new_tokens
        if limit is None:
            limit = max_new_frames
        if limit is None:
            limit = max_tokens
        if limit is None:
            limit = 2048
        return int(limit)

    def supports_tts_batch(
        self,
        *,
        stream: bool = False,
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        speed: Optional[float] = 1.0,
        gender: Optional[str] = None,
        pitch: Optional[float] = 1.0,
        **kwargs,
    ) -> bool:
        del kwargs
        if stream:
            return False
        if voice is not None or instruct is not None:
            return False
        if gender not in (None, "male"):
            return False
        if speed not in (None, 1.0) or pitch not in (None, 1.0):
            return False
        return True

    def supports_tts_continuous_batch(self, **kwargs) -> bool:
        return self.supports_tts_batch(**kwargs)

    def create_tts_batch_session(self, options):
        from .continuous_batching import HiggsAudioV3BatchSession

        return HiggsAudioV3BatchSession(self, options)

    def batch_generate(
        self,
        texts: list[str],
        voices: Optional[list[Optional[str]]] = None,
        instructs: Optional[list[Optional[str]]] = None,
        speeds: Optional[list[Optional[float]]] = None,
        genders: Optional[list[Optional[str]]] = None,
        pitches: Optional[list[Optional[float]]] = None,
        ref_audio=None,
        ref_text=None,
        references=None,
        ref_audios=None,
        ref_texts=None,
        ref_audio_codes=None,
        ref_audio_codes_list=None,
        max_new_tokens: Optional[int] = None,
        max_new_frames: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        fade_in_ms: float = 30.0,
        fade_out_ms: float = 15.0,
        stream: bool = False,
        **kwargs,
    ) -> Iterator[BatchGenerationResult]:
        del kwargs
        if stream:
            raise NotImplementedError(
                "Higgs Audio v3 batch streaming is not implemented."
            )

        batch_size = len(texts)
        if batch_size == 0:
            return

        voices = self._normalize_batch_arg("voices", voices, batch_size)
        instructs = self._normalize_batch_arg("instructs", instructs, batch_size)
        speeds = self._normalize_batch_arg("speeds", speeds, batch_size)
        genders = self._normalize_batch_arg("genders", genders, batch_size)
        pitches = self._normalize_batch_arg("pitches", pitches, batch_size)

        for index in range(batch_size):
            if voices[index] is not None:
                raise ValueError(
                    "Higgs Audio v3 batch_generate does not support voices"
                )
            if instructs[index] is not None:
                raise ValueError(
                    "Higgs Audio v3 batch_generate does not support instructs"
                )
            if genders[index] not in (None, "male"):
                raise ValueError(
                    "Higgs Audio v3 batch_generate does not support gender"
                )
            if speeds[index] not in (None, 1.0) or pitches[index] not in (None, 1.0):
                raise ValueError(
                    "Higgs Audio v3 batch_generate does not support speed or pitch"
                )

        if seed is not None:
            mx.random.seed(int(seed))

        limit = self._resolve_generation_limit(
            max_new_tokens=max_new_tokens,
            max_new_frames=max_new_frames,
            max_tokens=max_tokens,
        )
        start = time.perf_counter()

        references_by_sequence = self._normalize_batch_references(
            batch_size=batch_size,
            ref_audio=ref_audio,
            ref_text=ref_text,
            references=references,
            ref_audios=ref_audios,
            ref_texts=ref_texts,
            ref_audio_codes=ref_audio_codes,
            ref_audio_codes_list=ref_audio_codes_list,
        )

        states = []
        prompt_embeddings = []
        prompt_lengths = []
        for sequence_idx, text in enumerate(texts):
            prompt_embeds, _prompt_tokens = self._build_prompt_embeddings(
                text,
                references_by_sequence[sequence_idx],
            )
            mx.eval(prompt_embeds)
            prompt_embeddings.append(prompt_embeds)
            prompt_lengths.append(int(prompt_embeds.shape[1]))

            states.append(
                {
                    "sequence_idx": sequence_idx,
                    "sampler": HiggsSamplerState(
                        num_codebooks=self.config.audio_num_codebooks
                    ),
                    "delayed_rows": [],
                }
            )

        max_prompt_length = max(prompt_lengths)
        left_padding = [max_prompt_length - length for length in prompt_lengths]
        padded_prompt_embeddings = []
        for prompt_embeds, pad in zip(prompt_embeddings, left_padding):
            if pad:
                prompt_embeds = mx.pad(prompt_embeds, [(0, 0), (pad, 0), (0, 0)])
            padded_prompt_embeddings.append(prompt_embeds)
        prompt_batch = mx.concatenate(padded_prompt_embeddings, axis=0)
        batch_cache = [BatchKVCache(left_padding) for _ in self.layers]
        dummy = mx.zeros((batch_size, max_prompt_length), dtype=mx.int32)
        hidden = self.backbone(dummy, cache=batch_cache, input_embeddings=prompt_batch)
        last_hidden_batch = hidden[:, -1, :]
        mx.eval(last_hidden_batch)

        active = list(range(batch_size))
        for _ in range(limit):
            if not active:
                break

            logits_batch = self._audio_logits(last_hidden_batch)
            sampler_states = [states[index]["sampler"] for index in active]
            sampled_rows = self._step_batch_sampler(
                logits_batch,
                sampler_states,
                temperature=float(temperature),
                top_p=top_p,
                top_k=top_k,
            )

            next_active = []
            next_codes = []
            keep_positions = []
            for active_position, (state_index, codes) in enumerate(
                zip(active, sampled_rows)
            ):
                state = states[state_index]
                state["delayed_rows"].append(codes)
                if not state["sampler"].generation_done:
                    keep_positions.append(active_position)
                    next_active.append(state_index)
                    next_codes.append(codes)

            if not next_active:
                active = []
                break

            if len(next_active) != len(active):
                keep = mx.array(keep_positions, dtype=mx.int32)
                for layer_cache in batch_cache:
                    layer_cache.filter(keep)

            codes_batch = mx.stack(next_codes, axis=0)
            next_embed = self._embed_audio_codes(codes_batch)[:, None, :]
            decode_dummy = mx.zeros((len(next_active), 1), dtype=mx.int32)
            hidden = self.backbone(
                decode_dummy,
                cache=batch_cache,
                input_embeddings=next_embed,
            )
            last_hidden_batch = hidden[:, -1, :]
            active = next_active

        for state in states:
            audio = self._decode_audio(state["delayed_rows"])
            mx.eval(audio)
            audio = self._apply_fades(
                audio,
                fade_in_ms=fade_in_ms,
                fade_out_ms=fade_out_ms,
            )
            mx.eval(audio)

            elapsed = time.perf_counter() - start
            samples = int(audio.shape[0])
            duration_s = samples / self.sample_rate if self.sample_rate else 0.0
            yield BatchGenerationResult(
                audio=audio,
                sequence_idx=state["sequence_idx"],
                samples=samples,
                sample_rate=self.sample_rate,
                token_count=len(state["delayed_rows"]),
                audio_duration=_format_duration(duration_s),
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio=None,
        ref_text=None,
        references=None,
        ref_audios=None,
        ref_texts=None,
        ref_audio_codes=None,
        ref_audio_codes_list=None,
        max_new_tokens: Optional[int] = None,
        max_new_frames: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        fade_in_ms: float = 30.0,
        fade_out_ms: float = 15.0,
        stream: bool = False,
        **kwargs,
    ) -> Iterator[GenerationResult]:
        del voice, kwargs
        if seed is not None:
            mx.random.seed(int(seed))

        limit = self._resolve_generation_limit(
            max_new_tokens=max_new_tokens,
            max_new_frames=max_new_frames,
            max_tokens=max_tokens,
        )

        start = time.perf_counter()
        references_list = self._normalize_references(
            ref_audio=ref_audio,
            ref_text=ref_text,
            references=references,
            ref_audios=ref_audios,
            ref_texts=ref_texts,
            ref_audio_codes=ref_audio_codes,
            ref_audio_codes_list=ref_audio_codes_list,
        )
        prompt_embeds, prompt_tokens = self._build_prompt_embeddings(
            text, references_list
        )
        mx.eval(prompt_embeds)

        cache = make_prompt_cache(self)
        dummy = mx.zeros((1, prompt_embeds.shape[1]), dtype=mx.int32)
        hidden = self.backbone(dummy, cache=cache, input_embeddings=prompt_embeds)
        last_hidden = hidden[:, -1, :]

        state = HiggsSamplerState(num_codebooks=self.config.audio_num_codebooks)
        delayed_rows: list[mx.array] = []
        for _ in range(int(limit)):
            logits = self._audio_logits(last_hidden)[0]
            codes = step(
                logits,
                state,
                temperature=float(temperature),
                top_p=top_p,
                top_k=top_k,
                boc_id=self.config.audio_boc_token_id,
                eoc_id=self.config.audio_eoc_token_id,
            )
            delayed_rows.append(codes)
            if state.generation_done:
                break
            next_embed = self._embed_audio_codes(codes)[None]
            decode_dummy = mx.zeros((1, 1), dtype=mx.int32)
            hidden = self.backbone(
                decode_dummy,
                cache=cache,
                input_embeddings=next_embed,
            )
            last_hidden = hidden[:, -1, :]

        audio = self._decode_audio(delayed_rows)
        mx.eval(audio)
        audio = self._apply_fades(
            audio,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
        )
        mx.eval(audio)

        elapsed = time.perf_counter() - start
        samples = int(audio.shape[0])
        duration_s = samples / self.sample_rate if self.sample_rate else 0.0
        token_count = len(delayed_rows)
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=_format_duration(duration_s),
            real_time_factor=round(elapsed / duration_s, 3) if duration_s else 0.0,
            prompt={
                "tokens": prompt_tokens,
                "completion_tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / elapsed, 2) if elapsed > 0 else 0.0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0.0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=bool(stream),
            is_final_chunk=bool(stream),
        )

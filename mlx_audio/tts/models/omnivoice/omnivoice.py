import math
import re
import time
from pathlib import Path
from typing import Any, Generator, List, Optional, Union, cast

import mlx.core as mx
import mlx.nn as nn

from ..base import GenerationResult
from .backbone import BackboneConfig, OmniVoiceBackbone
from .config import OmniVoiceConfig

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

BatchInput = dict[str, mx.array]
PackedBatchData = dict[str, mx.array | list[int]]


def _combine_text(text: str, ref_text: Optional[str] = None) -> str:
    """Merge ref_text + text, normalise whitespace and CJK spacing."""
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()
    full_text = re.sub(r"[\r\n]+", "", full_text)
    full_text = re.sub(r"[ \t]+", " ", full_text)
    cjk = r"[\u4e00-\u9fff]"
    full_text = re.sub(rf"(?<={cjk})\s+|\s+(?={cjk})", "", full_text)
    return full_text


def _ensure_list(x, batch_size: int, auto_repeat: bool = True):
    if x is None:
        return [None] * batch_size
    if not isinstance(x, list):
        if auto_repeat:
            return [x] * batch_size
        raise ValueError(
            f"Expected list of length {batch_size}, got scalar. Pass a list or set auto_repeat=True."
        )
    if len(x) != batch_size:
        raise ValueError(f"Expected list of length {batch_size}, got {len(x)}")
    return x


def _pack_batch(
    inputs_list: list[BatchInput], target_lens: list[int], mask_id: int
) -> PackedBatchData:
    B = len(inputs_list)
    c_lens = [inp["input_ids"].shape[1] for inp in inputs_list]
    max_c_len = max(c_lens)
    max_u_len = max(target_lens)
    C = inputs_list[0]["input_ids"].shape[2]

    cond_rows = []
    cond_mask_rows = []
    uncond_rows = []
    uncond_mask_rows = []

    for i, inp in enumerate(inputs_list):
        cl = c_lens[i]
        tl = target_lens[i]

        cond_row = inp["input_ids"]
        pad_len = max_c_len - cl
        if pad_len > 0:
            cond_row = mx.concatenate(
                [cond_row, mx.full((1, pad_len, C), mask_id, dtype=mx.int32)], axis=1
            )
        cond_rows.append(cond_row)

        cond_mask_row = inp["audio_mask"]
        if pad_len > 0:
            cond_mask_row = mx.concatenate(
                [cond_mask_row, mx.zeros((1, pad_len), dtype=mx.bool_)], axis=1
            )
        cond_mask_rows.append(cond_mask_row)

        uncond_row = inp["input_ids"][0, -tl:, :]
        uncond_mask_row = inp["audio_mask"][0, -tl:]

        # If target length exceeds conditioned length, extend with full mask tokens.
        pad_u_len = max_u_len - tl
        if pad_u_len > 0:
            uncond_row = mx.concatenate(
                [
                    uncond_row,
                    mx.full((pad_u_len, C), mask_id, dtype=mx.int32),
                ],
                axis=0,
            )
            # Padded positions are still masked and participate in iterative decoding.
            uncond_mask_row = mx.concatenate(
                [
                    uncond_mask_row,
                    mx.ones((pad_u_len,), dtype=mx.bool_),
                ],
                axis=0,
            )

        uncond_rows.append(uncond_row[None, :, :])
        uncond_mask_rows.append(uncond_mask_row[None, :])

    cond_input_ids = mx.concatenate(cond_rows, axis=0)
    cond_audio_mask = mx.concatenate(cond_mask_rows, axis=0)
    uncond_input_ids = mx.concatenate(uncond_rows, axis=0)
    uncond_audio_mask = mx.concatenate(uncond_mask_rows, axis=0)

    return {
        "cond_input_ids": cond_input_ids,
        "cond_audio_mask": cond_audio_mask,
        "uncond_input_ids": uncond_input_ids,
        "uncond_audio_mask": uncond_audio_mask,
        "c_lens": c_lens,
        "target_lens": target_lens,
    }


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> mx.array:
    """Tokenize text, keeping nonverbal tags like [laughter] as atomic tokens."""
    parts: list[int] = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.extend(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.extend(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.extend(ids)
    if not parts:
        parts = tokenizer(text, add_special_tokens=False).input_ids
    return mx.array(parts, dtype=mx.int32)


class Model(nn.Module):
    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config

        llm_cfg = config.llm_config or {}
        self.backbone = OmniVoiceBackbone(
            BackboneConfig(
                **{
                    k: v
                    for k, v in llm_cfg.items()
                    if k in BackboneConfig.__dataclass_fields__
                }
            )
        )

        hidden = self.backbone.embed_tokens.weight.shape[-1]
        C = config.num_audio_codebook
        V = config.audio_vocab_size  # 1025 (includes mask token)

        # 8 independent embedding tables for 8 codebooks
        self.audio_embeddings: List[nn.Embedding] = [
            nn.Embedding(V, hidden) for _ in range(C)
        ]
        # 8 independent prediction heads
        self.audio_heads: List[nn.Linear] = [
            nn.Linear(hidden, V, bias=False) for _ in range(C)
        ]

    def _tokenize_style_and_text(
        self,
        text: str,
        language: str = "None",
        instruct: str = "None",
        text_tokenizer=None,
        denoise: bool = True,
        ref_text: Optional[str] = None,
    ) -> tuple[mx.array, mx.array]:
        if text_tokenizer is None:
            raise ValueError("text_tokenizer is required for text tokenization.")
        style_text = ""
        if denoise:
            style_text += "<|denoise|>"
        style_text += f"<|lang_start|>{language}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct}<|instruct_end|>"
        style_ids = mx.array(
            text_tokenizer(style_text, return_tensors="np").input_ids[0],
            dtype=mx.int32,
        )

        full_text = _combine_text(text, ref_text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_ids = _tokenize_with_nonverbal_tags(wrapped_text, text_tokenizer)

        return style_ids, text_ids

    def _prepare_inference_inputs(
        self,
        style_ids: mx.array,
        text_ids: mx.array,
        T: int,
        ref_tokens: Optional[mx.array] = None,
    ) -> dict[str, mx.array]:
        C = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id

        style_block = mx.broadcast_to(style_ids[None, :, None], (1, len(style_ids), C))
        text_block = mx.broadcast_to(text_ids[None, :, None], (1, len(text_ids), C))
        target_block = mx.full((1, T, C), mask_id, dtype=mx.int32)

        parts = [style_block, text_block]
        n_text = len(style_ids) + len(text_ids)

        if ref_tokens is not None:
            ref_block = ref_tokens[None]  # [1, T_ref, 8]
            parts.append(ref_block)
        parts.append(target_block)

        input_ids = mx.concatenate(parts, axis=1)
        L = input_ids.shape[1]

        audio_mask = mx.concatenate(
            [
                mx.zeros((1, n_text), dtype=mx.bool_),
                mx.ones((1, L - n_text), dtype=mx.bool_),
            ],
            axis=1,
        )

        return {"input_ids": input_ids, "audio_mask": audio_mask}

    def _prepare_embed_inputs(
        self, input_ids: mx.array, audio_mask: mx.array
    ) -> mx.array:
        text_embeds = self.backbone.embed_tokens(input_ids[:, :, 0])
        audio_embeds = sum(
            self.audio_embeddings[i](input_ids[:, :, i])
            for i in range(self.config.num_audio_codebook)
        )
        return mx.where(audio_mask[:, :, None], audio_embeds, text_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        audio_mask: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)
        hidden = self.backbone(inputs_embeds, attention_mask)
        logits = mx.stack(
            [
                self.audio_heads[i](hidden)
                for i in range(self.config.num_audio_codebook)
            ],
            axis=2,
        )
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remap k2-fsa/OmniVoice PyTorch keys to mlx-audio naming convention.

        Key transforms:
        - ``llm.*``               → ``backbone.*``
        - ``audio_embeddings.weight [8*V, H]`` → 8× ``audio_embeddings.N.weight [V, H]``
        - ``audio_heads.weight [8*V, H]``      → 8× ``audio_heads.N.weight [V, H]``
        - ``codebook_layer_offsets``           → dropped (not needed)
        """
        C = self.config.num_audio_codebook  # 8
        V = self.config.audio_vocab_size  # 1025
        result = {}
        for k, v in weights.items():
            if k == "codebook_layer_offsets":
                continue
            elif k == "audio_embeddings.weight":
                for i in range(C):
                    result[f"audio_embeddings.{i}.weight"] = v[i * V : (i + 1) * V]
            elif k == "audio_heads.weight":
                for i in range(C):
                    result[f"audio_heads.{i}.weight"] = v[i * V : (i + 1) * V]
            elif k.startswith("llm."):
                result["backbone." + k[4:]] = v
            else:
                result[k] = v
        return result

    def generate_batch(
        self,
        text: List[str],
        language: Union[str, List[str]] = "None",
        lang_code: Union[str, List[str]] = "None",
        ref_text: Optional[Union[str, List[str]]] = None,
        ref_audio=None,
        ref_tokens: Optional[Union[mx.array, List[mx.array]]] = None,
        duration_s: Optional[Union[float, List[float]]] = None,
        instruct: Union[str, List[str]] = "None",
        ref_audio_max_duration_s: float = 10.0,
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
        position_temperature: float = 5.0,
        layer_penalty_factor: float = 5.0,
        t_shift: float = 0.1,
        tokenizer=None,
        text_tokenizer=None,
        max_batch_size: int = 8,
        **kwargs,
    ) -> List[GenerationResult]:
        from .generation import PackedBatch, iterative_unmask_batch
        from .utils import create_voice_clone_prompt

        if not isinstance(text, list):
            text = [text]
        if not text:
            return []
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        if text_tokenizer is None:
            text_tokenizer = getattr(self, "text_tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self, "audio_tokenizer", None)

        batch_size = len(text)
        language_list = cast(list[str], _ensure_list(language, batch_size))
        lang_code_list = cast(list[str], _ensure_list(lang_code, batch_size))
        ref_text_list = cast(list[str | None], _ensure_list(ref_text, batch_size))
        ref_tokens_list = cast(
            list[mx.array | None], _ensure_list(ref_tokens, batch_size)
        )
        duration_list = cast(list[float | None], _ensure_list(duration_s, batch_size))
        instruct_list = cast(list[str], _ensure_list(instruct, batch_size))
        language_list = [
            (
                lang_code_list[i]
                if language_list[i] == "None" and lang_code_list[i] != "None"
                else language_list[i]
            )
            for i in range(batch_size)
        ]

        if ref_audio is not None:
            ref_audio_list = cast(list[Any], _ensure_list(ref_audio, batch_size))
            for i in range(batch_size):
                if ref_tokens_list[i] is not None or ref_audio_list[i] is None:
                    continue
                if tokenizer is None:
                    raise ValueError(
                        "tokenizer (HiggsAudioTokenizer) is required for voice cloning via ref_audio."
                    )
                if isinstance(ref_audio_list[i], (str, Path)):
                    ref_tokens_list[i] = create_voice_clone_prompt(
                        str(ref_audio_list[i]),
                        tokenizer=tokenizer,
                        max_duration_s=ref_audio_max_duration_s,
                    )
                else:
                    wav = ref_audio_list[i]
                    if not isinstance(wav, mx.array):
                        wav = mx.array(wav)
                    if wav.ndim == 1:
                        wav = wav[None, :, None]
                    ref_tokens_list[i] = tokenizer.encode(wav)[0]

        if text_tokenizer is None:
            raise ValueError(
                "text_tokenizer is required for generate_batch(). Pass an AutoTokenizer or load the model via post_load_hook."
            )

        tokens_per_sec = self.config.sample_rate / 960
        inputs_list = []
        target_lens = []

        for i in range(batch_size):
            has_ref = ref_tokens_list[i] is not None
            style_ids, text_ids = self._tokenize_style_and_text(
                text=text[i],
                language=language_list[i],
                instruct=instruct_list[i],
                text_tokenizer=text_tokenizer,
                denoise=has_ref,
                ref_text=ref_text_list[i],
            )

            duration_value = duration_list[i]
            if duration_value is None:
                from .duration import RuleDurationEstimator

                estimator = RuleDurationEstimator()
                raw_tokens = estimator.estimate_duration(
                    text[i] or "", "Nice to meet you.", 25
                )
                target_len = max(10, int(raw_tokens * 1.15))
            else:
                target_len = math.ceil(duration_value * tokens_per_sec)

            target_lens.append(target_len)
            inputs_list.append(
                self._prepare_inference_inputs(
                    style_ids, text_ids, target_len, ref_tokens_list[i]
                )
            )

        all_results = []
        for chunk_start in range(0, batch_size, max_batch_size):
            chunk_end = min(chunk_start + max_batch_size, batch_size)
            packed = cast(
                PackedBatch,
                cast(
                    object,
                    _pack_batch(
                        inputs_list[chunk_start:chunk_end],
                        target_lens[chunk_start:chunk_end],
                        self.config.audio_mask_id,
                    ),
                ),
            )

            start_time = time.time()
            token_list = iterative_unmask_batch(
                self,
                packed,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                class_temperature=class_temperature,
                position_temperature=position_temperature,
                layer_penalty_factor=layer_penalty_factor,
                t_shift=t_shift,
            )
            elapsed = time.time() - start_time

            for j, tokens in enumerate(token_list):
                idx = chunk_start + j
                token_count = target_lens[idx]
                if tokenizer is not None:
                    audio = tokenizer.decode(tokens).astype(mx.float32)
                else:
                    audio = mx.zeros((token_count * 960,), dtype=mx.float32)

                n_samples = token_count * 960
                audio_duration_s = n_samples / self.config.sample_rate
                rtf = audio_duration_s / elapsed if elapsed > 0 else 0.0
                d = int(audio_duration_s)
                duration_str = (
                    f"{d // 3600:02d}:{(d % 3600) // 60:02d}:{d % 60:02d}."
                    f"{int((audio_duration_s % 1) * 1000):03d}"
                )

                all_results.append(
                    GenerationResult(
                        audio=audio,
                        samples=n_samples,
                        sample_rate=self.config.sample_rate,
                        segment_idx=idx,
                        token_count=token_count,
                        audio_duration=duration_str,
                        real_time_factor=rtf,
                        prompt={
                            "tokens": token_count,
                            "tokens-per-sec": (
                                round(token_count / elapsed, 2) if elapsed > 0 else 0
                            ),
                        },
                        audio_samples={
                            "samples": n_samples,
                            "samples-per-sec": (
                                round(n_samples / elapsed, 2) if elapsed > 0 else 0
                            ),
                        },
                        processing_time_seconds=elapsed,
                        peak_memory_usage=mx.get_peak_memory() / 1e9,
                    )
                )

        return all_results

    def generate(
        self,
        text: Optional[str] = None,
        duration_s: Optional[float] = None,
        language: str = "None",
        lang_code: str = "None",
        instruct: str = "None",
        ref_audio=None,
        ref_text: Optional[str] = None,
        ref_audio_max_duration_s: float = 10.0,
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
        position_temperature: float = 5.0,
        layer_penalty_factor: float = 5.0,
        t_shift: float = 0.1,
        tokenizer=None,
        text_tokenizer=None,
        ref_tokens: Optional[mx.array] = None,
        input_ids: Optional[mx.array] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        from .generation import iterative_unmask
        from .utils import create_voice_clone_prompt

        if language == "None" and lang_code != "None":
            language = lang_code

        if text_tokenizer is None:
            text_tokenizer = getattr(self, "text_tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self, "audio_tokenizer", None)

        # --- voice cloning ---
        if ref_tokens is None and ref_audio is not None:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer (HiggsAudioTokenizer) is required for voice cloning via ref_audio."
                )
            if isinstance(ref_audio, (str, Path)):
                ref_tokens = create_voice_clone_prompt(
                    str(ref_audio),
                    tokenizer=tokenizer,
                    max_duration_s=ref_audio_max_duration_s,
                )
            else:
                wav = ref_audio
                if not isinstance(wav, mx.array):
                    wav = mx.array(wav)
                if wav.ndim == 1:
                    wav = wav[None, :, None]
                ref_tokens = tokenizer.encode(wav)[0]

        has_ref = ref_tokens is not None

        # --- text encoding ---
        if input_ids is not None:
            # Legacy path: caller provided pre-encoded flat input_ids [S]
            # Build a minimal unified input_ids from them
            C = self.config.num_audio_codebook
            mask_id = self.config.audio_mask_id
            tokens_per_sec = self.config.sample_rate / 960
            if duration_s is None:
                from .duration import RuleDurationEstimator

                _estimator = RuleDurationEstimator()
                raw_tokens = _estimator.estimate_duration(
                    text or "", "Nice to meet you.", 25
                )
                target_len = max(10, int(raw_tokens * 1.15))
            else:
                target_len = math.ceil(duration_s * tokens_per_sec)

            S = input_ids.shape[0]
            text_block = mx.broadcast_to(input_ids[None, :, None], (1, S, C))
            target_block = mx.full((1, target_len, C), mask_id, dtype=mx.int32)
            parts = [text_block]
            if ref_tokens is not None:
                parts.append(ref_tokens[None])
            parts.append(target_block)
            cond_input_ids = mx.concatenate(parts, axis=1)
            n_text = S
            L = cond_input_ids.shape[1]
            cond_audio_mask = mx.concatenate(
                [
                    mx.zeros((1, n_text), dtype=mx.bool_),
                    mx.ones((1, L - n_text), dtype=mx.bool_),
                ],
                axis=1,
            )
        else:
            if text_tokenizer is None:
                raise ValueError(
                    "text_tokenizer is required when input_ids is not provided. "
                    "Pass an AutoTokenizer or use input_ids directly."
                )

            style_ids, text_ids = self._tokenize_style_and_text(
                text=text or "",
                language=language,
                instruct=instruct,
                text_tokenizer=text_tokenizer,
                denoise=has_ref,
                ref_text=ref_text,
            )

            tokens_per_sec = self.config.sample_rate / 960
            if duration_s is None:
                from .duration import RuleDurationEstimator

                _estimator = RuleDurationEstimator()
                raw_tokens = _estimator.estimate_duration(
                    text or "", "Nice to meet you.", 25
                )
                target_len = max(10, int(raw_tokens * 1.15))
            else:
                target_len = math.ceil(duration_s * tokens_per_sec)

            inputs = self._prepare_inference_inputs(
                style_ids, text_ids, target_len, ref_tokens
            )
            cond_input_ids = inputs["input_ids"]
            cond_audio_mask = inputs["audio_mask"]

        start_time = time.time()
        tokens = iterative_unmask(
            self,
            cond_input_ids=cond_input_ids,
            cond_audio_mask=cond_audio_mask,
            T=target_len,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
            position_temperature=position_temperature,
            layer_penalty_factor=layer_penalty_factor,
            t_shift=t_shift,
        )
        elapsed = time.time() - start_time

        if tokenizer is not None:
            audio = tokenizer.decode(tokens).astype(mx.float32)
        else:
            audio = mx.zeros((target_len * 960,), dtype=mx.float32)

        n_samples = target_len * 960
        audio_duration_s = n_samples / self.config.sample_rate
        rtf = audio_duration_s / elapsed if elapsed > 0 else 0.0
        d = int(audio_duration_s)
        duration_str = f"{d // 3600:02d}:{(d % 3600) // 60:02d}:{d % 60:02d}.{int((audio_duration_s % 1) * 1000):03d}"

        yield GenerationResult(
            audio=audio,
            samples=n_samples,
            sample_rate=self.config.sample_rate,
            segment_idx=0,
            token_count=target_len,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": target_len,
                "tokens-per-sec": round(target_len / elapsed, 2) if elapsed > 0 else 0,
            },
            audio_samples={
                "samples": n_samples,
                "samples-per-sec": round(n_samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    @property
    def model_type(self) -> str:
        return self.config.model_type

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        """Load text tokenizer and HiggsAudio tokenizer after weight loading."""
        import warnings

        try:
            from transformers import AutoTokenizer

            model.text_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception as e:
            warnings.warn(f"Could not load text tokenizer: {e}")
            model.text_tokenizer = None

        try:
            from mlx_audio.codec.models.higgs_audio.higgs_audio import (
                HiggsAudioTokenizer,
            )

            model.audio_tokenizer = HiggsAudioTokenizer.from_pretrained(str(model_path))
        except Exception as e:
            warnings.warn(f"Could not load audio tokenizer: {e}")
            model.audio_tokenizer = None

        return model

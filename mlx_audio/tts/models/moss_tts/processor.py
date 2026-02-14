"""Prompt + codec processing for MOSS-TTS family variants."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.utils import load_audio

from .config import ModelConfig

AUDIO_PLACEHOLDER = "<|audio|>"
VALID_INPUT_TYPES = {"text", "pinyin", "ipa"}


def normalize_instruction(instruction: str) -> str:
    """Apply upstream-compatible normalization for instruction fields."""
    if not instruction:
        return instruction

    normalized = instruction.replace("\n", " ")
    normalized = re.sub(r"\[.*?\]", "", normalized)
    normalized = re.sub(r"\{.*?\}", "", normalized)

    for char in "【】《》（）『』「」～-_":
        normalized = normalized.replace(char, "，")

    normalized = re.sub(r"([，。！？,.!?;；])+", r"\1", normalized)
    if re.search(r"[\u4e00-\u9fff]", normalized):
        normalized = normalized.replace(",", "，")
    return normalized.strip()


def normalize_text(text: str) -> str:
    """Apply upstream-compatible normalization for user text fields."""
    if not text:
        return text

    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\[.*?\]", "", normalized)
    normalized = re.sub(r"\{.*?\}", "", normalized)

    for char in "【】《》（）『』「」～":
        normalized = normalized.replace(char, "，")

    normalized = normalized.replace('"', "")
    normalized = re.sub(r"([，。！？,.!?;；])+", r"\1", normalized)
    return normalized.strip()


@dataclass
class Message:
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class UserMessage(Message):
    text: Optional[str] = None
    reference: Optional[List[Optional[Union[str, mx.array]]]] = None
    instruction: Optional[str] = None
    tokens: Optional[int] = None
    quality: Optional[str] = None
    sound_event: Optional[str] = None
    ambient_sound: Optional[str] = None
    language: Optional[str] = None
    input_type: str = "text"

    def __post_init__(self):
        if self.input_type not in VALID_INPUT_TYPES:
            raise ValueError(
                f"Unsupported input_type '{self.input_type}'. "
                f"Expected one of {sorted(VALID_INPUT_TYPES)}"
            )

        template = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Input Type:
{input_type}
- Text:
{text}
</user_inst>"""

        references = []
        rendered_references = "None"
        if self.reference is not None:
            if not isinstance(self.reference, list):
                raise TypeError("reference must be a list when provided")
            chunks: List[str] = []
            for speaker_idx, item in enumerate(self.reference):
                if item is None:
                    chunks.append(f"[S{speaker_idx + 1}]: None")
                else:
                    chunks.append(f"[S{speaker_idx + 1}]:\n{AUDIO_PLACEHOLDER}")
                    references.append(item)
            if chunks:
                rendered_references = "\n".join(chunks)

        self._content = (
            template.replace("{reference}", str(rendered_references))
            .replace("{instruction}", str(self.instruction))
            .replace("{tokens}", str(self.tokens))
            .replace("{quality}", str(self.quality))
            .replace("{sound_event}", str(self.sound_event))
            .replace("{ambient_sound}", str(self.ambient_sound))
            .replace("{language}", str(self.language))
            .replace("{input_type}", str(self.input_type))
            .replace("{text}", str(self.text))
        )
        self._references = references

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": self._content,
            "audio_references": self._references,
        }


@dataclass
class AssistantMessage(Message):
    audio_codes_list: List[Union[str, mx.array]]
    content: str = AUDIO_PLACEHOLDER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "audio_references": self.audio_codes_list,
        }


USER_MESSAGE_FIELDS = (
    "text",
    "reference",
    "instruction",
    "tokens",
    "quality",
    "sound_event",
    "ambient_sound",
    "language",
    "input_type",
)


class MossTTSProcessor:
    """Runtime processor used by mlx-audio MOSS-TTS model implementations."""

    def __init__(
        self,
        tokenizer: Any,
        audio_tokenizer: Optional[MossAudioTokenizer],
        model_config: ModelConfig,
    ):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.model_config = model_config

        self.audio_user_slot_token = self._id_to_token(
            model_config.audio_user_slot_token_id
        )
        self.audio_assistant_gen_slot_token = self._id_to_token(
            model_config.audio_assistant_gen_slot_token_id
        )
        self.audio_assistant_delay_slot_token = self._id_to_token(
            model_config.audio_assistant_delay_slot_token_id
        )
        self.audio_start_token = self._id_to_token(model_config.audio_start_token_id)
        self.audio_end_token = self._id_to_token(model_config.audio_end_token_id)

    def _id_to_token(self, token_id: int) -> str:
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token, list):
            token = token[0] if token else ""
        if token is None:
            token = self.tokenizer.decode([int(token_id)])
        return str(token)

    @staticmethod
    def build_user_message(
        text: Optional[str] = None,
        reference: Optional[List[Optional[Union[str, mx.array]]]] = None,
        instruction: Optional[str] = None,
        tokens: Optional[int] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        language: Optional[str] = None,
        input_type: str = "text",
        normalize: bool = False,
    ) -> Dict[str, Any]:
        if normalize:
            if text is not None:
                text = normalize_text(text)
            if instruction is not None:
                instruction = normalize_instruction(instruction)
        if reference is not None and not isinstance(reference, list):
            reference = [reference]
        return UserMessage(
            text=text,
            reference=reference,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
            input_type=input_type,
        ).to_dict()

    @staticmethod
    def build_assistant_message(
        audio_codes_list: List[Union[str, mx.array]],
        content: str = AUDIO_PLACEHOLDER,
    ) -> Dict[str, Any]:
        return AssistantMessage(audio_codes_list=audio_codes_list, content=content).to_dict()

    @staticmethod
    def _parse_speaker_id(raw_speaker_id: Any) -> int:
        try:
            speaker_id = int(raw_speaker_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid speaker_id '{raw_speaker_id}'") from exc
        if speaker_id < 0:
            raise ValueError("speaker_id must be non-negative")
        return speaker_id

    def build_ttsd_continuation_messages(
        self,
        *,
        dialogue_text: str,
        speakers: Sequence[Mapping[str, Any]],
        instruction: Optional[str] = None,
        tokens: Optional[int] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        language: Optional[str] = None,
        input_type: str = "text",
        n_vq: Optional[int] = None,
        normalize_inputs: bool = False,
    ) -> List[Dict[str, Any]]:
        if not dialogue_text or not dialogue_text.strip():
            raise ValueError("dialogue_text must be a non-empty string")
        if len(speakers) == 0:
            raise ValueError("speakers must include at least one speaker definition")

        n_vq = self.model_config.n_vq if n_vq is None else int(n_vq)
        if n_vq <= 0:
            raise ValueError("n_vq must be positive")

        speaker_rows: Dict[int, Dict[str, Any]] = {}
        normalized_entries: List[Tuple[Mapping[str, Any], int]] = []
        for entry in speakers:
            if not isinstance(entry, Mapping):
                raise ValueError("Each speaker entry must be a mapping")
            parsed_speaker_id = self._parse_speaker_id(entry.get("speaker_id"))
            normalized_entries.append((entry, parsed_speaker_id))

        # Interpret ids per request: if any speaker uses `0`, treat the whole schema as
        # zero-based; otherwise treat it as one-based.
        use_zero_based_ids = any(speaker_id == 0 for _, speaker_id in normalized_entries)
        for entry, parsed_speaker_id in normalized_entries:
            speaker_id = parsed_speaker_id + 1 if use_zero_based_ids else parsed_speaker_id
            prior = speaker_rows.get(speaker_id, {})
            merged = dict(prior)
            merged.update(dict(entry))
            merged["speaker_id"] = speaker_id
            speaker_rows[speaker_id] = merged

        sorted_ids = sorted(speaker_rows.keys())
        references: List[Optional[Union[str, mx.array]]] = [None] * max(sorted_ids)
        prompt_lines: List[str] = []
        assistant_reference_inputs: List[Union[str, mx.array]] = []

        for speaker_id in sorted_ids:
            row = speaker_rows[speaker_id]
            ref_audio = row.get("ref_audio")
            if ref_audio is not None:
                references[speaker_id - 1] = ref_audio
                assistant_reference_inputs.append(ref_audio)

            ref_text = row.get("ref_text")
            if ref_text is None:
                ref_text = row.get("text")
            if ref_text is not None:
                normalized_text = str(ref_text).strip()
                if normalized_text:
                    prompt_lines.append(f"[S{speaker_id}] {normalized_text}")

        full_dialogue = " ".join(prompt_lines + [dialogue_text.strip()])
        user_message = self.build_user_message(
            text=full_dialogue,
            reference=references if any(item is not None for item in references) else None,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
            input_type=input_type,
            normalize=normalize_inputs,
        )
        messages: List[Dict[str, Any]] = [user_message]

        if assistant_reference_inputs:
            assistant_codes = self.encode_audios_from_reference(
                assistant_reference_inputs,
                n_vq=n_vq,
            )
            concatenated = mx.concatenate(assistant_codes, axis=0).astype(mx.int32)
            messages.append(
                self.build_assistant_message(audio_codes_list=[concatenated])
            )
            # End on a fresh user turn so downstream packing can run in generation mode
            # after priming speaker references through the assistant continuation payload.
            messages.append(
                self.build_user_message(
                    text=dialogue_text.strip(),
                    instruction=instruction,
                    tokens=tokens,
                    quality=quality,
                    sound_event=sound_event,
                    ambient_sound=ambient_sound,
                    language=language,
                    input_type=input_type,
                    normalize=normalize_inputs,
                )
            )

        return messages

    def _normalize_message(
        self,
        message: Union[Message, Dict[str, Any]],
        *,
        normalize_inputs: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(message, Message):
            return message.to_dict()
        if not isinstance(message, dict):
            raise TypeError("Message must be a Message or dict")
        if "role" not in message:
            raise ValueError("Message dict must include role")

        if "content" in message and "audio_references" in message:
            return message

        role = message["role"]
        if role == "user":
            kwargs = {field: message.get(field) for field in USER_MESSAGE_FIELDS}
            kwargs["normalize"] = normalize_inputs
            return self.build_user_message(**kwargs)
        if role == "assistant":
            return self.build_assistant_message(
                audio_codes_list=message.get("audio_references", []),
                content=message.get("content", AUDIO_PLACEHOLDER),
            )
        raise ValueError(f"Unsupported role '{role}'")

    @staticmethod
    def _replace_audio_placeholders(
        content: str,
        lengths: Sequence[int],
        audio_start_token: str,
        audio_end_token: str,
        slot_token: str,
        delay_slot_token: Optional[str] = None,
        n_vq: int = 1,
    ) -> str:
        result = content
        placeholder_count = result.count(AUDIO_PLACEHOLDER)
        if placeholder_count != len(lengths):
            raise ValueError(
                f"Audio placeholder count ({placeholder_count}) does not match "
                f"provided references ({len(lengths)})"
            )

        for length in lengths:
            if length < 0:
                raise ValueError("reference code length must be non-negative")
            if delay_slot_token is None:
                block = audio_start_token + (slot_token * length) + audio_end_token
            else:
                if length == 0:
                    block = audio_start_token + audio_end_token
                else:
                    block = (
                        audio_start_token
                        + (slot_token * length)
                        + (delay_slot_token * max(n_vq - 1, 0))
                        + audio_end_token
                    )
            result = result.replace(AUDIO_PLACEHOLDER, block, 1)
        return result

    @staticmethod
    def _normalize_preencoded_audio_codes(
        item: mx.array,
        *,
        n_vq: int,
    ) -> Optional[mx.array]:
        if item.ndim != 2:
            return None

        item_np = np.array(item)
        if not np.issubdtype(item_np.dtype, np.integer):
            return None

        rows = int(item.shape[0])
        cols = int(item.shape[1])
        if rows < n_vq and cols < n_vq:
            return None

        # `(NQ, T)` -> `(T, NQ)`
        if rows >= n_vq and cols < n_vq:
            return mx.array(item_np[:n_vq, :].T, dtype=mx.int32)
        # `(T, NQ)` already
        if cols >= n_vq and rows < n_vq:
            return mx.array(item_np[:, :n_vq], dtype=mx.int32)

        # Prefer exact axis matches when available.
        if rows == n_vq and cols != n_vq:
            return mx.array(item_np[:n_vq, :].T, dtype=mx.int32)
        if cols == n_vq and rows != n_vq:
            return mx.array(item_np[:, :n_vq], dtype=mx.int32)

        # If neither axis exactly matches n_vq, bias toward plausible codebook axis:
        # small, n_vq-aligned dimensions are likely codebook-major (`(NQ, T)`).
        if rows % n_vq == 0 and rows <= 64:
            return mx.array(item_np[:n_vq, :].T, dtype=mx.int32)
        if cols % n_vq == 0 and cols <= 64:
            return mx.array(item_np[:, :n_vq], dtype=mx.int32)
        if rows <= 64 and cols > 64:
            return mx.array(item_np[:n_vq, :].T, dtype=mx.int32)
        if cols <= 64 and rows > 64:
            return mx.array(item_np[:, :n_vq], dtype=mx.int32)

        # Fallback keeps backward compatibility with historical `(T, NQ)` assumptions.
        return mx.array(item_np[:, :n_vq], dtype=mx.int32)

    @staticmethod
    def _normalize_waveform_layout_to_time_major(wav: mx.array) -> mx.array:
        """Normalize a 2D waveform to `(T, C)` before mono downmix."""
        if wav.ndim != 2:
            return wav

        rows = int(wav.shape[0])
        cols = int(wav.shape[1])
        max_expected_channels = 8

        # Typical channel-first waveforms have a tiny channel axis (`C<=8`) and a much
        # longer time axis. Transpose those to `(T, C)` before downmix.
        if rows <= max_expected_channels and cols > rows:
            return wav.transpose(1, 0)

        # Typical time-major waveforms already have the channel axis in the last dim.
        if cols <= max_expected_channels and rows >= cols:
            return wav

        # Ambiguous fallback: smaller leading axis is usually channel-first.
        if rows < cols:
            return wav.transpose(1, 0)
        return wav

    def apply_delay_pattern(
        self,
        codes: mx.array,
        *,
        pad_code: Optional[int] = None,
    ) -> mx.array:
        if codes.ndim != 2:
            raise ValueError(f"Expected codes shape (T, NQ), got {codes.shape}")
        pad_code = (
            self.model_config.audio_pad_code if pad_code is None else int(pad_code)
        )
        time_steps = int(codes.shape[0])
        n_vq = int(codes.shape[1])
        delayed = np.full(
            (time_steps + n_vq - 1, n_vq),
            pad_code,
            dtype=np.array(codes).dtype,
        )
        source = np.array(codes)
        for channel_idx in range(n_vq):
            delayed[channel_idx : channel_idx + time_steps, channel_idx] = source[
                :, channel_idx
            ]
        return mx.array(delayed, dtype=codes.dtype)

    def apply_de_delay_pattern(self, delay_codes: mx.array) -> mx.array:
        if delay_codes.ndim != 2:
            raise ValueError(
                f"Expected delayed codes shape (T + NQ - 1, NQ), got {delay_codes.shape}"
            )
        total_steps = int(delay_codes.shape[0])
        n_vq = int(delay_codes.shape[1])
        if total_steps < n_vq:
            return mx.zeros((0, n_vq), dtype=delay_codes.dtype)

        time_steps = total_steps - n_vq + 1
        output = np.zeros((time_steps, n_vq), dtype=np.array(delay_codes).dtype)
        delayed = np.array(delay_codes)
        for channel_idx in range(n_vq):
            output[:, channel_idx] = delayed[
                channel_idx : channel_idx + time_steps, channel_idx
            ]
        return mx.array(output, dtype=delay_codes.dtype)

    def _split_audio_segments(self, audio_codes: mx.array) -> List[mx.array]:
        if audio_codes.ndim != 2:
            raise ValueError(
                f"Expected audio_codes shape (T, NQ), got {audio_codes.shape}"
            )
        pad_code = int(self.model_config.audio_pad_code)
        codes_np = np.array(audio_codes)
        if codes_np.size == 0:
            return []

        is_pad = np.all(codes_np == pad_code, axis=1)
        segments: List[mx.array] = []
        start = -1
        for idx, is_pad_row in enumerate(is_pad):
            if not is_pad_row and start < 0:
                start = idx
            if is_pad_row and start >= 0:
                segments.append(mx.array(codes_np[start:idx], dtype=audio_codes.dtype))
                start = -1
        if start >= 0:
            segments.append(mx.array(codes_np[start:], dtype=audio_codes.dtype))
        return segments

    def parse_generated_assistant_segments(
        self,
        audio_codes: mx.array,
        *,
        delayed: Optional[bool] = None,
    ) -> List[mx.array]:
        if delayed is None:
            delayed = not self.model_config.is_local_variant
        normalized = (
            self.apply_de_delay_pattern(audio_codes) if delayed else audio_codes
        ).astype(mx.int32)
        return self._split_audio_segments(normalized)

    def extract_complete_delay_rows(self, delay_codes: mx.array) -> mx.array:
        """
        Convert delayed rows to standard `(T, n_vq)` codes and keep only rows where
        all codebooks have emitted a real token (no pad values).
        """

        codes = self.apply_de_delay_pattern(delay_codes).astype(mx.int32)
        if codes.shape[0] == 0:
            return mx.zeros((0, int(delay_codes.shape[1])), dtype=mx.int32)

        codes_np = np.array(codes)
        keep = np.all(codes_np != int(self.model_config.audio_pad_code), axis=1)
        if not np.any(keep):
            return mx.zeros((0, int(delay_codes.shape[1])), dtype=mx.int32)
        return mx.array(codes_np[keep], dtype=mx.int32)

    def encode_audios_from_reference(
        self,
        references: Iterable[Union[str, mx.array]],
        *,
        n_vq: Optional[int] = None,
    ) -> List[mx.array]:
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not loaded")

        n_vq = self.model_config.n_vq if n_vq is None else int(n_vq)
        normalized_items: List[Tuple[str, int | mx.array]] = []
        wav_list: List[mx.array] = []
        for item in references:
            if isinstance(item, str):
                wav = load_audio(item, sample_rate=self.model_config.sampling_rate)
                normalized_items.append(("wav", len(wav_list)))
            elif isinstance(item, mx.array):
                preencoded_codes = self._normalize_preencoded_audio_codes(
                    item,
                    n_vq=n_vq,
                )
                if preencoded_codes is not None:
                    normalized_items.append(("codes", preencoded_codes))
                    continue
                wav = item
                normalized_items.append(("wav", len(wav_list)))
            else:
                raise TypeError(
                    "Each reference must be an audio path, waveform, or pre-encoded "
                    "audio codes with shape (T, NQ) or (NQ, T)."
                )

            if wav.ndim == 2:
                wav = self._normalize_waveform_layout_to_time_major(wav)
                wav = mx.mean(wav, axis=1)
            wav_list.append(wav.astype(mx.float32))

        wav_codes: List[mx.array] = []
        if wav_list:
            encoded = self.audio_tokenizer.batch_encode(
                wav_list,
                num_quantizers=n_vq,
            )
            if encoded.audio_codes is None or encoded.audio_codes_lengths is None:
                raise RuntimeError("audio tokenizer encode returned empty outputs")
            for batch_idx in range(int(encoded.audio_codes.shape[1])):
                length = int(encoded.audio_codes_lengths[batch_idx])
                codes = encoded.audio_codes[:, batch_idx, :length].transpose(1, 0)
                wav_codes.append(codes.astype(mx.int32))

        codes_list: List[mx.array] = []
        for source_type, payload in normalized_items:
            if source_type == "codes":
                codes_list.append(payload)  # type: ignore[arg-type]
            else:
                wav_idx = int(payload)
                codes_list.append(wav_codes[wav_idx])
        return codes_list

    def decode_audio_codes(
        self,
        audio_codes: mx.array,
        *,
        chunk_duration: Optional[float] = 8.0,
    ) -> mx.array:
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not loaded")
        if audio_codes.ndim != 2:
            raise ValueError(
                f"Expected audio_codes with shape (T, NQ), got {audio_codes.shape}"
            )

        decoded = self.audio_tokenizer.decode(
            audio_codes.transpose(1, 0),
            return_dict=True,
            chunk_duration=chunk_duration,
            num_quantizers=audio_codes.shape[1],
        )
        if decoded.audio is None or decoded.audio_lengths is None:
            raise RuntimeError("audio tokenizer decode returned empty outputs")
        samples = int(decoded.audio_lengths[0])
        return decoded.audio[0, 0, :samples]

    def _get_unified_codes(
        self,
        role: str,
        content: str,
        audio_codes_list: List[mx.array],
        n_vq: int,
    ) -> mx.array:
        is_delay_variant = not self.model_config.is_local_variant
        if role == "user":
            slot_token = self.audio_user_slot_token
            delay_slot_token = self.audio_user_slot_token if is_delay_variant else None
        else:
            slot_token = self.audio_assistant_gen_slot_token
            delay_slot_token = (
                self.audio_assistant_delay_slot_token if is_delay_variant else None
            )

        text = self._replace_audio_placeholders(
            content=content,
            lengths=[int(codes.shape[0]) for codes in audio_codes_list],
            audio_start_token=self.audio_start_token,
            audio_end_token=self.audio_end_token,
            slot_token=slot_token,
            delay_slot_token=delay_slot_token,
            n_vq=n_vq,
        )
        text_ids = mx.array(self.tokenizer.encode(text), dtype=mx.int32)
        text_length = int(text_ids.shape[0])

        if not audio_codes_list:
            audio_channels = mx.full(
                (text_length, n_vq),
                self.model_config.audio_pad_code,
                dtype=mx.int32,
            )
            return mx.concatenate([text_ids[:, None], audio_channels], axis=1)

        text_list = np.array(text_ids).tolist()
        start_positions = [
            idx
            for idx, token_id in enumerate(text_list)
            if token_id == self.model_config.audio_start_token_id
        ]
        end_positions = [
            idx
            for idx, token_id in enumerate(text_list)
            if token_id == self.model_config.audio_end_token_id
        ]
        if len(start_positions) != len(audio_codes_list) or len(end_positions) != len(
            audio_codes_list
        ):
            raise ValueError("Audio placeholders and references are misaligned")

        segments: List[mx.array] = []
        prefix_index = 0
        for start_idx, end_idx, codes in zip(start_positions, end_positions, audio_codes_list):
            start = int(start_idx)
            end = int(end_idx)
            packed_codes = codes[:, :n_vq].astype(mx.int32)
            if is_delay_variant:
                packed_codes = self.apply_delay_pattern(
                    packed_codes,
                    pad_code=self.model_config.audio_pad_code,
                )
            pad = mx.full(
                (start - prefix_index + 1, n_vq),
                self.model_config.audio_pad_code,
                dtype=mx.int32,
            )
            segments.extend([pad, packed_codes])
            prefix_index = end

        trailing = mx.full(
            (text_length - int(end_positions[-1]), n_vq),
            self.model_config.audio_pad_code,
            dtype=mx.int32,
        )
        segments.append(trailing)
        audio_channels = mx.concatenate(segments, axis=0)

        if text_length != int(audio_channels.shape[0]):
            text_ids = text_ids[: int(audio_channels.shape[0])]
        return mx.concatenate([text_ids[:, None], audio_channels], axis=1)

    def prepare_generation_inputs(
        self,
        messages: Union[Message, Dict[str, Any], List[Union[Message, Dict[str, Any]]]],
        *,
        n_vq: Optional[int] = None,
        apply_chat_template: bool = True,
        mode: str = "generation",
        normalize_inputs: bool = False,
    ) -> Dict[str, mx.array]:
        if isinstance(messages, (Message, dict)):
            messages = [messages]

        normalized = [
            self._normalize_message(message, normalize_inputs=normalize_inputs)
            for message in messages
        ]
        if not normalized:
            raise ValueError("At least one message is required")
        if mode not in {"generation", "continuation"}:
            raise ValueError(f"Unsupported mode '{mode}'")
        if mode == "generation" and normalized[-1]["role"] != "user":
            raise ValueError("Generation mode requires final message role='user'")
        if mode == "continuation" and normalized[-1]["role"] != "assistant":
            raise ValueError("Continuation mode requires final message role='assistant'")

        n_vq = self.model_config.n_vq if n_vq is None else int(n_vq)
        if n_vq <= 0:
            raise ValueError("n_vq must be positive")

        packed_parts: List[mx.array] = []
        for idx, message in enumerate(normalized):
            add_generation_prompt = mode == "generation" and idx == len(normalized) - 1
            content = message["content"]
            if apply_chat_template:
                try:
                    content = self.tokenizer.apply_chat_template(
                        [{"role": message["role"], "content": content}],
                        add_generation_prompt=add_generation_prompt,
                        tokenize=False,
                    )
                except TypeError:
                    try:
                        content = self.tokenizer.apply_chat_template(
                            [{"role": message["role"], "content": content}],
                            add_generation_prompt=add_generation_prompt,
                        )
                    except Exception:
                        content = str(content)
                except Exception:
                    content = str(content)
            else:
                content = str(content)

            audio_refs = message.get("audio_references", [])
            codes_list = self.encode_audios_from_reference(audio_refs, n_vq=n_vq)
            packed_parts.append(
                self._get_unified_codes(message["role"], content, codes_list, n_vq)
            )

        input_ids = mx.concatenate(packed_parts, axis=0)
        if mode == "generation":
            start_row = mx.full((1, 1 + n_vq), self.model_config.audio_pad_code, dtype=mx.int32)
            start_row[:, 0] = self.model_config.audio_start_token_id
            input_ids = mx.concatenate([input_ids, start_row], axis=0)
        attention_mask = mx.ones((input_ids.shape[0],), dtype=mx.bool_)

        return {
            "input_ids": input_ids[None, :, :],
            "attention_mask": attention_mask[None, :],
        }


__all__ = [
    "AUDIO_PLACEHOLDER",
    "AssistantMessage",
    "Message",
    "MossTTSProcessor",
    "UserMessage",
    "VALID_INPUT_TYPES",
]

"""Prompt + codec processing for MOSS-TTS family variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.utils import load_audio

from .config import ModelConfig

AUDIO_PLACEHOLDER = "<|audio|>"
VALID_INPUT_TYPES = {"text", "pinyin", "ipa"}


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
                    continue
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
    ) -> Dict[str, Any]:
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

    def _normalize_message(self, message: Union[Message, Dict[str, Any]]) -> Dict[str, Any]:
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
            block = audio_start_token + (slot_token * length) + audio_end_token
            result = result.replace(AUDIO_PLACEHOLDER, block, 1)
        return result

    def encode_audios_from_reference(
        self,
        references: Iterable[Union[str, mx.array]],
        *,
        n_vq: Optional[int] = None,
    ) -> List[mx.array]:
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not loaded")

        wav_list: List[mx.array] = []
        for item in references:
            if isinstance(item, str):
                wav = load_audio(item, sample_rate=self.model_config.sampling_rate)
            elif isinstance(item, mx.array):
                wav = item
            else:
                raise TypeError(
                    "Each reference must be an audio path or mlx.array waveform"
                )

            if wav.ndim == 2:
                wav = mx.mean(wav, axis=-1)
            wav_list.append(wav.astype(mx.float32))

        if not wav_list:
            return []

        encoded = self.audio_tokenizer.batch_encode(
            wav_list,
            num_quantizers=n_vq,
            return_dict=True,
        )
        if encoded.audio_codes is None or encoded.audio_codes_lengths is None:
            raise RuntimeError("audio tokenizer encode returned empty outputs")

        codes_list: List[mx.array] = []
        for batch_idx in range(int(encoded.audio_codes.shape[1])):
            length = int(encoded.audio_codes_lengths[batch_idx])
            codes = encoded.audio_codes[:, batch_idx, :length].transpose(1, 0)
            codes_list.append(codes.astype(mx.int32))
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
        if role == "user":
            slot_token = self.audio_user_slot_token
        else:
            slot_token = self.audio_assistant_gen_slot_token

        text = self._replace_audio_placeholders(
            content=content,
            lengths=[int(codes.shape[0]) for codes in audio_codes_list],
            audio_start_token=self.audio_start_token,
            audio_end_token=self.audio_end_token,
            slot_token=slot_token,
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
            pad = mx.full(
                (start - prefix_index + 1, n_vq),
                self.model_config.audio_pad_code,
                dtype=mx.int32,
            )
            segments.extend([pad, codes[:, :n_vq].astype(mx.int32)])
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
    ) -> Dict[str, mx.array]:
        if isinstance(messages, (Message, dict)):
            messages = [messages]

        normalized = [self._normalize_message(message) for message in messages]
        if not normalized:
            raise ValueError("At least one message is required")
        if normalized[-1]["role"] != "user":
            raise ValueError("Generation requires final message role='user'")

        n_vq = self.model_config.n_vq if n_vq is None else int(n_vq)
        if n_vq <= 0:
            raise ValueError("n_vq must be positive")

        packed_parts: List[mx.array] = []
        for idx, message in enumerate(normalized):
            add_generation_prompt = idx == len(normalized) - 1
            content = message["content"]
            if apply_chat_template:
                try:
                    content = self.tokenizer.apply_chat_template(
                        [{"role": message["role"], "content": content}],
                        add_generation_prompt=add_generation_prompt,
                        tokenize=False,
                    )
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

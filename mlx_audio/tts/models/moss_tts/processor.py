from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx

from .config import ModelConfig
from .text import normalize_tts_text

AUDIO_PLACEHOLDER = "<|audio|>"


@dataclass
class Message:
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class UserMessage(Message):
    text: Optional[str] = None
    reference: Optional[list[Optional[Any]]] = None
    instruction: Optional[str] = None
    tokens: Optional[int] = None
    quality: Optional[str] = None
    sound_event: Optional[str] = None
    ambient_sound: Optional[str] = None
    language: Optional[str] = None
    scene: Optional[str] = None
    include_scene: bool = False

    def __post_init__(self):
        fields = [
            ("Reference(s)", "{reference}"),
            ("Instruction", "{instruction}"),
            ("Tokens", "{tokens}"),
            ("Quality", "{quality}"),
            ("Sound Event", "{sound_event}"),
            ("Ambient Sound", "{ambient_sound}"),
            ("Language", "{language}"),
        ]
        if self.include_scene:
            fields.append(("Scene", "{scene}"))
        fields.append(("Text", "{text}"))
        template = (
            "<user_inst>\n"
            + "\n".join(f"- {label}:\n{placeholder}" for label, placeholder in fields)
            + "\n</user_inst>"
        )

        audio_codes_list = []
        if self.reference is None:
            reference = "None"
        elif isinstance(self.reference, list):
            reference_parts = []
            for speaker_idx, speaker_reference in enumerate(self.reference):
                if speaker_reference is None:
                    reference_parts.append(f"[S{speaker_idx + 1}]: None")
                else:
                    reference_parts.append(
                        f"[S{speaker_idx + 1}]:\n{AUDIO_PLACEHOLDER}"
                    )
                    audio_codes_list.append(speaker_reference)
            reference = "\n".join(reference_parts)
        else:
            raise TypeError("reference must be a list when it is not None")

        self._content = (
            template.replace("{reference}", str(reference))
            .replace("{instruction}", str(self.instruction))
            .replace("{tokens}", str(self.tokens))
            .replace("{quality}", str(self.quality))
            .replace("{sound_event}", str(self.sound_event))
            .replace("{ambient_sound}", str(self.ambient_sound))
            .replace("{language}", str(self.language))
            .replace("{scene}", str(self.scene))
            .replace("{text}", str(self.text))
        )
        self._audio_codes_list = audio_codes_list

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": "user",
            "content": self._content,
            "audio_codes_list": self._audio_codes_list,
        }


@dataclass
class AssistantMessage(Message):
    audio_codes_list: list[Any]
    content: str = AUDIO_PLACEHOLDER

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "audio_codes_list": self.audio_codes_list,
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
    "scene",
)


def apply_delay_pattern(codes: mx.array, pad_code: int) -> mx.array:
    if codes.ndim != 2:
        raise ValueError(f"Expected codes shape [frames, n_vq], got {codes.shape}")
    delayed = mx.full(
        (codes.shape[0] + codes.shape[1] - 1, codes.shape[1]),
        int(pad_code),
        dtype=codes.dtype,
    )
    for codebook_index in range(codes.shape[1]):
        delayed[codebook_index : codebook_index + codes.shape[0], codebook_index] = (
            codes[:, codebook_index]
        )
    return delayed


def apply_de_delay_pattern(delay_codes: mx.array) -> mx.array:
    if delay_codes.ndim != 2:
        raise ValueError(
            f"Expected delay_codes shape [frames, n_vq], got {delay_codes.shape}"
        )
    output_length = delay_codes.shape[0] - delay_codes.shape[1] + 1
    if output_length <= 0:
        return mx.zeros((0, delay_codes.shape[1]), dtype=delay_codes.dtype)
    tokens = mx.zeros((output_length, delay_codes.shape[1]), dtype=delay_codes.dtype)
    for codebook_index in range(delay_codes.shape[1]):
        tokens[:, codebook_index] = delay_codes[
            codebook_index : codebook_index + output_length, codebook_index
        ]
    return tokens


class MossTTSDelayProcessor:
    def __init__(
        self,
        tokenizer,
        model_config: ModelConfig,
        *,
        use_delay_pattern: bool = True,
        append_audio_start_for_generation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.use_delay_pattern = bool(use_delay_pattern)
        self.append_audio_start_for_generation = bool(append_audio_start_for_generation)
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
        self.include_scene = (
            not model_config.is_local_transformer and int(model_config.n_vq) == 16
        )

    def _id_to_token(self, token_id: int) -> str:
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token, list):
            return token[0] if token else ""
        return str(token)

    def build_user_message(
        self,
        text: Optional[str] = None,
        reference: Optional[list[Optional[Any]]] = None,
        instruction: Optional[str] = None,
        tokens: Optional[int] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        language: Optional[str] = None,
        scene: Optional[str] = None,
    ) -> dict[str, Any]:
        if reference is not None and not isinstance(reference, list):
            reference = [reference]
        text = normalize_tts_text(text)
        return UserMessage(
            text=text,
            reference=reference,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
            scene=scene,
            include_scene=self.include_scene,
        ).to_dict()

    @staticmethod
    def build_assistant_message(
        audio_codes_list: list[Any],
        content: str = AUDIO_PLACEHOLDER,
    ) -> dict[str, Any]:
        return AssistantMessage(
            audio_codes_list=audio_codes_list,
            content=content,
        ).to_dict()

    def _normalize_message(self, message: Message | dict[str, Any]) -> dict[str, Any]:
        if isinstance(message, Message):
            return message.to_dict()
        if not isinstance(message, dict):
            raise TypeError("Each message must be a Message or dict")
        if "role" not in message:
            raise ValueError("Message dict must include a role field")
        if "content" in message and "audio_codes_list" in message:
            return message
        role = message["role"]
        if role == "user":
            return self.build_user_message(
                **{key: message.get(key) for key in USER_MESSAGE_FIELDS}
            )
        if role == "assistant":
            return self.build_assistant_message(
                audio_codes_list=message.get("audio_codes_list", []),
                content=message.get("content", AUDIO_PLACEHOLDER),
            )
        raise ValueError(f"Unsupported role: {role}")

    @staticmethod
    def apply_chat_template(
        role: str, content: str, add_generation_prompt: bool
    ) -> str:
        rendered = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    @staticmethod
    def _replace_audio_placeholders(
        content: str,
        lengths: list[int],
        n_vq: int,
        gen_slot_token: str,
        delay_slot_token: str,
        audio_start_token: str,
        audio_end_token: str,
    ) -> str:
        if n_vq < 1:
            raise ValueError(f"n_vq must be >= 1, got {n_vq}")
        if content.count(AUDIO_PLACEHOLDER) != len(lengths):
            raise ValueError("Audio placeholders do not match audio code lengths")

        def build_audio_block(length: int) -> str:
            if length < 0:
                raise ValueError(f"length must be >= 0, got {length}")
            if length == 0:
                return f"{audio_start_token}{audio_end_token}"
            if delay_slot_token:
                return (
                    f"{audio_start_token}"
                    f"{gen_slot_token * length}"
                    f"{delay_slot_token * (n_vq - 1)}"
                    f"{audio_end_token}"
                )
            return f"{audio_start_token}{gen_slot_token * length}{audio_end_token}"

        lengths_iter = iter(lengths)
        return re.sub(
            re.escape(AUDIO_PLACEHOLDER),
            lambda _match: build_audio_block(next(lengths_iter)),
            content,
        )

    @staticmethod
    def _merge_consecutive_audio_placeholders(
        content: str,
        audio_codes_list: list[mx.array],
    ) -> tuple[str, list[mx.array]]:
        matches = list(re.finditer(re.escape(AUDIO_PLACEHOLDER), content))
        if len(matches) <= 1:
            return content, audio_codes_list
        if len(matches) != len(audio_codes_list):
            raise ValueError("Audio placeholders do not match audio codes")

        new_audio_codes = []
        parts = []
        last_pos = 0
        index = 0
        while index < len(matches):
            end_index = index
            while (
                end_index + 1 < len(matches)
                and content[
                    matches[end_index].end() : matches[end_index + 1].start()
                ].strip()
                == ""
            ):
                end_index += 1
            parts.append(content[last_pos : matches[index].start()])
            parts.append(AUDIO_PLACEHOLDER)
            last_pos = matches[end_index].end()
            if end_index == index:
                new_audio_codes.append(audio_codes_list[index])
            else:
                new_audio_codes.append(
                    mx.concatenate(audio_codes_list[index : end_index + 1], axis=0)
                )
            index = end_index + 1

        parts.append(content[last_pos:])
        return "".join(parts), new_audio_codes

    def _get_unified_codes(
        self,
        role: str,
        content: str,
        audio_codes_list: list[mx.array],
        truncation: bool,
    ) -> mx.array:
        if role == "user":
            audio_gen_slot_token = self.audio_user_slot_token
            audio_delay_slot_token = self.audio_user_slot_token
            truncation = False
        else:
            audio_gen_slot_token = self.audio_assistant_gen_slot_token
            audio_delay_slot_token = self.audio_assistant_delay_slot_token

        n_vq = int(self.model_config.n_vq)
        audio_codes_list = self._normalize_audio_codes_list(audio_codes_list, n_vq)
        if len(audio_codes_list) > 1 and AUDIO_PLACEHOLDER in content:
            content, audio_codes_list = self._merge_consecutive_audio_placeholders(
                content, audio_codes_list
            )
        content = self._replace_audio_placeholders(
            content=content,
            lengths=[int(audio_codes.shape[0]) for audio_codes in audio_codes_list],
            n_vq=n_vq,
            gen_slot_token=audio_gen_slot_token,
            delay_slot_token=(audio_delay_slot_token if self.use_delay_pattern else ""),
            audio_start_token=self.audio_start_token,
            audio_end_token=self.audio_end_token,
        )
        text_codes = mx.array(self.tokenizer.encode(content), dtype=mx.int32)

        text_list = text_codes.tolist()
        audio_start_indices = [
            i
            for i, token_id in enumerate(text_list)
            if token_id == self.model_config.audio_start_token_id
        ]
        audio_end_indices = [
            i
            for i, token_id in enumerate(text_list)
            if token_id == self.model_config.audio_end_token_id
        ]
        if len(audio_start_indices) != len(audio_codes_list) or len(
            audio_end_indices
        ) != len(audio_codes_list):
            raise ValueError(
                "Audio placeholders do not match the provided audio codes list"
            )

        if not audio_codes_list:
            delay_audio_codes = mx.full(
                (text_codes.shape[0], n_vq),
                self.model_config.audio_pad_code,
                dtype=mx.int32,
            )
        else:
            sections = []
            prefix_idx = 0
            for audio_start_idx, audio_end_idx, audio_codes in zip(
                audio_start_indices, audio_end_indices, audio_codes_list
            ):
                audio_codes = audio_codes.astype(mx.int32)
                if self.use_delay_pattern:
                    audio_codes = apply_delay_pattern(
                        audio_codes, self.model_config.audio_pad_code
                    )
                pad_codes = mx.full(
                    (audio_start_idx - prefix_idx + 1, n_vq),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                sections.extend([pad_codes, audio_codes])
                prefix_idx = audio_end_idx
            if truncation and self.use_delay_pattern:
                sections[-1] = sections[-1][: -(n_vq - 1), :]
            elif not truncation:
                sections.append(
                    mx.full(
                        (len(text_list) - audio_end_indices[-1], n_vq),
                        self.model_config.audio_pad_code,
                        dtype=mx.int32,
                    )
                )
            delay_audio_codes = mx.concatenate(sections, axis=0)

        if text_codes.shape[0] != delay_audio_codes.shape[0]:
            text_codes = text_codes[: delay_audio_codes.shape[0]]
        return mx.concatenate([text_codes[:, None], delay_audio_codes], axis=1)

    @staticmethod
    def _normalize_audio_codes_list(
        audio_codes_list: list[mx.array],
        n_vq: int,
    ) -> list[mx.array]:
        normalized = []
        for audio_codes in audio_codes_list:
            if audio_codes.ndim != 2:
                raise ValueError(
                    f"Expected audio codes shape [frames, n_vq], got {audio_codes.shape}"
                )
            if audio_codes.shape[1] < n_vq and audio_codes.shape[0] >= n_vq:
                audio_codes = audio_codes.transpose(1, 0)
            if audio_codes.shape[1] < n_vq:
                raise ValueError(
                    f"audio_codes channels ({audio_codes.shape[1]}) < "
                    f"model n_vq ({n_vq})"
                )
            normalized.append(audio_codes[:, :n_vq].astype(mx.int32))
        return normalized

    def __call__(
        self,
        conversations,
        *,
        mode: str = "generation",
        apply_chat_template: bool = True,
    ) -> dict[str, mx.array]:
        if mode not in {"generation", "continuation"}:
            raise ValueError("mode must be generation or continuation")
        if isinstance(conversations, (Message, dict)):
            conversations = [conversations]

        truncation = mode == "continuation"
        input_ids_list = []
        for conversation in conversations:
            if isinstance(conversation, (Message, dict)):
                conversation = [conversation]
            conversation = [
                self._normalize_message(message) for message in conversation
            ]
            if (mode == "generation") ^ (len(conversation) % 2 != 0):
                raise ValueError("Invalid conversation length for mode")
            if (mode == "generation") ^ (conversation[-1]["role"] == "user"):
                raise ValueError("Invalid final role for mode")

            unified_codes = []
            for message_idx, message in enumerate(conversation):
                add_generation_prompt = (
                    mode == "generation" and message_idx == len(conversation) - 1
                )
                content = str(message["content"])
                if apply_chat_template:
                    content = self.apply_chat_template(
                        message["role"], content, add_generation_prompt
                    )
                audio_codes_list = [
                    (
                        item
                        if isinstance(item, mx.array)
                        else mx.array(item, dtype=mx.int32)
                    )
                    for item in message.get("audio_codes_list", [])
                ]
                unified_codes.append(
                    self._get_unified_codes(
                        message["role"], content, audio_codes_list, truncation
                    )
                )
            input_ids = mx.concatenate(unified_codes, axis=0)
            if self.append_audio_start_for_generation and mode == "generation":
                audio_start_row = mx.full(
                    (1, input_ids.shape[-1]),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                audio_start_row[:, 0] = self.model_config.audio_start_token_id
                input_ids = mx.concatenate([input_ids, audio_start_row], axis=0)
            input_ids_list.append(input_ids)

        return self._pad(input_ids_list)

    def _pad(self, input_ids_list: list[mx.array]) -> dict[str, mx.array]:
        max_len = max(int(input_ids.shape[0]) for input_ids in input_ids_list)
        padded_ids = []
        masks = []
        for input_ids in input_ids_list:
            pad_len = max_len - int(input_ids.shape[0])
            if pad_len > 0:
                pad_rows = mx.full(
                    (pad_len, self.model_config.n_vq + 1),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                pad_rows[:, 0] = self.model_config.pad_token_id
                input_ids = mx.concatenate([pad_rows, input_ids], axis=0)
            mask = mx.concatenate(
                [
                    mx.zeros((pad_len,), dtype=mx.bool_),
                    mx.ones((max_len - pad_len,), dtype=mx.bool_),
                ]
            )
            padded_ids.append(input_ids)
            masks.append(mask)
        return {
            "input_ids": mx.stack(padded_ids, axis=0),
            "attention_mask": mx.stack(masks, axis=0),
        }


class MossTTSLocalProcessor(MossTTSDelayProcessor):
    def __init__(self, tokenizer, model_config: ModelConfig):
        super().__init__(
            tokenizer,
            model_config,
            use_delay_pattern=False,
            append_audio_start_for_generation=True,
        )


LOCAL_V15_USER_ROLE_PREFIX = "user\n"
LOCAL_V15_USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
LOCAL_V15_USER_TEMPLATE_AFTER_REFERENCE_SUFFIX = "\n- Text:\n"
LOCAL_V15_USER_TEMPLATE_SUFFIX = "\n</user_inst>"
LOCAL_V15_ASSISTANT_TURN_PREFIX = "\n"
LOCAL_V15_ASSISTANT_ROLE_PREFIX = "assistant\n"


def _normalize_template_value(value: Any) -> str:
    if value is None:
        return "None"
    value = str(value).strip()
    return value or "None"


def _render_local_v15_user_prompt_after_reference(
    *,
    language_code: object | None = None,
    prompt_fields: dict[str, Any] | None = None,
) -> str:
    fields = dict(prompt_fields or {})
    return (
        "\n- Instruction:\n"
        + _normalize_template_value(fields.get("instruction"))
        + "\n- Tokens:\n"
        + _normalize_template_value(fields.get("tokens"))
        + "\n- Quality:\n"
        + _normalize_template_value(fields.get("quality"))
        + "\n- Sound Event:\n"
        + _normalize_template_value(fields.get("sound_event"))
        + "\n- Ambient Sound:\n"
        + _normalize_template_value(fields.get("ambient_sound"))
        + "\n- Language:\n"
        + _normalize_template_value(fields.get("language", language_code))
        + LOCAL_V15_USER_TEMPLATE_AFTER_REFERENCE_SUFFIX
    )


@dataclass
class LocalV15UserMessage(Message):
    text: Optional[str] = None
    reference: Optional[list[Optional[Any]]] = None
    instruction: Optional[str] = None
    tokens: Optional[int] = None
    quality: Optional[str] = None
    sound_event: Optional[str] = None
    ambient_sound: Optional[str] = None
    language: Optional[str] = None

    def __post_init__(self):
        audio_codes_list = []
        if self.reference is None:
            reference = "None"
        else:
            reference_items = []
            for speaker_reference in self.reference:
                if speaker_reference is None:
                    continue
                reference_items.append(AUDIO_PLACEHOLDER)
                audio_codes_list.append(speaker_reference)
            reference = "\n".join(reference_items) if reference_items else "None"

        template = (
            "<user_inst>\n"
            "- Reference(s):\n{reference}\n"
            "- Instruction:\n{instruction}\n"
            "- Tokens:\n{tokens}\n"
            "- Quality:\n{quality}\n"
            "- Sound Event:\n{sound_event}\n"
            "- Ambient Sound:\n{ambient_sound}\n"
            "- Language:\n{language}\n"
            "- Text:\n{text}\n"
            "</user_inst>"
        )
        self._content = (
            template.replace("{reference}", str(reference))
            .replace("{instruction}", str(self.instruction))
            .replace("{tokens}", str(self.tokens))
            .replace("{quality}", str(self.quality))
            .replace("{sound_event}", str(self.sound_event))
            .replace("{ambient_sound}", str(self.ambient_sound))
            .replace("{language}", str(self.language))
            .replace("{text}", str(self.text))
        )
        self._audio_codes_list = audio_codes_list

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": "user",
            "content": self._content,
            "audio_codes_list": self._audio_codes_list,
            "text": self.text,
            "instruction": self.instruction,
            "tokens": self.tokens,
            "quality": self.quality,
            "sound_event": self.sound_event,
            "ambient_sound": self.ambient_sound,
            "language": self.language,
        }


class MossTTSLocalV15Processor:
    def __init__(self, tokenizer, model_config: ModelConfig):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.audio_user_slot_token = self._id_to_token(
            model_config.audio_user_slot_token_id
        )
        self.audio_assistant_slot_token = self._id_to_token(
            model_config.audio_assistant_slot_token_id
        )
        self.audio_start_token = self._id_to_token(model_config.audio_start_token_id)
        self.audio_end_token = self._id_to_token(model_config.audio_end_token_id)

    def _id_to_token(self, token_id: int) -> str:
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token, list):
            return token[0] if token else ""
        return str(token)

    @staticmethod
    def build_assistant_message(
        audio_codes_list: list[Any],
        content: str = AUDIO_PLACEHOLDER,
    ) -> dict[str, Any]:
        return AssistantMessage(
            audio_codes_list=audio_codes_list,
            content=content,
        ).to_dict()

    @staticmethod
    def build_user_message(
        text: Optional[str] = None,
        reference: Optional[list[Optional[Any]]] = None,
        instruction: Optional[str] = None,
        tokens: Optional[int] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        language: Optional[str] = None,
        scene: Optional[str] = None,
    ) -> dict[str, Any]:
        del scene
        if reference is not None and not isinstance(reference, list):
            reference = [reference]
        text = normalize_tts_text(text)
        return LocalV15UserMessage(
            text=text,
            reference=reference,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        ).to_dict()

    def _assert_fixed_nq(self, n_vq: int | None) -> int:
        config_nq = int(self.model_config.n_vq)
        if n_vq is not None and int(n_vq) != config_nq:
            raise ValueError(
                "MOSS-TTS-Local-Transformer-v1.5 uses the RVQ depth stored in "
                f"the model config. Expected n_vq={config_nq}, got {int(n_vq)}."
            )
        return config_nq

    def _encode_text(self, text: str) -> list[int]:
        try:
            return [
                int(token_id)
                for token_id in self.tokenizer.encode(
                    str(text), add_special_tokens=False
                )
            ]
        except TypeError:
            return [int(token_id) for token_id in self.tokenizer.encode(str(text))]

    def _build_text_rows(self, token_ids: list[int]) -> mx.array:
        rows = mx.full(
            (len(token_ids), int(self.model_config.n_vq) + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=mx.int32,
        )
        if token_ids:
            rows[:, 0] = mx.array(
                [int(token_id) for token_id in token_ids], dtype=mx.int32
            )
        return rows

    def _build_audio_rows(self, audio_tokens: mx.array, slot_token_id: int) -> mx.array:
        rows = mx.full(
            (int(audio_tokens.shape[0]), int(self.model_config.n_vq) + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=mx.int32,
        )
        if rows.shape[0] > 0:
            rows[:, 0] = int(slot_token_id)
            rows[:, 1:] = audio_tokens.astype(mx.int32)
        return rows

    def _user_prompt_prefix_ids(self) -> list[int]:
        return (
            [int(self.model_config.im_start_token_id)]
            + self._encode_text(LOCAL_V15_USER_ROLE_PREFIX)
            + self._encode_text(LOCAL_V15_USER_TEMPLATE_REFERENCE_PREFIX)
        )

    def _user_prompt_after_reference_ids(
        self,
        language_code: object | None,
        prompt_fields: dict[str, Any] | None,
    ) -> list[int]:
        return self._encode_text(
            _render_local_v15_user_prompt_after_reference(
                language_code=language_code,
                prompt_fields=prompt_fields,
            )
        )

    def _assistant_prompt_prefix_ids(self) -> list[int]:
        return (
            self._encode_text(LOCAL_V15_USER_TEMPLATE_SUFFIX)
            + [int(self.model_config.im_end_token_id)]
            + self._encode_text(LOCAL_V15_ASSISTANT_TURN_PREFIX)
            + [int(self.model_config.im_start_token_id)]
            + self._encode_text(LOCAL_V15_ASSISTANT_ROLE_PREFIX)
        )

    @staticmethod
    def _prompt_fields_from_user_message(message: dict[str, Any]) -> dict[str, Any]:
        fields = {}
        for key in (
            "instruction",
            "tokens",
            "quality",
            "sound_event",
            "ambient_sound",
            "language",
        ):
            if key in message and message.get(key) is not None:
                fields[key] = message.get(key)
        return fields

    def _normalize_audio_codes_list(
        self,
        audio_codes_list: list[Any],
        n_vq: int,
    ) -> list[mx.array]:
        normalized = []
        for audio_codes in audio_codes_list:
            if not isinstance(audio_codes, mx.array):
                audio_codes = mx.array(audio_codes, dtype=mx.int32)
            if audio_codes.ndim != 2 or int(audio_codes.shape[1]) != n_vq:
                raise ValueError(
                    f"audio code tensor must have shape [frames, {n_vq}], "
                    f"got {audio_codes.shape}"
                )
            normalized.append(audio_codes.astype(mx.int32))
        return normalized

    def _build_generation_or_voice_clone_codes(
        self,
        message: dict[str, Any],
        n_vq: int,
    ) -> mx.array:
        if "text" not in message:
            raise ValueError(
                "Direct MOSS-TTS-Local-Transformer-v1.5 generation requires "
                "messages built by build_user_message(...)."
            )
        text = "" if message.get("text") is None else str(message.get("text"))
        prompt_fields = self._prompt_fields_from_user_message(message)
        language_code = message.get("language")
        audio_codes_list = self._normalize_audio_codes_list(
            list(message.get("audio_codes_list", [])),
            n_vq,
        )
        text_token_ids = self._encode_text(text)

        if audio_codes_list:
            parts = [self._build_text_rows(self._user_prompt_prefix_ids())]
            for reference_codes in audio_codes_list:
                parts.append(
                    self._build_text_rows([int(self.model_config.audio_start_token_id)])
                )
                parts.append(
                    self._build_audio_rows(
                        reference_codes,
                        int(self.model_config.audio_user_slot_token_id),
                    )
                )
                parts.append(
                    self._build_text_rows([int(self.model_config.audio_end_token_id)])
                )
            parts.append(
                self._build_text_rows(
                    self._user_prompt_after_reference_ids(language_code, prompt_fields)
                    + text_token_ids
                    + self._assistant_prompt_prefix_ids()
                    + [int(self.model_config.audio_start_token_id)]
                )
            )
            return mx.concatenate(parts, axis=0)

        prompt_token_ids = (
            self._user_prompt_prefix_ids()
            + self._encode_text("None")
            + self._user_prompt_after_reference_ids(language_code, prompt_fields)
            + text_token_ids
            + self._assistant_prompt_prefix_ids()
            + [int(self.model_config.audio_start_token_id)]
        )
        return self._build_text_rows(prompt_token_ids)

    def _build_continuation_codes(
        self,
        conversation: list[dict[str, Any]],
        n_vq: int,
    ) -> mx.array:
        if len(conversation) < 2:
            raise ValueError(
                "continuation mode requires a user message followed by an "
                "assistant audio message."
            )
        user_message = conversation[-2]
        assistant_message = conversation[-1]
        if (
            user_message.get("role") != "user"
            or assistant_message.get("role") != "assistant"
        ):
            raise ValueError(
                "continuation mode requires the last two messages to be user, "
                "assistant."
            )
        if "text" not in user_message:
            raise ValueError(
                "Direct MOSS-TTS-Local-Transformer-v1.5 continuation requires "
                "user messages built by build_user_message(...)."
            )
        text = "" if user_message.get("text") is None else str(user_message.get("text"))
        prompt_fields = self._prompt_fields_from_user_message(user_message)
        language_code = user_message.get("language")
        prompt_token_ids = (
            self._user_prompt_prefix_ids()
            + self._encode_text("None")
            + self._user_prompt_after_reference_ids(language_code, prompt_fields)
            + self._encode_text(text)
            + self._assistant_prompt_prefix_ids()
            + [int(self.model_config.audio_start_token_id)]
        )
        audio_codes_list = self._normalize_audio_codes_list(
            list(assistant_message.get("audio_codes_list", [])),
            n_vq,
        )
        if not audio_codes_list:
            return self._build_text_rows(prompt_token_ids)
        if len(audio_codes_list) != 1:
            raise ValueError(
                "MOSS-TTS-Local-Transformer-v1.5 continuation mode expects one "
                "prompt audio item."
            )
        return mx.concatenate(
            [
                self._build_text_rows(prompt_token_ids),
                self._build_audio_rows(
                    audio_codes_list[0],
                    int(self.model_config.audio_assistant_slot_token_id),
                ),
            ],
            axis=0,
        )

    def _normalize_message(self, message: Message | dict[str, Any]) -> dict[str, Any]:
        if isinstance(message, Message):
            return message.to_dict()
        if not isinstance(message, dict):
            raise TypeError("Each message must be a Message or dict.")
        if "content" in message and "audio_codes_list" in message:
            return message
        role = message.get("role")
        if role == "user":
            return self.build_user_message(
                **{key: message.get(key) for key in USER_MESSAGE_FIELDS}
            )
        if role == "assistant":
            return self.build_assistant_message(
                audio_codes_list=message.get("audio_codes_list", []),
                content=message.get("content", AUDIO_PLACEHOLDER),
            )
        raise ValueError(f"Unsupported role: {role}")

    def _pad(self, input_ids_list: list[mx.array]) -> dict[str, mx.array]:
        max_len = max(int(input_ids.shape[0]) for input_ids in input_ids_list)
        padded_ids = []
        masks = []
        for input_ids in input_ids_list:
            pad_len = max_len - int(input_ids.shape[0])
            if pad_len > 0:
                pad_rows = mx.full(
                    (pad_len, self.model_config.n_vq + 1),
                    self.model_config.audio_pad_token_id,
                    dtype=mx.int32,
                )
                pad_rows[:, 0] = self.model_config.pad_token_id
                input_ids = mx.concatenate([pad_rows, input_ids], axis=0)
            mask = mx.concatenate(
                [
                    mx.zeros((pad_len,), dtype=mx.bool_),
                    mx.ones((max_len - pad_len,), dtype=mx.bool_),
                ]
            )
            padded_ids.append(input_ids)
            masks.append(mask)
        return {
            "input_ids": mx.stack(padded_ids, axis=0),
            "attention_mask": mx.stack(masks, axis=0),
        }

    def __call__(
        self,
        conversations,
        *,
        mode: str = "generation",
        apply_chat_template: bool = True,
        n_vq: int | None = None,
    ) -> dict[str, mx.array]:
        del apply_chat_template
        if mode not in {"generation", "continuation"}:
            raise ValueError("mode must be generation or continuation")
        n_vq = self._assert_fixed_nq(n_vq)
        if isinstance(conversations, (Message, dict)):
            conversations = [conversations]

        input_ids_list = []
        for conversation in conversations:
            if isinstance(conversation, (Message, dict)):
                conversation = [conversation]
            conversation = [
                self._normalize_message(message) for message in conversation
            ]
            if (mode == "generation") ^ (conversation[-1]["role"] == "user"):
                raise ValueError("generation mode must end with a user message.")
            if mode == "continuation" and conversation[-1]["role"] != "assistant":
                raise ValueError(
                    "continuation mode must end with an assistant message."
                )
            if mode == "generation":
                input_ids = self._build_generation_or_voice_clone_codes(
                    conversation[-1],
                    n_vq,
                )
            else:
                input_ids = self._build_continuation_codes(conversation, n_vq)
            input_ids_list.append(input_ids)

        return self._pad(input_ids_list)

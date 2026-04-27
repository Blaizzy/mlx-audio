from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

# pyright: reportMissingImports=false


AUDIO_PLACEHOLDER = "<|audio|>"


_USER_MESSAGE_TEMPLATE = """<user_inst>
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
- Text:
{text}
</user_inst>"""


def build_user_message(
    text: Optional[str] = None,
    reference: Optional[List[Optional[Any]]] = None,
    instruction: Optional[str] = None,
    tokens: Optional[int] = None,
    quality: Optional[str] = None,
    sound_event: Optional[str] = None,
    ambient_sound: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    if reference is None:
        reference_str = "None"
    elif isinstance(reference, list):
        refs: List[str] = []
        for speaker_idx, speaker_reference in enumerate(reference):
            if speaker_reference is not None:
                refs.append(f"[S{speaker_idx + 1}]:\n{AUDIO_PLACEHOLDER}")
        reference_str = "\n".join(refs) if len(refs) > 0 else "None"
    else:
        raise TypeError("`reference` should be a list when it is not None.")

    content = (
        _USER_MESSAGE_TEMPLATE.replace("{reference}", str(reference_str))
        .replace("{instruction}", str(instruction))
        .replace("{tokens}", str(tokens))
        .replace("{quality}", str(quality))
        .replace("{sound_event}", str(sound_event))
        .replace("{ambient_sound}", str(ambient_sound))
        .replace("{language}", str(language))
        .replace("{text}", str(text))
    )

    return content


def apply_delay_pattern(codes: mx.array, pad_code: int) -> mx.array:
    if codes.ndim != 2:
        raise ValueError("codes must have shape (T, NQ)")
    t, nq = int(codes.shape[0]), int(codes.shape[1])
    if nq < 1:
        raise ValueError("nq must be >= 1")

    cols = []
    for i in range(nq):
        pre = mx.full((i,), pad_code, dtype=codes.dtype)
        post = mx.full((nq - 1 - i,), pad_code, dtype=codes.dtype)
        cols.append(mx.concatenate([pre, codes[:, i], post], axis=0))
    return mx.stack(cols, axis=1)


def apply_de_delay_pattern(delay_codes: mx.array) -> mx.array:
    if delay_codes.ndim != 2:
        raise ValueError("delay_codes must have shape (T_delay, NQ)")
    t_delay, nq = int(delay_codes.shape[0]), int(delay_codes.shape[1])
    t = t_delay - nq + 1
    if t <= 0:
        return mx.zeros((0, nq), dtype=delay_codes.dtype)

    cols = [delay_codes[i : i + t, i] for i in range(nq)]
    return mx.stack(cols, axis=1)


def _id_to_token(tokenizer: Any, token_id: int) -> str:
    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        return str(int(token_id))
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    if isinstance(tok, list):
        return tok[0] if len(tok) > 0 else ""
    return str(tok)


def _replace_audio_placeholders(
    content: str,
    lengths: Sequence[int],
    *,
    n_vq: int,
    gen_slot_token: str,
    delay_slot_token: str,
    audio_start_token: str,
    audio_end_token: str,
) -> str:
    if int(n_vq) < 1:
        raise ValueError(f"n_vq must be >= 1, got {n_vq}")

    num_placeholders = int(content.count(AUDIO_PLACEHOLDER))
    if num_placeholders != int(len(lengths)):
        raise ValueError(
            f"Number of {AUDIO_PLACEHOLDER} ({num_placeholders}) does not match lengths ({len(lengths)})"
        )

    def build_audio_block(length: int) -> str:
        if int(length) < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if int(length) == 0:
            return f"{audio_start_token}{audio_end_token}"
        step_tokens = (gen_slot_token * int(length)) + (
            delay_slot_token * (int(n_vq) - 1)
        )
        return f"{audio_start_token}{step_tokens}{audio_end_token}"

    lengths_iter = iter([int(x) for x in lengths])

    def replacer(match: re.Match) -> str:
        _ = match
        length = next(lengths_iter)
        return build_audio_block(length)

    return re.sub(re.escape(AUDIO_PLACEHOLDER), replacer, content)


def _merge_consecutive_audio_placeholders(
    content: str, audio_codes_list: List[mx.array]
) -> Tuple[str, List[mx.array]]:
    matches = list(re.finditer(re.escape(AUDIO_PLACEHOLDER), content))
    if len(matches) <= 1:
        return content, audio_codes_list

    if len(matches) != len(audio_codes_list):
        raise ValueError(
            "Audio placeholders do not match the provided audio codes list."
        )

    merged_audio_codes_list: List[mx.array] = []
    new_parts: List[str] = []
    last_pos = 0
    i = 0
    while i < len(matches):
        j = i
        while (
            j + 1 < len(matches)
            and content[matches[j].end() : matches[j + 1].start()].strip() == ""
        ):
            j += 1

        new_parts.append(content[last_pos : matches[i].start()])
        new_parts.append(AUDIO_PLACEHOLDER)
        last_pos = matches[j].end()

        if j == i:
            merged_audio_codes_list.append(audio_codes_list[i])
        else:
            merged_audio_codes_list.append(
                mx.concatenate(audio_codes_list[i : j + 1], axis=0)
            )

        i = j + 1

    new_parts.append(content[last_pos:])
    return "".join(new_parts), merged_audio_codes_list


def _get_unified_codes(
    *,
    role: str,
    content: str,
    audio_codes_list: List[mx.array],
    truncation: bool,
    tokenizer: Any,
    config: Any,
) -> mx.array:
    if role == "user":
        audio_gen_slot_token = audio_delay_slot_token = _id_to_token(
            tokenizer, int(getattr(config, "audio_user_slot_token_id", 151654))
        )
        truncation = False
    else:
        audio_gen_slot_token = _id_to_token(
            tokenizer, int(getattr(config, "audio_assistant_gen_slot_token_id", 151656))
        )
        audio_delay_slot_token = _id_to_token(
            tokenizer,
            int(getattr(config, "audio_assistant_delay_slot_token_id", 151662)),
        )

    audio_start_token = _id_to_token(
        tokenizer, int(getattr(config, "audio_start_token_id", 151652))
    )
    audio_end_token = _id_to_token(
        tokenizer, int(getattr(config, "audio_end_token_id", 151653))
    )

    if len(audio_codes_list) > 0:
        n_vq = int(audio_codes_list[0].shape[1])
    else:
        n_vq = int(getattr(config, "n_vq", 32))

    if len(audio_codes_list) > 1 and AUDIO_PLACEHOLDER in content:
        content, audio_codes_list = _merge_consecutive_audio_placeholders(
            content, audio_codes_list
        )

    content = _replace_audio_placeholders(
        content,
        lengths=[int(c.shape[0]) for c in audio_codes_list],
        n_vq=n_vq,
        gen_slot_token=audio_gen_slot_token,
        delay_slot_token=audio_delay_slot_token,
        audio_start_token=audio_start_token,
        audio_end_token=audio_end_token,
    )

    token_ids: List[int] = [int(x) for x in tokenizer.encode(content)]
    audio_start_id = int(getattr(config, "audio_start_token_id", 151652))
    audio_end_id = int(getattr(config, "audio_end_token_id", 151653))
    audio_start_indices = [
        i for i, tid in enumerate(token_ids) if tid == audio_start_id
    ]
    audio_end_indices = [i for i, tid in enumerate(token_ids) if tid == audio_end_id]

    if len(audio_start_indices) != len(audio_codes_list) or len(
        audio_end_indices
    ) != len(audio_codes_list):
        raise ValueError(
            "Audio placeholders do not match the provided audio codes list. "
            f"starts={len(audio_start_indices)} ends={len(audio_end_indices)} codes={len(audio_codes_list)}"
        )

    pad_code = int(getattr(config, "audio_pad_code", 1024))

    if len(audio_codes_list) == 0:
        delay_audio_codes = mx.full((len(token_ids), n_vq), pad_code, dtype=mx.int32)
    else:
        parts: List[mx.array] = []
        prefix_idx = 0
        for audio_start_idx, audio_end_idx, audio_codes in zip(
            audio_start_indices, audio_end_indices, audio_codes_list
        ):
            delay_audio = apply_delay_pattern(audio_codes, pad_code)
            pad = mx.full(
                (int(audio_start_idx - prefix_idx + 1), n_vq),
                pad_code,
                dtype=mx.int32,
            )
            parts.extend([pad, delay_audio])
            prefix_idx = int(audio_end_idx)

        if truncation:
            if n_vq > 1 and int(parts[-1].shape[0]) >= (n_vq - 1):
                parts[-1] = parts[-1][: -(n_vq - 1), :]
        else:
            last_audio_end_idx = int(audio_end_indices[-1])
            pad = mx.full(
                (int(len(token_ids) - last_audio_end_idx), n_vq),
                pad_code,
                dtype=mx.int32,
            )
            parts.append(pad)

        delay_audio_codes = mx.concatenate(parts, axis=0)

    if len(token_ids) != int(delay_audio_codes.shape[0]):
        token_ids = token_ids[: int(delay_audio_codes.shape[0])]

    text_codes = mx.array(token_ids, dtype=mx.int32)
    unified = mx.concatenate([text_codes[:, None], delay_audio_codes], axis=1)
    return unified


def _apply_chat_template(
    tokenizer: Any, role: str, content: str, *, add_generation_prompt: bool
) -> str:
    messages = [{"role": role, "content": content}]

    def _manual_qwen_chat_template() -> str:
        rendered = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    @lru_cache(maxsize=1)
    def _get_qwen3_tokenizer() -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def _render_with_qwen3_chat_template() -> str:
        try:
            qwen_tok = _get_qwen3_tokenizer()
            return str(
                qwen_tok.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                )
            )
        except Exception:
            return _manual_qwen_chat_template()

    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return _render_with_qwen3_chat_template()
    except Exception:
        return _render_with_qwen3_chat_template()


def prepare_generation_input(
    text: str,
    tokenizer: Any,
    config: Any,
    *,
    reference_audio_codes: Optional[Sequence[mx.array]] = None,
    instruction: Optional[str] = None,
    tokens: Optional[int] = None,
    quality: Optional[str] = None,
    sound_event: Optional[str] = None,
    ambient_sound: Optional[str] = None,
    language: Optional[str] = None,
) -> mx.array:
    audio_codes_list: List[mx.array] = []
    if reference_audio_codes is not None:
        audio_codes_list = list(reference_audio_codes)

    user_content = build_user_message(
        text=text,
        reference=audio_codes_list if len(audio_codes_list) > 0 else None,
        instruction=instruction,
        tokens=tokens,
        quality=quality,
        sound_event=sound_event,
        ambient_sound=ambient_sound,
        language=language,
    )
    chat_text = _apply_chat_template(
        tokenizer, "user", user_content, add_generation_prompt=True
    )

    unified = _get_unified_codes(
        role="user",
        content=str(chat_text),
        audio_codes_list=audio_codes_list,
        truncation=False,
        tokenizer=tokenizer,
        config=config,
    )
    return unified[None, :, :]


def parse_output(
    generation_ids: mx.array, start_length: int, config: Any
) -> Tuple[mx.array, List[mx.array]]:
    if generation_ids.ndim == 3 and int(generation_ids.shape[0]) == 1:
        generation_ids = generation_ids[0]
    if generation_ids.ndim != 2:
        raise ValueError("generation_ids must have shape (L, 1+n_vq) or (1, L, 1+n_vq)")

    _ = int(start_length)
    pad_code = int(getattr(config, "audio_pad_code", 1024))

    text_ids = generation_ids[:, 0]
    delay_audio = generation_ids[:, 1:]
    audio_codes = apply_de_delay_pattern(delay_audio)
    if int(audio_codes.shape[0]) == 0:
        return text_ids, []

    audio_codes_np = np.array(audio_codes)
    is_pad_row_list = np.all(audio_codes_np == pad_code, axis=1).tolist()
    idx_list = [i for i, is_pad in enumerate(is_pad_row_list) if not bool(is_pad)]

    if len(idx_list) == 0:
        return text_ids, []

    segments: List[mx.array] = []
    run_start = idx_list[0]
    prev = idx_list[0]
    for cur in idx_list[1:]:
        if cur != prev + 1:
            segments.append(audio_codes[run_start : prev + 1])
            run_start = cur
        prev = cur
    segments.append(audio_codes[run_start : prev + 1])

    return text_ids, segments


def build_assistant_message(
    audio_codes_list: List[mx.array],
    content: str = AUDIO_PLACEHOLDER,
) -> dict:
    if not isinstance(audio_codes_list, list):
        raise TypeError("`audio_codes_list` must be a list of mx.array")

    placeholder_count = content.count(AUDIO_PLACEHOLDER)
    if placeholder_count == 0 and len(audio_codes_list) > 0:
        content = AUDIO_PLACEHOLDER

    return {
        "role": "assistant",
        "content": content,
        "audio_codes_list": audio_codes_list,
    }


def _normalize_message(message: dict) -> dict:
    if not isinstance(message, dict):
        raise TypeError("Each message must be a dict.")
    if "role" not in message:
        raise ValueError("Message dict must include a 'role' field.")
    if "content" in message and "audio_codes_list" in message:
        return message
    role = message["role"]
    if role == "user":
        user_fields = (
            "text",
            "reference",
            "instruction",
            "tokens",
            "quality",
            "sound_event",
            "ambient_sound",
            "language",
        )
        kwargs = {key: message.get(key) for key in user_fields}
        content = build_user_message(**kwargs)
        audio_codes_list: List[mx.array] = []
        ref = message.get("reference")
        if isinstance(ref, list):
            audio_codes_list = [r for r in ref if isinstance(r, mx.array)]
        return {
            "role": "user",
            "content": content,
            "audio_codes_list": audio_codes_list,
        }
    if role == "assistant":
        return build_assistant_message(
            audio_codes_list=message.get("audio_codes_list", []),
            content=message.get("content", AUDIO_PLACEHOLDER),
        )
    raise ValueError(f"Unsupported role: {role}")


def prepare_conversation_input(
    conversation: List[dict],
    tokenizer: Any,
    config: Any,
    *,
    mode: str = "generation",
) -> mx.array:
    if mode not in ("generation", "continuation"):
        raise ValueError(f"mode must be 'generation' or 'continuation', got {mode!r}")

    conversation = [_normalize_message(m) for m in conversation]

    if mode == "generation":
        if len(conversation) % 2 == 0:
            raise ValueError(
                "generation mode requires odd number of messages (last must be user)"
            )
        if conversation[-1]["role"] != "user":
            raise ValueError("generation mode requires last message to be user")
    else:
        if len(conversation) % 2 != 0:
            raise ValueError(
                "continuation mode requires even number of messages (last must be assistant)"
            )
        if conversation[-1]["role"] != "assistant":
            raise ValueError("continuation mode requires last message to be assistant")

    truncation = mode == "continuation"

    unified_parts: List[mx.array] = []
    for msg_idx, message in enumerate(conversation):
        is_last = msg_idx == len(conversation) - 1
        add_generation_prompt = mode == "generation" and is_last

        chat_text = _apply_chat_template(
            tokenizer,
            message["role"],
            message["content"],
            add_generation_prompt=add_generation_prompt,
        )

        msg_truncation = truncation and is_last

        unified = _get_unified_codes(
            role=message["role"],
            content=str(chat_text),
            audio_codes_list=message.get("audio_codes_list", []),
            truncation=msg_truncation,
            tokenizer=tokenizer,
            config=config,
        )
        unified_parts.append(unified)

    return mx.concatenate(unified_parts, axis=0)[None, :, :]


@dataclass
class AssistantMessage:
    content: str
    audio: List[mx.array]


@dataclass
class MossTTSProcessor:
    tokenizer: Any
    config: Any
    codec: Any | None = None

    def build_user_message(self, **kwargs: Any) -> str:
        return build_user_message(**kwargs)

    def build_assistant_message(
        self, audio_codes_list: List[mx.array], content: str = AUDIO_PLACEHOLDER
    ) -> dict:
        return build_assistant_message(
            audio_codes_list=audio_codes_list, content=content
        )

    def prepare_conversation_input(
        self,
        conversation: List[dict],
        *,
        mode: str = "generation",
    ) -> mx.array:
        return prepare_conversation_input(
            conversation=conversation,
            tokenizer=self.tokenizer,
            config=self.config,
            mode=mode,
        )

    def prepare_generation_input(
        self,
        text: str,
        *,
        reference_audio_codes: Optional[Sequence[mx.array]] = None,
        instruction: Optional[str] = None,
        tokens: Optional[int] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        language: Optional[str] = None,
    ) -> mx.array:
        return prepare_generation_input(
            text=text,
            tokenizer=self.tokenizer,
            config=self.config,
            reference_audio_codes=reference_audio_codes,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        )

    def parse_output(
        self, generation_ids: "mx.array", start_length: int
    ) -> Tuple["mx.array", List["mx.array"]]:
        return parse_output(
            generation_ids=generation_ids, start_length=start_length, config=self.config
        )

    def _decode_token_ids(self, token_ids: mx.array) -> str:
        token_ids_np = np.array(token_ids).reshape(-1)
        token_ids_list = [int(x) for x in token_ids_np]
        return str(self.tokenizer.decode(token_ids_list))

    def _parse_text_codes(self, start_length: int, text_codes: mx.array) -> str:
        text = self._decode_token_ids(text_codes)
        prefix = self._decode_token_ids(text_codes[:start_length])
        if text.startswith(prefix):
            text = text[len(prefix) :]
        else:
            text = text[len(prefix) :]

        audio_start_token = re.escape(
            _id_to_token(
                self.tokenizer,
                int(getattr(self.config, "audio_start_token_id", 151652)),
            )
        )
        audio_end_token = re.escape(
            _id_to_token(
                self.tokenizer, int(getattr(self.config, "audio_end_token_id", 151653))
            )
        )
        audio_gen_slot_token = re.escape(
            _id_to_token(
                self.tokenizer,
                int(getattr(self.config, "audio_assistant_gen_slot_token_id", 151656)),
            )
        )
        audio_delay_slot_token = re.escape(
            _id_to_token(
                self.tokenizer,
                int(
                    getattr(self.config, "audio_assistant_delay_slot_token_id", 151662)
                ),
            )
        )

        audio_pattern = re.compile(
            rf"(?:{audio_start_token})?"
            rf"(?:{audio_gen_slot_token})*"
            rf"(?:{audio_delay_slot_token})*"
            rf"{audio_end_token}"
        )

        def normalize_audio_segments(value: str) -> str:
            def repl(match: re.Match) -> str:
                seg = match.group(0)
                if (
                    _id_to_token(
                        self.tokenizer,
                        int(
                            getattr(
                                self.config, "audio_assistant_gen_slot_token_id", 151656
                            )
                        ),
                    )
                    in seg
                ):
                    return AUDIO_PLACEHOLDER
                return ""

            return audio_pattern.sub(repl, value)

        return normalize_audio_segments(text)

    def decode_audio_codes(
        self, audio_codes_list: Sequence[mx.array], *, codec: Any | None = None
    ) -> List[mx.array]:
        codec = codec if codec is not None else self.codec
        if codec is None:
            raise RuntimeError("codec is not set on processor.")

        wav_list: List[mx.array] = []
        for codes in audio_codes_list:
            if codes.ndim != 2:
                raise ValueError("audio codes must have shape (T, NQ)")
            if int(codes.shape[0]) <= 0:
                wav_list.append(mx.zeros((0,), dtype=mx.float32))
                continue
            codes_t = codes.transpose(1, 0)[:, None, :]
            dec = codec.decode(codes_t)
            audio = dec["audio"][0, 0].astype(mx.float32)
            wav_list.append(audio)
        return wav_list

    def _parse_audio_codes(
        self,
        start_length: int,
        delay_audio_codes: mx.array,
        *,
        codec: Any | None = None,
    ) -> List[mx.array]:
        _, audio_codes_list = parse_output(
            generation_ids=mx.concatenate(
                [
                    mx.zeros((int(delay_audio_codes.shape[0]), 1), dtype=mx.int32),
                    delay_audio_codes,
                ],
                axis=1,
            ),
            start_length=start_length,
            config=self.config,
        )
        if len(audio_codes_list) == 0:
            return []

        decoded_audio_list = self.decode_audio_codes(audio_codes_list, codec=codec)
        if start_length > 0 and len(decoded_audio_list) > 0:
            first_codes_length = int(audio_codes_list[0].shape[0])
            if first_codes_length > 0:
                trim_ratio = max(
                    0.0, min(float(start_length) / float(first_codes_length), 1.0)
                )
                first_audio = decoded_audio_list[0]
                if trim_ratio >= 1.0:
                    decoded_audio_list = decoded_audio_list[1:]
                elif trim_ratio > 0.0:
                    trim_samples = int(float(first_audio.shape[-1]) * trim_ratio)
                    decoded_audio_list[0] = first_audio[trim_samples:]
        return decoded_audio_list

    def decode(
        self, output: Sequence[Tuple[int, mx.array]], *, codec: Any | None = None
    ) -> List[Optional[AssistantMessage]]:
        generated_messages: List[Optional[AssistantMessage]] = []
        for start_length, generation_ids in output:
            if generation_ids.ndim == 3 and int(generation_ids.shape[0]) == 1:
                generation_ids = generation_ids[0]
            if generation_ids.ndim != 2:
                raise ValueError(
                    "generation_ids must have shape (L, 1+n_vq) or (1, L, 1+n_vq)"
                )

            content = self._parse_text_codes(int(start_length), generation_ids[:, 0])
            audio = self._parse_audio_codes(
                int(start_length), generation_ids[:, 1:], codec=codec
            )
            if content == "":
                generated_messages.append(None)
            else:
                generated_messages.append(
                    AssistantMessage(content=content, audio=audio)
                )
        return generated_messages

"""Shared request normalization for MOSS-TTS family variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MossNormalizedRequest:
    """Narrow-waist request contract used by upstream build_user_message()."""

    text: Optional[str] = None
    reference: Optional[List[Optional[Any]]] = None
    instruction: Optional[str] = None
    tokens: Optional[int] = None
    quality: Optional[str] = None
    sound_event: Optional[str] = None
    ambient_sound: Optional[str] = None
    language: Optional[str] = None

    def __post_init__(self):
        if self.tokens is not None and self.tokens <= 0:
            raise ValueError("tokens must be positive when provided")

    @classmethod
    def from_generate_kwargs(
        cls,
        *,
        text: Optional[str] = None,
        reference: Optional[List[Optional[Any]] | Optional[Any]] = None,
        instruction: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[Any] = None,
        tokens: Optional[int] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        language: Optional[str] = None,
    ) -> "MossNormalizedRequest":
        if instruction is not None and instruct is not None and instruction != instruct:
            raise ValueError(
                "Both instruction and instruct were provided with different values"
            )

        if reference is not None and ref_audio is not None:
            raise ValueError("Provide either reference or ref_audio, not both")

        normalized_reference = reference
        if normalized_reference is None and ref_audio is not None:
            normalized_reference = [ref_audio]
        elif normalized_reference is not None and not isinstance(
            normalized_reference, list
        ):
            normalized_reference = [normalized_reference]

        return cls(
            text=text,
            reference=normalized_reference,
            instruction=instruction if instruction is not None else instruct,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        )

    def to_user_message_kwargs(self) -> Dict[str, Optional[Any]]:
        return {
            "text": self.text,
            "reference": self.reference,
            "instruction": self.instruction,
            "tokens": self.tokens,
            "quality": self.quality,
            "sound_event": self.sound_event,
            "ambient_sound": self.ambient_sound,
            "language": self.language,
        }


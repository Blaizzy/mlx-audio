from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import mlx.core as mx

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
LEGACY_SYMBOL_VOCAB_SIZE = 192
BYTE_VOCAB_SIZE = 256
BYTE_TEXT_VOCAB_SIZE = LEGACY_SYMBOL_VOCAB_SIZE + BYTE_VOCAB_SIZE


_SILENCE_TOKENS_0_2S = mx.array(
    [
        [568, 778, 338, 524, 967, 360, 728, 550, 90],
        [568, 778, 10, 674, 364, 981, 741, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 804, 10, 674, 364, 981, 568, 378, 731],
        [568, 778, 721, 842, 264, 974, 989, 507, 308],
    ],
    dtype=mx.int32,
)


@dataclass(frozen=True)
class TTSPromptConfig:
    n_codebooks: int = 9
    audio_pad_id: int = 1025
    text_vocab: int = 519
    speaking_rate_num_buckets: int = 8
    quality_bucket_counts: tuple[int, ...] = (12, 12, 12, 8, 8, 8)
    speaker_background_num_buckets: int = 2
    accurate_mode_num_buckets: int = 1
    prepend_silence: bool = True

    def __post_init__(self) -> None:
        if self.n_codebooks <= 0:
            raise ValueError("n_codebooks must be positive")
        if self.audio_pad_id < 0:
            raise ValueError("audio_pad_id must be non-negative")
        if self.text_vocab < BYTE_TEXT_VOCAB_SIZE:
            raise ValueError(f"text_vocab must include byte IDs, got {self.text_vocab}")
        _conditioning_base_text_vocab(
            self.text_vocab,
            self.speaking_rate_num_buckets,
            self.quality_bucket_counts,
            self.speaker_background_num_buckets,
            self.accurate_mode_num_buckets,
            context="prompt configuration",
        )


def byte_text_vocab_size() -> int:
    return BYTE_TEXT_VOCAB_SIZE


def conditioned_text_vocab_size(
    speaking_rate_num_buckets: int = 0,
    quality_num_buckets: int = 0,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    counts = (
        int(speaking_rate_num_buckets),
        int(quality_num_buckets),
        int(speaker_background_num_buckets),
        int(accurate_mode_num_buckets),
    )
    if any(count < 0 for count in counts):
        raise ValueError("conditioning bucket counts must be non-negative")
    return BYTE_TEXT_VOCAB_SIZE + sum(counts)


def text_to_byte_ids(text: str) -> list[int]:
    return [
        BOS_ID,
        *(byte + LEGACY_SYMBOL_VOCAB_SIZE for byte in text.encode("utf-8")),
        EOS_ID,
    ]


def _normalize_quality_bucket_counts(counts: Iterable[int] | None) -> tuple[int, ...]:
    result = tuple(int(x) for x in (counts or ()))
    if any(x < 0 for x in result):
        raise ValueError("quality bucket counts must be non-negative")
    return result


def _conditioning_base_text_vocab(
    text_vocab: int,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Iterable[int] | None,
    speaker_background_num_buckets: int,
    accurate_mode_num_buckets: int,
    *,
    context: str,
) -> int:
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base = (
        int(text_vocab)
        - int(speaking_rate_num_buckets)
        - sum(counts)
        - int(speaker_background_num_buckets)
        - int(accurate_mode_num_buckets)
    )
    if base < 0:
        raise ValueError(f"text_vocab is too small for {context}")
    return base


def speaking_rate_token_id(
    text_vocab: int,
    speaking_rate_num_buckets: int,
    speaking_rate_bucket: int,
    quality_bucket_counts: Iterable[int] | None = (),
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    bucket = int(speaking_rate_bucket)
    count = int(speaking_rate_num_buckets)
    if bucket < 0 or bucket >= count:
        raise ValueError(f"speaking_rate_bucket must be in [0, {count - 1}]")
    return (
        _conditioning_base_text_vocab(
            text_vocab,
            count,
            quality_bucket_counts,
            speaker_background_num_buckets,
            accurate_mode_num_buckets,
            context="speaking-rate conditioning",
        )
        + bucket
    )


def quality_token_id(
    text_vocab: int,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Iterable[int],
    feature_idx: int,
    quality_bucket: int,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    feature = int(feature_idx)
    if feature < 0 or feature >= len(counts):
        raise ValueError(f"quality feature index out of range: {feature}")
    bucket = int(quality_bucket)
    if bucket < 0 or bucket >= counts[feature]:
        raise ValueError(
            f"quality bucket for feature {feature} must be in [0, {counts[feature] - 1}]"
        )
    base = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="quality conditioning",
    )
    return base + int(speaking_rate_num_buckets) + sum(counts[:feature]) + bucket


def speaker_background_token_id(
    text_vocab: int,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Iterable[int],
    clean: bool,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 0,
) -> int:
    if int(speaker_background_num_buckets) < 2:
        raise ValueError("speaker_background_num_buckets must be at least 2")
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="speaker-background conditioning",
    )
    return base + int(speaking_rate_num_buckets) + sum(counts) + (0 if clean else 1)


def accurate_mode_token_id(
    text_vocab: int,
    speaking_rate_num_buckets: int,
    quality_bucket_counts: Iterable[int],
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 1,
) -> int:
    if int(accurate_mode_num_buckets) <= 0:
        raise ValueError("accurate_mode_num_buckets must be positive")
    if int(speaker_background_num_buckets) < 2:
        raise ValueError("speaker_background_num_buckets must be at least 2")
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="accurate-mode conditioning",
    )
    return (
        base
        + int(speaking_rate_num_buckets)
        + sum(counts)
        + int(speaker_background_num_buckets)
    )


def shear(x: mx.array, pad: int) -> mx.array:
    if x.ndim != 2:
        raise ValueError(f"shear expects a 2-D array, got shape {x.shape}")
    t, c = x.shape
    padded = mx.concatenate(
        [mx.full((c - 1, c), int(pad), dtype=x.dtype), x],
        axis=0,
    )
    rows = (c - 1) + mx.arange(t)[:, None] - mx.arange(c)[None, :]
    cols = mx.arange(c)[None, :]
    return padded[rows, cols]


def shear_up(x: mx.array, pad: int) -> mx.array:
    if x.ndim < 2:
        raise ValueError(f"shear_up expects at least 2 dimensions, got shape {x.shape}")
    h, w = x.shape[-2:]
    rows = mx.arange(h)[:, None] + mx.arange(w)[None, :]
    valid = rows < h
    gather_rows = mx.minimum(rows, h - 1)
    prefix_dims = (1,) * (x.ndim - 2)
    gather_rows = gather_rows.reshape(prefix_dims + (h, w))
    valid = valid.reshape(prefix_dims + (h, w))
    gather_rows = mx.broadcast_to(gather_rows, x.shape)
    valid = mx.broadcast_to(valid, x.shape)
    gathered = mx.take_along_axis(x, gather_rows, axis=-2)
    return mx.where(valid, gathered, mx.array(int(pad), dtype=x.dtype))


def silence_prompt_tokens(config: TTSPromptConfig) -> list[list[int]]:
    sheared = shear(_SILENCE_TOKENS_0_2S[:, : config.n_codebooks], config.audio_pad_id)
    text_col = mx.full((sheared.shape[0], 1), config.text_vocab, dtype=mx.int32)
    return mx.concatenate([sheared, text_col], axis=1).tolist()


def make_speaker_slot(config: TTSPromptConfig) -> list[int]:
    return [config.audio_pad_id] * config.n_codebooks + [config.text_vocab]


def make_marker_slot(config: TTSPromptConfig, text_token: int) -> list[int]:
    return [config.audio_pad_id] * config.n_codebooks + [int(text_token)]


def _text_rows(
    tokens: Sequence[int],
    config: TTSPromptConfig,
    *,
    speaking_rate_bucket: int | None = None,
    quality_buckets: Sequence[int | None] | None = None,
) -> list[list[int]]:
    rows: list[list[int]] = []
    if speaking_rate_bucket is not None:
        rows.append(
            make_marker_slot(
                config,
                speaking_rate_token_id(
                    config.text_vocab,
                    config.speaking_rate_num_buckets,
                    speaking_rate_bucket,
                    config.quality_bucket_counts,
                    config.speaker_background_num_buckets,
                    config.accurate_mode_num_buckets,
                ),
            )
        )
    if quality_buckets is not None:
        for feature_idx, bucket in enumerate(quality_buckets):
            if bucket is None:
                continue
            rows.append(
                make_marker_slot(
                    config,
                    quality_token_id(
                        config.text_vocab,
                        config.speaking_rate_num_buckets,
                        config.quality_bucket_counts,
                        feature_idx,
                        int(bucket),
                        config.speaker_background_num_buckets,
                        config.accurate_mode_num_buckets,
                    ),
                )
            )
    rows.extend(make_marker_slot(config, token) for token in tokens)
    return rows


def tokens_to_prompt_tokens(
    tokens: Sequence[int],
    *,
    n_codebooks: int = 9,
    audio_pad_id: int = 1025,
    text_vocab: int = 519,
    speaking_rate_num_buckets: int = 8,
    speaking_rate_bucket: int | None = None,
    quality_bucket_counts: Iterable[int] = (12, 12, 12, 8, 8, 8),
    quality_buckets: Sequence[int | None] | None = None,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 1,
) -> list[list[int]]:
    config = TTSPromptConfig(
        n_codebooks=n_codebooks,
        audio_pad_id=audio_pad_id,
        text_vocab=text_vocab,
        speaking_rate_num_buckets=speaking_rate_num_buckets,
        quality_bucket_counts=_normalize_quality_bucket_counts(quality_bucket_counts),
        speaker_background_num_buckets=speaker_background_num_buckets,
        accurate_mode_num_buckets=accurate_mode_num_buckets,
        prepend_silence=False,
    )
    return _text_rows(
        list(tokens),
        config,
        speaking_rate_bucket=speaking_rate_bucket,
        quality_buckets=quality_buckets,
    )


def text_to_prompt_tokens(text: str, **kwargs) -> list[list[int]]:
    return tokens_to_prompt_tokens(text_to_byte_ids(text), **kwargs)


class TTSPromptBuilder:
    def __init__(self, config: TTSPromptConfig):
        self.config = config
        self._silence_tokens = (
            silence_prompt_tokens(config) if config.prepend_silence else []
        )

    def build_list(
        self,
        text: str,
        *,
        speaking_rate_bucket: int | None = None,
        quality_buckets: Sequence[int | None] | None = None,
    ) -> list[list[int]]:
        rows = _text_rows(
            text_to_byte_ids(text),
            self.config,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=quality_buckets,
        )
        if self._silence_tokens:
            rows.extend(self._silence_tokens)
        return rows

    def build(self, text: str, **kwargs) -> mx.array:
        return mx.array(self.build_list(text, **kwargs), dtype=mx.int32)

    def speaker_slot(self) -> list[int]:
        return make_speaker_slot(self.config)

    def speaker_marker_prefix(
        self,
        *,
        clean_speaker_background: bool = False,
        accurate_mode: bool = True,
    ) -> list[list[int]]:
        rows = [self.speaker_slot()]
        if self.config.speaker_background_num_buckets > 0:
            rows.append(
                make_marker_slot(
                    self.config,
                    speaker_background_token_id(
                        self.config.text_vocab,
                        self.config.speaking_rate_num_buckets,
                        self.config.quality_bucket_counts,
                        clean_speaker_background,
                        self.config.speaker_background_num_buckets,
                        self.config.accurate_mode_num_buckets,
                    ),
                )
            )
            if accurate_mode and self.config.accurate_mode_num_buckets > 0:
                rows.append(
                    make_marker_slot(
                        self.config,
                        accurate_mode_token_id(
                            self.config.text_vocab,
                            self.config.speaking_rate_num_buckets,
                            self.config.quality_bucket_counts,
                            self.config.speaker_background_num_buckets,
                            self.config.accurate_mode_num_buckets,
                        ),
                    )
                )
        return rows

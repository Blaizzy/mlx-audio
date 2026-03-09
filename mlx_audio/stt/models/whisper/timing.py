# Copyright © 2023 Apple Inc.
# -*- coding: utf-8 -*-

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import mlx.core as mx

from .audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND

if TYPE_CHECKING:
    from .whisper import Model


def median_filter(x: mx.array, filter_width: int) -> mx.array:
    """Apply a median filter of width `filter_width` along the last dimension of `x`"""
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        return x

    if (ndim := x.ndim) <= 2:
        x = x[None, None, :]

    assert (
        filter_width > 0 and filter_width % 2 == 1
    ), "`filter_width` should be an odd number"

    # Reflect padding along last dimension
    left_pad = x[..., 1 : pad_width + 1][..., ::-1]
    right_pad = x[..., -(pad_width + 1) : -1][..., ::-1]
    x = mx.concatenate([left_pad, x, right_pad], axis=-1)

    # Sliding window median filter
    x = x.astype(mx.float32)
    windows = mx.stack(
        [x[..., i : x.shape[-1] - filter_width + 1 + i] for i in range(filter_width)],
        axis=-1,
    )
    result = mx.median(windows, axis=-1)

    if ndim <= 2:
        result = result[0, 0]

    return result


def backtrace(trace: mx.array):
    trace_list = trace.tolist()
    rows = len(trace_list)
    cols = len(trace_list[0])
    i = rows - 1
    j = cols - 1

    # Set boundary conditions
    for c in range(cols):
        trace_list[0][c] = 2
    for r in range(rows):
        trace_list[r][0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        t = trace_list[i][j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        elif t == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result.reverse()
    # Return as two lists (text_indices, time_indices)
    text_indices = [r[0] for r in result]
    time_indices = [r[1] for r in result]
    return mx.array(text_indices), mx.array(time_indices)


def dtw(x: mx.array) -> tuple:
    """Dynamic time warping on an mx.array cost matrix."""
    x_list = x.tolist()
    N, M = x.shape
    N, M = int(N), int(M)

    cost = [[float("inf")] * (M + 1) for _ in range(N + 1)]
    trace = [[-1.0] * (M + 1) for _ in range(N + 1)]

    cost[0][0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1][j - 1]
            c1 = cost[i - 1][j]
            c2 = cost[i][j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i][j] = x_list[i - 1][j - 1] + c
            trace[i][j] = t

    return backtrace(mx.array(trace))


@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


def find_alignment(
    model: "Model",
    tokenizer,
    text_tokens: List[int],
    mel: mx.array,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> List[WordTiming]:
    if len(text_tokens) == 0:
        return []

    tokens = mx.array(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    )

    logits, cross_qk = model.forward_with_cross_qk(mel[None, :], tokens[None, :])
    # consider only the logits associated with predicting text
    sampled_logits = logits[0][len(tokenizer.sot_sequence) : -2, : tokenizer.eot]
    token_probs = mx.softmax(sampled_logits, precise=True, axis=-1)
    text_token_probs = mx.take_along_axis(
        token_probs, mx.array(text_tokens)[:, None], axis=1
    ).squeeze(1)

    # heads * tokens * frames
    weights = mx.stack(
        [cross_qk[_l][0, _h] for _l, _h in model.alignment_heads.tolist()]
    )
    weights = weights[:, :, : num_frames // 2]
    weights = mx.softmax(weights * qk_scale, axis=-1, precise=True)
    weights = weights.astype(mx.float32)
    mean = mx.mean(weights, axis=-2, keepdims=True)
    std = mx.var(weights, axis=-2, keepdims=True, ddof=0).sqrt()
    weights = (weights - mean) / std
    weights = median_filter(weights, medfilt_width)

    matrix = mx.mean(weights, axis=0)
    matrix = matrix[len(tokenizer.sot_sequence) : -1]
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        return []

    # Build word boundaries
    cumsum = [0]
    for t in word_tokens[:-1]:
        cumsum.append(cumsum[-1] + len(t))
    word_boundaries = mx.array(cumsum)

    # Compute jumps and jump times
    text_indices_list = text_indices.tolist()
    diffs = [1] + [
        text_indices_list[i] - text_indices_list[i - 1]
        for i in range(1, len(text_indices_list))
    ]
    jumps = [d != 0 for d in diffs]

    time_indices_list = time_indices.tolist()
    jump_times = (
        mx.array([t for t, j in zip(time_indices_list, jumps) if j]) / TOKENS_PER_SECOND
    )

    wb = word_boundaries.tolist()
    jt = jump_times.tolist()
    start_times = [jt[int(wb[i])] for i in range(len(wb) - 1)]
    end_times = [jt[int(wb[i])] for i in range(1, len(wb))]

    ttp = text_token_probs.tolist()
    word_probabilities = [
        sum(ttp[int(wb[i]) : int(wb[i + 1])]) / max(1, int(wb[i + 1]) - int(wb[i]))
        for i in range(len(wb) - 1)
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]


def merge_punctuations(alignment: List[WordTiming], prepended: str, appended: str):
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            # prepend it to the following word
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in appended:
            # append it to the previous word
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def add_word_timestamps(
    *,
    segments: List[dict],
    model: "Model",
    tokenizer,
    mel: mx.array,
    num_frames: int,
    prepend_punctuations: str = "\"'\u201c\u00bf([{-",
    append_punctuations: str = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]\u007d\u3001",
    last_speech_timestamp: float,
    **kwargs,
):
    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)
    durations = [t.end - t.start for t in alignment]
    nonzero_durations = [d for d in durations if d != 0]
    if len(nonzero_durations) > 0:
        sorted_d = sorted(nonzero_durations)
        mid = len(sorted_d) // 2
        median_duration = (
            sorted_d[mid]
            if len(sorted_d) % 2 == 1
            else (sorted_d[mid - 1] + sorted_d[mid]) / 2
        )
    else:
        median_duration = 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    if len(nonzero_durations) > 0:
        sentence_end_marks = ".。!！?？"
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=float(timing.probability),
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        if len(words) > 0:
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (
                    len(words) > 1
                    and words[1]["end"] - words[0]["start"] > max_duration * 2
                )
            ):
                if (
                    len(words) > 1
                    and words[1]["end"] - words[1]["start"] > max_duration
                ):
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            if (
                segment["start"] < words[0]["end"]
                and segment["start"] - 0.5 > words[0]["start"]
            ):
                words[0]["start"] = max(
                    0, min(words[0]["end"] - median_duration, segment["start"])
                )
            else:
                segment["start"] = words[0]["start"]

            if (
                segment["end"] > words[-1]["start"]
                and segment["end"] + 0.5 < words[-1]["end"]
            ):
                words[-1]["end"] = max(
                    words[-1]["start"] + median_duration, segment["end"]
                )
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES and HuggingFace Inc.
# Adapted from NeMo Speech ctc_timestamp_utils.py and Transformers
# processing_nemotron3_5_transcribe.py (Apache-2.0).
"""Native CTC head and speaker-aware word alignment, without PyTorch."""

import re
import unicodedata
from bisect import bisect_left, bisect_right
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_audio.vad.models.nemotron_diarization.config import EncoderConfig
from mlx_audio.vad.models.nemotron_diarization.nemotron_diarization import Encoder


class TimestampDecoder(nn.Module):
    def __init__(self, source):
        super().__init__()
        factor = source.get("upsampling_factor", 1)
        if type(factor) is not int or factor < 1:
            raise ValueError("CTC upsampling_factor must be a positive integer")
        if source.get("upsampling_mode", "conv_transpose") != "conv_transpose":
            raise ValueError("Expected transposed-convolution CTC upsampling")
        if not source.get("use_transformer", True) or not source.get("residual", True):
            raise ValueError("Expected a residual Transformer CTC head")
        width = source["feat_in"]
        self._factor = factor
        if factor > 1:
            self.upsample = nn.ConvTranspose1d(width, width, factor, stride=factor)
        self.residual_scale = mx.array(source.get("residual_scale", 0.1))
        self.transformer = Encoder(
            EncoderConfig.from_dict(
                {
                    **source,
                    "d_model": width,
                    "feat_in": width,
                    "subsampling_factor": 1,
                }
            )
        )
        del self.transformer.pre_encode
        self.decoder_layers = [nn.Linear(width, source["num_classes"] + 1)]

    def __call__(self, states):
        states = states.astype(self.transformer.final_norm.weight.dtype)
        if self._factor > 1:
            states = self.upsample(states)
        encoded = self.transformer(states, mx.array([states.shape[1]]))
        hidden = states + self.residual_scale.astype(states.dtype) * encoded
        return self.decoder_layers[0](hidden)


class CTCTimestamp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = TimestampDecoder(config["decoder_config"])

    def logits(self, states, encoder_config):
        """Match the ASR encoder's bounded windows, then trim context frames."""
        size = encoder_config.get("online_inference_length", 500)
        left = max(0, encoder_config.get("chunk_left_context", 50))
        right = max(0, encoder_config.get("chunk_right_context", 50))
        factor = self.decoder._factor
        for start in range(0, states.shape[1], size):
            end = min(start + size, states.shape[1])
            first, last = max(0, start - left), min(states.shape[1], end + right)
            logits = self.decoder(states[:, first:last])
            yield logits[0, (start - first) * factor : (end - first) * factor]


def load_alignment_tokenizer(model_path):
    """Prefer the released tokenizer; accept embedded NeMo SentencePiece too."""
    from transformers import PreTrainedTokenizerFast

    path = Path(model_path) / "alignment_tokenizer"
    if (path / "tokenizer.json").exists():
        return PreTrainedTokenizerFast.from_pretrained(path)
    proto = (
        (path / "tokenizer.model").read_bytes()
        if (path / "tokenizer.model").exists()
        else embedded_tokenizer(model_path)
    )
    if proto is None:
        return None
    import sentencepiece as spm

    class SentencePieceTokenizer:
        def __init__(self):
            self.processor = spm.SentencePieceProcessor(model_proto=proto)
            self.unk_token_id = self.processor.unk_id()

        def encode(self, text, add_special_tokens=False):
            return self.processor.encode(text)

        def convert_tokens_to_ids(self, token):
            return self.processor.piece_to_id(token)

    return SentencePieceTokenizer()


def embedded_tokenizer(model_path):
    from safetensors import safe_open

    for path in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(path, framework="numpy") as weights:
            key = "ctc_timestamp.tokenizer.model_proto"
            if key in weights.keys():  # noqa: SIM118 - safe_open is not a mapping
                return weights.get_tensor(key).tobytes()
    return None


SPEAKER_TAG = re.compile(r"<spk:(\d+)>", re.IGNORECASE)
KEPT_TAG = re.compile(r"(?i:<spk:\d+>)|<(?:ev|sty):[^<>]*>")
SCRIPT_LANGUAGES = [
    ("ko", "HANGUL"),
    ("ja", "HIRAGANA"),
    ("ja", "KATAKANA"),
    ("zh", "CJK UNIFIED"),
    ("zh", "CJK COMPATIBILITY"),
    ("th", "THAI"),
]


def tokenize_transcript(tokenizer, text, blank_id, language=None):
    parts = SPEAKER_TAG.split(text.replace("—", " "))
    tags = [0 if len(parts) == 1 else None] + [int(tag) for tag in parts[1::2]]
    words = [
        (word, tag) for tag, part in zip(tags, parts[::2]) for word in part.split()
    ]
    if language is None:
        names = [
            unicodedata.name(c, "")
            for word, _ in words
            if not KEPT_TAG.fullmatch(word)
            for c in word
        ]
        language = next(
            (
                lang
                for lang, prefix in SCRIPT_LANGUAGES
                if any(n.startswith(prefix) for n in names)
            ),
            None,
        )
    language = {"chinese": "zh", "japanese": "ja", "thai": "th"}.get(
        str(language).lower(), language
    )
    if language == "th":
        raise ValueError(
            "Thai word timestamps require a word segmenter and are not supported"
        )
    streams = {}
    for word, tag in words:
        if language in ("zh", "ja") and not KEPT_TAG.fullmatch(word):
            units = []
            for char in unicodedata.normalize("NFC", word):
                if unicodedata.category(char)[0] not in "LMN":
                    continue
                token = tokenizer.convert_tokens_to_ids(char)
                ids = [token]
                if token == tokenizer.unk_token_id:
                    boundary = tokenizer.convert_tokens_to_ids("▁")
                    ids = [
                        i
                        for i in tokenizer.encode(" " + char, add_special_tokens=False)
                        if i != boundary
                    ]
                if tokenizer.unk_token_id in ids:
                    raise ValueError(f"Unrepresentable alignment character: {char!r}")
                units.append((char, ids))
        else:
            units = [(word, tokenizer.encode(" " + word, add_special_tokens=False))]
        if any(not ids or min(ids) < 0 or max(ids) >= blank_id for _, ids in units):
            raise ValueError(
                "Alignment tokenizer produced IDs outside the CTC vocabulary"
            )
        streams.setdefault(tag, []).extend(units)
    return {tag: words for tag, words in streams.items() if words}


def _band_starts(greedy, target, states, width):
    """NeMo's monotonic unique-trigram anchors bound long-recording search."""

    def trigrams(tokens):
        result = {}
        for i in range(len(tokens) - 2):
            result.setdefault(tuple(tokens[i : i + 3]), []).append(i)
        return result

    frames = len(greedy)
    keep = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    times = np.flatnonzero(keep)
    left, right = trigrams(greedy[keep]), trigrams(target)
    pairs = sorted(
        (int(times[v[0] + 1]), 2 * (right[k][0] + 1) + 1)
        for k, v in left.items()
        if len(v) == 1 and len(right.get(k, [])) == 1
    )
    tails, indices, parents = [], [], []
    for i, (_, state) in enumerate(pairs):
        j = bisect_left(tails, state)
        parents.append(indices[j - 1] if j else -1)
        if j == len(tails):
            tails.append(state)
            indices.append(i)
        else:
            tails[j], indices[j] = state, i
    chosen = []
    i = indices[-1] if indices else -1
    while i >= 0:
        chosen.append(pairs[i])
        i = parents[i]
    points = (
        [(0, 0)]
        + [(t, s) for t, s in reversed(chosen) if 0 < t < frames - 1]
        + [(frames - 1, states - 1)]
    )
    center = np.interp(
        np.arange(frames), [p[0] for p in points], [p[1] for p in points]
    )
    starts = np.clip(np.rint(center).astype(np.int64) - width // 2, 0, states - width)
    starts = np.minimum.accumulate(starts - 2 * np.arange(frames)) + 2 * np.arange(
        frames
    )
    return np.maximum(
        starts, np.maximum(0, states - width - 2 * (frames - 1 - np.arange(frames)))
    )


def ctc_path(log_probs, target, probabilities):
    """CTC Viterbi path in compact columns (blank=0), with byte backpointers.

    Short inputs use exact search. Long inputs use NeMo's anchor-guided band;
    storage scales with frames times band width, rather than vocabulary size.
    """
    frames = np.flatnonzero(probabilities >= 0.01)
    target = np.asarray(target, dtype=np.int64)
    if len(frames) < len(target) + np.count_nonzero(target[1:] == target[:-1]):
        raise ValueError(
            "Transcript cannot be CTC-aligned to the available speech frames"
        )
    labels = np.zeros(2 * len(target) + 1, dtype=np.int64)
    labels[1::2] = target
    states = len(labels)
    width = (
        states
        if len(frames) < 1000
        else min(states, 2 * int(np.ceil(0.1 * states + 100)) + 1)
    )
    starts = _band_starts(log_probs[frames].argmax(-1), target, states, width)
    prior = np.log(np.maximum(probabilities, 1e-6))
    previous = np.full(width, -np.inf, np.float32)
    previous[:2] = log_probs[frames[0], labels[:2]] + np.array([0, prior[frames[0]]])
    pointers = np.zeros((len(frames), width), dtype=np.uint8)
    padded = np.full(width + 4, -np.inf, np.float32)
    can_skip = (labels != 0) & (labels != np.r_[labels[:2], labels[:-2]])
    for t in range(1, len(frames)):
        start = int(starts[t])
        delta = start - int(starts[t - 1])
        padded[2 : width + 2] = previous
        best = padded[2 + delta : 2 + delta + width].copy()
        one, two = padded[1 + delta : 1 + delta + width], padded[delta : delta + width]
        take = one > best
        np.copyto(best, one, where=take)
        pointers[t, take] = 1
        take = (two > best) & can_skip[start : start + width]
        np.copyto(best, two, where=take)
        pointers[t, take] = 2
        columns = labels[start : start + width]
        previous = (
            best
            + log_probs[frames[t], columns]
            + np.where(columns != 0, prior[frames[t]], 0)
        )
    ends = np.array([states - 1, states - 2])
    state = int(ends[previous[ends - starts[-1]].argmax()])
    if not np.isfinite(previous[state - starts[-1]]):
        raise ValueError("No finite CTC alignment path for the transcript")
    path = np.empty(len(frames), dtype=np.int64)
    for t in range(len(frames) - 1, -1, -1):
        path[t] = state
        if t:
            state -= int(pointers[t, state - starts[t]])
    return frames, path


def _speaker_mapping(log_probs, streams, probabilities):
    from scipy.optimize import linear_sum_assignment

    classes = log_probs.shape[1]
    greedy = log_probs.argmax(-1)
    onsets = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    counts = np.array(
        [
            np.bincount([i for _, ids in words for i in ids], minlength=classes)
            for words in streams.values()
        ],
        dtype=np.float64,
    )
    spoken = np.array(
        [
            np.bincount(greedy[onsets & (p >= 0.5)], minlength=classes)
            for p in probabilities.T
        ],
        dtype=np.float64,
    )
    idf = np.log1p(len(spoken) / (1 + (spoken > 0).sum(0)))
    a, b = np.sqrt(counts * idf), np.sqrt(spoken * idf)
    similarity = (
        a
        @ b.T
        / np.maximum(
            np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None], 1e-8
        )
    )
    similarity -= 0.1 * np.abs(
        np.log((counts.sum(1)[:, None] + 1) / (spoken.sum(1)[None] + 1))
    )
    mapping = dict.fromkeys(streams)
    for row, column in zip(*linear_sum_assignment(-similarity)):
        mapping[list(streams)[row]] = int(column)
    return mapping


def align_words(
    text,
    tokenizer,
    head,
    states,
    speaker_probabilities,
    duration,
    encoder_config,
    *,
    language=None,
    special_tokens=(),
    speaker_frame_seconds=None,
):
    """Align generated words, preserving transcript speaker IDs in the result."""
    for token in special_tokens:
        if not KEPT_TAG.fullmatch(token):
            text = text.replace(token, " ")
    blank = head.decoder.decoder_layers[0].weight.shape[0] - 1
    streams = tokenize_transcript(tokenizer, text, blank, language)
    if not streams:
        return []
    # Keep only the transcript's columns on the host. Project the head one
    # window at a time so hour-long recordings never retain T x vocabulary.
    columns = [blank] + sorted(
        {i for words in streams.values() for _, ids in words for i in ids}
    )
    remap = {token: i for i, token in enumerate(columns)}
    streams = {
        tag: [(word, [remap[i] for i in ids]) for word, ids in words]
        for tag, words in streams.items()
    }
    chunks = []
    for logits in head.logits(states, encoder_config):
        compact = logits[:, mx.array(columns)].astype(mx.float32)
        compact -= mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
        chunks.append(np.array(compact))
    log_probs = np.concatenate(chunks)
    step = duration / len(log_probs)
    factor = encoder_config.get("asr_encoder_cfg", {}).get("subsampling_factor", 1)
    pooled = speaker_probabilities.reshape(
        -1, factor, speaker_probabilities.shape[-1]
    ).mean(axis=1)
    positions = np.linspace(0, len(pooled) - 1, len(log_probs))
    probabilities = np.stack(
        [np.interp(positions, np.arange(len(pooled)), p) for p in pooled.T],
        axis=-1,
    )
    mapping = _speaker_mapping(log_probs, streams, probabilities)
    result = []
    for tag, words in streams.items():
        speaker = mapping[tag]
        if speaker is None:
            raise ValueError(f"No diarization speaker matched transcript speaker {tag}")
        activity = probabilities[:, speaker]
        frames, path = ctc_path(
            log_probs, [i for _, ids in words for i in ids], activity
        )
        timestamps = []
        index = 0
        for word, ids in words:
            first = np.searchsorted(path, 2 * index + 1)
            index += len(ids)
            last = np.searchsorted(path, 2 * index - 1, side="right") - 1
            start = float(frames[first] * step)
            end = min(float((frames[last] + 1) * step), start + 1.0, duration)
            timestamps.append(
                {"text": word, "start": start, "end": end, "speaker": tag}
            )
        native_step = speaker_frame_seconds or duration / len(speaker_probabilities)
        valid_frames = min(
            len(speaker_probabilities), max(1, int(duration / native_step))
        )
        active = speaker_probabilities[:valid_frames, speaker] > 0.5
        transitions = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        segments = list(
            zip(
                np.flatnonzero(transitions == 1) * native_step,
                np.minimum(np.flatnonzero(transitions == -1) * native_step, duration),
            )
        )
        # Match the reference's bounded snapping to diarization activity.
        segment_ids = []
        for word in timestamps:
            i = bisect_right([s[0] for s in segments], word["start"]) - 1
            segment_ids.append(i if i >= 0 and word["start"] < segments[i][1] else None)
        limit = 12 * head.decoder._factor * min(step, 0.08)
        for i, (word, segment) in enumerate(zip(timestamps, segment_ids)):
            if (
                segment is not None
                and segment_ids[i - 1 : i] != [segment]
                and 0 < word["start"] - segments[segment][0] <= limit
            ):
                word["start"] = float(segments[segment][0])
        for i, (word, segment) in enumerate(zip(timestamps, segment_ids)):
            if segment is not None and segment_ids[i + 1 : i + 2] != [segment]:
                end = segments[segment][1]
            else:
                end = (
                    timestamps[i + 1]["start"]
                    if i + 1 < len(timestamps)
                    else word["end"]
                )
            word["end"] = float(
                max(word["start"], min(end, word["start"] + limit, duration))
            )
        result.extend(timestamps)
    return sorted(result, key=lambda word: (word["start"], word["end"]))

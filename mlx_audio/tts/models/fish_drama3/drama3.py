"""Fish Audio Drama 3: a Qwen3.5-MoE backbone, a temporal codebook tower, and the
Fish S1 DAC codec.

Each decode step samples the frame's main token (EOS or a semantic token) from
the backbone, lets the tower sample codebooks 1..9, and feeds the frame back into
the backbone. Frames stream through the causal codec as they are produced.
"""

import copy
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.models.base import GenerationResult

from .backbone import build_backbone
from .chunking import reference_transcript, split_text, validate_text, with_speaker_tag
from .config import ModelConfig
from .tower import Tower

# Repetition-aware sampling from the training recipe: a semantic main token that
# repeats one of the last RAS_WINDOW main tokens is replaced by a second draw.
RAS_WINDOW = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5
DEFAULT_MAX_FRAMES = 1024
# SGLang-Omni's sampling defaults (sgl-project/sglang-omni#2890 and #2810). A
# top-k cap on the main token lets a take without a reference fall silent after
# a phrase and stay silent until the frame cap, most often at fish-speech's
# 0.7 / 0.7 / 30. The codebook heads keep a cap of 30, which holds the voice.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = None
DEFAULT_TOP_P = 0.9
CODEBOOK_TOP_K = 30
PREFILL_CHUNK = 256
# Frames per codec call when a chunk is decoded at once. The codec's working set
# grows with the frames of one call: a 1024-frame chunk needs about 20 GB on top
# of the weights in one call and about 4 GB in 64-frame pieces, no slower.
DECODE_FRAMES = 64
VOICE_CACHE_SIZE = 4
KEPT_PREFIXES = ("backbone.", "semantic_head.", "tower.")

AudioInput = Union[str, Path, mx.array, np.ndarray]


def sample_top_k_top_p(
    logits: mx.array,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    keep_crossing: bool = False,
) -> mx.array:
    """One id per row of logits [B, V], as the reference samples.

    Temperature applies before top-p; the cumulative probability runs over the
    full vocabulary; top_k None keeps every candidate. The token that crosses
    top_p is dropped unless it ranks first; keep_crossing keeps it, as the main
    token's draws do so that EOS stays reachable. Only the top_k candidates are
    sorted, since all others are dropped.
    """
    x = logits.astype(mx.float32) / max(temperature, MIN_TEMPERATURE)
    # At low temperature the logits share a huge offset that would swamp the
    # log-sum-exp in float32 and push every probability to 1.
    x = x - x.max(axis=-1, keepdims=True)
    k = x.shape[-1] if top_k is None else min(top_k, x.shape[-1])
    ids = mx.argpartition(-x, kth=k - 1, axis=-1)[..., :k]
    values = mx.take_along_axis(x, ids, axis=-1)
    order = mx.argsort(-values, axis=-1)
    values = mx.take_along_axis(values, order, axis=-1)
    ids = mx.take_along_axis(ids, order, axis=-1)
    probs = mx.exp(values - mx.logsumexp(x, axis=-1, keepdims=True))
    passed = mx.cumsum(probs, axis=-1, inclusive=not keep_crossing)
    dropped = (passed > top_p) & (mx.arange(k) > 0)
    choice = mx.random.categorical(mx.where(dropped, -mx.inf, values))
    return mx.take_along_axis(ids, choice[..., None], axis=-1)[..., 0]


def _copy_cache(cache):
    """An independent copy of a KVCache or ArraysCache. KVCache writes into its
    arrays in place, so a snapshot must not share them with a live cache."""
    clone = copy.copy(cache)
    if isinstance(cache, KVCache):
        if cache.keys is not None:
            clone.keys = cache.keys[..., : cache.offset, :]
            clone.values = cache.values[..., : cache.offset, :]
    else:
        clone.cache = list(cache.cache)
    return clone


@dataclass
class _Voice:
    """Encoded references and, once built, the caches of the system block."""

    references: list[tuple[str, mx.array]]
    prefix: Optional[tuple] = field(default=None, repr=False)


class _VoiceCache:
    """The few most recently used voices, keyed by reference content."""

    def __init__(self, size: int):
        self._voices: OrderedDict[str, _Voice] = OrderedDict()
        self._size = size

    def get(self, key: str, build: Callable[[], _Voice]) -> _Voice:
        voice = self._voices.pop(key, None) or build()
        self._voices[key] = voice
        while len(self._voices) > self._size:
            self._voices.popitem(last=False)
        return voice


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id
        tower_cfg = config.audio_decoder_config
        self.num_codebooks = tower_cfg.num_codebooks
        self.backbone = build_backbone(config.text_config)
        # Rows [semantic band..., EOS]: the only tokens a step may emit.
        band = self.semantic_end - self.semantic_begin + 1
        self.semantic_head = nn.Linear(
            config.text_config["hidden_size"], band + 1, bias=False
        )
        self.tower = Tower(tower_cfg, self.semantic_begin, self.semantic_end)
        self._candidate_ids = mx.concatenate(
            [
                mx.arange(self.semantic_begin, self.semantic_end + 1),
                mx.array([self.eos_id]),
            ]
        )
        self.tokenizer = None
        self.codec = None
        # Not an array, dict, or module, so it stays out of the module tree.
        self._voices = _VoiceCache(VOICE_CACHE_SIZE)

    @property
    def sample_rate(self) -> int:
        return self.codec.sample_rate

    @property
    def generate_config(self) -> dict:
        """The defaults the server uses for sampling fields a request leaves out.
        A top_k of every candidate leaves the main token uncapped, as None does."""
        return {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "top_k": int(self._candidate_ids.size),
        }

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        # The loader also reads codec.safetensors; the codec loads separately.
        return {k: v for k, v in weights.items() if k.startswith(KEPT_PREFIXES)}

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from tokenizers import Tokenizer

        from mlx_audio.codec.models.fish_s1_dac import DAC as FishS1DAC

        if model.config.speech_template_version != "reference-bracket-v1":
            raise ValueError(
                f"unsupported speech_template_version "
                f"{model.config.speech_template_version!r}"
            )
        tokenizer = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
        last_code = model.tower.vocab_size - 1
        for token, expected in (
            ("<|semantic:0|>", model.semantic_begin),
            (f"<|semantic:{last_code}|>", model.semantic_end),
            ("<|im_end|>", model.eos_id),
        ):
            if tokenizer.token_to_id(token) != expected:
                raise ValueError(f"tokenizer and config disagree on {token}")
        model.tokenizer = tokenizer
        model.codec = FishS1DAC.from_pretrained(str(model_path))
        # Evaluated here, before another stream or thread can read it.
        mx.eval(model._candidate_ids)
        return model

    # ---- prompt ------------------------------------------------------------

    def _voice(
        self,
        ref_audio: Union[AudioInput, list[AudioInput], None],
        ref_text: Union[str, list[str], None],
    ) -> _Voice:
        """Encoded references for a voice. The few most recent voices are kept,
        keyed by audio content, so repeated requests skip encoding and prefill."""
        audios = [] if ref_audio is None else ref_audio
        texts = [] if ref_text is None else ref_text
        if not isinstance(audios, (list, tuple)):
            audios = [audios]
        if isinstance(texts, str):
            texts = [texts]
        if len(audios) != len(texts):
            raise ValueError("every reference audio needs its transcript (ref_text)")

        samples = [self._samples(audio) for audio in audios]
        digest = hashlib.sha1()
        for audio, text in zip(samples, texts):
            for part in (audio.tobytes(), text.encode()):
                # The length prefix keeps bytes from moving between parts.
                digest.update(len(part).to_bytes(8, "little") + part)

        return self._voices.get(
            digest.hexdigest(),
            lambda: _Voice(
                [(t, self._encode_audio(a)) for a, t in zip(samples, texts)]
            ),
        )

    def _samples(self, audio: AudioInput) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            from mlx_audio.utils import load_audio

            audio = load_audio(str(audio), sample_rate=self.sample_rate)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _encode_audio(self, samples: np.ndarray) -> mx.array:
        """Mono samples at the codec rate -> codes [num_codebooks, frames]."""
        indices, lengths = self.codec.encode(mx.array(samples)[None, None, :])
        codes = indices[0, :, : int(lengths[0].item())].astype(mx.int32)
        mx.eval(codes)
        # Temporaries sized by this reference rarely fit a later allocation;
        # cached, those of every reference of a voice would add up.
        mx.clear_cache()
        return codes

    def _check_codes(self, codes: mx.array, location: str) -> None:
        size = self.tower.vocab_size
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks:
            raise ValueError(
                f"{location}: codes must be [{self.num_codebooks}, frames]; "
                f"got {codes.shape}"
            )
        if codes.shape[1] == 0:
            raise ValueError(f"{location}: codes are empty")
        if not mx.issubdtype(codes.dtype, mx.integer):
            raise ValueError(f"{location}: codes must be integers; got {codes.dtype}")
        if codes.min().item() < 0 or codes.max().item() >= size:
            raise ValueError(f"{location}: codes must be in [0, {size})")

    def build_frames(
        self,
        references: list[tuple[str, mx.array]],
        history: list[tuple[str, mx.array]],
        target: str,
    ) -> tuple[mx.array, int]:
        """The reference-bracket-v1 prompt as frame rows [T, 1 + num_codebooks],
        and how many leading rows form the system block.

        Text rows are [token, 0, ...]; audio rows are [c0 + semantic_begin, c0..c9].
        Every string is tokenized on its own: tokenizing the joined prompt changes
        the ids at the seams. The system block only depends on the references, so
        its cache state is reused across chunks and calls.
        """
        validate_text(target, "text")
        segments: list[Union[str, mx.array]] = ["<|im_start|>system\n"]
        if references:
            segments.append(
                "convert the provided text to speech reference to the following:"
            )
            for index, (transcript, codes) in enumerate(references, 1):
                location = f"reference {index}"
                validate_text(transcript, location)
                self._check_codes(codes, location)
                segments += [
                    f"\n\n[Reference {index}]\n"
                    f"Text: {reference_transcript(transcript)}\nSpeech: ",
                    codes,
                ]
        else:
            segments.append("convert the provided text to speech")
        segments.append("<|im_end|>\n")
        system_segments = len(segments)
        for index, (turn_text, codes) in enumerate(history, 1):
            validate_text(turn_text, f"history {index}")
            self._check_codes(codes, f"history {index}")
            segments += [
                "<|im_start|>user\n",
                with_speaker_tag(turn_text),
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
                codes,
                "<|im_end|>\n",
            ]
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(target),
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ]

        width = 1 + self.num_codebooks
        rows, system_rows = [], 0
        for index, segment in enumerate(segments):
            if isinstance(segment, str):
                ids = self.tokenizer.encode(segment).ids
                block = np.zeros((len(ids), width), dtype=np.int32)
                block[:, 0] = ids
            else:
                codes = np.array(segment, dtype=np.int32)
                block = np.concatenate([codes[:1] + self.semantic_begin, codes]).T
            rows.append(block)
            if index < system_segments:
                system_rows += len(block)
        return mx.array(np.concatenate(rows)), system_rows

    # ---- decoding ----------------------------------------------------------

    def _embed(self, frames: mx.array) -> mx.array:
        text = self.backbone.model.embed_tokens(frames[..., 0])
        return self.tower.embed_backbone(frames, text)

    def _prefill(self, frames: mx.array, caches, start: int, stop: int) -> mx.array:
        """Feed rows [start, stop) into both caches; returns row stop-1's backbone
        hidden state [1, 1, D].

        The tower is teacher-forced with the following row, so it covers
        [start, min(stop, T - 1)).
        """
        backbone_cache, tower_cache = caches
        frames = frames[None]
        last = frames.shape[1] - 1
        hidden = None
        for begin in range(start, stop, PREFILL_CHUNK):
            end = min(begin + PREFILL_CHUNK, stop)
            hidden = self.backbone.model(
                None,
                cache=backbone_cache,
                input_embeddings=self._embed(frames[:, begin:end]),
            )
            labelled = min(end, last)
            if labelled > begin:
                self.tower.prefill(
                    hidden[:, : labelled - begin],
                    frames[:, begin:labelled],
                    frames[:, begin + 1 : labelled + 1],
                    tower_cache,
                )
            mx.eval(
                hidden,
                [c.state for c in backbone_cache],
                [c.state for c in tower_cache],
            )
        return hidden[:, -1:]

    def _start_caches(self, frames: mx.array, system_rows: int, voice: _Voice):
        """Fresh caches holding the system block, copied from the voice's snapshot."""
        if voice.prefix is None:
            caches = (self.backbone.make_cache(), self.tower.make_cache())
            self._prefill(frames, caches, 0, system_rows)
            voice.prefix = tuple([_copy_cache(c) for c in group] for group in caches)
        return tuple([_copy_cache(c) for c in group] for group in voice.prefix)

    def _step(self, hidden, frame, history, caches, temperature, top_k, top_p):
        """One frame: (frame [1, 1, 11], next hidden, history, main token [1]),
        all lazy."""
        backbone_cache, tower_cache = caches
        logits = self.semantic_head(hidden[:, 0]).astype(mx.float32)
        first = self._candidate_ids[
            sample_top_k_top_p(logits, temperature, top_k, top_p, keep_crossing=True)
        ]
        retry = self._candidate_ids[
            sample_top_k_top_p(
                logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, keep_crossing=True
            )
        ]
        repeated = (history[None, :] == first[:, None]).any(axis=-1)
        main = mx.where(repeated & self.tower.is_semantic(first), retry, first)
        c0 = mx.clip(main - self.semantic_begin, 0, self.tower.vocab_size - 1)
        semantic = self.tower.is_semantic(main)[:, None]
        head_top_k = CODEBOOK_TOP_K if top_k is None else min(top_k, CODEBOOK_TOP_K)

        def sample(head_logits: mx.array) -> mx.array:
            ids = sample_top_k_top_p(head_logits[:, 0], temperature, head_top_k, top_p)
            return ids[:, None]

        codes = self.tower.residual_codes(
            hidden, c0[:, None], semantic, frame, tower_cache, sample
        )
        new_frame = mx.concatenate([main[:, None], *codes], axis=-1)
        new_frame = new_frame[:, None, :].astype(mx.int32)
        next_hidden = self.backbone.model(
            None, cache=backbone_cache, input_embeddings=self._embed(new_frame)
        )
        history = mx.concatenate([history[1:], main])
        return new_frame, next_hidden, history, main

    def _frames(
        self,
        frames: mx.array,
        system_rows: int,
        voice: _Voice,
        max_frames: int,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
    ) -> Iterator[mx.array]:
        """Codes [num_codebooks] of each generated frame until EOS.

        Step n + 1 is queued before step n's EOS is read, so the GPU never waits
        for the host; the one wasted step after EOS only touches this call's
        caches. The EOS check reads step n's own `main` array: slicing the frame
        instead would build a new op that waits behind the queued step.

        The model often ends a turn with <|semantic:0|> right before EOS; that
        frame decodes to a click after the trailing silence, so it is held back
        one step and dropped when EOS follows.
        """
        caches = self._start_caches(frames, system_rows, voice)
        hidden = self._prefill(frames, caches, system_rows, frames.shape[0])
        args = (caches, temperature, top_k, top_p)
        pending = self._step(hidden, frames[None, -1:], frames[-RAS_WINDOW:, 0], *args)
        mx.async_eval(pending)
        held = None
        for n in range(max_frames):
            frame, hidden, history, main = pending
            if n + 1 < max_frames:
                pending = self._step(hidden, frame, history, *args)
                mx.async_eval(pending)
            token = main.item()
            if token == self.eos_id:
                return
            if held is not None:
                yield held
            held = frame[0, 0, 1:]
            if token != self.semantic_begin:
                yield held
                held = None
        if held is not None:
            yield held

    def generate(
        self,
        text: str,
        ref_audio: Union[AudioInput, list[AudioInput], None] = None,
        ref_text: Union[str, list[str], None] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        streaming_interval: float = 0.32,
        first_chunk_frames: Optional[int] = None,
        speed: Optional[float] = 1.0,
        instruct: Optional[str] = None,
        **kwargs,
    ) -> Iterator[GenerationResult]:
        """Speech for text, optionally in the voice of reference audio.

        Long text is generated chunk by chunk; each chunk sees the first and the
        previous chunk as history, and max_tokens caps the frames of one chunk.
        Sampling arguments left as None use SGLang-Omni's defaults; a top_k
        caps the main token, and the codebook heads at most 30.
        With stream=True, audio is yielded every streaming_interval seconds;
        first_chunk_frames sets a smaller first piece for a faster first audio.
        Delivery cues go inline, e.g. "[whispers]". Other keyword arguments that
        the CLI and server pass to every model (voice, lang_code, ...) do not
        apply to Drama 3 and are ignored.
        """
        if speed not in (None, 1.0):
            raise ValueError("Drama 3 does not support speed control")
        if instruct:
            raise ValueError(
                "Drama 3 takes delivery cues inline, e.g. [whispers], not instruct"
            )
        validate_text(text, "text")
        max_tokens = max_tokens or DEFAULT_MAX_FRAMES
        sampling = (
            DEFAULT_TEMPERATURE if temperature is None else temperature,
            DEFAULT_TOP_K if top_k is None else top_k,
            DEFAULT_TOP_P if top_p is None else top_p,
        )
        voice = self._voice(ref_audio, ref_text)
        frame_rate = self.sample_rate / self.codec.frame_length
        interval = max(1, round(streaming_interval * frame_rate))
        first = first_chunk_frames or interval
        self.codec.reset_streaming_state()

        chunks = split_text(text)
        pinned: Optional[tuple[str, mx.array]] = None
        latest: Optional[tuple[str, mx.array]] = None
        started = time.perf_counter()
        try:
            for index, chunk in enumerate(chunks):
                is_last = index == len(chunks) - 1
                history = [turn for turn in (pinned, latest) if turn is not None]
                frames, system_rows = self.build_frames(
                    voice.references, history, chunk
                )
                codes: list[mx.array] = []
                emitted = 0
                for frame in self._frames(
                    frames, system_rows, voice, max_tokens, *sampling
                ):
                    codes.append(frame)
                    due = first if index == 0 and emitted == 0 else interval
                    if stream and len(codes) - emitted >= due:
                        yield self._result(
                            codes[emitted:], index, started, stream, False
                        )
                        emitted, started = len(codes), time.perf_counter()
                if codes:
                    turn = (chunk, mx.stack(codes, axis=1))
                    if pinned is None:
                        pinned = turn
                    else:
                        latest = turn
                if len(codes) > emitted or is_last:
                    # An empty last chunk still closes the stream.
                    result = self._result(
                        codes[emitted:], index, started, stream, is_last
                    )
                    # Buffers sized by this prompt and chunk rarely fit a later one,
                    # so MLX's cache would keep them up to its GC limit, about 110 GB
                    # on a 128 GB Mac. Cleared before the yield, as a caller may stop
                    # at the final chunk.
                    mx.clear_cache()
                    yield result
                    started = time.perf_counter()
                else:
                    # The chunk ended with its last streamed piece.
                    mx.clear_cache()
        finally:
            # A caller that stops early, or an error, skips the clear above.
            mx.clear_cache()

    def _result(
        self,
        codes: list[mx.array],
        segment_idx: int,
        started: float,
        streaming: bool,
        final: bool,
    ) -> GenerationResult:
        pieces = []
        for begin in range(0, len(codes), DECODE_FRAMES):
            frames = mx.stack(codes[begin : begin + DECODE_FRAMES], axis=1)[None]
            pieces.append(self.codec.streaming_step(frames)[0, 0])
            # Evaluated piece by piece, so only one piece's working set is alive.
            mx.eval(pieces[-1])
        audio = mx.concatenate(pieces) if pieces else mx.zeros((0,), mx.float32)
        mx.eval(audio)
        elapsed = max(time.perf_counter() - started, 1e-6)
        samples = audio.shape[0]
        seconds = samples / self.sample_rate
        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=segment_idx,
            token_count=len(codes),
            audio_duration=f"{seconds:.2f}s",
            real_time_factor=seconds / elapsed,
            prompt={"tokens": len(codes), "tokens-per-sec": len(codes) / elapsed},
            audio_samples={"samples": samples, "samples-per-sec": samples / elapsed},
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=streaming,
            is_final_chunk=final,
        )

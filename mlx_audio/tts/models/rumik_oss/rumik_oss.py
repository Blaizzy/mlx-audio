"""rumik-oss 1: multilingual Indic text-to-speech.

rumik-oss 1 extends Cohere's Tiny Aya Fire (a ``cohere2`` decoder) with
flattened Mimi codec tokens. Text conditioning and audio generation share one
autoregressive sequence: for every 80 ms audio frame the model emits eight
codebook tokens in codebook order before moving to the next frame. The frozen
Mimi codec turns the regrouped frames back into 24 kHz audio.

Prompt format (the tokenizer adds ``[BOS]``)::

    <text>Ira: <description="happy, Hindi accent, steady pace"> नमस्ते<audio>

Reference implementation: https://huggingface.co/rumik-ai/rumik-oss-1
"""

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder
from mlx_audio.lm.generate import generate_step
from mlx_audio.lm.models.cohere2 import Model as Cohere2Model
from mlx_audio.lm.models.cohere2 import ModelArgs as Cohere2ModelArgs
from mlx_audio.lm.sample_utils import make_sampler

from ..base import GenerationResult

# The codec bundled with rumik-oss-1 is byte-identical to kyutai/mimi; the
# kyutai checkpoint below is the same weights in the layout our loader expects.
MIMI_REPO = "kyutai/moshiko-pytorch-bf16"

_STOP_PREDICTOR_KEY = re.compile(r"^stop_predictor\.(\d+)\.")


@dataclass
class ModelConfig(Cohere2ModelArgs):
    model_type: str = "rumik_oss"
    # Audio vocabulary. Unit tokens are laid out code-major, quantizer-minor:
    # <0_0> <0_1> ... <0_7> <1_0> ... so for an id in [first_unit_id, last_unit_id]
    #   code      = (id - first_unit_id) // num_quantizers
    #   quantizer = (id - first_unit_id) %  num_quantizers
    num_quantizers: int = 8
    codebook_size: int = 2048
    audio_start_token_id: Optional[int] = None
    audio_end_token_id: Optional[int] = None
    text_start_token_id: Optional[int] = None
    first_unit_id: Optional[int] = None
    last_unit_id: Optional[int] = None
    frame_rate_hz: float = 12.5
    speakers: List[str] = field(
        default_factory=lambda: ["Ira", "Aisha", "Siya", "Zoya"]
    )
    sample_rate: int = 24000
    # Hugging Face repo holding the Mimi codec weights (kyutai layout).
    mimi_repo: str = MIMI_REPO
    # Injected by the loader; used to locate the tokenizer.
    model_path: Optional[str] = None

    def __post_init__(self):
        if self.last_unit_id is None and self.first_unit_id is not None:
            self.last_unit_id = (
                self.first_unit_id + self.codebook_size * self.num_quantizers - 1
            )


class Model(Cohere2Model):
    """Cohere2 backbone plus rumik-oss 1's stop predictor and codec plumbing."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        hidden = int(config.hidden_size)
        mid = max(64, hidden // 4)
        # Mirrors the PyTorch nn.Sequential: LayerNorm -> Linear -> GELU -> Linear.
        self.stop_predictor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )
        # Underscore-prefixed attributes are invisible to nn.Module parameters().
        self._last_hidden: Optional[mx.array] = None
        self._tokenizer = None
        self._mimi: Optional[Mimi] = None
        self._streaming_decoder: Optional[MimiStreamingDecoder] = None
        # Audio-only output head (see ``audio_head``); built lazily.
        self._use_audio_head = False
        self._audio_head_params = None
        self._audio_rows: Optional[List[int]] = None

    # ------------------------------------------------------------------ forward

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        hidden = self.model(inputs, cache)  # post final-norm hidden states
        # The stop predictor reads the last position's normed hidden state, the
        # same tensor the logits are computed from.
        self._last_hidden = hidden[:, -1, :]
        if self._use_audio_head:
            logits = self._audio_logits(hidden)
        else:
            logits = self.model.embed_tokens.as_linear(hidden)
        return logits * self.args.logit_scale

    # -------------------------------------------------------------- audio head
    #
    # Inside an <audio> span the model may only emit the 16,384 unit tokens plus
    # </audio>, yet the tied output head spans the whole 277k text+audio vocab.
    # Reading that matrix dominates per-step memory traffic (0.57B of the 3.35B
    # parameters) and the following softmax/top-k run over 277k columns. Slicing
    # the head to the audio rows gives identical logits for every legal token
    # (verified bit-exact against ``as_linear`` for both float and quantized
    # embeddings) while skipping the rest.

    @property
    def audio_rows(self) -> List[int]:
        """Vocabulary ids covered by the audio head: units then ``</audio>`` last."""
        if self._audio_rows is None:
            c = self.config
            self._audio_rows = list(
                range(int(c.first_unit_id), int(c.last_unit_id) + 1)
            )
            self._audio_rows.append(self.audio_end_id)
        return self._audio_rows

    def _build_audio_head(self):
        emb = self.model.embed_tokens
        rows = mx.array(self.audio_rows)
        if isinstance(emb, nn.QuantizedEmbedding):
            self._audio_head_params = dict(
                weight=emb["weight"][rows],
                scales=emb["scales"][rows],
                biases=None if emb.get("biases") is None else emb["biases"][rows],
                group_size=emb.group_size,
                bits=emb.bits,
                mode=emb.mode,
            )
        else:
            self._audio_head_params = dict(weight=emb.weight[rows])
        mx.eval(
            *[v for v in self._audio_head_params.values() if isinstance(v, mx.array)]
        )

    def _audio_logits(self, hidden: mx.array) -> mx.array:
        if self._audio_head_params is None:
            self._build_audio_head()
        p = self._audio_head_params
        if "scales" in p:
            return mx.quantized_matmul(
                hidden,
                p["weight"],
                scales=p["scales"],
                biases=p["biases"],
                transpose=True,
                group_size=p["group_size"],
                bits=p["bits"],
                mode=p["mode"],
            )
        return hidden @ p["weight"].T

    def make_sliced_sampler(self, sampler):
        """Wrap a sampler over sliced logits so it returns vocabulary ids.

        ``generate_step`` feeds each sampled token back into the model and
        yields it to the caller, so the sampler must emit real token ids, not
        columns of the sliced head.
        """
        rows = mx.array(self.audio_rows, dtype=mx.int32)

        def sliced_sampler(logprobs: mx.array) -> mx.array:
            return rows[sampler(logprobs)]

        return sliced_sampler

    @contextmanager
    def audio_head(self, enabled: bool = True):
        """Route ``__call__`` through the audio-only head while active.

        Logits then index ``audio_rows`` (column ``i`` is token ``audio_rows[i]``)
        rather than the full vocabulary. Pair with ``make_sliced_sampler`` so
        sampled tokens are mapped back to vocabulary ids.
        """
        previous = self._use_audio_head
        self._use_audio_head = bool(enabled)
        try:
            yield
        finally:
            self._use_audio_head = previous

    def stop_probability(self, hidden: mx.array) -> mx.array:
        """P(utterance is over) for hidden states of shape ``(..., hidden_size)``."""
        return mx.sigmoid(self.stop_predictor(hidden))

    # ------------------------------------------------------------- weights I/O

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            if k == "lm_head.weight":
                # Embeddings are tied; the head is the embedding matrix.
                continue
            # PyTorch nn.Sequential indexes children directly; MLX under `layers`.
            k = _STOP_PREDICTOR_KEY.sub(r"stop_predictor.layers.\1.", k)
            sanitized[k] = v
        return sanitized

    def model_quant_predicate(self, path, module):
        # The stop head is ~1M params and threshold-sensitive: keep it unquantized.
        return not path.startswith("stop_predictor")

    # -------------------------------------------------------------- properties

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self) -> int:
        return int(self.config.sample_rate)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.config.model_path is None:
                raise ValueError(
                    "ModelConfig.model_path is unset; cannot locate the tokenizer."
                )
            from transformers import AutoTokenizer

            # The repo's auto_map points at custom *model* code; the tokenizer
            # itself is a stock CohereTokenizer, so skip the remote-code prompt.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=False
            )
        return self._tokenizer

    @property
    def mimi(self) -> Mimi:
        if self._mimi is None:
            mimi = Mimi.from_pretrained(self.config.mimi_repo)
            mimi.eval()
            self._mimi = mimi
            self._streaming_decoder = MimiStreamingDecoder(mimi)
        return self._mimi

    @property
    def streaming_decoder(self) -> MimiStreamingDecoder:
        self.mimi  # ensure loaded
        return self._streaming_decoder

    # ------------------------------------------------------------ audio vocab

    @property
    def audio_end_id(self) -> int:
        return int(self.config.audio_end_token_id)

    def is_unit_token(self, token_id: int) -> bool:
        c = self.config
        return int(c.first_unit_id) <= token_id <= int(c.last_unit_id)

    def push_audio_token(
        self, token_id: int, frame: List[int]
    ) -> Tuple[List[int], Optional[List[int]]]:
        """Feed one generated id into the round-robin frame assembler.

        Returns ``(frame_in_progress, completed_frame_or_None)``. A token that
        breaks the codebook round robin drops the partial frame and resyncs, so
        one bad token costs one frame rather than the rest of the clip. This is
        the incremental form of the reference ``audio_tokens_to_codes``.
        """
        c = self.config
        q_count = int(c.num_quantizers)
        if not self.is_unit_token(token_id):
            return [], None
        code, q = divmod(token_id - int(c.first_unit_id), q_count)
        if q == len(frame):
            frame = frame + [code]
            if len(frame) == q_count:
                return [], frame
            return frame, None
        return ([code] if q == 0 else []), None

    def audio_tokens_to_codes(self, token_ids: List[int]) -> mx.array:
        """Generated ids -> ``[1, num_quantizers, num_frames]`` codec codes."""
        frames: List[List[int]] = []
        frame: List[int] = []
        for tid in token_ids:
            tid = int(tid)
            if tid == self.audio_end_id:
                break
            frame, done = self.push_audio_token(tid, frame)
            if done is not None:
                frames.append(done)
        if not frames:
            raise ValueError(
                f"no complete codec frame in {len(token_ids)} tokens "
                f"(need at least {self.config.num_quantizers})"
            )
        return self._frames_to_codes(frames)

    @staticmethod
    def _frames_to_codes(frames: List[List[int]]) -> mx.array:
        # frames: [T, Q] -> codes: [1, Q, T]
        return mx.array(frames, dtype=mx.int32).T[None]

    # ---------------------------------------------------------------- prompts

    def resolve_voice(self, voice: Optional[str]) -> str:
        speakers = list(self.config.speakers)
        if voice is None:
            return speakers[0]
        for name in speakers:
            if name.lower() == str(voice).strip().lower():
                return name
        raise ValueError(f"Unknown voice {voice!r}. Available: {', '.join(speakers)}")

    def build_prompt(
        self, text: str, voice: Optional[str] = None, description: Optional[str] = None
    ) -> str:
        speaker = self.resolve_voice(voice)
        text = text.strip()
        if description and "<description=" not in text:
            text = f'<description="{description.strip()}"> {text}'
        return f"<text>{speaker}: {text}<audio>"

    def encode_prompt(self, prompt: str) -> mx.array:
        # The tokenizer prepends [BOS] itself.
        return mx.array(self.tokenizer(prompt)["input_ids"])

    # ------------------------------------------------------------- generation

    def make_audio_logits_processor(self, min_tokens: int):
        """Restrict sampling to the audio vocabulary and apply the stop predictor.

        Per step, in this order (matching the reference ``generate_audio``):
          1. before ``min_tokens`` generated tokens, ``</audio>`` is masked out
             and the stop predictor is not consulted;
          2. afterwards, if the stop predictor fires (p > 0.5) on the last
             hidden state, ``</audio>`` is forced;
          3. otherwise sampling is restricted to unit tokens plus ``</audio>``.
        """
        c = self.config
        vocab = int(c.vocab_size)
        first, last, end_id = (
            int(c.first_unit_id),
            int(c.last_unit_id),
            self.audio_end_id,
        )

        ids = mx.arange(vocab)
        units = (ids >= first) & (ids <= last)
        allowed = units | (ids == end_id)
        forced_end = mx.where(ids == end_id, 0.0, -mx.inf)
        state = {"step": 0}

        def processor(tokens: mx.array, logits: mx.array) -> mx.array:
            step = state["step"]
            state["step"] += 1
            if step < min_tokens:
                return mx.where(units, logits, -mx.inf)
            stop = self.stop_probability(self._last_hidden) > 0.5  # (B, 1)
            restricted = mx.where(allowed, logits, -mx.inf)
            return mx.where(stop, forced_end.astype(logits.dtype), restricted)

        return processor

    def make_sliced_logits_processor(self, min_tokens: int):
        """``make_audio_logits_processor`` for logits produced by ``audio_head``.

        Every column is already a legal audio token, so only the ``</audio>``
        column (the last one) needs the min-tokens gate and the stop forcing.
        """
        n = len(self.audio_rows)
        end_col = n - 1
        cols = mx.arange(n)
        not_end = cols != end_col
        forced_end = mx.where(not_end, -mx.inf, 0.0)
        state = {"step": 0}

        def processor(tokens: mx.array, logits: mx.array) -> mx.array:
            step = state["step"]
            state["step"] += 1
            if step < min_tokens:
                return mx.where(not_end, logits, -mx.inf)
            stop = self.stop_probability(self._last_hidden) > 0.5  # (B, 1)
            return mx.where(stop, forced_end.astype(logits.dtype), logits)

        return processor

    def _decode_frames_streaming(self, frames: List[List[int]]) -> mx.array:
        codes = self._frames_to_codes(frames)
        return self.streaming_decoder.decode_frames(codes).reshape(-1)

    def _decode_frames(self, frames: List[List[int]]) -> mx.array:
        codes = self._frames_to_codes(frames)
        return self.mimi.decode(codes).reshape(-1)

    def _result(
        self,
        audio: mx.array,
        start_time: float,
        token_count: int,
        prompt_tokens: int,
        segment_idx: int = 0,
        is_streaming_chunk: bool = False,
        is_final_chunk: bool = False,
    ) -> GenerationResult:
        samples = int(audio.shape[0])
        sr = self.sample_rate
        duration = samples / sr
        elapsed = time.perf_counter() - start_time
        rtf = duration / elapsed if elapsed > 0 else 0.0
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        s = int(duration % 60)
        ms = int((duration % 1) * 1000)
        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=sr,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}",
            real_time_factor=rtf,
            prompt={
                "tokens": prompt_tokens,
                "tokens-per-sec": round(token_count / elapsed, 2) if elapsed > 0 else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        min_tokens: int = 8,
        instruct: Optional[str] = None,
        description: Optional[str] = None,
        stream: bool = False,
        streaming_interval: float = 0.5,
        verbose: bool = False,
        slice_head: bool = True,
        **kwargs,
    ):
        """Synthesize ``text`` and yield :class:`GenerationResult` objects.

        Args:
            text: Text to speak. May embed ``<description="...">`` and inline
                ``<laugh>``, ``<chuckle>``, ``<sigh>`` tags.
            voice: One of the speakers in the config (default: the first).
            temperature, top_k, top_p: Sampling controls (reference: 0.8 / 30 / 1.0).
            max_tokens: Cap on generated audio tokens (100 tokens per second).
            min_tokens: Tokens to generate before ``</audio>`` may be emitted.
            instruct / description: Delivery description, e.g.
                ``"excited, Hindi accent, fast pace"``. ``instruct`` is the
                mlx-audio CLI spelling; ``description`` wins if both are given.
            stream: Yield audio as it is generated instead of once at the end.
            streaming_interval: Seconds of audio per streamed chunk.
            slice_head: Compute logits only over the audio vocabulary (see
                ``audio_head``). Same outputs, fewer bytes per step; disable
                only for benchmarking against the full head.
        """
        description = description or instruct
        prompt = self.build_prompt(text, voice, description)
        prompt_ids = self.encode_prompt(prompt)
        prompt_tokens = int(prompt_ids.shape[0])

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p if 0.0 < top_p < 1.0 else 0.0,
            top_k=int(top_k) if top_k and top_k > 0 else 0,
        )
        if slice_head:
            processors = [self.make_sliced_logits_processor(int(min_tokens))]
            sampler = self.make_sliced_sampler(sampler)
        else:
            processors = [self.make_audio_logits_processor(int(min_tokens))]

        frame_rate = float(self.config.frame_rate_hz)
        frames_per_chunk = max(1, int(round(streaming_interval * frame_rate)))

        frames: List[List[int]] = []
        frame: List[int] = []
        start = time.perf_counter()

        if stream:
            self.streaming_decoder.reset()

        with self.audio_head(slice_head):
            steps = generate_step(
                prompt_ids,
                self,
                max_tokens=int(max_tokens),
                sampler=sampler,
                logits_processors=processors,
            )
            yield from self._consume(
                steps,
                frames,
                frame,
                frames_per_chunk,
                stream,
                start,
                prompt_tokens,
                frame_rate,
                verbose,
            )

    def _consume(
        self,
        steps,
        frames,
        frame,
        frames_per_chunk,
        stream,
        start,
        prompt_tokens,
        frame_rate,
        verbose,
    ):
        decoded = 0
        token_count = 0
        for token, _ in steps:
            token_count += 1
            if token == self.audio_end_id:
                break
            frame, done = self.push_audio_token(token, frame)
            if done is not None:
                frames.append(done)

            if stream and len(frames) - decoded >= frames_per_chunk:
                audio = self._decode_frames_streaming(frames[decoded:])
                decoded = len(frames)
                yield self._result(
                    audio,
                    start,
                    token_count,
                    prompt_tokens,
                    is_streaming_chunk=True,
                )
                start = time.perf_counter()
                token_count = 0

            if verbose and token_count % 200 == 0:
                print(f"{len(frames)} frames ({len(frames) / frame_rate:.1f}s)")

        if verbose:
            print(f"{len(frames)} frames -> {len(frames) / frame_rate:.2f}s of audio")

        if stream:
            if len(frames) > decoded:
                audio = self._decode_frames_streaming(frames[decoded:])
            else:
                audio = mx.zeros((0,), dtype=mx.float32)
            yield self._result(
                audio,
                start,
                token_count,
                prompt_tokens,
                is_streaming_chunk=True,
                is_final_chunk=True,
            )
        else:
            if not frames:
                raise ValueError("The model produced no complete audio frame.")
            audio = self._decode_frames(frames)
            yield self._result(audio, start, token_count, prompt_tokens)

        mx.clear_cache()

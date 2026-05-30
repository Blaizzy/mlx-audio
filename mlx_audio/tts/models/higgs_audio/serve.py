"""Higgs Audio v2 — MLX serve engine.

High-level entry point that takes ChatML-style messages (with optional
reference audio for voice cloning) and produces PCM output. Mirrors the
PyTorch `boson_multimodal.serve.serve_engine.HiggsAudioServeEngine` shape
but uses the MLX port end-to-end.

Usage:

    server = HiggsAudioServer.from_pretrained(
        model_path="/path/to/higgs-v2-3b",
        codec_path="/path/to/higgs-mlx-codec",
        # tokenizer_path defaults to model_path
    )
    pcm_24k, sr = server.generate(
        target_text="Hello.",
        reference_audio_path="/path/to/voice.wav",     # optional
        reference_text="Reference transcript.",        # optional, paired with ref audio
        temperature=0.7, top_p=0.95,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import mlx.core as mx
import numpy as np

from ....codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
from .config import HiggsAudioConfig
from .generation import (
    build_delay_pattern_mask,
    lookup_audio_embedding,
    revert_delay_pattern,
)
from .higgs_audio import HiggsAudioModel


@dataclass
class HiggsAudioGenerationResult:
    pcm: np.ndarray  # float32 in [-1, 1], mono
    sampling_rate: int  # 24000 for Higgs v2
    num_frames_raw: int
    num_frames_aligned: int
    stop_reason: str


def _load_wav_as_24k_mono(path: str) -> np.ndarray:
    """Load audio as float32 mono at 24 kHz.

    Higgs codec expects 24 kHz mono input.
    """
    from mlx_audio.audio_io import read as audio_read
    from mlx_audio.utils import resample_audio

    samples, sr = audio_read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 24000:
        samples = resample_audio(samples, sr, 24000)
    return samples


@dataclass
class ReferenceContext:
    """Precomputed reference-audio prompt components.

    Produced once by `encode_reference()`; reusable across any number of
    `build_prompt()` / `HiggsAudioServer.generate()` calls that share the
    same voice. Skips codec.encode, delay-pattern wrap, and prefix-text
    embedding on every call — roughly 300-500 ms per call at q8 on M5 Max
    for a 3-5 s reference clip.
    """

    prefix_emb: mx.array  # [prefix_len, H] — text prefix through <|audio_out_bos|>
    audio_emb: mx.array  # [T_ref_delayed, H] — ref codes as summed codebook embeds
    prefix_len: int
    T_ref_delayed: int
    T_ref: int  # raw ref-code length, pre-delay-pattern (for info)
    ref_text: str  # what was baked into prefix_emb


def encode_reference(
    ref_audio_24k: np.ndarray,
    ref_text: str,
    *,
    config: HiggsAudioConfig,
    tokenizer,
    codec,
    embed_tokens,  # nn.Embedding, text vocab
    audio_codebook_embeddings,  # nn.Embedding, audio codebook table
) -> ReferenceContext:
    """Encode reference audio + its transcript into cacheable prompt pieces.

    This is the expensive half of prompt assembly: codec.encode on the
    reference wav, delay-pattern wrap, and the audio codebook embedding
    lookup. Output is deterministic in the inputs — cache it once per voice.
    """
    K = config.audio_num_codebooks
    stride = config.audio_codebook_size + 2

    ref_codes = codec.encode(mx.array(ref_audio_24k).reshape(1, -1, 1))
    mx.eval(ref_codes)
    ref_codes = ref_codes[0].T.astype(mx.int32)  # [K, T_ref]
    T_ref = ref_codes.shape[1]

    bos_col = mx.full((K, 1), config.audio_stream_bos_id, dtype=mx.int32)
    eos_col = mx.full((K, 1), config.audio_stream_eos_id, dtype=mx.int32)
    ref_wrapped = mx.concatenate([bos_col, ref_codes, eos_col], axis=1)
    ref_delayed = build_delay_pattern_mask(
        ref_wrapped,
        bos_token_id=config.audio_stream_bos_id,
        pad_token_id=config.audio_stream_eos_id,
    )
    T_ref_d = ref_delayed.shape[1]

    prefix = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{ref_text or ''}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|audio_out_bos|>"
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_emb = embed_tokens(mx.array([prefix_ids], dtype=mx.int32))[0]
    audio_emb = lookup_audio_embedding(audio_codebook_embeddings, ref_delayed, stride)
    mx.eval(prefix_emb, audio_emb)  # materialize so first generate() is fast

    return ReferenceContext(
        prefix_emb=prefix_emb,
        audio_emb=audio_emb,
        prefix_len=len(prefix_ids),
        T_ref_delayed=T_ref_d,
        T_ref=T_ref,
        ref_text=ref_text or "",
    )


def _build_prompt_voice_clone(
    target_text: str,
    reference: ReferenceContext,
    *,
    tokenizer,
    embed_tokens,
) -> Tuple[mx.array, mx.array, dict]:
    """Assemble (inputs_embeds, audio_out_mask, info) from a cached reference."""
    middle = (
        "<|audio_eos|><|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{target_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|audio_out_bos|>"
    )
    middle_ids = tokenizer.encode(middle, add_special_tokens=False)
    middle_emb = embed_tokens(mx.array([middle_ids], dtype=mx.int32))[0]

    full_embeds = mx.concatenate(
        [reference.prefix_emb, reference.audio_emb, middle_emb], axis=0
    )[None]
    audio_out_mask = mx.concatenate(
        [
            mx.zeros((reference.prefix_len,), dtype=mx.bool_),
            mx.ones((reference.T_ref_delayed,), dtype=mx.bool_),
            mx.zeros((len(middle_ids),), dtype=mx.bool_),
        ],
        axis=0,
    )[None]
    info = {
        "mode": "voice_clone",
        "T_ref": reference.T_ref,
        "T_ref_delayed": reference.T_ref_delayed,
        "text_len": reference.prefix_len + len(middle_ids),
    }
    return full_embeds, audio_out_mask, info


def _build_prompt_smart_voice(
    target_text: str,
    *,
    tokenizer,
    embed_tokens,
) -> Tuple[mx.array, mx.array, dict]:
    """Assemble the no-reference prompt. Less reliable than voice_clone."""
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{target_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|audio_out_bos|>"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_embeds = embed_tokens(mx.array([prompt_ids], dtype=mx.int32))
    audio_out_mask = mx.zeros((1, len(prompt_ids)), dtype=mx.bool_)
    info = {"mode": "smart_voice", "text_len": len(prompt_ids)}
    return full_embeds, audio_out_mask, info


def build_prompt(
    target_text: str,
    *,
    ref_text: Optional[str],
    ref_audio_24k,  # np.ndarray (float32, mono, 24kHz) or None
    config: HiggsAudioConfig,
    tokenizer,
    codec,
    embed_tokens,  # nn.Embedding, text vocab
    audio_codebook_embeddings,  # nn.Embedding, audio codebook table
) -> Tuple[mx.array, mx.array, dict]:
    """Build (inputs_embeds [1,T,H], audio_out_mask [1,T], info) for a Higgs
    generation step.

    Voice-clone mode (ref_audio_24k provided): ChatML layout
      user: ref_text / assistant: <ref audio> / user: target / assistant: <|audio_out_bos|>
    Smart-voice mode (no ref_audio_24k): target only, assistant kicks off at <|audio_out_bos|>.

    For repeated calls with the same reference, prefer `encode_reference()`
    + `HiggsAudioServer.prepare_reference()` — this entry point re-runs
    codec.encode every call.
    """
    if ref_audio_24k is None:
        return _build_prompt_smart_voice(
            target_text, tokenizer=tokenizer, embed_tokens=embed_tokens
        )
    reference = encode_reference(
        ref_audio_24k,
        ref_text or "",
        config=config,
        tokenizer=tokenizer,
        codec=codec,
        embed_tokens=embed_tokens,
        audio_codebook_embeddings=audio_codebook_embeddings,
    )
    return _build_prompt_voice_clone(
        target_text, reference, tokenizer=tokenizer, embed_tokens=embed_tokens
    )


class HiggsAudioServer:
    """Serve engine — composes model + codec + tokenizer for end-to-end generation."""

    def __init__(
        self,
        model: HiggsAudioModel,
        codec: HiggsAudioTokenizer,
        tokenizer,  # HF AutoTokenizer, kept loose to avoid transformers import here
        config: HiggsAudioConfig,
    ):
        self.model = model
        self.codec = codec
        self.tokenizer = tokenizer
        self.config = config
        # Populated by prepare_reference(); used by _build_prompt when no
        # explicit reference_audio_path overrides it on a per-call basis.
        self._reference_cache: Optional[ReferenceContext] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        codec_path: str,
        tokenizer_path: Optional[str] = None,
    ) -> "HiggsAudioServer":
        import mlx.nn as nn
        from transformers import AutoTokenizer

        # Accept either a local directory or a Hugging Face repo id. Repo ids
        # look like "owner/name" and do not resolve to an existing local path.
        def _resolve(path: str) -> Path:
            p = Path(path)
            if p.exists():
                return p
            if "/" in path and not path.startswith(("/", "./", "../", "~")):
                from huggingface_hub import snapshot_download

                return Path(snapshot_download(repo_id=path))
            return p  # will error later with a clear "not found"

        model_dir = _resolve(model_path)
        codec_dir = _resolve(codec_path)
        tokenizer_dir = _resolve(tokenizer_path) if tokenizer_path else model_dir
        raw_config = json.loads((model_dir / "config.json").read_text())
        config = HiggsAudioConfig.from_dict(raw_config)

        model = HiggsAudioModel(config)

        # If config declares a quantization block, quantize the model skeleton
        # BEFORE loading weights so the QuantizedLinear parameter shapes match
        # the saved checkpoint. The skip list mirrors what the checkpoint was
        # built with — protected layers stay at bf16.
        q = raw_config.get("quantization")
        if q is not None:
            skip = set(q.get("class_predicate_skip", []))

            def _predicate(name: str, module: nn.Module) -> bool:
                if not isinstance(module, (nn.Linear, nn.Embedding)):
                    return False
                return not any(s in name for s in skip)

            nn.quantize(
                model,
                group_size=q["group_size"],
                bits=q["bits"],
                class_predicate=_predicate,
            )

        # Accept either a sharded bosonai-style layout (index + shards) or a
        # single-file weights.safetensors / model.safetensors dump.
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            idx = json.loads(index_path.read_text())
            shards = sorted(set(idx["weight_map"].values()))
            weights: dict = {}
            for shard in shards:
                weights.update(mx.load(str(model_dir / shard)))
        else:
            single = None
            for candidate in ("model.safetensors", "weights.safetensors"):
                if (model_dir / candidate).exists():
                    single = model_dir / candidate
                    break
            if single is None:
                raise FileNotFoundError(
                    f"No weights found in {model_dir}: expected "
                    "model.safetensors.index.json, model.safetensors, "
                    "or weights.safetensors"
                )
            weights = mx.load(str(single))

        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())

        codec = HiggsAudioTokenizer.from_pretrained(str(codec_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        return cls(model=model, codec=codec, tokenizer=tokenizer, config=config)

    # ------------------------------------------------------------------
    # Reference caching
    # ------------------------------------------------------------------

    def prepare_reference(
        self,
        reference_audio_path: str,
        reference_text: str,
    ) -> ReferenceContext:
        """Pre-encode a reference voice once and cache it on this server.

        Subsequent `generate()` / `generate_stream()` calls reuse the cached
        encoding automatically, as long as no explicit `reference_audio_path=`
        kwarg is passed. Saves the codec.encode + delay-pattern + prefix
        embedding work on every call — ~300-500 ms per call at q8 on M5 Max.

        Call `clear_reference()` to drop the cache (e.g. switching voices).
        """
        ref_audio_24k = _load_wav_as_24k_mono(reference_audio_path)
        self._reference_cache = encode_reference(
            ref_audio_24k,
            reference_text,
            config=self.config,
            tokenizer=self.tokenizer,
            codec=self.codec,
            embed_tokens=self.model.embed_tokens,
            audio_codebook_embeddings=self.model.audio_codebook_embeddings,
        )
        return self._reference_cache

    def clear_reference(self) -> None:
        """Drop the cached reference. Subsequent generate() calls without
        explicit reference args fall back to smart-voice mode."""
        self._reference_cache = None

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        target_text: str,
        reference_text: Optional[str],
        reference_audio_path: Optional[str],
    ) -> Tuple[mx.array, mx.array, dict]:
        """Dispatch prompt assembly across three paths:

        1. Explicit `reference_audio_path` → re-encode from scratch
           (backwards-compatible one-shot path).
        2. `self._reference_cache` set → reuse cached encoding.
        3. Neither → smart-voice mode (no reference).
        """
        if reference_audio_path is not None:
            ref_audio_24k = _load_wav_as_24k_mono(reference_audio_path)
            return build_prompt(
                target_text,
                ref_text=reference_text,
                ref_audio_24k=ref_audio_24k,
                config=self.config,
                tokenizer=self.tokenizer,
                codec=self.codec,
                embed_tokens=self.model.embed_tokens,
                audio_codebook_embeddings=self.model.audio_codebook_embeddings,
            )
        if self._reference_cache is not None:
            return _build_prompt_voice_clone(
                target_text,
                self._reference_cache,
                tokenizer=self.tokenizer,
                embed_tokens=self.model.embed_tokens,
            )
        return _build_prompt_smart_voice(
            target_text,
            tokenizer=self.tokenizer,
            embed_tokens=self.model.embed_tokens,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        target_text: str,
        *,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_new_frames: int = 900,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        ras_win_len: Optional[int] = 7,
        ras_max_repeat: int = 2,
        sampling_warmup_frames: int = 0,
        fade_in_ms: float = 5.0,
        fade_out_ms: float = 5.0,
    ) -> HiggsAudioGenerationResult:
        """Synthesize target_text through the MLX port, optionally voice-cloned.

        Returns PCM at 24 kHz mono as float32 in [-1, 1].
        """
        full_embeds, audio_out_mask, _info = self._build_prompt(
            target_text, reference_text, reference_audio_path
        )

        aligned, gen_info = self.model.generate(
            inputs_embeds=full_embeds,
            audio_out_mask=audio_out_mask,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=ras_win_len,
            ras_max_repeat=ras_max_repeat,
            sampling_warmup_frames=sampling_warmup_frames,
        )
        mx.eval(aligned)

        # Codec decode: expects [T, K] layout (per codec_roundtrip reference path).
        pcm = self.codec.decode(aligned.T)
        mx.eval(pcm)
        pcm_np = np.array(pcm).astype(np.float32).reshape(-1)

        # Short linear fade-in/out masks sample-zero/sample-last rounding clicks
        # from quantized variants. At 24 kHz a 5 ms fade is 120 samples — below
        # the onset perception threshold, inaudible on clean material.
        sr = 24000
        n_in = int(fade_in_ms * sr / 1000.0)
        n_out = int(fade_out_ms * sr / 1000.0)
        if n_in > 0 and pcm_np.size > n_in:
            pcm_np[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        if n_out > 0 and pcm_np.size > n_out:
            pcm_np[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)

        return HiggsAudioGenerationResult(
            pcm=pcm_np,
            sampling_rate=24000,
            num_frames_raw=gen_info["num_frames_raw"],
            num_frames_aligned=gen_info["num_frames_aligned"],
            stop_reason=gen_info["stop_reason"],
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        target_text: str,
        *,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_new_frames: int = 900,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        ras_win_len: Optional[int] = 7,
        ras_max_repeat: int = 2,
        sampling_warmup_frames: int = 0,
        chunk_ms: float = 640.0,
        fade_in_ms: float = 5.0,
        fade_out_ms: float = 5.0,
    ) -> Iterator[np.ndarray]:
        """Yield PCM chunks (float32, 24 kHz mono) suitable for Pipecat.

        Current shape: full-generate then chunk. TTFB = full generation wall
        time; per-chunk quality is identical to non-streaming generate().

        Mid-generation emission was tried (re-decode every N frames on the
        accumulated sequence) but produced audible discontinuities at chunk
        boundaries — the codec is a neural vocoder whose output at a given
        sample depends on full-sequence context, so re-decode at chunk N
        vs N+1 produces slightly different PCM for the same positions.
        Proper mid-generation streaming requires overlap-add or an
        incremental-decoder API on the codec; deferred to follow-up.
        """
        result = self.generate(
            target_text,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=ras_win_len,
            ras_max_repeat=ras_max_repeat,
            sampling_warmup_frames=sampling_warmup_frames,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
        )
        samples_per_chunk = int(chunk_ms * result.sampling_rate / 1000.0)
        pcm = result.pcm
        for i in range(0, pcm.size, samples_per_chunk):
            yield pcm[i : i + samples_per_chunk]

    def generate_stream_overlap_add(
        self,
        target_text: str,
        *,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_new_frames: int = 900,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        ras_win_len: Optional[int] = 7,
        ras_max_repeat: int = 2,
        sampling_warmup_frames: int = 0,
        emit_every_frames: int = 16,
        overlap_ms: float = 40.0,
        fade_in_ms: float = 5.0,
        fade_out_ms: float = 5.0,
    ) -> Iterator[np.ndarray]:
        """Mid-generation streaming via overlap-add. Yields PCM chunks.

        Thin wrapper around :func:`iter_overlap_add_pcm` that builds the prompt
        from the server's tokenizer/codec/cache state and strips the per-chunk
        meta. See the helper for the algorithm rationale.
        """
        full_embeds, audio_out_mask, _info = self._build_prompt(
            target_text, reference_text, reference_audio_path
        )
        for chunk, _meta in iter_overlap_add_pcm(
            model=self.model,
            codec=self.codec,
            config=self.config,
            full_embeds=full_embeds,
            audio_out_mask=audio_out_mask,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=ras_win_len,
            ras_max_repeat=ras_max_repeat,
            sampling_warmup_frames=sampling_warmup_frames,
            emit_every_frames=emit_every_frames,
            overlap_ms=overlap_ms,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
        ):
            yield chunk


# ----------------------------------------------------------------------
# Streaming helper
# ----------------------------------------------------------------------
#
# Shared between :meth:`HiggsAudioServer.generate_stream_overlap_add` (rich
# kwarg-style API yielding raw PCM) and the framework-conforming
# :class:`mlx_audio.tts.models.higgs_audio.model.Model.generate` (yields
# ``GenerationResult`` when ``stream=True``). Both call this helper; the
# overlap-add algorithm lives in exactly one place.


def iter_overlap_add_pcm(
    *,
    model,  # HiggsAudioModel — exposes _generate_raw_frames
    codec,  # HiggsAudioTokenizer — decodes [K, T] codes → PCM
    config: HiggsAudioConfig,
    full_embeds: mx.array,
    audio_out_mask: mx.array,
    max_new_frames: int = 900,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.95,
    top_k: Optional[int] = None,
    ras_win_len: Optional[int] = 7,
    ras_max_repeat: int = 2,
    sampling_warmup_frames: int = 0,
    emit_every_frames: int = 16,
    overlap_ms: float = 40.0,
    fade_in_ms: float = 5.0,
    fade_out_ms: float = 5.0,
    sample_rate: int = 24000,
) -> Iterator[Tuple[np.ndarray, dict]]:
    """Mid-generation overlap-add streaming. Yields ``(pcm_chunk, meta)``.

    ``meta`` contains:

    - ``is_final`` (bool): True iff this is the last chunk of the stream.
    - ``frames_total`` (int): cumulative raw codec frames generated so far.

    Re-decodes the accumulated codec sequence every ``emit_every_frames``
    codec frames. The neural vocoder's output at a given sample depends on
    full-sequence context — re-decoding at length ``N`` vs length ``N+K``
    produces slightly different PCM near the right edge of the shorter
    decode. Strategy: hold the last ``overlap_ms`` of each decode as a
    buffer. On the next decode, linearly crossfade the buffered tail with
    the new decode's same-position samples and emit the smoothed region.
    This replaces the edge-affected tail of each decode with the
    full-context version from the next decode.

    Cost: ``O(T² / emit_every_frames)`` decode work — at
    ``emit_every_frames=16`` (~640 ms) this is a few percent of generation
    cost. TTFB shrinks from full-generation wall time to
    ``~(emit_every_frames / 25) s``.
    """
    overlap_samples = int(overlap_ms * sample_rate / 1000.0)
    K = config.audio_num_codebooks
    audio_codebook_size = config.audio_codebook_size

    n_fade_in = int(fade_in_ms * sample_rate / 1000.0)
    n_fade_out = int(fade_out_ms * sample_rate / 1000.0)

    frames_raw: list[mx.array] = []
    overlap_tail: Optional[np.ndarray] = None
    emitted_samples = 0
    raw_frames_at_last_emit = 0
    is_first_chunk = True
    done_via_eos = False

    def _decode_current() -> Optional[np.ndarray]:
        sequence = mx.stack(frames_raw, axis=1).astype(mx.int32)
        aligned = revert_delay_pattern(sequence)
        if aligned.shape[1] < 2:
            return None
        aligned = aligned[:, 1:-1]
        if aligned.shape[1] == 0:
            return None
        aligned = mx.clip(aligned, 0, audio_codebook_size - 1)
        pcm = codec.decode(aligned.T)
        mx.eval(pcm)
        return np.array(pcm).astype(np.float32).reshape(-1)

    for tok, meta in model._generate_raw_frames(
        full_embeds,
        audio_out_mask,
        max_new_frames=max_new_frames,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        ras_win_len=ras_win_len,
        ras_max_repeat=ras_max_repeat,
        sampling_warmup_frames=sampling_warmup_frames,
    ):
        frames_raw.append(tok)

        done_via_eos = (
            meta.get("num_remaining_delays") is not None
            and meta["num_remaining_delays"] <= 0
        )

        have_enough = len(frames_raw) > K + 1
        frames_since_last = len(frames_raw) - raw_frames_at_last_emit
        should_emit = have_enough and (
            done_via_eos or frames_since_last >= emit_every_frames
        )
        if not should_emit:
            continue
        raw_frames_at_last_emit = len(frames_raw)

        pcm_np = _decode_current()
        if pcm_np is None:
            continue

        if is_first_chunk:
            if n_fade_in > 0 and pcm_np.size > n_fade_in:
                pcm_np[:n_fade_in] *= np.linspace(0.0, 1.0, n_fade_in, dtype=np.float32)
            if done_via_eos:
                if n_fade_out > 0 and pcm_np.size > n_fade_out:
                    pcm_np[-n_fade_out:] *= np.linspace(
                        1.0, 0.0, n_fade_out, dtype=np.float32
                    )
                yield pcm_np.copy(), {
                    "is_final": True,
                    "frames_total": len(frames_raw),
                }
                return
            if pcm_np.size > overlap_samples:
                yield pcm_np[:-overlap_samples].copy(), {
                    "is_final": False,
                    "frames_total": len(frames_raw),
                }
                overlap_tail = pcm_np[-overlap_samples:].copy()
                emitted_samples = pcm_np.size - overlap_samples
            else:
                overlap_tail = pcm_np.copy()
                emitted_samples = 0
            is_first_chunk = False
            continue

        # Subsequent chunk: crossfade overlap_tail with new decode's
        # same-position samples, then emit middle region.
        if overlap_tail is None or pcm_np.size < emitted_samples + overlap_samples:
            continue
        new_overlap_region = pcm_np[emitted_samples : emitted_samples + overlap_samples]
        fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
        crossfaded = overlap_tail * fade_out + new_overlap_region * fade_in

        if done_via_eos:
            tail_pcm = pcm_np[emitted_samples + overlap_samples :].copy()
            if n_fade_out > 0 and tail_pcm.size > n_fade_out:
                tail_pcm[-n_fade_out:] *= np.linspace(
                    1.0, 0.0, n_fade_out, dtype=np.float32
                )
            yield np.concatenate([crossfaded, tail_pcm]), {
                "is_final": True,
                "frames_total": len(frames_raw),
            }
            return

        new_middle_start = emitted_samples + overlap_samples
        new_middle_end = pcm_np.size - overlap_samples
        if new_middle_end > new_middle_start:
            middle = pcm_np[new_middle_start:new_middle_end].copy()
            yield np.concatenate([crossfaded, middle]), {
                "is_final": False,
                "frames_total": len(frames_raw),
            }
            overlap_tail = pcm_np[-overlap_samples:].copy()
            emitted_samples = new_middle_end
        else:
            yield crossfaded.copy(), {
                "is_final": False,
                "frames_total": len(frames_raw),
            }
            overlap_tail = pcm_np[-overlap_samples:].copy()
            emitted_samples = pcm_np.size - overlap_samples

    # Post-loop flush: generator exited without EOS (hit max_new_frames).
    # Emit remaining frames as a "final" chunk with fade-out.
    if done_via_eos or raw_frames_at_last_emit >= len(frames_raw):
        return
    pcm_np = _decode_current()
    if pcm_np is None:
        return

    if is_first_chunk:
        # Nothing yet emitted — full utterance fits in one final chunk.
        if n_fade_in > 0 and pcm_np.size > n_fade_in:
            pcm_np[:n_fade_in] *= np.linspace(0.0, 1.0, n_fade_in, dtype=np.float32)
        if n_fade_out > 0 and pcm_np.size > n_fade_out:
            pcm_np[-n_fade_out:] *= np.linspace(1.0, 0.0, n_fade_out, dtype=np.float32)
        yield pcm_np.copy(), {"is_final": True, "frames_total": len(frames_raw)}
        return

    if overlap_tail is None or pcm_np.size < emitted_samples + overlap_samples:
        return
    new_overlap_region = pcm_np[emitted_samples : emitted_samples + overlap_samples]
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
    crossfaded = overlap_tail * fade_out + new_overlap_region * fade_in
    tail_pcm = pcm_np[emitted_samples + overlap_samples :].copy()
    if n_fade_out > 0 and tail_pcm.size > n_fade_out:
        tail_pcm[-n_fade_out:] *= np.linspace(1.0, 0.0, n_fade_out, dtype=np.float32)
    yield np.concatenate([crossfaded, tail_pcm]), {
        "is_final": True,
        "frames_total": len(frames_raw),
    }

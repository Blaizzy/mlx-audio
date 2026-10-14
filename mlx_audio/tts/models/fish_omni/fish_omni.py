"""Fish Audio Drama 3 (`fish_omni`) for MLX.

The backbone is a Qwen3.5-MoE hybrid (Gated DeltaNet + gated full attention)
text model; the codebook tower (tower.py) turns each backbone hidden state into
the 10 codes of one 44.1 kHz codec frame, and the S2 Pro codec decodes them.

Reference: fish_speech/models/text2semantic/fish_omni (generate.py, tower.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs
from mlx_audio.tts.models.base import GenerationResult

from .config import ModelConfig
from .prompt import (
    DRAMA3_CHUNK_BYTES,
    DRAMA3_CHUNK_MAX_NEW_TOKENS,
    History,
    PromptTooLongError,
    build_prompt,
    split_text,
)
from .tower import (
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    RAS_WINDOW_FRAMES,
    AudioDecoder,
    sample_top_k_top_p,
)

PREFILL_CHUNK_SIZE = 128


def load_reference(path, sample_rate: int = 44100) -> mx.array:
    """Mono reference audio at `sample_rate`, resampled with the polyphase FIR.

    Drama 3 is sensitive to how a reference is resampled: miniaudio's linear
    upsampler (what `load_audio` uses when the target rate is higher) leaves
    imaging noise that the codec encodes, and generation from such a voice
    often runs on past the text. Decoding at the native rate and resampling
    here keeps the codes in distribution.
    """
    from mlx_audio.audio_io import read as audio_read
    from mlx_audio.resample import resample_audio_array

    samples, native_rate = audio_read(str(path), dtype="float32", nchannels=1)
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    return mx.array(resample_audio_array(samples, native_rate, sample_rate))


WEIGHT_PREFIXES = ("backbone.", "lm_head.", "tower.")


@dataclass
class DramaGenerationResult(GenerationResult):
    """A chunk's audio plus its codes, reusable as a `previous_takes` entry."""

    codes: Optional[mx.array] = None  # [num_codebooks, frames]
    text: str = ""


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class Model(nn.Module):
    # The CLI then hands over reference paths, which load_reference resamples.
    preserve_ref_audio_path = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        text_args = TextModelArgs.from_dict(config.text_config)
        layer_types = config.text_config.get("layer_types")
        if layer_types is not None:
            expected = [
                (
                    "full_attention"
                    if (i + 1) % text_args.full_attention_interval == 0
                    else "linear_attention"
                )
                for i in range(text_args.num_hidden_layers)
            ]
            if list(layer_types) != expected:
                raise ValueError("layer_types is not a fixed full-attention interval")
        self.backbone = Qwen3_5TextModel(text_args)

        # The main token is only ever drawn from eos + the semantic band, so the
        # converter keeps just those lm_head rows, in increasing id order.
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id
        if not self.eos_id < self.semantic_begin:
            raise ValueError("expected eos below the semantic band")
        self.num_candidates = 1 + self.semantic_end - self.semantic_begin + 1
        self.lm_head = nn.Linear(text_args.hidden_size, self.num_candidates, bias=False)
        self.tower = AudioDecoder(
            config.audio_decoder_config, self.semantic_begin, self.semantic_end
        )
        self.tokenizer = None
        self.codec = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def model_type(self) -> str:
        return "fish_omni"

    def sanitize(self, weights: dict) -> dict:
        """Keep the converted model tensors; the codec loads separately."""
        if any(k.startswith("model.language_model.") for k in weights):
            raise ValueError(
                "this is the original Drama 3 checkpoint; convert it first with "
                "python -m mlx_audio.tts.models.fish_omni.convert"
            )
        return {k: v for k, v in weights.items() if k.startswith(WEIGHT_PREFIXES)}

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from tokenizers import Tokenizer

        from mlx_audio.codec.models.fish_s1_dac import DAC as FishS1DAC

        model_path = Path(model_path)
        model.tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        for token, expected in (
            ("<|semantic:0|>", model.semantic_begin),
            (
                f"<|semantic:{model.config.audio_decoder_config.vocab_size - 1}|>",
                model.semantic_end,
            ),
            ("<|im_end|>", model.eos_id),
        ):
            if model.tokenizer.token_to_id(token) != expected:
                raise ValueError(f"tokenizer/config mismatch for {token}")
        if (model_path / "codec.safetensors").exists():
            model.codec = FishS1DAC.from_pretrained(str(model_path))
        return model

    # ------------------------------------------------------------------ decode

    def candidate_to_token(self, index: mx.array) -> mx.array:
        """lm_head row -> token id (row 0 is eos, then the semantic band)."""
        return mx.where(index == 0, self.eos_id, self.semantic_begin + index - 1)

    def _backbone_step(self, frames: mx.array, cache) -> mx.array:
        embeddings = self.tower.embed_backbone(
            frames, self.backbone.embed_tokens(frames[..., 0])
        )
        return self.backbone(None, cache, input_embeddings=embeddings)

    def prefill(self, prompt: mx.array, backbone_cache, tower_cache) -> mx.array:
        """Run prompt rows [1, P, W] through both caches; returns the last hidden."""
        rows = prompt.shape[1]
        hidden = None
        for start in range(0, rows, PREFILL_CHUNK_SIZE):
            end = min(start + PREFILL_CHUNK_SIZE, rows)
            hidden = self._backbone_step(prompt[:, start:end], backbone_cache)
            paired_end = min(end, rows - 1)
            if paired_end > start:
                self.tower.prefill(
                    hidden[:, : paired_end - start],
                    prompt[:, start:paired_end],
                    prompt[:, start + 1 : paired_end + 1],
                    tower_cache,
                )
            mx.eval(
                hidden,
                [c.state for c in backbone_cache],
                [c.state for c in tower_cache],
            )
        return hidden[:, -1:]

    def sample_main(
        self,
        hidden: mx.array,
        window: mx.array,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> mx.array:
        """Main token [1] over eos + semantics, with repetition-aware resampling."""
        logits = self.lm_head(hidden)[:, -1]
        first = self.candidate_to_token(
            sample_top_k_top_p(logits, temperature, top_k, top_p)
        )
        retry = self.candidate_to_token(
            sample_top_k_top_p(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)
        )
        # Only audio tokens are resampled: the window starts with prompt rows,
        # which hold the user turn's eos, and a stop must not be redrawn.
        repeated = (window[None, :] == first[:, None]).any(axis=-1)
        repeated = repeated & self.tower.is_semantic(first)
        return mx.where(repeated, retry, first)

    def generate_frames(
        self,
        prompt: np.ndarray,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        """Yield sampled frames [1, 1, 1 + N] after the prompt rows, until eos."""
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1; got {top_k}")
        backbone_cache = self.backbone.make_cache()
        tower_cache = self.tower.make_cache()
        frames = mx.array(prompt)[None]
        hidden = self.prefill(frames, backbone_cache, tower_cache)
        frame = frames[:, -1:]
        window = mx.array(prompt[-RAS_WINDOW_FRAMES:, 0])

        for step in range(max_new_tokens):
            main = self.sample_main(hidden, window, temperature, top_k, top_p)
            codes = self.tower.decode_step(
                main[:, None],
                hidden,
                frame,
                tower_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            frame = mx.concatenate([main[:, None, None], codes], axis=-1)
            mx.eval(frame)
            if frame[0, 0, 0].item() == self.eos_id:
                return
            yield frame
            window = mx.concatenate([window[1:], main])
            if step + 1 < max_new_tokens:
                hidden = self._backbone_step(frame, backbone_cache)

    def generate_codes(
        self,
        prompt: np.ndarray,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Optional[mx.array]:
        """Codes [num_codebooks, frames] for one prompt, or None if eos came first."""
        frames = list(
            self.generate_frames(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )
        if not frames:
            return None
        return mx.concatenate(frames, axis=1)[0, :, 1:].T

    # --------------------------------------------------------------- codec I/O

    def encode_reference(self, audio: mx.array) -> mx.array:
        """Codes [num_codebooks, frames] of 44.1 kHz reference audio."""
        if self.codec is None:
            raise ValueError("Codec not loaded. Call post_load_hook first.")
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[None, :, :]
        if audio.shape[1] != 1:
            audio = mx.mean(audio, axis=1, keepdims=True)
        indices, feature_lengths = self.codec.encode(audio)
        return indices[0, :, : int(feature_lengths[0].item())]

    def decode_codes(self, codes: mx.array) -> mx.array:
        if self.codec is None:
            raise ValueError("Codec not loaded. Call post_load_hook first.")
        lengths = mx.array([codes.shape[1]], dtype=mx.int32)
        audio, audio_lengths = self.codec.decode(codes[None], lengths)
        return audio[0, 0, : int(audio_lengths[0].item())]

    def build_chunk_prompt(
        self,
        text: str,
        prompt_texts: Sequence[str],
        prompt_tokens: Sequence[mx.array],
        history: History,
    ) -> np.ndarray:
        """The prompt for one chunk with as much history as fits."""
        cfg = self.config.audio_decoder_config
        error = None
        for turns in history.candidates():
            try:
                return build_prompt(
                    text,
                    self.tokenizer,
                    num_codebooks=cfg.num_codebooks,
                    codebook_size=cfg.vocab_size,
                    semantic_start=self.semantic_begin,
                    prompt_texts=prompt_texts,
                    prompt_tokens=prompt_tokens,
                    history=turns,
                )
            except PromptTooLongError as too_long:
                error = too_long
        raise error

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        max_tokens: int = DRAMA3_CHUNK_MAX_NEW_TOKENS,
        # The official Drama 3 server's defaults (sglang-omni speech_request.py).
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 30,
        chunk_length: int = DRAMA3_CHUNK_BYTES,
        single_chunk: bool = False,
        ref_codes: Optional[List[mx.array]] = None,
        ref_texts: Optional[List[str]] = None,
        previous_takes: Optional[List[Tuple[str, mx.array]]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Yield one DramaGenerationResult per text chunk.

        References come from `ref_audio` (a path, or 44.1 kHz audio) + `ref_text`,
        or from pre-encoded `ref_codes` + `ref_texts`. `previous_takes` holds
        earlier (text, codes) takes, e.g. `(result.text, result.codes)`: they go
        into the prompt as history, so the model edits by continuation.
        """
        del voice, kwargs
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        prompt_texts = list(ref_texts or [])
        prompt_tokens = [mx.array(c) for c in (ref_codes or [])]
        if ref_audio is not None:
            audios = ref_audio if isinstance(ref_audio, (list, tuple)) else [ref_audio]
            texts = ref_text if isinstance(ref_text, (list, tuple)) else [ref_text]
            if len(texts) != len(audios) or not all(texts):
                raise ValueError("every ref_audio needs its transcript in ref_text")
            for audio, transcript in zip(audios, texts):
                if isinstance(audio, (str, Path)):
                    audio = load_reference(audio, self.sample_rate)
                prompt_tokens.append(self.encode_reference(audio))
                prompt_texts.append(transcript)

        chunks = [text] if single_chunk else split_text(text, chunk_length) or [text]
        takes = list(previous_takes or [])
        history = History(recent_turns=max(1, len(takes) - 1))
        for take_text, take_codes in takes:
            history.push(take_text, mx.array(take_codes))
        for segment_idx, chunk in enumerate(chunks):
            start_time = time.perf_counter()
            prompt = self.build_chunk_prompt(
                chunk, prompt_texts, prompt_tokens, history
            )
            codes = self.generate_codes(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if codes is None:
                continue
            history.push(chunk, codes)
            audio = self.decode_codes(codes)
            mx.eval(audio)

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            audio_duration = audio.shape[0] / self.sample_rate
            if verbose:
                print(
                    f"chunk {segment_idx + 1}/{len(chunks)}: {len(prompt)}-row prompt, "
                    f"{codes.shape[1]} frames in {elapsed:.2f}s "
                    f"({codes.shape[1] / elapsed:.1f} frames/s)"
                )
            yield DramaGenerationResult(
                codes=codes,
                text=chunk,
                audio=audio,
                samples=int(audio.shape[0]),
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=int(codes.shape[1]),
                audio_duration=_format_duration(audio_duration),
                real_time_factor=audio_duration / elapsed,
                prompt={
                    "tokens": int(len(prompt)),
                    "tokens-per-sec": len(prompt) / elapsed,
                },
                audio_samples={
                    "samples": int(audio.shape[0]),
                    "samples-per-sec": audio.shape[0] / elapsed,
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=float(mx.get_peak_memory() / 1e9),
            )

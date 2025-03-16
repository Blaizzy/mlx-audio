from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from mlx_audio.codec import Mimi

from ..base import GenerationResult
from .model import SesameModel

try:
    from .watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
except ImportError:
    print(
        "Watermarking module not found. Please install silentcipher to use watermarking."
    )

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
TOKENIZER_REPO = "unsloth/Llama-3.2-1B"


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: mx.array


def load_llama3_tokenizer(path_or_hf_repo: str):
    tokenizer = AutoTokenizer.from_pretrained(path_or_hf_repo)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id),
            (f"{eos}", tokenizer.eos_token_id),
        ],
    )
    return tokenizer


class Model:
    def __init__(
        self,
        config: Dict,
    ):
        self._model = SesameModel(config)
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer(TOKENIZER_REPO)
        mimi = Mimi.from_pretrained(MIMI_REPO)
        self._audio_tokenizer = mimi

        try:
            self._watermarker = load_watermarker()
        except Exception:
            self._watermarker = None

        self.sample_rate = mimi.cfg.sample_rate

    def _tokenize_text_segment(
        self, text: str, speaker: int
    ) -> Tuple[mx.array, mx.array]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = mx.zeros((len(text_tokens), 33)).astype(mx.int32)
        text_frame_mask = mx.zeros((len(text_tokens), 33)).astype(mx.bool_)
        text_frame[:, -1] = mx.array(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame)
        frame_masks.append(text_frame_mask)

        return mx.concat(frame_tokens, axis=0), mx.concat(frame_masks, axis=0)

    def _tokenize_audio(self, audio: mx.array) -> Tuple[mx.array, mx.array]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio_tokens = self._audio_tokenizer.encode(
            mx.expand_dims(mx.expand_dims(audio, 0), 0)
        )[0]

        # add EOS frame
        eos_frame = mx.zeros((audio_tokens.shape[0], 1))
        audio_tokens = mx.concat([audio_tokens, eos_frame], axis=1)

        audio_frame = mx.zeros((audio_tokens.shape[1], 33)).astype(mx.int32)
        audio_frame_mask = mx.zeros((audio_tokens.shape[1], 33)).astype(mx.bool_)
        audio_frame[:, :-1] = audio_tokens.swapaxes(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return mx.concat(frame_tokens, axis=0), mx.concat(frame_masks, axis=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[mx.array, mx.array]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(
            segment.text, segment.speaker
        )
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return mx.concat([text_tokens, audio_tokens], axis=0), mx.concat(
            [text_masks, audio_masks], axis=0
        )

    def sanitize(self, weights):
        return weights

    def load_weights(self, weights):
        self._model.load_weights(weights)
        mx.eval(self._model.parameters())
        self._model.eval()

    def generate(
        self,
        text: str,
        speaker: int = 0,
        context: List[Segment] = [],
        max_audio_length_ms: float = 90_000,
        sampler: Callable[..., mx.array] = None,
        ref_audio: mx.array = None,
        ref_text: str = None,
        **kwargs,
    ):
        self._model.reset_caches()

        # if reference audio is provided, use it as the first segment

        if len(context) == 0:
            context = [Segment(speaker=speaker, text=ref_text, audio=ref_audio)]

        start_time = time.time()

        sampler = sampler or make_sampler(temp=0.9, top_k=50)
        max_audio_frames = int(max_audio_length_ms / 80)

        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
            text, speaker
        )
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int32)
        prompt_tokens_mask = mx.concat(tokens_mask, axis=0).astype(mx.bool_)

        samples = []
        curr_tokens = mx.expand_dims(prompt_tokens, axis=0)
        curr_tokens_mask = mx.expand_dims(prompt_tokens_mask, axis=0)
        curr_pos = mx.expand_dims(mx.arange(0, prompt_tokens.shape[0]), axis=0).astype(
            mx.int32
        )

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.shape[1] >= max_seq_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}"
            )

        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, sampler
            )
            if mx.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = mx.expand_dims(
                mx.concat([sample, mx.zeros((1, 1)).astype(mx.int32)], axis=1), axis=1
            )
            curr_tokens_mask = mx.expand_dims(
                mx.concat(
                    [
                        mx.ones_like(sample).astype(mx.bool_),
                        mx.zeros((1, 1)).astype(mx.bool_),
                    ],
                    axis=1,
                ),
                axis=1,
            )
            curr_pos = curr_pos[:, -1:] + 1

        transposed = mx.transpose(mx.stack(samples), axes=[1, 2, 0])
        audio = self._audio_tokenizer.decode(transposed).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        if self._watermarker is not None:
            audio = watermark(
                self._watermarker,
                audio,
                self.sample_rate,
                CSM_1B_GH_WATERMARK,
            )
            audio = mx.array(audio, dtype=mx.float32)

        mx.eval(audio)

        segment_time = time.time() - start_time

        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0, "No audio generated"

        # Calculate token count
        token_count = curr_tokens.shape[2]

        # Calculate audio duration in seconds
        sample_rate = 24000  # Assuming 24kHz sample rate, adjust if different
        audio_duration_seconds = samples / sample_rate

        # Calculate real-time factor (RTF)
        rtf = segment_time / audio_duration_seconds if audio_duration_seconds > 0 else 0

        # Format duration as HH:MM:SS.mmm
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        return [
            GenerationResult(
                audio=audio,
                samples=samples,
                segment_idx=0,
                token_count=token_count,
                audio_duration=duration_str,
                real_time_factor=round(rtf, 2),
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": (
                        round(token_count / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                processing_time_seconds=segment_time,
                peak_memory_usage=mx.metal.get_peak_memory() / 1e9,
            )
        ]

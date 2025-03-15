from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import mlx.core as mx

from huggingface_hub import snapshot_download

from mlx_lm.tokenizer_utils import load_tokenizer

from mlx_audio.codec import Mimi
from .model import Model, ModelArgs

MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"
TOKENIZER_REPO = "mlx-community/Llama-3.2-1B-Instruct-bf16"


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: mx.array


def get_model_path_for_tokenizer(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=["*.json"],
            )
        )
    return model_path


def load_llama3_tokenizer(path_or_hf_repo: str, revision: Optional[str] = None):
    model_path = get_model_path_for_tokenizer(path_or_hf_repo, revision)
    tokenizer = load_tokenizer(model_path)
    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer(TOKENIZER_REPO)
        mimi = Mimi.from_pretrained(DEFAULT_REPO, MIMI_NAME)
        self._audio_tokenizer = mimi

        #  self._watermarker = load_watermarker()

        self.sample_rate = mimi.cfg.sample_rate

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[mx.array, mx.array]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = mx.zeros((len(text_tokens), 33)).astype(mx.int64)
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
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = mx.zeros(audio_tokens.size(0), 1)
        audio_tokens = mx.concat([audio_tokens, eos_frame], axis=1)

        audio_frame = mx.zeros(audio_tokens.size(1), 33).astype(mx.int64)
        audio_frame_mask = mx.zeros(audio_tokens.size(1), 33).astype(mx.bool_)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return mx.concat(frame_tokens, axis=0), mx.concat(frame_masks, axis=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[mx.array, mx.array]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return mx.concat([text_tokens, audio_tokens], axis=0), mx.concat([text_masks, audio_masks], axis=0)

    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> mx.array:
        #  TODO: self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int32)
        prompt_tokens_mask = mx.concat(tokens_mask, axis=0).astype(mx.bool_)

        samples = []
        curr_tokens = mx.expand_dims(prompt_tokens, axis=0)
        curr_tokens_mask = mx.expand_dims(prompt_tokens_mask, axis=0)
        curr_pos = mx.expand_dims(mx.arange(0, prompt_tokens.shape[0]), axis=0).astype(mx.int32)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.shape[1] >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if mx.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = mx.expand_dims(mx.concat([sample, mx.zeros((1, 1)).astype(mx.int32)], axis=1), axis=1)
            curr_tokens_mask = mx.expand_dims(
                mx.concat([mx.ones_like(sample).astype(mx.bool_), mx.zeros((1, 1)).astype(mx.bool_)], axis=1), axis=1
            )
            curr_pos = curr_pos[:, -1:] + 1

        transposed = mx.transpose(mx.stack(samples), axes=[1, 2, 0])
        audio = self._audio_tokenizer.decode(transposed)[None, None, :]

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        # audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        # audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def load_csm_1b(ckpt_path: str = "ckpt.pt") -> Generator:
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(model_args)

    # TODO: Load model from checkpoint
    # state_dict = torch.load(ckpt_path)
    # model.load_state_dict(state_dict)

    generator = Generator(model)
    return generator

#  Debugging

if __name__ == "__main__":
    generator = load_csm_1b()
    text = "Hello world."
    speaker = 0
    context = []
    audio = generator.generate(text, speaker, context)
    print(audio)

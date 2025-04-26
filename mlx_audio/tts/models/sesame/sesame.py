from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelArgs
from mlx_lm.sample_utils import make_sampler
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from transformers import AutoTokenizer

from mlx_audio.codec import Mimi

from ..base import GenerationResult
from .attention import Attention

try:
    from .watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
except ImportError:
    print(
        "Watermarking module not found. Please install silentcipher to use watermarking."
    )

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
TOKENIZER_REPO = "unsloth/Llama-3.2-1B"


def create_causal_mask(seq_len: int) -> mx.array:
    return mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))


def index_causal_mask(mask: mx.array, input_pos: mx.array) -> mx.array:
    mask_indexed = mx.take(mask, input_pos, axis=0)

    seq_len = input_pos.shape[1]
    mask_indexed = mask_indexed[:, :, :seq_len]

    # reshape to (batch_size, 1, seq_len, seq_len) for broadcasting across heads
    return mx.expand_dims(mask_indexed, axis=1)


@dataclass
class SesameModelArgs:
    model_type: str
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


def create_llama_model_args(flavor: str) -> LlamaModelArgs:
    if flavor == "llama-1B":
        return LlamaModelArgs(
            model_type="llama",
            num_hidden_layers=16,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=64,
            hidden_size=2048,
            intermediate_size=8192,
            rms_norm_eps=1e-5,
            vocab_size=128_256,
            max_position_embeddings=2048,
            attention_bias=False,
            mlp_bias=False,
            rope_theta=500_000,
            rope_scaling={
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
    elif flavor == "llama-100M":
        return LlamaModelArgs(
            model_type="llama",
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=128,
            hidden_size=1024,
            intermediate_size=8192,
            rms_norm_eps=1e-5,
            vocab_size=128_256,
            max_position_embeddings=2048,
            attention_bias=False,
            mlp_bias=False,
            rope_theta=500_000,
            rope_scaling={
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
    else:
        raise ValueError(f"Unknown flavor: {flavor}")


class SesameModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = SesameModelArgs(**config)
        self.args = args

        backbone_args = create_llama_model_args(args.backbone_flavor)
        decoder_args = create_llama_model_args(args.decoder_flavor)

        self.backbone = LlamaModel(backbone_args)
        self.decoder = LlamaModel(decoder_args)

        backbone_dim = backbone_args.hidden_size
        decoder_dim = decoder_args.hidden_size

        self.backbone.embed_tokens = nn.Identity()
        self.decoder.embed_tokens = nn.Identity()

        for layer in self.backbone.layers:
            layer.self_attn = Attention(backbone_args)
        for layer in self.decoder.layers:
            layer.self_attn = Attention(decoder_args)

        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            args.audio_vocab_size * args.audio_num_codebooks, backbone_dim
        )

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = mx.zeros(
            (args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size)
        )

        self._backbone_causal_mask = None
        self._decoder_causal_mask = None

        self.backbone_cache = None
        self.decoder_cache = None
        self.caches_enabled = False

    def setup_caches(self, max_batch_size: int):
        backbone_args = create_llama_model_args(self.args.backbone_flavor)

        self._backbone_causal_mask = create_causal_mask(
            backbone_args.max_position_embeddings
        )
        self._decoder_causal_mask = create_causal_mask(self.args.audio_num_codebooks)

        self.backbone_cache = make_prompt_cache(self.backbone)
        self.decoder_cache = make_prompt_cache(self.decoder)
        self.caches_enabled = True

    def caches_are_enabled(self):
        return self.caches_enabled

    def reset_caches(self):
        if self.backbone_cache is not None:
            self.backbone_cache = make_prompt_cache(self.backbone)

        if self.decoder_cache is not None:
            self.decoder_cache = make_prompt_cache(self.decoder)

    def generate_frame(
        self,
        tokens: mx.array,
        tokens_mask: mx.array,
        input_pos: mx.array,
        sampler: Callable[..., mx.array],
    ) -> mx.array:
        assert self.caches_are_enabled(), "backbone caches are not enabled"

        curr_backbone_mask = index_causal_mask(self._backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * mx.expand_dims(tokens_mask, -1)
        h = mx.sum(masked_embeds, axis=2)
        h = self.backbone(h, mask=curr_backbone_mask, cache=self.backbone_cache)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = mx.expand_dims(sampler(c0_logits), axis=-1)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = mx.concat([mx.expand_dims(last_h, 1), c0_embed], axis=1)
        curr_sample = c0_sample
        curr_pos = mx.arange(curr_h.shape[1], dtype=mx.int32)
        curr_pos = mx.expand_dims(curr_pos, 0)
        curr_pos = mx.broadcast_to(curr_pos, (curr_h.shape[0], curr_h.shape[1]))

        # reset decoder cache for new frame

        self.decoder_cache = make_prompt_cache(self.decoder)

        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = index_causal_mask(self._decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h),
                mask=curr_decoder_mask,
                cache=self.decoder_cache,
            )

            ci_logits = mx.matmul(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = mx.expand_dims(sampler(ci_logits), axis=-1)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = mx.concat([curr_sample, ci_sample], axis=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def _embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: mx.array) -> mx.array:
        text_embeds = self.text_embeddings(tokens[:, :, -1])
        text_embeds = mx.expand_dims(text_embeds, axis=-2)

        codebook_indices = mx.arange(self.args.audio_num_codebooks, dtype=mx.int32)
        codebook_offsets = codebook_indices * self.args.audio_vocab_size

        audio_tokens = tokens[:, :, :-1] + mx.reshape(codebook_offsets, (1, 1, -1))
        audio_embeds_flat = self.audio_embeddings(audio_tokens.flatten())

        audio_embeds = mx.reshape(
            audio_embeds_flat,
            (tokens.shape[0], tokens.shape[1], self.args.audio_num_codebooks, -1),
        )

        return mx.concat([audio_embeds, text_embeds], axis=-2)


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


class Model(nn.Module):
    def __init__(
        self,
        config: Dict,
    ):
        self.model = SesameModel(config)
        self.model.setup_caches(1)

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
        sanitized_weights = {}

        for k, v in weights.items():
            if "model." in k:
                k = k.replace("model.", "")

            sanitized_weights[k] = v

        return sanitized_weights

    def load_weights(self, weights, strict: bool = True):
        self.model.load_weights(weights, strict=strict)

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
        self.model.reset_caches()

        # if reference audio is provided, use it as the first segment

        if len(context) == 0 and ref_audio is not None and ref_text is not None:
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

        for _ in tqdm(range(max_audio_frames)):
            sample = self.model.generate_frame(
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
                sample_rate=sample_rate,
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
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )
        ]

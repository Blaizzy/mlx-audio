from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelArgs
from mlx_lm.sample_utils import make_sampler
from scipy import signal
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from transformers import AutoTokenizer

from mlx_audio.audio_io import read as audio_read
from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder

from ..base import GenerationResult
from .attention import Attention

try:
    from .watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
except ImportError:
    pass

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
TOKENIZER_REPO = "unsloth/Llama-3.2-1B"


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled


@dataclass
class DepthDecoderConfig:
    attention_bias: bool
    attention_dropout: float
    backbone_hidden_size: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: str
    num_attention_heads: int
    num_codebooks: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: dict
    rope_theta: int
    use_cache: bool
    vocab_size: int

    def __init__(
        self,
        attention_bias: bool,
        attention_dropout: float,
        backbone_hidden_size: int,
        head_dim: int,
        hidden_act: str,
        hidden_size: int,
        initializer_range: float,
        intermediate_size: int,
        max_position_embeddings: int,
        mlp_bias: bool,
        model_type: str,
        num_attention_heads: int,
        num_codebooks: int,
        num_hidden_layers: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        rope_scaling: dict,
        rope_theta: int,
        use_cache: bool,
        vocab_size: int,
        **kwargs,
    ):
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.backbone_hidden_size = backbone_hidden_size
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.mlp_bias = mlp_bias
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_codebooks = num_codebooks
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.vocab_size = vocab_size


@dataclass
class SesameModelArgs:
    model_type: str
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    attention_bias: bool
    attention_dropout: float
    audio_eos_token_id: int
    audio_token_id: int
    bos_token_id: int
    codebook_eos_token_id: int
    codebook_pad_token_id: int
    depth_decoder_config: DepthDecoderConfig
    head_dim: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    num_attention_heads: int
    num_codebooks: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    rms_norm_eps: float
    rope_scaling: dict
    rope_theta: int
    tie_codebooks_embeddings: bool
    tie_word_embeddings: bool
    use_cache: bool
    vocab_size: int

    def __init__(
        self,
        **kwargs,
    ):
        depth_cfg = kwargs.pop("depth_decoder_config", None)
        rope_cfg = kwargs.pop("rope_scaling", None)

        self.depth_decoder_config = (
            DepthDecoderConfig(
                **{
                    **depth_cfg,
                    "rope_scaling": depth_cfg["rope_scaling"],
                }
            )
            if depth_cfg
            else None
        )
        self.rope_scaling = rope_cfg

        for k, v in kwargs.items():
            setattr(self, k, v)


def create_llama_model_args_for_backbone(cfg) -> LlamaModelArgs:
    return LlamaModelArgs(
        model_type="llama",
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        vocab_size=int(cfg.text_vocab_size),
        max_position_embeddings=cfg.max_position_embeddings,
        attention_bias=cfg.attention_bias,
        mlp_bias=cfg.mlp_bias,
        rope_theta=cfg.rope_theta,
        rope_scaling=cfg.rope_scaling,
    )


def create_llama_model_args_for_decoder(decoder_cfg) -> LlamaModelArgs:
    return LlamaModelArgs(
        model_type="llama",
        num_hidden_layers=decoder_cfg.num_hidden_layers,
        num_attention_heads=decoder_cfg.num_attention_heads,
        num_key_value_heads=decoder_cfg.num_key_value_heads,
        head_dim=decoder_cfg.head_dim,
        hidden_size=decoder_cfg.hidden_size,
        intermediate_size=decoder_cfg.intermediate_size,
        rms_norm_eps=decoder_cfg.rms_norm_eps,
        vocab_size=decoder_cfg.vocab_size,
        max_position_embeddings=decoder_cfg.max_position_embeddings,
        attention_bias=decoder_cfg.attention_bias,
        mlp_bias=decoder_cfg.mlp_bias,
        rope_theta=decoder_cfg.rope_theta,
        rope_scaling=decoder_cfg.rope_scaling,
    )


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

        try:
            backbone_args = create_llama_model_args_for_backbone(args)
            decoder_args = create_llama_model_args_for_decoder(
                args.depth_decoder_config
            )
        except Exception:
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

        self.backbone_cache = None
        self.decoder_cache = None
        self.caches_enabled = False

    def setup_caches(self, max_batch_size: int):
        try:
            backbone_args = create_llama_model_args_for_backbone(self.args)
        except Exception:
            backbone_args = create_llama_model_args(self.args.backbone_flavor)

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

        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * mx.expand_dims(tokens_mask, -1)
        h = mx.sum(masked_embeds, axis=2)
        h = self.backbone(h, cache=self.backbone_cache)

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
            decoder_h = self.decoder(
                self.projection(curr_h),
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
        super().__init__()
        self.model = SesameModel(config)
        self.model.setup_caches(1)

        self.tokenizer_repo = config.get("text_tokenizer")
        if self.tokenizer_repo:
            self._text_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_repo)
        else:
            self._text_tokenizer = load_llama3_tokenizer(TOKENIZER_REPO)

        mimi = Mimi.from_pretrained(MIMI_REPO)
        mimi.eval()

        self._audio_tokenizer = mimi
        self._streaming_decoder = MimiStreamingDecoder(mimi)

        try:
            self._watermarker = load_watermarker()
        except Exception:
            self._watermarker = None

        self._sample_rate = mimi.cfg.sample_rate

    def model_quant_predicate(self, p, m):
        """
        Model modules to skip during quantization
        """
        return not p.startswith("_audio_tokenizer")

    @property
    def layers(self):
        """Return the backbone layers of the model."""
        return self.model.backbone.layers

    @property
    def sample_rate(self):
        return self._sample_rate

    def _tokenize_text_segment(
        self, text: str, speaker: int
    ) -> Tuple[mx.array, mx.array]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(
            f"[{speaker}]{text}", return_tensors="mlx"
        ).squeeze(0)
        text_frame = mx.zeros((len(text_tokens), 33)).astype(mx.int32)
        text_frame_mask = mx.zeros((len(text_tokens), 33)).astype(mx.bool_)
        text_frame[:, -1] = text_tokens
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame)
        frame_masks.append(text_frame_mask)

        return mx.concat(frame_tokens, axis=0), mx.concat(frame_masks, axis=0)

    def _tokenize_audio(
        self, audio: mx.array, add_eos: bool = True
    ) -> Tuple[mx.array, mx.array]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio_tokens = self._audio_tokenizer.encode(audio[None, None, ...])[0]

        # add EOS frame
        if add_eos:
            eos_frame = mx.zeros((audio_tokens.shape[0], 1))
            audio_tokens = mx.concat([audio_tokens, eos_frame], axis=1)

        audio_frame = mx.zeros((audio_tokens.shape[1], 33)).astype(mx.int32)
        audio_frame_mask = mx.zeros((audio_tokens.shape[1], 33)).astype(mx.bool_)
        audio_frame[:, :-1] = audio_tokens.swapaxes(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return mx.concat(frame_tokens, axis=0), mx.concat(frame_masks, axis=0)

    def _tokenize_segment(
        self, segment: Segment, add_eos: bool = True
    ) -> Tuple[mx.array, mx.array]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(
            segment.text, segment.speaker
        )
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio, add_eos=add_eos)

        return mx.concat([text_tokens, audio_tokens], axis=0), mx.concat(
            [text_masks, audio_masks], axis=0
        )

    def sanitize(self, weights):
        sanitized_weights = {}

        for k, v in weights.items():
            if not k.startswith("model."):
                k = "model." + k

            if "attn" in k and not "self_attn" in k:
                k = k.replace("attn", "self_attn")
                k = k.replace("output_proj", "o_proj")

            if "mlp" in k:
                k = k.replace("w1", "gate_proj")
                k = k.replace("w2", "down_proj")
                k = k.replace("w3", "up_proj")

            if "sa_norm" in k or "mlp_norm" in k:
                k = k.replace("sa_norm", "input_layernorm").replace("scale", "weight")
                k = k.replace("mlp_norm", "post_attention_layernorm").replace(
                    "scale", "weight"
                )

            if "decoder.norm" in k or "backbone.norm" in k:
                k = k.replace("scale", "weight")

            sanitized_weights[k] = v

        return sanitized_weights

    def prepare_prompt(
        self, text: str, speaker: int, audio_path: str, sample_rate: int
    ) -> Segment:
        audio, sr = audio_read(audio_path)
        if sr != sample_rate:
            audio = resample_audio(audio, sr, sample_rate)
        return Segment(text=text, speaker=speaker, audio=mx.array(audio))

    def default_speaker_prompt(
        self, voice: str, repo_id="sesame/csm-1b"
    ) -> List[Segment]:
        SPEAKER_PROMPTS = {
            "conversational_a": {
                "text": (
                    "like revising for an exam I'd have to try and like keep up the momentum because I'd "
                    "start really early I'd be like okay I'm gonna start revising now and then like "
                    "you're revising for ages and then I just like start losing steam I didn't do that "
                    "for the exam we had recently to be fair that was a more of a last minute scenario "
                    "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
                    "sort of start the day with this not like a panic but like a"
                ),
            },
            "conversational_b": {
                "text": (
                    "like a super Mario level. Like it's very like high detail. And like, once you get "
                    "into the park, it just like, everything looks like a computer game and they have all "
                    "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
                    "will have like a question block. And if you like, you know, punch it, a coin will "
                    "come out. So like everyone, when they come into the park, they get like this little "
                    "bracelet and then you can go punching question blocks around."
                ),
            },
        }

        prompt_path = hf_hub_download(repo_id=repo_id, filename=f"prompts/{voice}.wav")

        try:
            prompt_text_path = hf_hub_download(
                repo_id=repo_id, filename=f"prompts/{voice}.txt"
            )
            prompt_text = Path(prompt_text_path).read_text()
        except Exception:
            prompt_text = SPEAKER_PROMPTS[voice]["text"]

        prompt = self.prepare_prompt(prompt_text, 0, prompt_path, 24_000)
        return [prompt]

    def generate_result(
        self, samples, start_time: float, stream: bool = False
    ) -> GenerationResult:
        token_count = len(samples)
        transposed = mx.transpose(mx.stack(samples), axes=[1, 2, 0])

        # decode in 10 second max chunks to avoid excessive memory usage
        tokens_per_batch = min(token_count, int(12.5 * 5))
        all_audio = []
        for i in range(0, transposed.shape[2], tokens_per_batch):
            batch_tokens = transposed[:, :, i : i + tokens_per_batch]
            audio = (
                self._streaming_decoder.decode_frames(batch_tokens)
                .squeeze(0)
                .squeeze(0)
            )
            all_audio.append(audio)
        audio = mx.concat(all_audio, axis=0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        if self._watermarker is not None:
            audio = watermark(
                self._watermarker,
                audio,
                self._sample_rate,
                CSM_1B_GH_WATERMARK,
            )
            audio = mx.array(audio, dtype=mx.float32)

        mx.eval(audio)

        segment_time = time.perf_counter() - start_time

        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0, "No audio generated"

        # Calculate audio duration in seconds
        sample_rate = 24000
        audio_duration_seconds = samples / sample_rate

        # Calculate real-time factor (RTF)
        rtf = segment_time / audio_duration_seconds if audio_duration_seconds > 0 else 0

        # Format duration as HH:MM:SS.mmm
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        return GenerationResult(
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

    def generate(
        self,
        text: List[str] | str,
        voice: Optional[str] = None,
        speaker: int = 0,
        context: List[Segment] = [],
        split_pattern: Optional[str] = r"\n+",
        sampler: Callable[..., mx.array] = None,
        max_audio_length_ms: float = 90_000,
        ref_audio: mx.array = None,
        ref_text: str = None,
        stream: bool = False,
        streaming_interval: float = 0.5,
        voice_match: bool = True,
        **kwargs,
    ):
        # if reference audio is provided, use it as the first segment
        if len(context) == 0 and ref_audio is not None and ref_text is not None:
            context = [Segment(speaker=speaker, text=ref_text, audio=ref_audio)]
        elif ref_audio is None:
            # otherwise, use the provided or default voice
            if voice is None:
                voice = "conversational_a"
            context = self.default_speaker_prompt(
                voice,
                repo_id=(
                    "sesame/csm-1b" if not self.tokenizer_repo else self.tokenizer_repo
                ),
            )

        sampler = sampler or make_sampler(temp=0.9, top_k=50)
        max_audio_frames = int(max_audio_length_ms / 80)
        streaming_interval_tokens = int(streaming_interval * 12.5)

        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]

        for prompt in text:
            if voice_match:
                generation_text = (context[0].text + " " + prompt).strip()
                current_context = [
                    Segment(
                        speaker=speaker, text=generation_text, audio=context[0].audio
                    )
                ]

            start_time = time.perf_counter()

            self.model.reset_caches()
            if stream:
                self._streaming_decoder.reset()

            tokens, tokens_mask = [], []
            for segment in current_context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(
                    segment, add_eos=not voice_match
                )
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

            if not voice_match:
                gen_segment_tokens, gen_segment_tokens_mask = (
                    self._tokenize_text_segment(prompt, speaker)
                )
                tokens.append(gen_segment_tokens)
                tokens_mask.append(gen_segment_tokens_mask)

            prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int32)
            prompt_tokens_mask = mx.concat(tokens_mask, axis=0).astype(mx.bool_)

            samples = []
            curr_tokens = mx.expand_dims(prompt_tokens, axis=0)
            curr_tokens_mask = mx.expand_dims(prompt_tokens_mask, axis=0)
            curr_pos = mx.expand_dims(
                mx.arange(0, prompt_tokens.shape[0]), axis=0
            ).astype(mx.int32)
            generated_frame_count = 0
            yielded_frame_count = 0

            max_seq_len = 2048 - max_audio_frames
            if curr_tokens.shape[1] >= max_seq_len:
                raise ValueError(
                    f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}"
                )

            with tqdm() as pbar:
                for _ in range(max_audio_frames):
                    sample = self.model.generate_frame(
                        curr_tokens, curr_tokens_mask, curr_pos, sampler
                    )
                    if mx.all(sample == 0):
                        break  # eos

                    samples.append(sample)

                    curr_tokens = mx.expand_dims(
                        mx.concat([sample, mx.zeros((1, 1)).astype(mx.int32)], axis=1),
                        axis=1,
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
                    generated_frame_count += 1
                    pbar.update()

                    # send a partial result in streaming mode
                    if (
                        stream
                        and (generated_frame_count - yielded_frame_count)
                        >= streaming_interval_tokens
                    ):
                        yielded_frame_count = generated_frame_count
                        yield self.generate_result(samples, start_time, stream=True)
                        samples = []
                        start_time = time.perf_counter()

                if len(samples) > 0:
                    yield self.generate_result(samples, start_time, stream=stream)

                # Clear cache after each segment to avoid memory leaks
                mx.clear_cache()

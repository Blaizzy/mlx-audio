import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import dacite
import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
import tqdm
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gpt2 import ModelArgs as GPT2Args
from mlx_lm.sample_utils import make_sampler

from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.indextts import normalize
from mlx_audio.tts.models.indextts.attention import LearnedPositionEncoding
from mlx_audio.tts.models.indextts.bigvgan import (
    BigVGANConditioning,
    BigVGANConditioningConfig,
)
from mlx_audio.tts.models.indextts.conformer import Conformer, ConformerArgs
from mlx_audio.tts.models.indextts.gpt2 import GPT2Model
from mlx_audio.tts.models.indextts.mel import log_mel_spectrogram
from mlx_audio.tts.models.indextts.perceiver import PerceiverResampler


@dataclass
class GPTConfig:
    model_dim: int
    heads: int
    layers: int
    max_mel_tokens: int
    max_text_tokens: int

    # special tokens
    number_text_tokens: int
    number_mel_codes: int
    start_mel_token: int
    stop_mel_token: int
    start_text_token: int
    stop_text_token: int

    # conditioner
    use_mel_codes_as_input: bool
    mel_length_compression: int
    condition_type: str
    condition_module: ConformerArgs
    max_conditioning_inputs: int = 1
    condition_num_latent: int = 32


@dataclass
class ModelArgs:
    bigvgan: BigVGANConditioningConfig
    gpt: GPTConfig
    tokenizer_name: str | Path
    sample_rate: int = 24000


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        if isinstance(args, dict):
            args = dacite.from_dict(ModelArgs, args)

        if not args.gpt.use_mel_codes_as_input:
            raise NotImplementedError(
                "use_mel_codes_as_input=false is not supported. Please open a new issue in mlx-audio to get this model supported."
            )
        if args.gpt.condition_type != "conformer_perceiver":
            raise NotImplementedError(
                f"condition_type={args.gpt.condition_type} is not supported. Please open a new issue in mlx-audio to get this model supported."
            )

        self.args = args
        self.sample_rate = args.sample_rate

        try:
            self.tokenizer = spm.SentencePieceProcessor(
                model_file=huggingface_hub.hf_hub_download(  # type: ignore
                    str(args.tokenizer_name), "tokenizer.model"
                )
            )
        except Exception:
            self.tokenizer = spm.SentencePieceProcessor(
                model_file=str(  # type: ignore
                    (Path(args.tokenizer_name) / "tokenizer.model").resolve()
                )
            )

        self.bigvgan = BigVGANConditioning(args.bigvgan)

        self.text_embedding = nn.Embedding(
            args.gpt.number_text_tokens + 1, args.gpt.model_dim
        )
        self.mel_embedding = nn.Embedding(args.gpt.number_mel_codes, args.gpt.model_dim)
        self.mel_pos_embedding = LearnedPositionEncoding(
            args.gpt.max_mel_tokens + 2 + args.gpt.max_conditioning_inputs,
            args.gpt.model_dim,
        )
        self.text_pos_embedding = LearnedPositionEncoding(
            args.gpt.max_text_tokens + 2, args.gpt.model_dim
        )

        self.text_head = nn.Linear(args.gpt.model_dim, args.gpt.number_text_tokens + 1)
        self.mel_head = nn.Linear(args.gpt.model_dim, args.gpt.number_mel_codes)

        self.conditioning_encoder = Conformer(args.gpt.condition_module)
        self.perceiver_encoder = PerceiverResampler(
            args.gpt.model_dim,
            n_dim_context=args.gpt.condition_module.output_size,
            n_ff_mult=args.gpt.condition_module.perceiver_mult,
            n_heads=args.gpt.condition_module.attention_heads,
            n_latents=args.gpt.condition_num_latent,
        )
        self.gpt = GPT2Model(
            GPT2Args(
                "gpt2",
                1,
                args.gpt.model_dim,
                args.gpt.heads,
                args.gpt.layers,
                1,
                1e-5,
                1,
            )
        )

        self.final_norm = nn.LayerNorm(args.gpt.model_dim)

        # patching
        self.gpt.wpe = nn.Identity()  # type: ignore
        self.gpt.wte = nn.Identity()  # type: ignore

    def sanitize(self, weights: dict[str, mx.array]):
        already_sanitized = all(
            ("num_batches_tracked" not in key) for key in weights.keys()
        )
        if already_sanitized:
            return weights

        bigvgan_prefixes = [
            "ups.",
            "speaker_encoder.",
            "resblocks.",
            "conv_pre.",
            "conv_post.",
            "conds.",
            "cond_layer.",
            "activation_post.",
        ]

        gpt_weights = {
            k: v
            for k, v in weights.items()
            if not any(k.startswith(prefix) for prefix in bigvgan_prefixes)
        }
        bigvgan_weights = {
            k: v
            for k, v in weights.items()
            if any(k.startswith(prefix) for prefix in bigvgan_prefixes)
        }

        new_gpt_weights = {}

        for key, value in gpt_weights.items():
            if "pos_enc" in key:
                continue  # it should calculate self

            if "conv" in key:
                if value.ndim == 3:
                    value = value.transpose(0, 2, 1)
                elif value.ndim == 4:
                    value = value.transpose(0, 2, 3, 1)

            if "perceiver_encoder.norm.gamma" in key:
                key = "perceiver_encoder.norm.weight"

            new_gpt_weights[key] = value

        for i in range(self.args.gpt.layers):
            if f"gpt.h.{i}.attn.bias" in new_gpt_weights:
                del new_gpt_weights[f"gpt.h.{i}.attn.bias"]
            if f"gpt.h.{i}.attn.c_attn.weight" in new_gpt_weights:
                new_gpt_weights[f"gpt.h.{i}.attn.c_attn.weight"] = new_gpt_weights[
                    f"gpt.h.{i}.attn.c_attn.weight"
                ].transpose(1, 0)
            if f"gpt.h.{i}.attn.c_proj.weight" in new_gpt_weights:
                new_gpt_weights[f"gpt.h.{i}.attn.c_proj.weight"] = new_gpt_weights[
                    f"gpt.h.{i}.attn.c_proj.weight"
                ].transpose(1, 0)
            if f"gpt.h.{i}.mlp.c_fc.weight" in new_gpt_weights:
                new_gpt_weights[f"gpt.h.{i}.mlp.c_fc.weight"] = new_gpt_weights[
                    f"gpt.h.{i}.mlp.c_fc.weight"
                ].transpose(1, 0)
            if f"gpt.h.{i}.mlp.c_proj.weight" in new_gpt_weights:
                new_gpt_weights[f"gpt.h.{i}.mlp.c_proj.weight"] = new_gpt_weights[
                    f"gpt.h.{i}.mlp.c_proj.weight"
                ].transpose(1, 0)

        for i in range(2):  # hard coded in original impl
            if f"perceiver_encoder.layers.{i}.0.to_q.weight" in new_gpt_weights:
                new_gpt_weights[f"perceiver_encoder.layers.{i}.0.linear_q.weight"] = (
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.0.to_q.weight"]
                )
                del new_gpt_weights[f"perceiver_encoder.layers.{i}.0.to_q.weight"]
            if f"perceiver_encoder.layers.{i}.0.to_kv.weight" in new_gpt_weights:
                (
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.0.linear_k.weight"],
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.0.linear_v.weight"],
                ) = mx.split(
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.0.to_kv.weight"],
                    2,
                    axis=0,
                )
                del new_gpt_weights[f"perceiver_encoder.layers.{i}.0.to_kv.weight"]
            if f"perceiver_encoder.layers.{i}.0.to_out.weight" in new_gpt_weights:
                new_gpt_weights[f"perceiver_encoder.layers.{i}.0.linear_out.weight"] = (
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.0.to_out.weight"]
                )
                del new_gpt_weights[f"perceiver_encoder.layers.{i}.0.to_out.weight"]

            if f"perceiver_encoder.layers.{i}.1.0.weight" in new_gpt_weights:
                new_gpt_weights[f"perceiver_encoder.layers.{i}.1.w_1.weight"] = (
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.1.0.weight"]
                )
                del new_gpt_weights[f"perceiver_encoder.layers.{i}.1.0.weight"]
            if f"perceiver_encoder.layers.{i}.1.2.weight" in new_gpt_weights:
                new_gpt_weights[f"perceiver_encoder.layers.{i}.1.w_2.weight"] = (
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.1.2.weight"]
                )
                del new_gpt_weights[f"perceiver_encoder.layers.{i}.1.2.weight"]
            if f"perceiver_encoder.layers.{i}.1.0.bias" in new_gpt_weights:
                new_gpt_weights[f"perceiver_encoder.layers.{i}.1.w_1.bias"] = (
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.1.0.bias"]
                )
                del new_gpt_weights[f"perceiver_encoder.layers.{i}.1.0.bias"]
            if f"perceiver_encoder.layers.{i}.1.2.bias" in new_gpt_weights:
                new_gpt_weights[f"perceiver_encoder.layers.{i}.1.w_2.bias"] = (
                    new_gpt_weights[f"perceiver_encoder.layers.{i}.1.2.bias"]
                )
                del new_gpt_weights[f"perceiver_encoder.layers.{i}.1.2.bias"]

        new_bigvgan_weight = {
            "bigvgan." + k: v for k, v in self.bigvgan.sanitize(bigvgan_weights).items()
        }

        return {**new_gpt_weights, **new_bigvgan_weight}

    def get_conditioning(self, mel: mx.array) -> mx.array:  # (b, c, t)
        latent = self.conditioning_encoder(mel)
        return self.perceiver_encoder(latent)

    def prepare_input_embedding(
        self,
        prompts: List[str],
        ref_audio: Optional[mx.array],
        ref_mel: Optional[mx.array] = None,
    ) -> mx.array:
        if ref_audio is not None:
            ref_mel = log_mel_spectrogram(ref_audio)

        if ref_mel is None:
            raise ValueError("Must provide one of ref_audio or ref_mel")

        conditioning = self.get_conditioning(ref_mel)
        # for case with multiple batch, and single ref_audio
        conditioning = mx.repeat(conditioning, len(prompts), axis=0)

        tokenized = [
            self.tokenizer.encode(
                normalize.tokenize_by_CJK_char(normalize.normalize(prompt))
            )
            for prompt in prompts
        ]  # type: ignore

        longest = max((len(tokens) for tokens in tokenized)) + 3

        embedding = mx.zeros(
            (len(tokenized), longest + conditioning.shape[1], self.args.gpt.model_dim)
        )

        for idx, tokens in enumerate(tokenized):
            # append tokens
            tokens.insert(0, self.args.gpt.start_text_token)
            tokens.append(self.args.gpt.stop_text_token)
            tokens.append(self.args.gpt.start_mel_token)
            length = len(tokens)

            tokens = mx.array(tokens)[None, :]

            text_embedding = self.text_embedding(tokens) + self.text_pos_embedding(
                tokens
            )
            embedding[idx : idx + 1, longest - length :, :] = mx.concat(
                [conditioning, text_embedding], axis=1
            )

        return embedding

    def generate_result(
        self,
        audio: mx.array,
        start_time: float,
        token_count: int,
        **kwargs,
    ) -> GenerationResult:
        audio = audio.squeeze(0).squeeze(0)

        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0, "No audio generated"

        sample_rate = self.sample_rate
        audio_duration_seconds = samples / sample_rate

        elapsed_time = time.perf_counter() - start_time
        rtf = audio_duration_seconds / elapsed_time

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
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },  # type: ignore
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def generate(
        self,
        text: str,
        ref_audio: Optional[mx.array],
        ref_mel: Optional[mx.array] = None,
        verbose: bool = False,
        max_tokens: int = 5000,
        sampler: Optional[Callable[..., mx.array]] = None,
        **kwargs,
    ):
        if ref_audio is not None:
            ref_mel = log_mel_spectrogram(ref_audio)

        if ref_mel is None:
            raise ValueError("Must provide one of ref_audio or ref_mel")

        time_start = time.perf_counter()

        embedding = self.prepare_input_embedding([text], None, ref_mel)

        cache = [KVCache() for _ in range(self.args.gpt.layers)]
        sampler = sampler or make_sampler(temp=0.8, top_k=30)

        inputs = embedding
        generated_tokens = []
        latent_states = []

        mel_position = 0

        for _ in range(max_tokens) if not verbose else tqdm.trange(max_tokens):
            hidden_states = self.gpt(inputs, cache=cache)

            hidden_states = self.final_norm(hidden_states)

            latent_states.append(hidden_states[:, -1:, :])
            mel_logits = self.mel_head(hidden_states[:, -1:, :])

            next_token = sampler(mel_logits)

            if next_token.item() == self.args.gpt.stop_mel_token:
                break

            generated_tokens.append(next_token.item())

            mel_emb = self.mel_embedding(next_token) + self.mel_pos_embedding(
                next_token, embedding.shape[1] + mel_position
            )

            inputs = mel_emb
            mel_position += 1

        latent_states = mx.concat(latent_states, axis=-2)

        audio = self.bigvgan(
            latent_states.transpose(0, 2, 1),
            ref_mel.transpose(0, 2, 1),
        )

        yield self.generate_result(audio, time_start, latent_states.shape[1])

        mx.clear_cache()

from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gpt2 import GPT2Model
from mlx_lm.models.gpt2 import ModelArgs as GPT2Args

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGANConfig
from mlx_audio.tts.models.indextts.attention import LearnedPositionEncoding
from mlx_audio.tts.models.indextts.conformer import Conformer, ConformerArgs
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
    bigvgan: BigVGANConfig
    gpt: GPTConfig


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        if not args.gpt.use_mel_codes_as_input:
            raise NotImplementedError(
                "use_mel_codes_as_input=false is not supported. Please open a new issue in mlx-audio to get this model supported."
            )
        if args.gpt.condition_type != "conformer_perceiver":
            raise NotImplementedError(
                f"condition_type={args.gpt.condition_type} is not supported. Please open a new issue in mlx-audio to get this model supported."
            )

        self.args = args

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
        new_weights = {}

        for key, value in weights.items():
            if "pos_enc" in key:
                continue  # it should calculate self

            if "conv" in key:
                if value.ndim == 3:
                    value = value.transpose(0, 2, 1)
                elif value.ndim == 4:
                    value = value.transpose(0, 2, 3, 1)

            if "perceiver_encoder.norm.gamma" in key:
                key = "perceiver_encoder.norm.weight"

            new_weights[key] = value

        for i in range(self.args.gpt.layers):
            if f"gpt.h.{i}.attn.bias" in new_weights:
                del new_weights[f"gpt.h.{i}.attn.bias"]
            if f"gpt.h.{i}.attn.c_attn.weight" in new_weights:
                new_weights[f"gpt.h.{i}.attn.c_attn.weight"] = new_weights[
                    f"gpt.h.{i}.attn.c_attn.weight"
                ].transpose(1, 0)
            if f"gpt.h.{i}.attn.c_proj.weight" in new_weights:
                new_weights[f"gpt.h.{i}.attn.c_proj.weight"] = new_weights[
                    f"gpt.h.{i}.attn.c_proj.weight"
                ].transpose(1, 0)
            if f"gpt.h.{i}.mlp.c_fc.weight" in new_weights:
                new_weights[f"gpt.h.{i}.mlp.c_fc.weight"] = new_weights[
                    f"gpt.h.{i}.mlp.c_fc.weight"
                ].transpose(1, 0)
            if f"gpt.h.{i}.mlp.c_proj.weight" in new_weights:
                new_weights[f"gpt.h.{i}.mlp.c_proj.weight"] = new_weights[
                    f"gpt.h.{i}.mlp.c_proj.weight"
                ].transpose(1, 0)

        for i in range(2):  # hard coded in original impl
            # attn
            if f"perceiver_encoder.layers.{i}.0.to_q.weight" in new_weights:
                new_weights[f"perceiver_encoder.layers.{i}.0.linear_q.weight"] = (
                    new_weights[f"perceiver_encoder.layers.{i}.0.to_q.weight"]
                )
                del new_weights[f"perceiver_encoder.layers.{i}.0.to_q.weight"]
            if f"perceiver_encoder.layers.{i}.0.to_kv.weight" in new_weights:
                (
                    new_weights[f"perceiver_encoder.layers.{i}.0.linear_k.weight"],
                    new_weights[f"perceiver_encoder.layers.{i}.0.linear_v.weight"],
                ) = mx.split(
                    new_weights[f"perceiver_encoder.layers.{i}.0.to_kv.weight"],
                    2,
                    axis=0,
                )
                del new_weights[f"perceiver_encoder.layers.{i}.0.to_kv.weight"]
            if f"perceiver_encoder.layers.{i}.0.to_out.weight" in new_weights:
                new_weights[f"perceiver_encoder.layers.{i}.0.linear_out.weight"] = (
                    new_weights[f"perceiver_encoder.layers.{i}.0.to_out.weight"]
                )
                del new_weights[f"perceiver_encoder.layers.{i}.0.to_out.weight"]

            # ffn
            if f"perceiver_encoder.layers.{i}.1.0.weight" in new_weights:
                new_weights[f"perceiver_encoder.layers.{i}.1.w_1.weight"] = new_weights[
                    f"perceiver_encoder.layers.{i}.1.0.weight"
                ]
                del new_weights[f"perceiver_encoder.layers.{i}.1.0.weight"]
            if f"perceiver_encoder.layers.{i}.1.2.weight" in new_weights:
                new_weights[f"perceiver_encoder.layers.{i}.1.w_2.weight"] = new_weights[
                    f"perceiver_encoder.layers.{i}.1.2.weight"
                ]
                del new_weights[f"perceiver_encoder.layers.{i}.1.2.weight"]
            if f"perceiver_encoder.layers.{i}.1.0.bias" in new_weights:
                new_weights[f"perceiver_encoder.layers.{i}.1.w_1.bias"] = new_weights[
                    f"perceiver_encoder.layers.{i}.1.0.bias"
                ]
                del new_weights[f"perceiver_encoder.layers.{i}.1.0.bias"]
            if f"perceiver_encoder.layers.{i}.1.2.bias" in new_weights:
                new_weights[f"perceiver_encoder.layers.{i}.1.w_2.bias"] = new_weights[
                    f"perceiver_encoder.layers.{i}.1.2.bias"
                ]
                del new_weights[f"perceiver_encoder.layers.{i}.1.2.bias"]

        return new_weights

    def get_conditioning(self, mel: mx.array) -> mx.array:
        latent = self.conditioning_encoder(mel)
        return self.perceiver_encoder(latent)

    def prepare_input_ids(
        self,
        prompts: List[str],
        conditioned: mx.array,
    ):
        pass

    def generate(
        self,
        text: str,
        voice: mx.array,
        speed: float = 1.0,
        **kwargs,
    ):
        mel = voice
        conditioned = self.get_conditioning(mel[None, :, :])
        pass

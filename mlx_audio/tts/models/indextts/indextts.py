from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gpt2 import ModelArgs as GPT2Args

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGANConfig
from mlx_audio.tts.models.indextts.attention import LearnedPositionEncoding
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

        self.tokenizer = spm.SentencePieceProcessor(
            model_file="./bpe.model"  # type: ignore
        )  # temporary

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

    def get_conditioning(self, mel: mx.array) -> mx.array:  # (b, c, t)
        latent = self.conditioning_encoder(mel)
        return self.perceiver_encoder(latent)

    def prepare_input_embedding(
        self,
        prompts: List[str],
        ref_audio: mx.array,
    ) -> mx.array:
        conditioning = self.get_conditioning(log_mel_spectrogram(ref_audio))
        # for case with multiple batch, and single ref_audio
        conditioning = mx.broadcast_to(
            conditioning, (len(prompts), *conditioning.shape[1:])
        )
        tokenized = [self.tokenizer.encode(prompt) for prompt in prompts]  # type: ignore

        longest = max((len(tokens) for tokens in tokenized)) + 2

        embedding = mx.zeros(
            (len(tokenized), longest + conditioning.shape[1], self.args.gpt.model_dim)
        )

        for idx, tokens in enumerate(tokenized):
            # append tokens
            tokens.insert(0, self.args.gpt.start_text_token)
            tokens.append(self.args.gpt.start_mel_token)
            length = len(tokens)

            tokens = mx.array(tokens)[None, :]

            text_embedding = self.text_embedding(tokens) + self.text_pos_embedding(
                tokens
            )
            embedding[idx : idx + 1, longest - length :, :] = mx.concat(
                [conditioning[idx : idx + 1], text_embedding], axis=1
            )

        return embedding

    def generate(
        self,
        text: str,
        ref_audio: mx.array,
        max_tokens: int = 5000,
        temperature: float = 1.0,
        **kwargs,
    ):
        embedding = self.prepare_input_embedding([text], ref_audio)

        cache = [KVCache() for _ in range(self.args.gpt.layers)]

        inputs = embedding
        generated_tokens = []
        latent_states = []

        mel_position = 0

        for _ in range(max_tokens):
            hidden_states = self.gpt(inputs, cache=cache)

            hidden_states = self.final_norm(hidden_states)

            latent_states.append(hidden_states[:, -1:, :])
            mel_logits = self.mel_head(hidden_states[:, -1:, :])

            if temperature > 0:
                probs = mx.softmax(mel_logits / temperature, axis=-1)
                next_token = mx.random.categorical(mx.log(probs))
            else:
                next_token = mx.argmax(mel_logits, axis=-1)

            if next_token.item() == self.args.gpt.stop_mel_token:
                break

            generated_tokens.append(next_token.item())

            mel_emb = self.mel_embedding(next_token)

            position = mx.array([[embedding.shape[1] + mel_position]])
            mel_emb = mel_emb + self.mel_pos_embedding(position)

            inputs = mel_emb
            mel_position += 1

        latent_states = mx.concat(latent_states, axis=-2)

        return mx.array(generated_tokens)  # TODO: bigvgan decode

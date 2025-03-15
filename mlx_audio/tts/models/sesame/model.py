from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import Model as LlamaBaseModel
from mlx_lm.models.llama import ModelArgs as LlamaModelArgs


def create_causal_mask(seq_len: int) -> mx.array:
    return mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))


def index_causal_mask(mask: mx.array, input_pos: mx.array) -> mx.array:
    mask_indexed = mx.take(mask, input_pos, axis=0)

    seq_len = input_pos.shape[1]
    mask_indexed = mask_indexed[:, :, :seq_len]

    # reshape to (batch_size, 1, seq_len, seq_len) for broadcasting across heads
    return mx.expand_dims(mask_indexed, axis=1)


def multinomial_sample_one(probs: mx.array) -> mx.array:
    return mx.random.categorical(probs, num_samples=1)


def sample_topk(logits: mx.array, topk: int, temperature: float) -> mx.array:
    logits = logits / temperature
    top_values = mx.topk(logits, k=topk, axis=-1)
    threshold = top_values[:, topk - 1 : topk]
    mask = logits < threshold
    logits = mx.where(mask, mx.ones_like(logits) * -float("inf"), logits)
    probs = mx.softmax(logits, axis=-1)
    sample = multinomial_sample_one(probs)
    return sample


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    rope_theta: float = 10000
    rope_traditional: bool = False
    max_position_embeddings: int = 2048
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False


def create_llama_model_args(flavor: str) -> LlamaModelArgs:
    if flavor == "llama-1B":
        return LlamaModelArgs(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=16,
            intermediate_size=8192,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            vocab_size=128_256,
            max_position_embeddings=2048,
            rope_theta=500_000,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
        )
    elif flavor == "llama-100M":
        return LlamaModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=8192,
            num_attention_heads=8,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=128_256,
            max_position_embeddings=2048,
            rope_theta=500_000,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
        )
    else:
        raise ValueError(f"Unknown flavor: {flavor}")


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x


def prepare_transformer(model: LlamaBaseModel) -> Tuple[LlamaBaseModel, int]:
    hidden_size = model.args.hidden_size
    model.model.embed_tokens = Identity()
    model.lm_head = Identity()
    return model, hidden_size


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        backbone_args = create_llama_model_args(args.backbone_flavor)
        decoder_args = create_llama_model_args(args.decoder_flavor)

        self.backbone, backbone_dim = prepare_transformer(LlamaBaseModel(backbone_args))
        self.decoder, decoder_dim = prepare_transformer(LlamaBaseModel(decoder_args))

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
            for cache in self.backbone_cache:
                if hasattr(cache, "keys"):
                    cache.keys = None
                    cache.values = None
                    cache.offset = 0

        if self.decoder_cache is not None:
            for cache in self.decoder_cache:
                if hasattr(cache, "keys"):
                    cache.keys = None
                    cache.values = None
                    cache.offset = 0

    def generate_frame(
        self,
        tokens: mx.array,
        tokens_mask: mx.array,
        input_pos: mx.array,
        temperature: float,
        topk: int,
    ) -> mx.array:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len)
            temperature: sampling temperature
            topk: top-k sampling parameter

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        assert self.caches_are_enabled(), "backbone caches are not enabled"

        curr_backbone_mask = index_causal_mask(self._backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * mx.expand_dims(tokens_mask, -1)
        h = mx.sum(masked_embeds, axis=2)
        h = self.backbone(h, mask=curr_backbone_mask, cache=self.backbone_cache)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = mx.concat([mx.expand_dims(last_h, 1), c0_embed], axis=1)
        curr_sample = mx.array(c0_sample)
        curr_pos = mx.arange(curr_h.shape[1], dtype=mx.int32)
        curr_pos = mx.expand_dims(curr_pos, 0)
        curr_pos = mx.broadcast_to(curr_pos, (curr_h.shape[0], curr_h.shape[1]))

        # reset decoder cache for new frame

        for cache in self.decoder_cache:
            if hasattr(cache, "keys"):
                cache.keys = None
                cache.values = None
                cache.offset = 0

        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = index_causal_mask(self._decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h),
                mask=curr_decoder_mask,
                cache=self.decoder_cache,
            )

            ci_logits = mx.matmul(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = mx.concat([curr_sample, ci_sample], axis=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def _embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: mx.array) -> mx.array:
        text_embeds = self.text_embeddings(tokens[:, :, -1])
        text_embeds = mx.expand_dims(text_embeds, -2)

        codebook_indices = mx.arange(self.args.audio_num_codebooks, dtype=mx.int32)
        codebook_offsets = codebook_indices * self.args.audio_vocab_size

        audio_tokens = tokens[:, :, :-1] + mx.reshape(codebook_offsets, (1, 1, -1))
        audio_tokens_flat = mx.reshape(audio_tokens, (-1,))
        audio_embeds_flat = self.audio_embeddings(audio_tokens_flat)

        audio_embeds = mx.reshape(
            audio_embeds_flat,
            (tokens.shape[0], tokens.shape[1], self.args.audio_num_codebooks, -1),
        )

        return mx.concat([audio_embeds, text_embeds], axis=-2)

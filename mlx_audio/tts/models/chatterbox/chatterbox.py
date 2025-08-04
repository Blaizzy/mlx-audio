# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import torch
import torch.nn.functional as F
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelConfig
from tqdm import tqdm
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

from .cond_enc import T3Cond, T3CondEnc
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from .inference.t3_hf_backend import T3HuggingfaceBackend


@dataclass
class ModelConfig(LlamaModelConfig):
    tokenizer_name: str = "ResembleAI/chatterbox"
    sample_rate: int = 24000
    vocab_size: int = 8
    max_position_embeddings: int = 131072
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    attn_implementation: str = "sdpa"
    head_dim: int = 64
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    mlp_bias: bool = False
    model_type: str = "llama"
    num_key_value_heads: int = 16
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    rope_scaling: dict = None
    rope_theta: float = 500000.0
    torch_dtype: str = "bfloat16"
    use_cache: bool = True

    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = dict(
                factor=8.0,
                high_freq_factor=4.0,
                low_freq_factor=1.0,
                original_max_position_embeddings=8192,
                rope_type="llama3",
            )
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


def _ensure_BOT_EOT(text_tokens: mx.array, hp):
    B = text_tokens.size(0)
    assert (
        text_tokens == hp.start_text_token
    ).int().sum() >= B, "missing start_text_token"
    assert (
        text_tokens == hp.stop_text_token
    ).int().sum() >= B, "missing stop_text_token"


from typing import Union

import torch
from torch import Tensor, nn


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)

    def __call__(self, x):
        """
        Returns positional embeddings for index 0 up to the length of x
        """
        sl = x.shape[1]
        return self.emb(mx.arange(0, sl)).astype(self.emb.weight.dtype)

    def get_fixed_embedding(self, idx: "Union[int, Tensor]"):
        """
        Args:
            idx: scalar int or an integer tensor of shape (T,) or (B, T)
        Returns:
            positional embeddings for given indices, shape (B, T, dim), ie (1, 1, dim) for int input
        """
        idx = idx if isinstance(idx, mx.array) else mx.array(idx)
        idx = mx.atleast_2d(idx)
        assert idx.ndim == 2
        return self.emb(idx).astype(self.emb.weight.dtype)  # (B, T, dim)


class Model(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config
        self.tfmr = LlamaModel(self.config)
        self.dim = self.config.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(self.config)
        self.text_emb = nn.Embedding(self.config.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.config.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if self.config.input_pos_emb == "learned":
            max_text_seq_len = self.config.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = self.config.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(
            self.dim, self.config.text_tokens_dict_size, bias=False
        )
        self.speech_head = nn.Linear(
            self.dim, self.config.speech_tokens_dict_size, bias=False
        )
        self.compiled = False

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if (
            t3_cond.cond_prompt_speech_tokens is not None
            and t3_cond.cond_prompt_speech_emb is None
        ):
            t3_cond.cond_prompt_speech_emb = self.speech_emb(
                t3_cond.cond_prompt_speech_tokens
            ) + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        speech_tokens: mx.array,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        text_emb[:, 1] = 0  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.config.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = mx.stack(
            [
                mx.concatenate((ce, te, se), axis=0)
                for ce, te, se in zip(cond_emb, text_emb, speech_emb)
            ]
        )  # (B, length, dim)

        return embeds, len_cond

    def __call__(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        text_token_lens: mx.array,
        speech_tokens: mx.array,
        speech_token_lens: mx.array,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.config)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[
            -1
        ]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.shape[1]
        len_speech = speech_tokens.shape[1]
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = mx.zeros((B, len_text, dim), dtype=dtype)
        speech_latents = mx.zeros((B, len_speech, dim), dtype=dtype)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, : ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, : stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return {
            "text_logits": text_logits,
            "text_latents": text_latents,
            "speech_logits": speech_logits,
            "speech_latents": speech_latents,
            "hidden_states": hidden_states,
        }

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        text_token_lens: mx.array,
        speech_tokens: mx.array,
        speech_token_lens: mx.array,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.__call__(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        mask_text = (
            mx.arange(len_text)[None] >= text_token_lens[:, None]
        )  # (B, len_text)
        mask_speech = (
            mx.arange(len_speech)[None] >= speech_token_lens[:, None]
        )  # (B, len_speech)
        masked_text = mx.where(mask_text, IGNORE_ID, text_tokens)
        masked_speech = mx.where(mask_speech, IGNORE_ID, speech_tokens)
        loss_text = mx.cross_entropy(
            out.text_logits, masked_text, ignore_index=IGNORE_ID
        )
        loss_speech = mx.cross_entropy(
            out.speech_logits, masked_speech, ignore_index=IGNORE_ID
        )

        return loss_text, loss_speech

    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        initial_speech_tokens: Optional[mx.array] = None,
        # misc conditioning
        prepend_prompt_speech_tokens: Optional[mx.array] = None,
        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        length_penalty=1.0,
        repetition_penalty=2.0,
        cfg_weight=0,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.config)
        text_tokens = mx.atleast_2d(text_tokens).astype(mx.long)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.config.start_speech_token * mx.ones_like(
                text_tokens[:, :1]
            )

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        self.compiled = False

        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if not self.compiled:
            alignment_stream_analyzer = AlignmentStreamAnalyzer(
                self.tfmr,
                None,
                text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                alignment_layer_idx=9,  # TODO: hparam or something?
                eos_idx=self.hp.stop_speech_token,
            )
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        bos_token = mx.array([[self.config.start_speech_token]], dtype=mx.long)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb(bos_token)

        # batch_size=2 for CFG
        bos_embed = mx.concatenate([bos_embed, bos_embed], axis=1)

        # Combine condition and BOS token for the initial input
        inputs_embeds = mx.concatenate([embeds, bos_embed], axis=1)

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token
        predicted = []  # To store the predicted tokens

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=repetition_penalty
        )

        cache = []
        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits = output.logits[:, -1, :]

            # CFG
            logits_cond = logits[0:1]
            logits_uncond = logits[1:2]
            logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
            logits = logits.squeeze(1)

            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature

            # Apply repetition penalty and top‑p filtering.
            logits = repetition_penalty_processor(generated_ids, logits)
            logits = top_p_warper(None, logits)

            # Convert logits to probabilities and sample the next token.
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids = mx.concatenate([generated_ids, next_token], axis=1)

            # Check for EOS token.
            if next_token.reshape(-1) == self.config.stop_speech_token:
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = (
                next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
            )

            #  For CFG
            next_token_embed = mx.concatenate([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values

        # Concatenate all predicted tokens along the sequence dimension.
        predicted_tokens = mx.concatenate(predicted, axis=1)  # shape: (B, num_tokens)
        return predicted_tokens

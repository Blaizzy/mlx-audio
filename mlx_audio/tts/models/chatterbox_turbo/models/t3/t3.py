# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from .cond_enc import T3Cond, T3CondEnc
from .gpt2 import GPT2Config, GPT2Model, create_gpt2_config
from .t3_config import T3Config


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using GPT2 as backbone.
    MLX port optimized for Apple Silicon.
    """

    def __init__(self, hp: Optional[T3Config] = None):
        super().__init__()
        if hp is None:
            hp = T3Config.turbo()
        self.hp = hp

        # Create GPT2 config
        self.cfg = create_gpt2_config()
        self.dim = self.cfg.hidden_size

        # GPT2 backbone
        self.tfmr = GPT2Model(self.cfg)

        # Conditioning encoder
        self.cond_enc = T3CondEnc(hp)

        # Text and speech embeddings
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # Output heads
        self.text_head = nn.Linear(self.dim, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.dim, hp.speech_tokens_dict_size, bias=True)

    def prepare_conditioning(self, t3_cond: T3Cond) -> mx.array:
        """
        Prepare conditioning embeddings.
        Token cond data needs to be embedded here.
        """
        if (
            t3_cond.cond_prompt_speech_tokens is not None
            and t3_cond.cond_prompt_speech_emb is None
        ):
            t3_cond.cond_prompt_speech_emb = self.speech_emb(
                t3_cond.cond_prompt_speech_tokens
            )

        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        speech_tokens: mx.array,
    ) -> Tuple[mx.array, int]:
        """Prepare input embeddings for the transformer."""
        # Get conditioning embeddings
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)

        # Get text and speech embeddings
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)

        len_cond = cond_emb.shape[1]

        # Expand cond_emb if needed
        if cond_emb.shape[0] != text_emb.shape[0]:
            cond_emb = mx.broadcast_to(
                cond_emb, (text_emb.shape[0], cond_emb.shape[1], cond_emb.shape[2])
            )

        # Concatenate: [cond, text, speech]
        embeds = mx.concatenate([cond_emb, text_emb, speech_emb], axis=1)

        return embeds, len_cond

    def inference_turbo(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 1000,
    ) -> mx.array:
        """
        Turbo inference: generate speech tokens from text tokens.

        Args:
            t3_cond: Conditioning data
            text_tokens: Input text tokens (B, T)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            max_gen_len: Maximum generation length

        Returns:
            Generated speech tokens
        """
        # Ensure batch dimension
        if text_tokens.ndim == 1:
            text_tokens = text_tokens[None, :]

        # Initial speech token
        B = text_tokens.shape[0]
        speech_start_token = (
            mx.ones((B, 1), dtype=mx.int32) * self.hp.start_speech_token
        )

        # Prepare initial embeddings
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
        )

        # Initial forward pass
        hidden_states, cache = self.tfmr(inputs_embeds=embeds, cache=None)

        # Get first speech prediction
        speech_hidden = hidden_states[:, -1:, :]
        speech_logits = self.speech_head(speech_hidden)

        # Sample first token
        generated_speech_tokens = []
        next_speech_token = self._sample_token(
            speech_logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_tokens=None,
            repetition_penalty=repetition_penalty,
        )
        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        # Generation loop
        for _ in tqdm(range(max_gen_len), desc="Generating speech tokens"):
            # Get embedding for current token
            current_speech_embed = self.speech_emb(current_speech_token)

            # Forward pass with cache
            hidden_states, cache = self.tfmr(
                inputs_embeds=current_speech_embed,
                cache=cache,
            )

            # Get logits
            speech_logits = self.speech_head(hidden_states)

            # Gather generated tokens for repetition penalty
            all_generated = mx.concatenate(generated_speech_tokens, axis=1)

            # Sample next token
            next_speech_token = self._sample_token(
                speech_logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_tokens=all_generated,
                repetition_penalty=repetition_penalty,
            )

            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token

            # Check for EOS
            if mx.all(next_speech_token == self.hp.stop_speech_token):
                break

        # Concatenate all tokens
        all_tokens = mx.concatenate(generated_speech_tokens, axis=1)

        # Remove EOS token if present
        if all_tokens.shape[1] > 0:
            # Check last token
            last_token = all_tokens[0, -1]
            if last_token == self.hp.stop_speech_token:
                all_tokens = all_tokens[:, :-1]

        return all_tokens

    def _sample_token(
        self,
        logits: mx.array,
        temperature: float,
        top_k: int,
        top_p: float,
        generated_tokens: Optional[mx.array],
        repetition_penalty: float,
    ) -> mx.array:
        """Sample a token from logits with various sampling strategies."""
        # Apply repetition penalty
        if generated_tokens is not None and repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, generated_tokens, repetition_penalty
            )

        # Apply temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Apply top-k
        if top_k > 0:
            logits = self._top_k_filtering(logits, top_k)

        # Apply top-p
        if top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)

        # Sample - mx.random.categorical expects logits (unnormalized), NOT probabilities
        # It applies softmax internally
        next_token = mx.random.categorical(logits)

        return next_token[:, None]

    def _apply_repetition_penalty(
        self,
        logits: mx.array,
        generated_tokens: mx.array,
        penalty: float,
    ) -> mx.array:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits

        # Create a mask for tokens that have been generated
        vocab_size = logits.shape[-1]
        token_mask = mx.zeros((vocab_size,))

        # Mark generated tokens
        flat_tokens = generated_tokens.reshape(-1)
        for token in flat_tokens.tolist():
            if 0 <= token < vocab_size:
                token_mask = mx.concatenate(
                    [token_mask[:token], mx.array([1.0]), token_mask[token + 1 :]]
                )

        # Apply penalty: if score < 0, multiply by penalty; if > 0, divide by penalty
        penalized = mx.where(logits < 0, logits * penalty, logits / penalty)
        logits = mx.where(token_mask[None, :] > 0, penalized, logits)

        return logits

    def _top_k_filtering(self, logits: mx.array, top_k: int) -> mx.array:
        """Filter logits to only keep top-k values."""
        if top_k <= 0:
            return logits

        # Get top-k indices
        top_k = min(top_k, logits.shape[-1])

        # Find the k-th largest value as threshold
        # argpartition puts the k largest at the end
        partitioned = mx.argpartition(logits, -top_k, axis=-1)
        kth_indices = partitioned[:, -top_k : -top_k + 1]  # The k-th largest index
        kth_values = mx.take_along_axis(logits, kth_indices, axis=-1)

        # Mask: keep values >= kth value
        mask = logits >= kth_values

        # Apply mask (set non-top-k to -inf)
        logits = mx.where(mask, logits, mx.array(-float("inf")))

        return logits

    def _top_p_filtering(self, logits: mx.array, top_p: float) -> mx.array:
        """Filter logits using nucleus (top-p) sampling."""
        if top_p >= 1.0:
            return logits

        # Sort logits in descending order
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

        # Compute cumulative probabilities
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Remove tokens with cumulative probability above threshold
        # Shift the cumulative probs to keep at least one token
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift right to keep first token above threshold
        sorted_indices_to_remove = mx.concatenate(
            [
                mx.zeros((logits.shape[0], 1), dtype=mx.bool_),
                sorted_indices_to_remove[:, :-1],
            ],
            axis=-1,
        )

        # Set removed tokens to -inf
        sorted_logits = mx.where(sorted_indices_to_remove, -float("inf"), sorted_logits)

        # Scatter back
        # Create inverse permutation
        inverse_indices = mx.argsort(sorted_indices, axis=-1)
        logits = mx.take_along_axis(sorted_logits, inverse_indices, axis=-1)

        return logits

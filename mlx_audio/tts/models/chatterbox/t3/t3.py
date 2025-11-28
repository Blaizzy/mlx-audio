# Ported from https://github.com/resemble-ai/chatterbox

from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelConfig
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from ..config import LLAMA_CONFIGS, T3Config
from .cond_enc import T3Cond, T3CondEnc
from .learned_pos_emb import LearnedPositionEmbeddings


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using LLaMA as backbone.

    Generates speech tokens from text tokens, conditioned on speaker embeddings
    and optional emotion/prompt controls.
    """

    def __init__(self, hp: Optional[T3Config] = None):
        super().__init__()

        if hp is None:
            hp = T3Config.english_only()

        self.hp = hp

        # Create LLaMA config from our T3 config
        llama_config_dict = LLAMA_CONFIGS[hp.llama_config_name].copy()
        self.cfg = LlamaModelConfig(**llama_config_dict)

        # LLaMA transformer backbone
        self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size

        # Conditioning encoder
        self.cond_enc = T3CondEnc(hp)

        # Text and speech token embeddings
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # Learned position embeddings (optional)
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # Output projection heads
        self.text_head = nn.Linear(
            self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False
        )
        self.speech_head = nn.Linear(
            self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize PyTorch weights for MLX.

        T3 uses a LLaMA backbone which mlx_lm already handles.
        We need to handle:
        - Mapping tfmr.* weights to the LLaMA model format (tfmr.X -> tfmr.model.X)
        - Conv1d weight transposition in conditioning encoder

        Note: Perceiver weights use the same naming convention (to_q, to_k, to_v, proj_out)
        as the original PyTorch implementation, so no renaming is needed.

        This method is idempotent - it checks shapes before transposing to support
        both PyTorch-format and pre-converted MLX-format weights.
        """
        import re

        from mlx.utils import tree_flatten

        new_weights = {}

        # Get expected shapes from model for idempotent transposition
        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            new_key = key

            # === Transformer weight name mapping ===
            # PyTorch uses: tfmr.layers.X, tfmr.embed_tokens, tfmr.norm
            # mlx_lm LlamaModel uses: model.layers.X, model.embed_tokens, model.norm
            # So we need: tfmr.layers.X -> tfmr.model.layers.X, etc.
            # Check if already converted (idempotent)
            if key.startswith("tfmr.") and not key.startswith("tfmr.model."):
                # Map tfmr.X to tfmr.model.X for transformer internal components
                # These are: layers, embed_tokens, norm
                patterns_to_prefix = [
                    r"^tfmr\.layers\.",
                    r"^tfmr\.embed_tokens\.",
                    r"^tfmr\.norm\.",
                ]
                for pattern in patterns_to_prefix:
                    if re.match(pattern, key):
                        new_key = re.sub(r"^tfmr\.", "tfmr.model.", key)
                        break

            # Conv1d weight transposition (idempotent)
            # Only transpose if shape doesn't match expected MLX format
            # PyTorch Conv1d: (out_channels, in_channels, kernel_size)
            # MLX Conv1d: (out_channels, kernel_size, in_channels)
            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 3:
                if (
                    new_key in curr_weights
                    and value.shape != curr_weights[new_key].shape
                ):
                    value = value.swapaxes(1, 2)

            new_weights[new_key] = value

        # Delegate to LLaMA sanitizer for transformer weights if it has one
        if hasattr(self.tfmr, "sanitize"):
            tfmr_weights = {
                k: v for k, v in new_weights.items() if k.startswith("tfmr.")
            }
            other_weights = {
                k: v for k, v in new_weights.items() if not k.startswith("tfmr.")
            }
            tfmr_weights = self.tfmr.sanitize(tfmr_weights)
            new_weights = {**other_weights, **tfmr_weights}

        return new_weights

    def prepare_conditioning(self, t3_cond: T3Cond) -> mx.array:
        """
        Prepare conditioning embeddings from T3Cond.

        Args:
            t3_cond: Conditioning information

        Returns:
            Conditioning embeddings (B, cond_len, dim)
        """
        # Embed speech prompt tokens if provided
        if (
            t3_cond.cond_prompt_speech_tokens is not None
            and t3_cond.cond_prompt_speech_emb is None
        ):
            t3_cond.cond_prompt_speech_emb = self.speech_emb(
                t3_cond.cond_prompt_speech_tokens
            ) + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)

        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        speech_tokens: mx.array,
        cfg_weight: float = 0.0,
    ) -> Tuple[mx.array, int]:
        """
        Prepare input embeddings for the transformer.

        Args:
            t3_cond: Conditioning information
            text_tokens: Text token IDs (B, text_len)
            speech_tokens: Speech token IDs (B, speech_len)
            cfg_weight: Classifier-free guidance weight

        Returns:
            Tuple of (embeddings, conditioning_length)
        """
        # Prepare conditioning embeddings
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)

        # Text embeddings
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)

        # CFG: zero out second batch item for unconditional
        if cfg_weight > 0.0 and text_emb.shape[0] > 1:
            text_emb = mx.concatenate(
                [
                    text_emb[:1],
                    mx.zeros_like(text_emb[1:2]),
                ],
                axis=0,
            )

        # Speech embeddings
        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)

        # Add position embeddings if using learned positions
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)

        len_cond = cond_emb.shape[1]

        # Broadcast conditioning if batch sizes don't match
        if cond_emb.shape[0] != text_emb.shape[0]:
            cond_emb = mx.broadcast_to(
                cond_emb, (text_emb.shape[0],) + cond_emb.shape[1:]
            )

        # Broadcast speech embeddings if batch sizes don't match (e.g., CFG)
        if speech_emb.shape[0] != text_emb.shape[0]:
            speech_emb = mx.broadcast_to(
                speech_emb, (text_emb.shape[0],) + speech_emb.shape[1:]
            )

        # Concatenate: [conditioning | text | speech]
        embeds = mx.concatenate([cond_emb, text_emb, speech_emb], axis=1)

        return embeds, len_cond

    def __call__(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        text_token_lens: mx.array,
        speech_tokens: mx.array,
        speech_token_lens: mx.array,
        cache=None,
    ) -> dict:
        """
        Forward pass through T3 model.

        Args:
            t3_cond: Conditioning information
            text_tokens: Text token IDs (B, text_len)
            text_token_lens: Valid lengths for each text sequence (B,)
            speech_tokens: Speech token IDs (B, speech_len)
            speech_token_lens: Valid lengths for each speech sequence (B,)
            cache: Optional KV cache for inference

        Returns:
            Dictionary with text_logits, speech_logits, and hidden_states
        """
        # Prepare input embeddings
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # Forward through LLaMA backbone
        # Note: mlx_lm's LlamaModel takes input_embeddings differently
        hidden_states = self.tfmr.model(
            inputs=None,
            cache=cache,
            input_embeddings=embeds,
        )

        # Extract text and speech portions of hidden states
        B = text_tokens.shape[0]
        len_text = text_tokens.shape[1]
        len_speech = speech_tokens.shape[1]
        dim = hidden_states.shape[-1]

        # Allocate output tensors
        text_latents = mx.zeros((B, len_text, dim))
        speech_latents = mx.zeros((B, len_speech, dim))

        # Split hidden states by sequence position
        for i in range(B):
            ttl = int(text_token_lens[i])
            stl = int(speech_token_lens[i])

            text_start = len_cond
            text_end = len_cond + ttl

            speech_start = len_cond + len_text
            speech_end = speech_start + stl

            text_latents = (
                mx.concatenate(
                    [
                        text_latents[:i],
                        hidden_states[i : i + 1, text_start:text_end, :],
                        text_latents[i + 1 :],
                    ],
                    axis=0,
                )
                if i < B
                else text_latents
            )

            speech_latents = (
                mx.concatenate(
                    [
                        speech_latents[:i],
                        hidden_states[i : i + 1, speech_start:speech_end, :],
                        speech_latents[i + 1 :],
                    ],
                    axis=0,
                )
                if i < B
                else speech_latents
            )

        # Project to vocabulary
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return {
            "text_logits": text_logits,
            "text_latents": text_latents,
            "speech_logits": speech_logits,
            "speech_latents": speech_latents,
            "hidden_states": hidden_states,
        }

    def inference(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        initial_speech_tokens: Optional[mx.array] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
    ) -> mx.array:
        """
        Generate speech tokens from text tokens.

        This matches the original PyTorch implementation which uses:
        - KV caching for efficient generation
        - Position embeddings for each generated token
        - Repetition penalty, min_p, and top_p filtering
        - Classifier-free guidance (CFG)

        Args:
            t3_cond: Conditioning information
            text_tokens: Text token IDs (1D or 2D)
            initial_speech_tokens: Optional initial speech tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor
            cfg_weight: Classifier-free guidance weight

        Returns:
            Generated speech tokens (B, T)
        """
        # Ensure text_tokens is 2D
        if text_tokens.ndim == 1:
            text_tokens = mx.expand_dims(text_tokens, 0)

        # Default initial speech token (BOS)
        bos_token = mx.array([[self.hp.start_speech_token]], dtype=mx.int32)

        # Prepare conditioning and text embeddings
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=bos_token,
            cfg_weight=cfg_weight,
        )

        # Create BOS embedding with position embedding at position 0
        bos_embed = self.speech_emb(bos_token)  # (1, 1, dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # For CFG, duplicate BOS embed
        if cfg_weight > 0.0:
            bos_embed = mx.concatenate([bos_embed, bos_embed], axis=0)

        # Combine conditioning+text embeddings with BOS token
        # embeds already has [cond | text | bos] structure from prepare_input_embeds
        # We need to replace the speech part with our position-embedded BOS
        # Actually, let's rebuild the input properly
        cond_emb = self.prepare_conditioning(t3_cond)  # (1, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)

        if cfg_weight > 0.0:
            # Zero out second batch for unconditional
            text_emb = mx.concatenate(
                [
                    text_emb[:1],
                    mx.zeros_like(text_emb[:1]),
                ],
                axis=0,
            )

        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)

        # Broadcast conditioning if needed
        if cond_emb.shape[0] != text_emb.shape[0]:
            cond_emb = mx.broadcast_to(
                cond_emb, (text_emb.shape[0],) + cond_emb.shape[1:]
            )

        # Build initial input: [cond | text | bos]
        input_embeddings = mx.concatenate([cond_emb, text_emb, bos_embed], axis=1)

        # Create KV cache
        cache = make_prompt_cache(self.tfmr)

        # Create sampler and logits processors
        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p)
        logits_processors = make_logits_processors(
            logit_bias=None,
            repetition_penalty=repetition_penalty,
            repetition_context_size=max_new_tokens,  # Use all generated tokens for repetition
        )

        # Track generated tokens (Python list for efficiency - avoid growing mx.array each step)
        generated_ids = [self.hp.start_speech_token]

        # Initial forward pass to fill cache
        hidden = self.tfmr.model(
            inputs=None, input_embeddings=input_embeddings, cache=cache
        )

        # Generation loop
        for step in range(max_new_tokens):
            # Get logits for last position
            logits = self.speech_head(hidden[:, -1:, :])  # (B, 1, vocab)
            logits = logits.squeeze(1)  # (B, vocab)

            # Apply CFG
            if cfg_weight > 0.0 and logits.shape[0] > 1:
                cond_logits = logits[0:1, :]
                uncond_logits = logits[1:2, :]
                logits = cond_logits + cfg_weight * (cond_logits - uncond_logits)
            else:
                logits = logits[0:1, :]

            # Apply logits processors (repetition penalty)
            # Lazily convert Python list to array - processor only uses last N tokens anyway
            for processor in logits_processors:
                tokens_for_penalty = mx.array([generated_ids], dtype=mx.int32)
                logits = processor(tokens_for_penalty, logits)

            # Sample next token using the sampler (handles temperature, top_p, min_p)
            next_token = sampler(logits)
            # Sampler returns 1D array (B,) for each batch item
            next_token_id = int(next_token[0])

            # Check for EOS
            if next_token_id == self.hp.stop_speech_token:
                generated_ids.append(next_token_id)
                break

            generated_ids.append(next_token_id)

            # Create embedding for next token with position embedding
            next_token_embed = self.speech_emb(mx.array([[next_token_id]]))
            next_token_embed = (
                next_token_embed + self.speech_pos_emb.get_fixed_embedding(step + 1)
            )

            # For CFG, duplicate
            if cfg_weight > 0.0:
                next_token_embed = mx.concatenate(
                    [next_token_embed, next_token_embed], axis=0
                )

            # Forward pass with cache (only new token)
            hidden = self.tfmr.model(
                inputs=None, input_embeddings=next_token_embed, cache=cache
            )

            # Pipeline: start computing next step while current finishes
            mx.async_eval(hidden)

        return mx.array([generated_ids])

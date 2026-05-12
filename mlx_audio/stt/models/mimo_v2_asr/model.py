"""
MiMo-V2.5-ASR: MLX-native implementation.

Ported from ``src/mimo_audio/modeling_mimo_audio.py``.

Architecture
------------
Input: [B, audio_channels+1, seq_len] where:
  Row 0: text token IDs (group_size tokens per LLM position)
  Rows 1-8: speech token IDs for 8 audio channels

           speech_input_ids [B, 8, T]
             │
             ▼
┌─────────────────────────────────────────────────┐
│ Speech Embeddings (8× Embedding → sum)          │
│   → [B, T//group_size, group_size, 1024]        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ Input Local Transformer (6 layers, d=1024)      │
│   full attention (non-causal)                   │
│   → [B, T//group_size, group_size, 1024]        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ speech_group_downcast: Linear(1024*4 → 4096)   │
│   + text_embedding                              │
│   → [B, T//group_size, 4096]                    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ Qwen2 LLM (36 layers, d=4096)                   │
│   → hidden_states [B, T, 4096]                  │
└────────────┬────────────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
 text_logits      hidden_states_downcast
 [B, 1, 151680]    → local_hidden [B, 1, 1024]
                         │
                         ▼ (if text_token == empty)
              ┌──────────────────────────┐
              │ Local Transformer (16L)  │
              │  delay-pattern decoding  │
              │  → [B, group_size, 8]    │
              └──────────────────────────┘
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import MiMoAudioConfig


def _make_kv_cache(num_layers: int):
    """Create an MLX-LM KV cache list for Qwen2-style autoregressive decoding."""
    from mlx_lm.models.cache import KVCache

    return [KVCache() for _ in range(num_layers)]


# ── Qwen2 imports (from mlx_lm) ─────────────────────────────────────

def _get_qwen2_model_and_args():
    """Import Qwen2Model and ModelArgs from mlx_lm."""
    try:
        from mlx_lm.models.qwen2 import Model as Qwen2Model, ModelArgs
        return Qwen2Model, ModelArgs
    except ImportError:
        raise ImportError(
            "mlx_lm is required for Phase 3. Install with: pip install mlx-lm"
        )


def _to_qwen2_args(config: MiMoAudioConfig, **overrides) -> "ModelArgs":
    """Convert our MiMoAudioConfig to mlx_lm's Qwen2 ModelArgs."""
    _, ModelArgs = _get_qwen2_model_and_args()
    return ModelArgs(
        model_type="qwen2",
        hidden_size=overrides.get("hidden_size", config.hidden_size),
        num_hidden_layers=overrides.get("num_hidden_layers", config.num_hidden_layers),
        intermediate_size=overrides.get("intermediate_size", config.intermediate_size),
        num_attention_heads=overrides.get("num_attention_heads", config.num_attention_heads),
        rms_norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        num_key_value_heads=overrides.get("num_key_value_heads", config.num_key_value_heads),
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=overrides.get("rope_theta", config.rope_theta),
        rope_traditional=False,
        rope_scaling=config.rope_scaling,
        tie_word_embeddings=False,
    )


# ── Causal mask helper ──────────────────────────────────────────────

def create_causal_mask(seq_len: int, offset: int = 0) -> mx.array:
    """Create a causal attention mask for local transformer decoding."""
    mask = mx.triu(
        mx.full((seq_len, seq_len), -1e9, dtype=mx.float32),
        k=1 + offset,
    )
    return mask


# ── Sampler ─────────────────────────────────────────────────────────

class MiMoSampler:
    """
    Text token sampler with temperature, top-k, top-p filtering.

    Matches the reference MiMoSampler in modeling_mimo_audio.py.
    """

    def __init__(
        self,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def process(self, scores: mx.array) -> mx.array:
        """Apply temperature, top-k, top-p to logits."""
        if self.temperature > 0:
            scores = scores / self.temperature

        if self.top_k > 0 and self.top_k < scores.shape[-1]:
            # Zero out scores below top-k
            topk_vals = mx.sort(scores)[:, -self.top_k]
            mask = scores < topk_vals[:, None]
            scores = mx.where(mask, mx.array(float("-inf"), dtype=scores.dtype), scores)

        if 0.0 < self.top_p < 1.0:
            sorted_indices = mx.argsort(scores, axis=-1)
            sorted_scores = mx.take_along_axis(scores, sorted_indices, axis=-1)
            cumulative_probs = mx.softmax(sorted_scores, axis=-1).cumsum(axis=-1)
            sorted_remove = cumulative_probs <= (1 - self.top_p)
            sorted_remove[:, -1] = False
            remove = mx.zeros_like(sorted_remove)
            remove = mx.put_along_axis(remove, sorted_indices, sorted_remove, axis=-1)
            scores = mx.where(remove, mx.array(float("-inf"), dtype=scores.dtype), scores)

        return scores

    def sample(self, scores: mx.array, removed_tokens: Optional[List[int]] = None) -> mx.array:
        """
        Sample from logits.

        Parameters
        ----------
        scores : mx.array, shape (batch, vocab_size)
        removed_tokens : list[int], optional — tokens to disallow

        Returns
        -------
        mx.array, shape (batch,) — sampled token indices
        """
        scores = self.process(scores)

        # Remove forbidden tokens
        for t in (removed_tokens or []):
            if t < scores.shape[-1]:
                scores[:, t] = float("-inf")

        if self.do_sample:
            probs = mx.softmax(scores, axis=-1)
            # Sample via categorical distribution
            return mx.random.categorical(probs, axis=-1)

        return mx.argmax(scores, axis=-1)


# ── MiMoAudioMLX ────────────────────────────────────────────────────

class MiMoAudioMLX(nn.Module):
    """
    MiMo-V2.5-ASR model (Qwen2 LLM + speech token processing).

    Parameters
    ----------
    config : MiMoAudioConfig
        Model configuration.
    """

    def __init__(self, config: MiMoAudioConfig):
        super().__init__()
        self.config = config
        _, ModelArgs = _get_qwen2_model_and_args()
        # Import Qwen2Model (the inner class, no wrapper) for direct access to hidden states
        from mlx_lm.models.qwen2 import Qwen2Model

        # ── Qwen2 LLM backbone (36 layers, d=4096) ──
        llm_args = _to_qwen2_args(config)
        self.model = Qwen2Model(llm_args)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── Speech embeddings ──
        speech_vocab_sizes = config.parsed_speech_vocab_sizes()
        empty_ids = config.parsed_speech_empty_ids()
        self.speech_vocab_sizes = speech_vocab_sizes
        self.speech_empty_ids = empty_ids
        self.delay_pattern = config.parsed_delay_pattern()
        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        self.speech_embeddings = [
            nn.Embedding(
                speech_vocab_sizes[i],
                config.input_local_dim,
            )
            for i in range(config.audio_channels)
        ]

        # ── Input Local Transformer (6 layers, d=1024, full attention) ──
        input_local_args = MiMoAudioMLX._build_local_args(config, is_input=True)
        self.input_local_transformer = Qwen2Model(input_local_args)
        # Match reference: we pass embeddings directly, no embed_tokens needed
        self.input_local_transformer.embed_tokens = None
        self.input_local_transformer.vocab_size = 0

        # ── Downcast / Upcast projections ──
        self.speech_group_downcast = nn.Linear(
            config.input_local_dim * config.group_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_states_downcast = nn.Linear(
            config.hidden_size,
            config.local_dim,
            bias=False,
        )

        # ── Local Transformer (16 layers, d=1024, causal) ──
        local_args = MiMoAudioMLX._build_local_args(config, is_input=False)
        self.local_transformer = Qwen2Model(local_args)
        # Match reference: we pass embeddings directly, no embed_tokens needed
        self.local_transformer.embed_tokens = None
        self.local_transformer.vocab_size = 0

        # ── Local LM heads (8× per-channel vocab projection) ──
        self.local_transformer_lm_heads = [
            nn.Linear(config.local_dim, speech_vocab_sizes[i], bias=False)
            for i in range(config.audio_channels)
        ]

        # ── Speech embeddings → Local input mapping (if dims differ) ──
        if config.input_local_dim != config.local_dim:
            self.speech_embeddings_to_local = nn.Linear(
                config.input_local_dim, config.local_dim, bias=False
            )
        else:
            self.speech_embeddings_to_local = None

    @staticmethod
    def _build_local_args(config: MiMoAudioConfig, is_input: bool):
        """Build ModelArgs for a local transformer."""
        _, ModelArgs = _get_qwen2_model_and_args()
        if is_input:
            return ModelArgs(
                model_type="qwen2",
                hidden_size=config.input_local_dim,
                num_hidden_layers=config.input_local_layers,
                intermediate_size=config.input_local_intermediate_size,
                num_attention_heads=config.input_local_attn_heads,
                rms_norm_eps=config.rms_norm_eps,
                vocab_size=config.vocab_size,
                num_key_value_heads=config.input_local_attn_heads,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.local_rotary_base,
                rope_traditional=False,
                tie_word_embeddings=False,
            )
        else:
            return ModelArgs(
                model_type="qwen2",
                hidden_size=config.local_dim,
                num_hidden_layers=config.local_layers,
                intermediate_size=config.local_ffn_dim,
                num_attention_heads=config.local_attn_heads,
                rms_norm_eps=config.rms_norm_eps,
                vocab_size=config.vocab_size,
                num_key_value_heads=config.local_attn_heads,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.local_rotary_base,
                rope_traditional=False,
                tie_word_embeddings=False,
            )

    # ==================================================================
    #  Core forward pass
    # ==================================================================

    def _prepare_input_embeds(
        self,
        input_ids: mx.array,  # [B, audio_channels+1, new_T]
    ) -> mx.array:
        """
        Build the combined input embeddings for the Qwen2 LLM.

        Steps:
        1. Split into text IDs (row 0) and speech IDs (rows 1-8)
        2. Sum speech embeddings across channels
        3. Apply input local transformer (group-wise, full attention)
        4. speech_group_downcast + text_embedding → [B, T_groups, 4096]
        """
        B = input_ids.shape[0]
        group_size = self.group_size
        audio_c = self.audio_channels

        # Text IDs: first row, every group_size-th element
        text_input_ids = input_ids[:, 0, ::group_size]  # [B, T_groups]
        # Speech IDs: rows 1-8, grouped
        speech_input_ids = input_ids[:, 1:, :]  # [B, audio_c, new_T]
        T_total = speech_input_ids.shape[2]
        T_groups = T_total // group_size

        # Reshape: [B, audio_c, T_groups*gs] → [B, T_groups, audio_c, gs]
        speech_input_ids = speech_input_ids.reshape(
            B, audio_c, T_groups, group_size
        ).transpose(0, 2, 1, 3)  # [B, T_groups, audio_c, group_size]

        # Mark speech positions (where text token is empty)
        is_speech = (text_input_ids == self.config.empty_idx)  # [B, T_groups]

        # Speech embeddings: sum across audio channels
        speech_embeds = mx.zeros(
            (B, T_groups, group_size, self.config.input_local_dim),
            dtype=mx.float32,
        )

        for i in range(audio_c):
            empty_id = self.speech_empty_ids[i]
            embed = self.speech_embeddings[i]
            cur_ids = speech_input_ids[:, :, i, :]  # [B, T_groups, group_size]
            cur_embeds = embed(cur_ids)  # [B, T_groups, group_size, input_local_dim]

            # Zero out padding positions
            empty_mask = (cur_ids == empty_id)
            cur_embeds = mx.where(
                empty_mask[..., None],
                mx.zeros_like(cur_embeds),
                cur_embeds,
            )
            speech_embeds = speech_embeds + cur_embeds

        # Zero out non-speech positions
        speech_embeds = speech_embeds * is_speech[:, :, None, None].astype(mx.float32)

        # ── Input Local Transformer (full attention, group-wise) ──
        speech_embeds = self._apply_input_local_transformer(speech_embeds)
        speech_embeds = speech_embeds * is_speech[:, :, None, None].astype(mx.float32)

        # ── speech_group_downcast: flatten group_size dim → 4096 ──
        speech_grouped = speech_embeds.reshape(B, T_groups, -1)  # [B, T_groups, input_dim*gs]
        speech_group_embeds = self.speech_group_downcast(speech_grouped)  # [B, T_groups, 4096]

        # ── Text embeddings ──
        text_embeds = self.model.embed_tokens(text_input_ids)  # [B, T_groups, 4096]

        # Zero out empty text positions
        text_zero_mask = (text_input_ids == self.config.empty_idx)
        text_embeds = mx.where(
            text_zero_mask[..., None],
            mx.zeros_like(text_embeds),
            text_embeds,
        )

        return text_embeds + speech_group_embeds  # [B, T_groups, 4096]

    def _apply_input_local_transformer(self, speech_embeddings: mx.array) -> mx.array:
        """
        Apply input local transformer to speech embeddings.

        speech_embeddings: [B, T_groups, group_size, hidden_size]

        Process each group independently through the 6-layer transformer
        with FULL attention (non-causal).

        Returns: [B, T_groups, group_size, hidden_size]
        """
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Flatten batch and group dims: [B*T_groups, group_size, hidden_size]
        x = speech_embeddings.reshape(B * T_groups, group_size, hidden_size)

        # Manual forward through input local transformer layers
        # with full (non-causal) attention: mask=None
        mask = None  # no mask = full attention
        cache = [None] * len(self.input_local_transformer.layers)
        h = x
        for layer, c in zip(self.input_local_transformer.layers, cache):
            h = layer(h, mask, c)
        output = self.input_local_transformer.norm(h)

        # Reshape back
        return output.reshape(B, T_groups, group_size, hidden_size)

    def __call__(
        self,
        input_ids: mx.array,  # [B, audio_channels+1, new_T]
        mask: Optional[mx.array] = None,  # [B, T_group] or None
        cache=None,
    ) -> Tuple[mx.array, mx.array, Optional[object]]:
        """
        Forward pass.

        Returns
        -------
        text_logits : mx.array, shape (B, 1, vocab_size)
        local_hidden_states : mx.array, shape (B, 1, local_dim)
        cache : updated KV cache (mutated in-place)
        """
        inputs_embeds = self._prepare_input_embeds(input_ids)  # [B, T_groups, 4096]
        if cache is None:
            cache = _make_kv_cache(len(self.model.layers))

        # Qwen2 LLM forward (inner Qwen2Model, no lm_head wrapper)
        # Pass input_embeddings since we already computed them
        # inputs is a required positional arg but unused when input_embeddings is set
        dummy_inputs = mx.zeros(inputs_embeds.shape[:2], dtype=mx.int32)
        hidden_states = self.model(
            dummy_inputs,
            input_embeddings=inputs_embeds,
            cache=cache,
        )

        # Last position only
        last_hidden = hidden_states[:, -1:, :]  # [B, 1, 4096]

        text_logits = self.lm_head(last_hidden)  # [B, 1, vocab_size]
        local_hidden = self.hidden_states_downcast(last_hidden)  # [B, 1, local_dim]

        return text_logits, local_hidden, cache

    # ==================================================================
    #  Local Transformer (delay-pattern speech token generation)
    # ==================================================================

    def local_forward(
        self,
        local_embeds: mx.array,  # [B, 1, local_dim]
        local_sampler: Optional[MiMoSampler] = None,
    ) -> mx.array:
        """
        Generate speech tokens using local transformer with delay pattern.

        Decodes group_size × audio_channels speech tokens autoregressively
        using the delay_pattern to interleave channel predictions.

        Parameters
        ----------
        local_embeds : mx.array, shape (B, 1, local_dim)
            Initial hidden state from the LLM (at empty text position).

        Returns
        -------
        mx.array, shape (B, group_size, audio_channels), dtype int32
            Generated speech token IDs.
        """
        B = local_embeds.shape[0]
        delay_iters = self.group_size + max(self.delay_pattern)  # 4 + 7 = 11

        cache = _make_kv_cache(len(self.local_transformer.layers))

        local_tokens = mx.zeros(
            (B, self.group_size, self.audio_channels),
            dtype=mx.int32,
        )

        if local_sampler is None:
            local_sampler = MiMoSampler(do_sample=False)

        current_input = local_embeds  # [B, 1, local_dim]

        for t in range(delay_iters):
            # Forward through local transformer (cache persists across iterations)
            hidden_state = self.local_transformer(
                mx.array([[0]], dtype=mx.int32),
                input_embeddings=current_input,
                cache=cache,
            )
            last_step = hidden_state[:, -1:, :]  # [B, 1, local_dim]

            # Prepare next input (will be filled with embeddings)
            current_input = mx.zeros_like(local_embeds)

            for idx in range(self.audio_channels):
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size

                if cur_start <= t < cur_end:
                    cur_empty = self.speech_empty_ids[idx]
                    cur_lm_head = self.local_transformer_lm_heads[idx]
                    cur_scores = cur_lm_head(last_step)[:, -1, :]  # [B, cb_size]

                    # Sample token (avoiding empty)
                    cur_token = local_sampler.sample(cur_scores, removed_tokens=[cur_empty])
                    # cur_token: [B]

                    # Store
                    row_idx = t - cur_start
                    if row_idx < self.group_size:
                        local_tokens[:, row_idx, idx] = cur_token[:B]

                    # Embed for next step
                    cur_embed = self.speech_embeddings[idx](cur_token[:, None])  # [B, 1, input_local_dim]
                    if self.speech_embeddings_to_local is not None:
                        cur_embed = self.speech_embeddings_to_local(cur_embed)
                    current_input = current_input + cur_embed

        return local_tokens  # [B, group_size, audio_channels]

    # ==================================================================
    #  Generation
    # ==================================================================

    def generate(
        self,
        input_ids: mx.array,  # [B, audio_channels+1, prompt_len]
        max_new_tokens: int = 256,
        global_sampler: Optional[MiMoSampler] = None,
        local_sampler: Optional[MiMoSampler] = None,
        stop_tokens: Optional[List[int]] = None,
        streamer=None,
    ) -> mx.array:
        """
        Autoregressive generation with dual (text + speech) sampling.

        Parameters
        ----------
        input_ids : mx.array
            Prompt with shape [B, audio_channels+1, prompt_len].
        max_new_tokens : int
            Maximum number of new LLM steps (not total tokens).
        global_sampler : MiMoSampler, optional
            Sampler for text tokens.
        local_sampler : MiMoSampler, optional
            Sampler for speech tokens.

        Returns
        -------
        mx.array, shape (B, (audio_channels+1), total_len)
            Full input_ids with generated tokens appended.
        """
        B = input_ids.shape[0]
        step_size = (self.audio_channels + 1) * self.group_size  # 9 * 4 = 36

        if global_sampler is None:
            global_sampler = MiMoSampler(do_sample=False)
        if local_sampler is None:
            local_sampler = MiMoSampler(do_sample=False)

        cache = _make_kv_cache(len(self.model.layers))
        current_ids = input_ids

        for step in range(max_new_tokens):
            # Forward: first step processes full prompt, subsequent steps only new tokens
            if step == 0:
                # Prefill: process entire prompt
                text_logits, local_hidden, cache = self(current_ids, cache=cache)
            else:
                # Incremental: only pass new tokens (last group_size positions per channel)
                new_ids = current_ids[:, :, -self.group_size:]  # [B, ac+1, gs]
                text_logits, local_hidden, cache = self(new_ids, cache=cache)

            # Sample text token
            text_scores = text_logits[:, -1, :]  # [B, vocab_size]
            next_text_token = global_sampler.sample(text_scores)  # [B]

            # Check if we need speech tokens
            need_speech = (next_text_token[0] == self.config.empty_idx)

            if need_speech:
                next_speech_tokens = self.local_forward(
                    local_hidden, local_sampler=local_sampler
                )  # [B, group_size, audio_channels]
            else:
                # Zero/padding speech tokens
                empty_arr = mx.array(self.speech_empty_ids, dtype=mx.int32)
                next_speech_tokens = mx.broadcast_to(
                    empty_arr[None, None, :],
                    (B, self.group_size, self.audio_channels),
                )

            # Build next step tokens
            next_text = mx.broadcast_to(
                next_text_token[:, None, None],
                (B, self.group_size, 1),
            )  # [B, group_size, 1]

            next_tokens = mx.concatenate(
                [next_text, next_speech_tokens], axis=-1
            )  # [B, group_size, audio_channels+1]

            # Reshape to match current_ids layout: [B, audio_channels+1, group_size]
            next_tokens = next_tokens.transpose(0, 2, 1)  # [B, audio_channels+1, group_size]

            # Append
            current_ids = mx.concatenate([current_ids, next_tokens], axis=-1)

            # Check stop
            if stop_tokens and next_text_token[0].item() in stop_tokens:
                break

        return current_ids

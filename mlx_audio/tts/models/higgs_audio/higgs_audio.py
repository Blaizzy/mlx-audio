"""Higgs Audio v2 (Boson AI) — MLX port.

Llama-3.2-3B backbone with a dual-FFN decoder layer (text path and audio path
share self-attention; LN + MLP are per-path, routed by `audio_out_mask`). Audio
tokens are emitted via a delay pattern (codebook i lags by i frames) — see
`generation.build_delay_pattern_mask` and `revert_delay_pattern`.

The first generated audio frame must be all-`audio_stream_bos_id` (synthetic);
sampling from the bos-text-position audio_logits collapses to stream-EOS on
half the codebooks. See `Model.generate` for the full ramp-in / ramp-out state
machine.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import MLP as LlamaMLP
from mlx_lm.models.llama import Attention as LlamaAttention
from mlx_lm.models.llama import ModelArgs as LlamaModelArgs

from .config import HiggsAudioConfig
from .generation import (
    greedy_sample_audio,
    lookup_audio_embedding,
    revert_delay_pattern,
    sample_audio,
)


def _llama_args_from_text_config(text_cfg) -> LlamaModelArgs:
    return LlamaModelArgs(
        model_type="llama",
        hidden_size=text_cfg.hidden_size,
        num_hidden_layers=text_cfg.num_hidden_layers,
        intermediate_size=text_cfg.intermediate_size,
        num_attention_heads=text_cfg.num_attention_heads,
        num_key_value_heads=text_cfg.num_key_value_heads,
        rms_norm_eps=text_cfg.rms_norm_eps,
        vocab_size=text_cfg.vocab_size,
        rope_theta=text_cfg.rope_theta,
        rope_scaling=text_cfg.rope_scaling,
        tie_word_embeddings=text_cfg.tie_word_embeddings,
    )


class HiggsDualFFNDecoderLayer(nn.Module):
    """One Llama-style decoder layer with dual-path norm+MLP for text vs audio tokens.

    Routing: positions where audio_out_mask==1 use the audio variants of the
    input layernorm, post-attention layernorm, and MLP. All other positions use
    the text variants. The self_attention module is shared.

    For v2 3B with use_audio_out_self_attention=0 and no fast_forward, every
    layer (0..27) is this kind of block.
    """

    def __init__(self, llama_args: LlamaModelArgs, text_cfg):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(
            text_cfg.hidden_size, eps=text_cfg.rms_norm_eps
        )
        self.audio_input_layernorm = nn.RMSNorm(
            text_cfg.hidden_size, eps=text_cfg.rms_norm_eps
        )
        self.self_attn = LlamaAttention(llama_args)
        self.post_attention_layernorm = nn.RMSNorm(
            text_cfg.hidden_size, eps=text_cfg.rms_norm_eps
        )
        self.audio_post_attention_layernorm = nn.RMSNorm(
            text_cfg.hidden_size, eps=text_cfg.rms_norm_eps
        )
        self.mlp = LlamaMLP(llama_args)
        self.audio_mlp = LlamaMLP(llama_args)

    def __call__(
        self,
        x: mx.array,
        audio_out_mask: mx.array,  # [B, T] bool
        attn_mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        # Pre-attention norm split: compute both paths, select by mask.
        mask_expanded = audio_out_mask[..., None]  # [B, T, 1]
        h_text_norm = self.input_layernorm(x)
        h_audio_norm = self.audio_input_layernorm(x)
        h_norm = mx.where(mask_expanded, h_audio_norm, h_text_norm)

        # Shared attention
        attn_out = self.self_attn(h_norm, attn_mask, cache)
        h = x + attn_out

        # Post-attention: split norm + split MLP by mask.
        post_text = self.post_attention_layernorm(h)
        post_audio = self.audio_post_attention_layernorm(h)
        mlp_text_out = self.mlp(post_text)
        mlp_audio_out = self.audio_mlp(post_audio)
        mlp_out = mx.where(mask_expanded, mlp_audio_out, mlp_text_out)

        return h + mlp_out


class HiggsAudioDecoderProjector(nn.Module):
    """Final projection: hidden states → (text_logits, audio_logits).

    For v2 3B with audio_decoder_proj_num_layers=0 this is literally two linear
    heads with no intermediate transformer layers.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        hidden_size = config.text_config.hidden_size
        vocab_size = config.text_config.vocab_size
        audio_out_dim = config.audio_num_codebooks * (config.audio_codebook_size + 2)

        self.text_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.audio_lm_head = nn.Linear(hidden_size, audio_out_dim, bias=False)
        self.audio_num_codebooks = config.audio_num_codebooks
        self.audio_codebook_plus2 = config.audio_codebook_size + 2

    def __call__(
        self, hidden_states: mx.array, audio_out_mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Project hidden states to text + audio logits.

        Returns (text_logits [B, T, vocab], audio_logits [B, T, K, C+2]).
        audio_logits is None only when audio_out_mask is None and we want to
        skip the audio head entirely (pure text forward). Otherwise we compute
        audio logits for all positions — caller filters to audio positions by
        mask. This is cheap (audio head is one Linear layer) and avoids MLX's
        unsupported boolean-mask indexing.
        """
        text_logits = self.text_lm_head(hidden_states)

        if audio_out_mask is None:
            return text_logits, None

        # audio_flat: [B, T, K * (C+2)]
        audio_flat = self.audio_lm_head(hidden_states)
        B, T = hidden_states.shape[:2]
        audio_logits = audio_flat.reshape(
            B, T, self.audio_num_codebooks, self.audio_codebook_plus2
        )
        return text_logits, audio_logits


class HiggsAudioModel(nn.Module):
    """End-to-end Higgs Audio v2 for MLX.

    NOT fully wired yet — see TODOs below. This scaffold defines the structure
    and the forward for a single decoder-layer pass with dual-FFN routing.

    TODO (M3 continuation):
      - Wire up the full forward: embed → stack of HiggsDualFFNDecoderLayer → norm
      - Audio codebook embedding lookup: audio tokens (from the codec) index a
        separate table; resulting embeddings replace embed_tokens[audio_pos]
      - Position embeddings + RoPE + attention mask with cache compatibility
      - sanitize(): weight-name conversion from bosonai safetensors to this
        module's state dict (M2 already did this for the pure-backbone keys;
        need to extend for dual-FFN audio_* keys + decoder_proj + codebook_embed)

    TODO (M4):
      - generate_delta_stream async loop — ChatML prompt + reference audio →
        tokenized context → autoregressive decode yielding [K] codebook frames
        with delay pattern applied
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        llama_args = _llama_args_from_text_config(config.text_config)

        self.embed_tokens = nn.Embedding(
            config.text_config.vocab_size, config.text_config.hidden_size
        )

        # Separate embedding table for audio codebook tokens.
        # Per-codebook has (codebook_size + 2) entries (values 0..C-1 plus BOS/EOS).
        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * (config.audio_codebook_size + 2),
            config.text_config.hidden_size,
        )

        self.layers = [
            HiggsDualFFNDecoderLayer(llama_args, config.text_config)
            for _ in range(config.text_config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        audio_out_mask: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
        cache=None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass.

        Pass either `input_ids` (text tokens — looked up via `embed_tokens`) OR
        `inputs_embeds` (pre-computed hidden states, e.g. for audio positions
        where summed codebook embeddings replace a vocab lookup).

        Args:
            input_ids: [B, T] int32 — text-vocab token ids.
            inputs_embeds: [B, T, hidden] float — pre-computed embeddings.
                Exactly one of input_ids / inputs_embeds must be provided.
            audio_out_mask: [B, T] bool — 1 at positions that should be routed
                through audio_mlp + audio_* layernorms. None → text-only (no
                audio_logits computed).
            attn_mask: [B, T, T] or broadcastable causal mask. Auto-created as
                causal if None.
            cache: optional kv-cache (list-per-layer) for incremental decoding.

        Returns:
            (text_logits [B, T, vocab], audio_logits [B, T, K, C+2] or None).
        """
        assert (input_ids is None) != (
            inputs_embeds is None
        ), "pass exactly one of input_ids or inputs_embeds"
        if input_ids is not None:
            B, T = input_ids.shape
            h = self.embed_tokens(input_ids)
        else:
            B, T = inputs_embeds.shape[:2]
            h = inputs_embeds

        caller_wants_audio = audio_out_mask is not None
        # Dual-FFN layers always need a mask; synthesize all-False when caller passes None.
        layer_mask = (
            audio_out_mask if caller_wants_audio else mx.zeros((B, T), dtype=mx.bool_)
        )

        if attn_mask is None and T > 1:
            attn_mask = create_causal_mask(T, offset=0)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, layer_mask, attn_mask, c)

        h = self.norm(h)

        # Only compute audio logits when caller explicitly passed a mask.
        proj_mask = audio_out_mask if caller_wants_audio else None
        text_logits, audio_logits = self.audio_decoder_proj(h, proj_mask)
        return text_logits, audio_logits

    def sanitize(self, weights: dict) -> dict:
        """Convert bosonai safetensors keys → our MLX state dict.

        Higgs's safetensors already use HuggingFace Llama naming with the
        Higgs-specific audio_* additions, so this is essentially a pass-through.
        Kept as a method so sanitize() can grow if we later add transposes
        (e.g. if any conv-style weights need layout remapping).
        """
        return dict(weights)

    # ------------------------------------------------------------------
    # Generation — AUDIO_INIT + delay-pattern ramp-in + EOS ramp-out
    # ------------------------------------------------------------------

    def _generate_raw_frames(
        self,
        inputs_embeds: mx.array,
        audio_out_mask: mx.array,
        *,
        max_new_frames: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        ras_win_len: Optional[int],
        ras_max_repeat: int,
        sampling_warmup_frames: int,
    ) -> Iterator[Tuple[mx.array, dict]]:
        """Run prefill then yield ([K] int32 frame, info) per generation step.

        Frames are emitted in *delay-pattern* form (not aligned). Caller is
        responsible for stacking, revert_delay_pattern, and boundary trim.

        Implements the Higgs v2 generation state machine documented at
        project_higgs_audio_init_delay_pattern.md:
          - frame 0: forced all audio_stream_bos_id (AUDIO_INIT — NOT sampled)
          - frames 1..K-1: ramp-in (sample [0..num_delay], force tail to BOS)
          - full sampling once num_delay >= K-1
          - any EOS in a frame triggers a K-frame EOS ramp-out, then stop
        """
        cfg = self.config
        K = cfg.audio_num_codebooks
        BOS = cfg.audio_stream_bos_id
        EOS = cfg.audio_stream_eos_id
        stride = cfg.audio_codebook_size + 2

        # Prefill. Discard logits (they come from a text position that was
        # never trained for direct audio prediction — use AUDIO_INIT instead).
        cache = make_prompt_cache(self)
        _, _ = self(
            inputs_embeds=inputs_embeds, audio_out_mask=audio_out_mask, cache=cache
        )
        mx.eval(*[c.state for c in cache])  # force prefill graph materialization

        # Frame 0 = synthetic all-BOS (AUDIO_INIT).
        frame0 = mx.full((K,), BOS, dtype=mx.int32)
        yield frame0, {"step": 0, "source": "audio_init", "num_delay": 0}

        num_delay = 0
        num_remaining_delays: Optional[int] = None
        step_mask = mx.ones((1, 1), dtype=mx.bool_)
        prev_frame = frame0

        # RAS window: per-codebook rolling history of recent sampled tokens.
        # Seeded with frame 0 (all-BOS) so the rolling window is already filled
        # enough to check from frame 1. We track python ints (small footprint).
        ras_enabled = ras_win_len is not None and ras_win_len > 0
        # rows are codebooks, columns are recent frames (most recent last)
        ras_window: list[list[int]] = [[BOS] for _ in range(K)] if ras_enabled else []

        for step in range(max_new_frames):
            last = prev_frame.reshape(K, 1)
            embed = lookup_audio_embedding(self.audio_codebook_embeddings, last, stride)
            _, audio_logits = self(
                inputs_embeds=embed[None], audio_out_mask=step_mask, cache=cache
            )
            # Greedy during warmup — pins trajectory through the low-context
            # ramp-in region where quantization noise otherwise amplifies
            # sampling variance into divergent outputs. After warmup, switch to
            # temperature+top_p for natural prosody variance.
            if step < sampling_warmup_frames:
                sampled = greedy_sample_audio(audio_logits)
            else:
                sampled = sample_audio(
                    audio_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
            mx.eval(sampled)
            tok_list = sampled[0, 0].tolist()  # list of K ints

            # --- RAS (repetition-avoidance sampling) BEFORE delay-pattern forcing.
            # For each codebook, if the sampled token has appeared >= max_repeat
            # times in the recent window, resample that codebook greedily (temp=0).
            # This catches first-token-dominance loops that quantization noise
            # triggers especially in the low-context prefix window.
            if ras_enabled:
                resample_mask_needed = False
                per_cb_resample: list[bool] = []
                for cb_i in range(K):
                    window = ras_window[cb_i][-ras_win_len:]
                    count = sum(1 for v in window if v == tok_list[cb_i])
                    need = count >= ras_max_repeat
                    per_cb_resample.append(need)
                    if need:
                        resample_mask_needed = True
                if resample_mask_needed:
                    greedy = greedy_sample_audio(audio_logits)
                    mx.eval(greedy)
                    greedy_tok = greedy[0, 0].tolist()
                    for cb_i in range(K):
                        if per_cb_resample[cb_i]:
                            tok_list[cb_i] = greedy_tok[cb_i]

            if cfg.use_delay_pattern:
                # Ramp-in: force tail codebooks to BOS.
                if num_delay + 1 < K:
                    for k_i in range(num_delay + 1, K):
                        tok_list[k_i] = BOS
                    num_delay += 1

                # Ramp-out in progress.
                if num_remaining_delays is not None:
                    force_until = K - num_remaining_delays
                    for k_i in range(force_until):
                        tok_list[k_i] = EOS
                    num_remaining_delays -= 1
                else:
                    # Check whether any codebook emitted EOS → start ramp-out.
                    eos_positions = [i for i, v in enumerate(tok_list) if v == EOS]
                    if eos_positions:
                        last_eos = eos_positions[-1]
                        for k_i in range(last_eos):
                            tok_list[k_i] = EOS
                        num_remaining_delays = K - last_eos - 1

            tok_arr = mx.array(tok_list, dtype=mx.int32)
            if ras_enabled:
                for cb_i in range(K):
                    ras_window[cb_i].append(tok_list[cb_i])
                    if len(ras_window[cb_i]) > ras_win_len + 4:
                        # keep window bounded
                        ras_window[cb_i] = ras_window[cb_i][-ras_win_len:]
            info = {
                "step": step + 1,
                "source": "sampled",
                "num_delay": num_delay,
                "num_remaining_delays": num_remaining_delays,
            }
            yield tok_arr, info
            prev_frame = tok_arr

            if (
                cfg.use_delay_pattern
                and num_remaining_delays is not None
                and num_remaining_delays <= 0
            ):
                return

    def generate(
        self,
        inputs_embeds: mx.array,
        audio_out_mask: mx.array,
        *,
        max_new_frames: int = 900,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = None,
        ras_win_len: Optional[int] = 7,
        ras_max_repeat: int = 2,
        sampling_warmup_frames: int = 0,
        trim_boundaries: bool = True,
    ) -> Tuple[mx.array, dict]:
        """Generate audio codebook tokens from a prefill of audio_out_mask=[...].

        Args:
            inputs_embeds: [1, T_prompt, hidden] prompt embeddings with any
                text + reference-audio context already stitched.
            audio_out_mask: [1, T_prompt] bool — True at positions routed
                through the audio dual-FFN path.
            max_new_frames: hard cap on generation length.
            temperature / top_p / top_k: sampling controls.
            trim_boundaries: drop the synthetic BOS-seed (col 0) and EOS-seal
                (col -1) columns after revert_delay_pattern. Default True;
                required for clean codec decode (otherwise those columns
                clip to codec token 1023 → audible click at sample-zero and end).

        Returns:
            (aligned_tokens, info):
                aligned_tokens: [K, T_audio] int32, codec-ready
                info: dict with num_frames, stop_reason, timing hooks, etc.
        """
        frames = []
        stop_reason = "max-frames"
        for tok, meta in self._generate_raw_frames(
            inputs_embeds,
            audio_out_mask,
            max_new_frames=max_new_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=ras_win_len,
            ras_max_repeat=ras_max_repeat,
            sampling_warmup_frames=sampling_warmup_frames,
        ):
            frames.append(tok)
            if (
                meta.get("num_remaining_delays") is not None
                and meta["num_remaining_delays"] <= 0
            ):
                stop_reason = f"eos-ramp-complete-at-frame-{meta['step']}"
        if len(frames) - 1 >= max_new_frames and stop_reason == "max-frames":
            stop_reason = f"max-frames-{max_new_frames}"

        sequence = mx.stack(frames, axis=1).astype(mx.int32)  # [K, N]
        aligned = revert_delay_pattern(sequence)  # [K, N-K+1]
        if trim_boundaries and aligned.shape[1] >= 2:
            aligned = aligned[:, 1:-1]
        aligned = mx.clip(aligned, 0, self.config.audio_codebook_size - 1)
        info = {
            "num_frames_raw": sequence.shape[1],
            "num_frames_aligned": aligned.shape[1],
            "stop_reason": stop_reason,
        }
        return aligned, info

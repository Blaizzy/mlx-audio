import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.base import GenerationResult

from .codec import CodecDecoder, CodecEncoder, create_segment_attention_mask
from .config import ModelConfig
from .diffusion_head import DiffusionHead
from .gray_code import decode_gray_code_to_time, encode_time_with_gray_code
from .llama import LlamaModel
from .text_utils import normalize_text


@dataclass
class EncoderOutput:
    audio: mx.array
    audio_len: mx.array
    text: List[str]
    token_positions: mx.array
    token_values: mx.array
    sample_rate: int = 24000
    text_tokens: Optional[mx.array] = None
    text_tokens_len: Optional[mx.array] = None
    token_masks: Optional[mx.array] = None


class Model(nn.Module):
    """TADA: Text-Audio Dual Alignment TTS model.

    Architecture:
        - Llama backbone for autoregressive text generation
        - Flow matching diffusion head for acoustic feature + duration prediction
        - Codec decoder for waveform synthesis
        - Codec encoder + aligner for reference audio encoding
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Llama backbone
        self.model = LlamaModel(config)

        # TADA-specific embeddings
        self.acoustic_proj = nn.Linear(config.acoustic_dim, config.hidden_size)
        self.time_start_embed = nn.Embedding(
            config.num_time_classes, config.hidden_size
        )
        self.time_end_embed = nn.Embedding(config.num_time_classes, config.hidden_size)
        self.acoustic_mask_emb = nn.Embedding(2, config.hidden_size)

        # Flow matching diffusion head
        self.num_time_bits = math.ceil(math.log2(config.num_time_classes))
        self.time_dim = 2 * self.num_time_bits
        latent_size = config.acoustic_dim + self.time_dim

        head_hidden = (
            config.hidden_size
            if config.bottleneck_dim is None
            else config.bottleneck_dim
        )
        self.prediction_head = DiffusionHead(
            hidden_size=head_hidden,
            latent_size=latent_size,
            head_layers=config.head_layers,
            head_ffn_ratio=config.head_ffn_ratio,
            rms_norm_eps=config.rms_norm_eps,
        )

        if config.bottleneck_dim is not None:
            self.bottleneck_proj = nn.Linear(config.hidden_size, config.bottleneck_dim)
        else:
            self.bottleneck_proj = None

        # Codec decoder (weights loaded from tada-codec in post_load_hook)
        self.decoder = CodecDecoder(
            hidden_dim=config.decoder_hidden_dim,
            embed_dim=config.decoder_embed_dim,
            d_model=config.decoder_d_model,
            strides=config.decoder_strides,
            num_attn_layers=config.decoder_num_attn_layers,
            num_attn_heads=config.decoder_num_attn_heads,
            attn_dim_feedforward=config.decoder_attn_dim_feedforward,
            block_attention=config.decoder_block_attention,
        )

        # Encoder and aligner (loaded in post_load_hook)
        self._encoder = None
        self._aligner = None
        self._tokenizer = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def num_eos_tokens(self):
        return self.config.shift_acoustic

    def _get_eos_ids(self) -> List[int]:
        eos = self.config.eos_token_id
        if isinstance(eos, list):
            ids = list(eos)
        else:
            ids = [eos]
        # Also include <|eot_id|> (128009) as stop token for Llama chat
        if self._tokenizer is not None:
            eot_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id not in ids:
                ids.append(eot_id)
        return ids

    def _lm_head_forward(self, hidden_states: mx.array) -> mx.array:
        return hidden_states @ self.model.embed_tokens.weight.T

    # ========================================================================
    # Flow matching
    # ========================================================================

    @staticmethod
    def _scheduled_cfg(base_scale: float, t: float, schedule: str) -> float:
        if schedule == "constant" or base_scale == 1.0:
            return base_scale
        if schedule == "linear":
            return 1.0 + (base_scale - 1.0) * (1.0 - t)
        if schedule == "cosine":
            return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))
        return base_scale

    @staticmethod
    def _build_time_schedule(num_steps: int, schedule: str) -> mx.array:
        if schedule == "cosine":
            u = mx.linspace(0, 1, num_steps + 1)
            return 0.5 * (1 - mx.cos(math.pi * u))
        if schedule == "logsnr":
            log_snr = mx.linspace(5.0, -5.0, num_steps + 1)
            t_span = mx.sigmoid(-log_snr / 2)
            # Ensure exact endpoints
            t_span = mx.concatenate([mx.array([0.0]), t_span[1:-1], mx.array([1.0])])
            return t_span
        return mx.linspace(0, 1, num_steps + 1)

    def _compute_velocity(
        self,
        speech_input: mx.array,
        t: mx.array,
        cond_input: mx.array,
        neg_cond_input: mx.array,
        acoustic_cfg: float,
        duration_cfg: float,
    ) -> mx.array:
        bottleneck = (
            self.bottleneck_proj if self.bottleneck_proj is not None else lambda x: x
        )

        if acoustic_cfg != 1.0:
            speech_combined = mx.concatenate([speech_input, speech_input], axis=0)
            t_combined = mx.repeat(t, speech_input.shape[0] * 2).astype(
                speech_input.dtype
            )
            cond_pos = cond_input.squeeze(1) if cond_input.ndim == 3 else cond_input
            cond_neg = (
                neg_cond_input.squeeze(1)
                if neg_cond_input.ndim == 3
                else neg_cond_input
            )
            cond_combined = mx.concatenate([cond_pos, cond_neg], axis=0)

            velocity_combined = self.prediction_head(
                speech_combined,
                t_combined,
                condition=bottleneck(cond_combined),
            )

            B = speech_input.shape[0]
            velocity_pos = velocity_combined[:B]
            velocity_neg = velocity_combined[B:]

            acoustic_dim = self.config.acoustic_dim
            velocity = mx.concatenate(
                [
                    (
                        velocity_neg[..., :acoustic_dim]
                        + acoustic_cfg
                        * (
                            velocity_pos[..., :acoustic_dim]
                            - velocity_neg[..., :acoustic_dim]
                        )
                    ),
                    (
                        velocity_neg[..., acoustic_dim:]
                        + duration_cfg
                        * (
                            velocity_pos[..., acoustic_dim:]
                            - velocity_neg[..., acoustic_dim:]
                        )
                    ),
                ],
                axis=-1,
            )
        else:
            cond = cond_input.squeeze(1) if cond_input.ndim == 3 else cond_input
            velocity = self.prediction_head(
                speech_input,
                mx.repeat(t, speech_input.shape[0]).astype(speech_input.dtype),
                condition=bottleneck(cond),
            )
        return velocity

    def _solve_flow_matching(
        self,
        speech: mx.array,
        cond: mx.array,
        neg_cond: mx.array,
        num_steps: int = 20,
        acoustic_cfg_scale: float = 1.6,
        duration_cfg_scale: float = 1.0,
        cfg_schedule: str = "cosine",
        time_schedule: str = "logsnr",
    ) -> mx.array:
        t_span = self._build_time_schedule(num_steps, time_schedule)
        t_curr = t_span[0]

        for i in range(1, len(t_span)):
            dt = t_span[i] - t_curr
            t_val = float(t_curr.item())
            a_cfg = self._scheduled_cfg(acoustic_cfg_scale, t_val, cfg_schedule)
            d_cfg = self._scheduled_cfg(duration_cfg_scale, t_val, cfg_schedule)

            velocity = self._compute_velocity(
                speech, t_curr, cond, neg_cond, a_cfg, d_cfg
            )
            speech = speech + dt * velocity
            t_curr = t_span[i]

        return speech

    # ========================================================================
    # Forward pass
    # ========================================================================

    def forward_one_step(
        self,
        input_ids: mx.array,
        acoustic_features: mx.array,
        acoustic_masks: mx.array,
        time_len_before: mx.array,
        time_len_after: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        compute_logits: bool = True,
    ):
        inputs_embeds = (
            self.model.embed_tokens(input_ids)
            + self.acoustic_proj(acoustic_features)
            + self.acoustic_mask_emb(acoustic_masks.astype(mx.int32))
            + self.time_start_embed(time_len_before)
            + self.time_end_embed(time_len_after)
        )

        last_hidden, new_cache = self.model(inputs_embeds=inputs_embeds, cache=cache)

        logits = self._lm_head_forward(last_hidden) if compute_logits else None
        return last_hidden, logits, new_cache

    def _build_prompt_inputs_embeds(
        self,
        input_ids: mx.array,
        prompt_acoustic_features: Optional[mx.array],
        prompt_acoustic_masks: Optional[mx.array],
        prompt_time_len_before: Optional[mx.array],
        prompt_time_len_after: Optional[mx.array],
        prompt_len: int,
    ) -> mx.array:
        B = input_ids.shape[0]
        shift = self.config.shift_acoustic

        token_emb = self.model.embed_tokens(input_ids[:, :prompt_len])

        acoustic_full = mx.zeros((B, prompt_len, self.config.acoustic_dim))
        masks_full = mx.zeros((B, prompt_len), dtype=mx.int32)
        if prompt_acoustic_features is not None and prompt_acoustic_masks is not None:
            n_ac = min(prompt_len - shift - 1, prompt_acoustic_features.shape[1])
            if n_ac > 0:
                # Build by explicit indexing
                indices = list(range(shift + 1, shift + 1 + n_ac))
                for idx_out, idx_in in enumerate(range(n_ac)):
                    acoustic_full = acoustic_full.at[:, indices[idx_out]].add(
                        prompt_acoustic_features[:, idx_in]
                    )
                    masks_full = masks_full.at[:, indices[idx_out]].add(
                        prompt_acoustic_masks[:, idx_in].astype(mx.int32)
                    )

        acoustic_emb = self.acoustic_proj(acoustic_full) + self.acoustic_mask_emb(
            masks_full
        )

        time_before = mx.zeros((B, prompt_len), dtype=mx.int32)
        time_after = mx.zeros((B, prompt_len), dtype=mx.int32)
        if prompt_time_len_before is not None and prompt_time_len_after is not None:
            n_t = min(
                prompt_len - shift - 1,
                prompt_time_len_before.shape[1] - 1,
            )
            if n_t > 0:
                for idx_out, idx_in in enumerate(range(1, 1 + n_t)):
                    pos = shift + 1 + idx_out
                    time_before = time_before.at[:, pos].add(
                        prompt_time_len_before[:, idx_in].astype(mx.int32)
                    )
                    time_after = time_after.at[:, pos].add(
                        prompt_time_len_after[:, idx_in].astype(mx.int32)
                    )

        time_emb = self.time_start_embed(time_before) + self.time_end_embed(time_after)

        return token_emb + acoustic_emb + time_emb

    # ========================================================================
    # Text sampling
    # ========================================================================

    def _sample_next_token(
        self,
        logits: mx.array,
        input_ids: mx.array,
        temperature: float = 0.6,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        pad_token_id: int = -1,
        text_only_logits: Optional[mx.array] = None,
        text_only_logit_scale: float = 0.0,
    ) -> mx.array:
        token_logits = logits[:, -1, :]

        if pad_token_id >= 0:
            token_logits = token_logits.at[:, pad_token_id].add(
                mx.array(float("-inf")) - token_logits[:, pad_token_id]
            )

        # Blend text-only logits (after pad suppression, matching PyTorch order)
        if text_only_logits is not None and text_only_logit_scale > 0.0:
            scale = text_only_logit_scale
            token_logits = (text_only_logits[:, -1, :] * scale + token_logits) / (
                scale + 1
            )

        if repetition_penalty != 1.0:
            # Vectorized repetition penalty
            prev_tokens = input_ids[0]
            scores = token_logits[0]
            prev_scores = scores[prev_tokens]
            penalty = mx.where(
                prev_scores < 0,
                mx.array(repetition_penalty),
                mx.array(1.0 / repetition_penalty),
            )
            new_scores = prev_scores * penalty
            token_logits = token_logits.at[0, prev_tokens].add(new_scores - prev_scores)

        token_logits = token_logits / temperature

        if top_k > 0:
            k = min(top_k, token_logits.shape[-1])
            top_vals = mx.sort(token_logits, axis=-1)[:, -k:]
            threshold = top_vals[:, 0:1]
            token_logits = mx.where(
                token_logits < threshold,
                mx.array(float("-inf")),
                token_logits,
            )

        if 0.0 < top_p < 1.0:
            # Vectorized top-p: sort descending, mask in sorted space
            sorted_indices = mx.argsort(-token_logits, axis=-1)
            sorted_logits = mx.take_along_axis(token_logits, sorted_indices, axis=-1)
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            # Mask tokens where cumulative prob exceeds top_p
            sorted_mask = cumulative_probs - mx.softmax(sorted_logits, axis=-1) >= top_p
            sorted_logits = mx.where(
                sorted_mask, mx.array(float("-inf")), sorted_logits
            )
            # Scatter back using argsort of indices
            restore_indices = mx.argsort(sorted_indices, axis=-1)
            token_logits = mx.take_along_axis(sorted_logits, restore_indices, axis=-1)

        probs = mx.softmax(token_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs + 1e-10))
        return mx.expand_dims(next_token, -1)

    # ========================================================================
    # Waveform decoding
    # ========================================================================

    def _decode_wav(self, encoded: mx.array, time_before: mx.array) -> mx.array:
        time_before = time_before[: encoded.shape[0] + 1]
        if time_before.shape[0] == 0:
            return mx.zeros((1, 0, 1))

        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            n_zeros = max(0, int(time_before[pos].item()) - 1)
            if n_zeros > 0:
                encoded_expanded.append(mx.zeros((n_zeros, encoded.shape[-1])))
            encoded_expanded.append(mx.expand_dims(encoded[pos], 0))

        n_trailing = int(time_before[-1].item())
        if n_trailing > 0:
            encoded_expanded.append(mx.zeros((n_trailing, encoded.shape[-1])))

        encoded_expanded = mx.concatenate(encoded_expanded, axis=0)
        encoded_expanded = mx.expand_dims(encoded_expanded, 0)  # (1, T, D)

        token_masks = (mx.sqrt(mx.sum(encoded_expanded**2, axis=-1)) != 0).astype(
            mx.int32
        )

        return self.decoder.generate(encoded_expanded, token_masks)

    # ========================================================================
    # Encoder (loaded separately)
    # ========================================================================

    def _load_decoder(self, codec_path: str):
        """Load decoder from tada-codec weights (separate from main model)."""
        codec_model_path = Path(codec_path)
        if not codec_model_path.exists():
            from huggingface_hub import snapshot_download

            codec_model_path = Path(
                snapshot_download(codec_path, local_files_only=False)
            )

        decoder_path = codec_model_path / "decoder"
        if decoder_path.exists():
            decoder_weights = _load_codec_weights(decoder_path, component="decoder")
            self.decoder.load_weights(list(decoder_weights.items()), strict=False)
            self.decoder.eval()
        else:
            print(f"Warning: Decoder path not found: {decoder_path}")

    def _load_encoder(self, codec_path: str):
        """Load encoder from tada-codec weights using transformers for aligner."""
        codec_model_path = Path(codec_path)
        if not codec_model_path.exists():
            from huggingface_hub import snapshot_download

            codec_model_path = Path(
                snapshot_download(codec_path, local_files_only=False)
            )

        # Load encoder weights
        encoder_path = codec_model_path / "encoder"
        if encoder_path.exists():
            self._encoder = CodecEncoder(
                hidden_dim=self.config.encoder_hidden_dim,
                embed_dim=self.config.encoder_embed_dim,
                strides=self.config.encoder_strides,
                num_attn_layers=self.config.encoder_num_attn_layers,
                num_attn_heads=self.config.encoder_num_attn_heads,
                attn_dim_feedforward=self.config.encoder_attn_dim_feedforward,
                block_attention=self.config.encoder_block_attention,
                std=self.config.encoder_std,
                acoustic_mean=self.config.acoustic_mean,
                acoustic_std=self.config.acoustic_std,
            )
            encoder_weights = _load_codec_weights(encoder_path, component="encoder")
            self._encoder.load_weights(list(encoder_weights.items()), strict=False)
            self._encoder.eval()

        # Load aligner using transformers
        try:
            self._load_aligner(codec_model_path)
        except Exception as e:
            print(f"Warning: Could not load aligner: {e}")
            print("Voice cloning from reference audio will not be available.")

    def _load_aligner(self, codec_model_path: Path):
        """Load wav2vec2-based aligner using transformers."""
        import json
        import tempfile

        import torch
        from safetensors.torch import load_file as torch_load_file
        from transformers import AutoTokenizer, Wav2Vec2Config, Wav2Vec2ForCTC

        aligner_path = codec_model_path / "aligner"
        if not aligner_path.exists():
            return

        # Load raw weights and strip 'encoder.' prefix
        raw_weights = torch_load_file(str(aligner_path / "model.safetensors"))
        stripped = {}
        for k, v in raw_weights.items():
            new_key = k.replace("encoder.", "", 1) if k.startswith("encoder.") else k
            stripped[new_key] = v

        # Resolve weight normalization for pos_conv_embed
        resolved = {}
        skip = set()
        for k, v in stripped.items():
            if "parametrizations.weight.original0" in k:
                k1 = k.replace("original0", "original1")
                if k1 in stripped:
                    g = v
                    d = stripped[k1]
                    flat = d.reshape(d.shape[0], -1)
                    norm = torch.sqrt(torch.sum(flat**2, dim=1, keepdim=True))
                    norm = norm.reshape(d.shape[0], *([1] * (len(d.shape) - 1)))
                    effective = g * d / (norm + 1e-12)
                    new_key = k.replace(".parametrizations.weight.original0", ".weight")
                    resolved[new_key] = effective
                    skip.add(k)
                    skip.add(k1)
            elif "parametrizations.weight.original1" in k:
                skip.add(k)
        for k, v in stripped.items():
            if k not in skip:
                resolved[k] = v

        # Create Wav2Vec2ForCTC with correct config (24 layers, hidden=1024)
        w2v_config = Wav2Vec2Config(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            conv_dim=[512, 512, 512, 512, 512, 512, 512],
            conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            conv_stride=[5, 2, 2, 2, 2, 2, 2],
            vocab_size=128256,
            do_stable_layer_norm=True,
        )
        self._aligner_model = Wav2Vec2ForCTC(w2v_config)
        self._aligner_model.load_state_dict(resolved, strict=False)
        self._aligner_model.eval()

        # Use Llama tokenizer for text tokenization
        self._tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct"
        )

    def encode_reference(
        self,
        audio: mx.array,
        text: str,
        sample_rate: int = 24000,
    ) -> EncoderOutput:
        """Encode reference audio for voice cloning.

        Args:
            audio: Reference audio waveform, shape (T,) or (1, T)
            text: Transcript of the reference audio
            sample_rate: Sample rate of the audio

        Returns:
            EncoderOutput with acoustic features and alignment info
        """
        if self._tokenizer is None or self._aligner_model is None:
            raise RuntimeError(
                "Encoder/aligner not loaded. Ensure HumeAI/tada-codec is available."
            )

        import torch
        import torchaudio

        # Prepare audio
        if audio.ndim == 1:
            audio_np = np.array(audio)
        else:
            audio_np = np.array(audio[0])

        audio_torch = torch.from_numpy(audio_np).float().unsqueeze(0)

        # Resample to 24kHz if needed
        if sample_rate != 24000:
            audio_torch = torchaudio.functional.resample(
                audio_torch, sample_rate, 24000
            )
            sample_rate = 24000

        # Resample to 16kHz for aligner
        audio_16k = torchaudio.functional.resample(audio_torch, 24000, 16000)

        # Normalize text and tokenize
        text = normalize_text(text)
        text_tokens = self._tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        )

        # Run aligner
        with torch.no_grad():
            logits = self._aligner_model(audio_16k).logits  # (1, T, vocab)

        # DP alignment
        token_positions, token_masks = _align_text_tokens(
            logits[0].numpy(),
            text_tokens[0].numpy(),
            audio_torch.shape[-1],
            sample_rate,
        )

        token_positions_mx = mx.array(token_positions).reshape(1, -1)
        token_masks_mx = mx.array(token_masks).reshape(1, -1)

        # Run encoder
        audio_mx = mx.array(audio_np).reshape(1, -1)
        token_values = self._encoder.forward(
            audio_mx,
            token_positions_mx,
            token_masks_mx,
            sample=True,
        )

        text_tokens_mx = mx.array(text_tokens[0].numpy()).reshape(1, -1)

        return EncoderOutput(
            audio=audio_mx,
            audio_len=mx.array([audio_mx.shape[-1]]),
            text=[text],
            text_tokens=text_tokens_mx,
            text_tokens_len=mx.array([text_tokens_mx.shape[-1]]),
            token_positions=token_positions_mx,
            token_values=token_values,
            token_masks=token_masks_mx,
        )

    # ========================================================================
    # Generation
    # ========================================================================

    def generate(
        self,
        text: str,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        voice: Optional[str] = None,
        temperature: float = 0.6,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        acoustic_cfg_scale: float = 1.6,
        duration_cfg_scale: float = 1.0,
        cfg_schedule: str = "cosine",
        noise_temperature: float = 0.9,
        num_flow_matching_steps: int = 20,
        time_schedule: str = "logsnr",
        num_transition_steps: int = 5,
        max_tokens: int = 1024,
        speed_up_factor: Optional[float] = None,
        text_only_logit_scale: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            ref_audio: Reference audio for voice cloning, shape (T,) at 24kHz
            ref_text: Transcript of reference audio
            voice: Voice preset name (not used yet)
            temperature: Text sampling temperature
            acoustic_cfg_scale: CFG scale for acoustic features
            duration_cfg_scale: CFG scale for duration
            noise_temperature: Initial noise scaling
            num_flow_matching_steps: Number of ODE steps
            max_tokens: Maximum generation steps
            speed_up_factor: If set, re-runs generation with time durations
                scaled by 1/factor for speed control (>1 = faster, <1 = slower)
            text_only_logit_scale: Scale for blending text-only logits with main
                logits during token sampling (0 = disabled)

        Yields:
            GenerationResult with audio and metadata
        """
        start_time = time.perf_counter()

        text = normalize_text(text)

        # Encode reference audio if provided
        prompt = None
        if ref_audio is not None and ref_text is not None:
            prompt = self.encode_reference(ref_audio, ref_text)
        elif self._encoder is not None:
            # Create empty prompt
            prompt = EncoderOutput(
                audio=mx.zeros((1, 0)),
                audio_len=mx.zeros((1,)),
                text=[""],
                text_tokens=mx.zeros((1, 0), dtype=mx.int32),
                text_tokens_len=mx.zeros((1,), dtype=mx.int32),
                token_positions=mx.zeros((1, 0), dtype=mx.int32),
                token_values=mx.zeros((1, 0, self.config.acoustic_dim)),
            )

        if prompt is None:
            prompt = EncoderOutput(
                audio=mx.zeros((1, 0)),
                audio_len=mx.zeros((1,)),
                text=[""],
                text_tokens=mx.zeros((1, 0), dtype=mx.int32),
                text_tokens_len=mx.zeros((1,), dtype=mx.int32),
                token_positions=mx.zeros((1, 0), dtype=mx.int32),
                token_values=mx.zeros((1, 0, self.config.acoustic_dim)),
            )

        # Build input IDs with special tokens
        prompt_text = prompt.text[0] if prompt.text[0] else ""
        full_text = prompt_text + text

        tokenizer = self._tokenizer
        text_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        # Add BOS + system prompt + assistant header
        bos_id = tokenizer.bos_token_id
        eos_ids = self._get_eos_ids()
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        start_header = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

        system_tokens = tokenizer.encode("system", add_special_tokens=False)
        assistant_tokens = tokenizer.encode("assistant", add_special_tokens=False)

        # Prefix tokens: BOS is separate, prefix_text_tokens follows
        prefix_text_tokens = (
            [start_header]
            + system_tokens
            + [end_header, eot_id, start_header]
            + assistant_tokens
            + [end_header]
        )
        prefix = [bos_id] + prefix_text_tokens
        prefix_len = len(prefix_text_tokens)  # Without BOS, matches PyTorch

        # EOS tokens at end
        eos_suffix = [eot_id] * self.num_eos_tokens

        input_ids_list = prefix + text_tokens + eos_suffix
        input_ids = mx.array([input_ids_list], dtype=mx.int32)

        # Compute prompt acoustic features and timing
        prompt_acoustic_features = prompt.token_values
        prompt_acoustic_masks = None
        prompt_time_before = None
        prompt_time_after = None

        has_prompt_audio = prompt.token_values.shape[1] > 0

        if has_prompt_audio:
            prompt_acoustic_masks = mx.ones(
                prompt_acoustic_features.shape[:2], dtype=mx.int32
            )

            # Compute time gaps from token positions
            token_positions = prompt.token_positions

            # Compute time gaps (match reference _build_inputs: no audio_feat_len)
            tp = np.array(token_positions[0])
            tp_padded = np.insert(tp, 0, 1)
            raw_gaps = np.clip(tp - tp_padded[:-1], 0, self.config.num_time_classes - 1)
            time_gaps = np.insert(raw_gaps, 0, 0)

            prompt_time_before = mx.array(time_gaps[:-1], dtype=mx.int32).reshape(1, -1)
            prompt_time_after = mx.array(time_gaps[1:], dtype=mx.int32).reshape(1, -1)

            # Pad for prefix tokens
            prompt_acoustic_features = mx.pad(
                prompt_acoustic_features,
                [(0, 0), (prefix_len, 0), (0, 0)],
            )
            prompt_acoustic_masks = mx.pad(
                prompt_acoustic_masks,
                [(0, 0), (prefix_len, 0)],
            )
            prompt_time_before = mx.pad(
                prompt_time_before,
                [(0, 0), (prefix_len, 0)],
            )
            prompt_time_after = mx.pad(
                prompt_time_after,
                [(0, 0), (prefix_len, 0)],
            )

            # Trim transition steps
            if (
                num_transition_steps > 0
                and prompt_acoustic_features.shape[1] > num_transition_steps
            ):
                prompt_acoustic_features = prompt_acoustic_features[
                    :, :-num_transition_steps, :
                ]
                prompt_acoustic_masks = prompt_acoustic_masks[:, :-num_transition_steps]
                prompt_time_before = prompt_time_before[:, :-num_transition_steps]
                prompt_time_after = prompt_time_after[:, :-num_transition_steps]

        # Shift mask left by 1 (matches PyTorch behavior)
        if has_prompt_audio and prompt_acoustic_masks is not None:
            prompt_acoustic_masks = mx.concatenate(
                [
                    prompt_acoustic_masks[:, 1:],
                    mx.ones_like(prompt_acoustic_masks[:, :1]),
                ],
                axis=-1,
            )

        # Mask prompt text tokens for prefill (reference: _build_inputs)
        # LLM should not see raw text in the prompt region — only structural tokens
        if has_prompt_audio and prompt_acoustic_features is not None:
            pad_token_id = self._tokenizer.convert_tokens_to_ids(
                "<|finetune_right_pad_id|>"
            )
            prompt_token_len = prompt_acoustic_features.shape[1]
            prompt_ids = input_ids[:, :prompt_token_len]
            is_start = prompt_ids == start_header
            is_end = prompt_ids == end_header
            header_depth = mx.cumsum(is_start.astype(mx.int32), axis=1) - mx.cumsum(
                is_end.astype(mx.int32), axis=1
            )
            in_header = (header_depth > 0) | is_start | is_end
            is_structural = (
                in_header
                | (prompt_ids == eot_id)
                | (prompt_ids == bos_id)
                | (prompt_ids == 128001)
            )
            masked_prompt = mx.where(
                is_structural,
                prompt_ids,
                mx.full(prompt_ids.shape, pad_token_id, dtype=mx.int32),
            )
            input_ids = mx.concatenate(
                [masked_prompt, input_ids[:, prompt_token_len:]], axis=1
            )

        # Run autoregressive generation
        audio_result = self._generate_loop(
            input_ids=input_ids,
            prompt_acoustic_features=(
                prompt_acoustic_features if has_prompt_audio else None
            ),
            prompt_acoustic_masks=prompt_acoustic_masks,
            prompt_time_before=prompt_time_before,
            prompt_time_after=prompt_time_after,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            acoustic_cfg_scale=acoustic_cfg_scale,
            duration_cfg_scale=duration_cfg_scale,
            cfg_schedule=cfg_schedule,
            noise_temperature=noise_temperature,
            num_flow_matching_steps=num_flow_matching_steps,
            time_schedule=time_schedule,
            speed_up_factor=speed_up_factor,
            text_only_logit_scale=text_only_logit_scale,
            verbose=verbose,
            has_prompt_audio=has_prompt_audio,
            num_prompt_features=(
                prompt_acoustic_features.shape[1] if has_prompt_audio else 0
            ),
            num_transition_steps=num_transition_steps,
            prefix_len=prefix_len,
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if audio_result is not None and audio_result.size > 0:
            samples = audio_result.shape[0]
            audio_duration = samples / self.sample_rate
        else:
            samples = 0
            audio_duration = 0
            audio_result = mx.array([])

        duration_str = f"{int(audio_duration // 3600):02d}:{int(audio_duration % 3600 // 60):02d}:{int(audio_duration % 60):02d}.{int((audio_duration % 1) * 1000):03d}"
        rtf = audio_duration / elapsed if elapsed > 0 else 0

        yield GenerationResult(
            audio=audio_result,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=input_ids.shape[1],
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": input_ids.shape[1],
                "tokens-per-sec": (
                    round(input_ids.shape[1] / elapsed, 2) if elapsed > 0 else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _generate_loop(
        self,
        input_ids: mx.array,
        prompt_acoustic_features: Optional[mx.array],
        prompt_acoustic_masks: Optional[mx.array],
        prompt_time_before: Optional[mx.array],
        prompt_time_after: Optional[mx.array],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        acoustic_cfg_scale: float,
        duration_cfg_scale: float,
        cfg_schedule: str,
        noise_temperature: float,
        num_flow_matching_steps: int,
        time_schedule: str,
        speed_up_factor: Optional[float] = None,
        text_only_logit_scale: float = 0.0,
        verbose: bool = False,
        has_prompt_audio: bool = False,
        num_prompt_features: int = 0,
        num_transition_steps: int = 5,
        num_extra_steps: int = 0,
        prefix_len: int = 0,
    ) -> Optional[mx.array]:
        B = input_ids.shape[0]
        shift = self.config.shift_acoustic

        # Text-driven generation: run for input_ids length + num_extra_steps
        # When num_extra_steps > 0, strip EOS tokens and let model generate freely
        if num_extra_steps > 0:
            input_ids = input_ids[:, : -self.num_eos_tokens]

        num_steps = min(input_ids.shape[1] + num_extra_steps, max_tokens)

        pad_token_id = -1
        try:
            pad_token_id = self._tokenizer.convert_tokens_to_ids(
                "<|finetune_right_pad_id|>"
            )
        except Exception:
            pass

        # Negative conditioning: match PyTorch default (negative_step_output)
        # When CFG is active, run a double batch with text-masked negative input
        tokenizer = self._tokenizer
        need_neg_batch = acoustic_cfg_scale != 1.0
        use_text_only_logit_scale = text_only_logit_scale > 0.0
        if need_neg_batch:
            start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
            end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # Determine prefill length
        prompt_len = input_ids.shape[1]
        prefill_len = 0
        if has_prompt_audio and prompt_acoustic_features is not None:
            # n_ac/n_t must leave room for post-prefill state:
            #   acoustic_features[:, n_prefill_frames-1] must be valid
            #   time_before[:, n_prefill_frames] must be valid
            # where n_prefill_frames = prefill_len - shift = n_max + 1
            n_ac = min(prompt_len - shift - 1, prompt_acoustic_features.shape[1])
            n_t = 0
            if prompt_time_before is not None:
                n_t = min(
                    prompt_len - shift - 1,
                    prompt_time_before.shape[1] - 1,
                )
            n_frames_cap = (
                max(0, prompt_time_before.shape[1] - 2)
                if prompt_time_before is not None
                else 0
            )
            n_max = min(n_ac, n_t, n_frames_cap) if n_ac > 0 and n_t > 0 else 0
            if n_max > 0:
                prefill_len = min(prompt_len, shift + n_max + 1)

        cache = None
        all_acoustic_features = []
        all_time_before = []

        # Current step state
        acoustic_features = mx.zeros((B, 1, self.config.acoustic_dim))
        acoustic_masks = mx.zeros((B, 1), dtype=mx.int32)
        time_len_before = mx.zeros((B, 1), dtype=mx.int32)
        time_len_after = mx.zeros((B, 1), dtype=mx.int32)

        neg_cond = mx.zeros((B, self.config.hidden_size))
        step_start = 0

        # Prefill prompt
        if prefill_len > 0:
            inputs_embeds_prefill = self._build_prompt_inputs_embeds(
                input_ids,
                prompt_acoustic_features,
                prompt_acoustic_masks,
                prompt_time_before,
                prompt_time_after,
                prefill_len,
            )

            # Double batch for negative conditioning during prefill
            if need_neg_batch:
                combined_embeds = mx.concatenate(
                    [inputs_embeds_prefill, inputs_embeds_prefill], axis=0
                )
            else:
                combined_embeds = inputs_embeds_prefill

            # Add text-only batch element (same text tokens, zero acoustic/masks/timing)
            if use_text_only_logit_scale:
                text_only_prefill = (
                    self.model.embed_tokens(input_ids[:, :prefill_len])
                    + self.acoustic_proj(
                        mx.zeros((B, prefill_len, self.config.acoustic_dim))
                    )
                    + self.acoustic_mask_emb(mx.zeros((B, prefill_len), dtype=mx.int32))
                    + self.time_start_embed(mx.zeros((B, prefill_len), dtype=mx.int32))
                    + self.time_end_embed(mx.zeros((B, prefill_len), dtype=mx.int32))
                )
                combined_embeds = mx.concatenate(
                    [combined_embeds, text_only_prefill], axis=0
                )

            hidden, cache = self.model(inputs_embeds=combined_embeds)
            mx.eval(hidden, *[c for pair in cache for c in pair])

            n_prefill_frames = prefill_len - shift

            for i in range(n_prefill_frames):
                all_acoustic_features.append(prompt_acoustic_features[:, i : i + 1])
            for i in range(n_prefill_frames):
                all_time_before.append(prompt_time_before[:, i + 1 : i + 2])

            acoustic_features = prompt_acoustic_features[
                :, n_prefill_frames - 1 : n_prefill_frames
            ]
            acoustic_masks = prompt_acoustic_masks[
                :, n_prefill_frames - 1 : n_prefill_frames
            ]
            time_len_before = prompt_time_before[
                :, n_prefill_frames : n_prefill_frames + 1
            ]
            time_len_after = prompt_time_after[
                :, n_prefill_frames : n_prefill_frames + 1
            ]

            step_start = prefill_len

        last_time_before = None

        for step in range(step_start, num_steps):
            # Use input tokens when available, otherwise use last generated token
            input_slice = (
                input_ids[:, step : step + 1]
                if step < input_ids.shape[1]
                else input_ids[:, -1:]
            )

            # Only compute logits when generating beyond input tokens
            need_logits = num_extra_steps > 0 and step >= input_ids.shape[1] - 1

            if need_neg_batch:
                # Create negative input: replace text tokens with pad, keep structural
                is_structural = (
                    (input_slice == start_header_id)
                    | (input_slice == end_header_id)
                    | (input_slice == eot_id)
                )
                neg_input_slice = mx.where(
                    is_structural,
                    input_slice,
                    mx.full(input_slice.shape, pad_token_id, dtype=input_slice.dtype),
                )
                combined_input = mx.concatenate([input_slice, neg_input_slice], axis=0)
                combined_acoustic = mx.concatenate(
                    [acoustic_features, acoustic_features], axis=0
                )
                combined_masks = mx.concatenate(
                    [acoustic_masks, acoustic_masks], axis=0
                )
                combined_time_before = mx.concatenate(
                    [time_len_before, time_len_before], axis=0
                )
                combined_time_after = mx.concatenate(
                    [time_len_after, time_len_after], axis=0
                )
                if use_text_only_logit_scale:
                    combined_input = mx.concatenate(
                        [combined_input, input_slice], axis=0
                    )
                    combined_acoustic = mx.concatenate(
                        [combined_acoustic, mx.zeros_like(acoustic_features)],
                        axis=0,
                    )
                    combined_masks = mx.concatenate(
                        [combined_masks, mx.zeros_like(acoustic_masks)], axis=0
                    )
                    combined_time_before = mx.concatenate(
                        [combined_time_before, mx.zeros_like(time_len_before)],
                        axis=0,
                    )
                    combined_time_after = mx.concatenate(
                        [combined_time_after, mx.zeros_like(time_len_after)],
                        axis=0,
                    )
                hidden, logits, cache = self.forward_one_step(
                    combined_input,
                    combined_acoustic,
                    combined_masks,
                    combined_time_before,
                    combined_time_after,
                    cache=cache,
                    compute_logits=need_logits,
                )
                mx.eval(hidden, *[c for pair in cache for c in pair] if cache else [])
                neg_cond = hidden[B : 2 * B]
                cond = hidden[:B]
                text_only_logits = (
                    logits[-B:]
                    if (logits is not None and use_text_only_logit_scale)
                    else None
                )
                logits = logits[:B] if logits is not None else None
            else:
                if use_text_only_logit_scale:
                    combined_input = mx.concatenate([input_slice, input_slice], axis=0)
                    combined_acoustic = mx.concatenate(
                        [acoustic_features, mx.zeros_like(acoustic_features)],
                        axis=0,
                    )
                    combined_masks = mx.concatenate(
                        [acoustic_masks, mx.zeros_like(acoustic_masks)], axis=0
                    )
                    combined_time_before = mx.concatenate(
                        [time_len_before, mx.zeros_like(time_len_before)], axis=0
                    )
                    combined_time_after = mx.concatenate(
                        [time_len_after, mx.zeros_like(time_len_after)], axis=0
                    )
                    hidden, logits, cache = self.forward_one_step(
                        combined_input,
                        combined_acoustic,
                        combined_masks,
                        combined_time_before,
                        combined_time_after,
                        cache=cache,
                        compute_logits=need_logits,
                    )
                    mx.eval(hidden)
                    if cache:
                        mx.eval(*[c for pair in cache for c in pair])
                    cond = hidden[:B]
                    text_only_logits = logits[-B:] if logits is not None else None
                    logits = logits[:B] if logits is not None else None
                else:
                    hidden, logits, cache = self.forward_one_step(
                        input_slice,
                        acoustic_features,
                        acoustic_masks,
                        time_len_before,
                        time_len_after,
                        cache=cache,
                        compute_logits=need_logits,
                    )
                    mx.eval(hidden)
                    if cache:
                        mx.eval(*[c for pair in cache for c in pair])
                    cond = hidden
                    text_only_logits = None

            # Flow matching to generate acoustic features + duration
            total_dim = self.config.acoustic_dim + self.time_dim
            speech = mx.random.normal((B, total_dim)) * noise_temperature

            speech = self._solve_flow_matching(
                speech=speech,
                cond=cond,
                neg_cond=neg_cond,
                num_steps=num_flow_matching_steps,
                acoustic_cfg_scale=acoustic_cfg_scale,
                duration_cfg_scale=duration_cfg_scale,
                cfg_schedule=cfg_schedule,
                time_schedule=time_schedule,
            )

            # Extract duration from gray code
            time_gray = speech[..., -self.time_dim :]
            predicted_time_before = decode_gray_code_to_time(
                time_gray[..., : self.num_time_bits], self.num_time_bits
            ).reshape(1, 1)
            predicted_time_after = decode_gray_code_to_time(
                time_gray[..., self.num_time_bits :], self.num_time_bits
            ).reshape(1, 1)

            # Sample next token when generating beyond input
            if num_extra_steps > 0 and step >= input_ids.shape[1] - 1:
                if logits is not None:
                    next_token = self._sample_next_token(
                        logits,
                        input_ids,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=pad_token_id,
                        text_only_logits=text_only_logits,
                        text_only_logit_scale=text_only_logit_scale,
                    )
                    input_ids = mx.concatenate(
                        [input_ids, next_token.astype(mx.int32)], axis=1
                    )

                    # Check EOS
                    if int(next_token[0, 0].item()) in self._get_eos_ids():
                        break

            # Update acoustic features for next step
            if step >= shift:
                if (
                    has_prompt_audio
                    and prompt_acoustic_features is not None
                    and step - shift < prompt_acoustic_features.shape[1]
                ):
                    acoustic_features = prompt_acoustic_features[
                        :, step - shift : step - shift + 1
                    ]
                    acoustic_masks = prompt_acoustic_masks[
                        :, step - shift : step - shift + 1
                    ]
                else:
                    acoustic_features = mx.expand_dims(
                        speech[..., : self.config.acoustic_dim], 0
                    )
                    acoustic_masks = mx.ones((B, 1), dtype=mx.int32)

                all_acoustic_features.append(acoustic_features)

                if (
                    has_prompt_audio
                    and prompt_time_before is not None
                    and step - shift < prompt_time_before.shape[1] - 1
                ):
                    time_len_before = prompt_time_before[
                        :, step - shift + 1 : step - shift + 2
                    ]
                    time_len_after = prompt_time_after[
                        :, step - shift + 1 : step - shift + 2
                    ]
                else:
                    time_len_before = predicted_time_before.astype(mx.int32)
                    time_len_after = predicted_time_after.astype(mx.int32)

                all_time_before.append(time_len_before)
                last_time_before = time_len_before

        if not all_acoustic_features:
            return None

        # Add trailing time
        if last_time_before is not None:
            all_time_before.append(last_time_before)

        # If speed_up_factor is set, re-run with scaled durations (two-pass)
        if speed_up_factor is not None and all_time_before:
            first_pass_time = mx.concatenate(all_time_before, axis=1)
            scaled_time = mx.round(
                first_pass_time.astype(mx.float32) / speed_up_factor
            ).astype(mx.int32)

            # Build prompt_time tensors for second pass
            # Index 0 is unused padding; indices 1..N map to steps shift..end
            second_pass_time_before = mx.concatenate(
                [mx.zeros_like(scaled_time[:, :1]), scaled_time], axis=1
            )
            # time_after[i] = time_before[i+1], last position = 1
            second_pass_time_after = mx.concatenate(
                [scaled_time, mx.ones_like(scaled_time[:, :1])], axis=1
            )

            return self._generate_loop(
                input_ids=input_ids,
                prompt_acoustic_features=prompt_acoustic_features,
                prompt_acoustic_masks=prompt_acoustic_masks,
                prompt_time_before=second_pass_time_before,
                prompt_time_after=second_pass_time_after,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                acoustic_cfg_scale=acoustic_cfg_scale,
                duration_cfg_scale=duration_cfg_scale,
                cfg_schedule=cfg_schedule,
                noise_temperature=noise_temperature,
                num_flow_matching_steps=num_flow_matching_steps,
                time_schedule=time_schedule,
                speed_up_factor=None,
                text_only_logit_scale=text_only_logit_scale,
                verbose=verbose,
                has_prompt_audio=has_prompt_audio,
                num_prompt_features=num_prompt_features,
                num_transition_steps=num_transition_steps,
                num_extra_steps=num_extra_steps,
                prefix_len=prefix_len,
            )

        # Stack features and decode
        acoustic_features_all = mx.concatenate(all_acoustic_features, axis=1)
        time_before_all = mx.concatenate(all_time_before, axis=1)

        # De-normalize acoustic features
        acoustic_features_all = (
            acoustic_features_all * self.config.acoustic_std + self.config.acoustic_mean
        )

        # Skip prompt/prefix features for output (matches PyTorch behavior)
        if has_prompt_audio:
            skip = num_prompt_features + num_transition_steps - 1
        else:
            # For zero-shot: skip structural prefix features
            zero_shot_prompt_tokens = max(0, prefix_len - num_transition_steps)
            skip = zero_shot_prompt_tokens + num_transition_steps - 1

        if skip > 0 and skip < acoustic_features_all.shape[1]:
            encoded = acoustic_features_all[:, skip:]
            time_before = time_before_all[:, skip:]
        else:
            encoded = acoustic_features_all
            time_before = time_before_all

        # Decode waveform
        wav = self._decode_wav(encoded[0], time_before[0])
        mx.eval(wav)

        # Remove leading silence
        wav_flat = wav.reshape(-1)
        if time_before.shape[1] > 0:
            leading_frames = int(time_before[0, 0].item())
            leading_samples = int(self.sample_rate * leading_frames / 50)
            if leading_samples > 0 and leading_samples < wav_flat.shape[0]:
                wav_flat = wav_flat[leading_samples:]

        return wav_flat

    # ========================================================================
    # Weight sanitization
    # ========================================================================

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}

        # First pass: resolve weight normalization
        resolved = {}
        skip_keys = set()
        for k, v in weights.items():
            if "parametrizations.weight.original0" in k:
                k1 = k.replace("original0", "original1")
                if k1 in weights:
                    weight_g = v  # (D, 1, 1) magnitude
                    weight_v = weights[k1]  # (D, C, K) direction
                    # Compute effective weight: weight_g * weight_v / ||weight_v||
                    flat = weight_v.reshape(weight_v.shape[0], -1)
                    norm = mx.sqrt(mx.sum(flat**2, axis=1, keepdims=True))
                    norm = norm.reshape(
                        weight_v.shape[0],
                        *([1] * (len(weight_v.shape) - 1)),
                    )
                    effective = weight_g * weight_v / (norm + 1e-12)
                    new_key = k.replace(".parametrizations.weight.original0", ".weight")
                    resolved[new_key] = effective
                    skip_keys.add(k)
                    skip_keys.add(k1)
            elif "parametrizations.weight.original1" in k:
                skip_keys.add(k)

        # Merge resolved weights
        for k, v in weights.items():
            if k not in skip_keys:
                resolved[k] = v

        # Second pass: key renaming and transposition
        for k, v in resolved.items():
            new_key = k

            # Skip buffers and tied weights
            if "_precomputed_mask" in k or "rope_freqs" in k:
                continue
            if k == "lm_head.weight":
                continue  # tied with embed_tokens.weight

            # Rename _decoder -> decoder (in case weights include decoder)
            if new_key.startswith("_decoder."):
                new_key = "decoder." + new_key[9:]

            # Rename model.* -> model.* (Llama backbone)
            # (keys already match)

            # Handle Sequential index mapping for prediction_head
            new_key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.",
                r".t_embedder.mlp.layers.\1.",
                new_key,
            )
            new_key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.weight$",
                r".t_embedder.mlp.layers.\1.weight",
                new_key,
            )
            new_key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.",
                r".adaLN_modulation.layers.\1.",
                new_key,
            )
            new_key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.weight$",
                r".adaLN_modulation.layers.\1.weight",
                new_key,
            )

            # Handle decoder FFN Sequential: ffn.0 -> ffn_in, ffn.3 -> ffn_out
            if "local_attention_decoder" in new_key:
                new_key = re.sub(r"\.ffn\.0\.", ".ffn_in.", new_key)
                new_key = re.sub(r"\.ffn\.3\.", ".ffn_out.", new_key)

            # Handle Snake1d alpha: squeeze (1, C, 1) -> (C,)
            if ".alpha" in new_key and len(v.shape) == 3:
                v = v.squeeze()

            # Handle Conv1d / ConvTranspose1d weight transposition
            if len(v.shape) == 3 and "weight" in new_key:
                is_conv_transpose = bool(
                    re.match(
                        r".*wav_decoder\.model\.[1-9]\d*\.block\.1\.weight$",
                        new_key,
                    )
                )
                if is_conv_transpose:
                    # PyTorch ConvTranspose1d: (C_in, C_out, K) -> MLX (C_out, K, C_in)
                    v = mx.transpose(v, axes=(1, 2, 0))
                else:
                    # PyTorch Conv1d: (C_out, C_in, K) -> MLX (C_out, K, C_in)
                    v = mx.transpose(v, axes=(0, 2, 1))

            sanitized[new_key] = v

        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path) -> "Model":
        """Load encoder, decoder, and tokenizer after model weights are loaded."""
        model_path = Path(model_path)

        # Load tokenizer
        try:
            from transformers import AutoTokenizer

            model._tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct"
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

        # Load decoder and encoder from tada-codec
        import os

        codec_path = os.environ.get("TADA_CODEC_PATH", "HumeAI/tada-codec")

        # Load decoder (decoder weights are NOT in main model)
        try:
            model._load_decoder(codec_path)
        except Exception as e:
            print(f"Warning: Could not load decoder: {e}")

        # Load encoder
        try:
            model._load_encoder(codec_path)
        except Exception as e:
            print(f"Warning: Could not load encoder: {e}")

        return model


# ============================================================================
# Helper functions
# ============================================================================


def _load_codec_weights(path: Path, component: str = "encoder") -> Dict[str, mx.array]:
    """Load and sanitize codec component weights."""
    import glob as glob_mod

    sanitized = {}
    weight_files = sorted(glob_mod.glob(str(path / "*.safetensors")))

    if not weight_files:
        return sanitized

    weights = {}
    for wf in weight_files:
        w = mx.load(wf)
        weights.update(w)

    # Resolve weight normalization
    resolved = {}
    skip_keys = set()
    for k, v in weights.items():
        if "parametrizations.weight.original0" in k:
            k1 = k.replace("original0", "original1")
            if k1 in weights:
                weight_g = v
                weight_v = weights[k1]
                flat = weight_v.reshape(weight_v.shape[0], -1)
                norm = mx.sqrt(mx.sum(flat**2, axis=1, keepdims=True))
                norm = norm.reshape(
                    weight_v.shape[0],
                    *([1] * (len(weight_v.shape) - 1)),
                )
                effective = weight_g * weight_v / (norm + 1e-12)
                new_key = k.replace(".parametrizations.weight.original0", ".weight")
                resolved[new_key] = effective
                skip_keys.add(k)
                skip_keys.add(k1)
        elif "parametrizations.weight.original1" in k:
            skip_keys.add(k)

    for k, v in weights.items():
        if k not in skip_keys:
            resolved[k] = v

    # Key renaming and transposition
    for k, v in resolved.items():
        new_key = k

        if "_precomputed_mask" in k or "rope_freqs" in k:
            continue

        # FFN Sequential mapping
        new_key = re.sub(r"\.ffn\.0\.", ".ffn_in.", new_key)
        new_key = re.sub(r"\.ffn\.3\.", ".ffn_out.", new_key)

        # Snake alpha
        if ".alpha" in new_key and len(v.shape) == 3:
            v = v.squeeze()

        # Conv transposition
        if len(v.shape) == 3 and "weight" in new_key:
            if component == "encoder":
                # Encoder has EncoderBlock with stride conv at block.4
                is_stride_conv = bool(
                    re.match(
                        r".*wav_encoder\.block\.[1-9]\d*\.block\.4\.weight$",
                        new_key,
                    )
                )
                # All encoder convs are Conv1d (no ConvTranspose1d)
                v = mx.transpose(v, axes=(0, 2, 1))
            else:
                # Decoder
                is_conv_transpose = bool(
                    re.match(
                        r".*wav_decoder\.model\.[1-9]\d*\.block\.1\.weight$",
                        new_key,
                    )
                )
                if is_conv_transpose:
                    v = mx.transpose(v, axes=(1, 2, 0))
                else:
                    v = mx.transpose(v, axes=(0, 2, 1))

        sanitized[new_key] = v

    return sanitized


def _align_text_tokens(
    logits: np.ndarray,
    text_tokens: np.ndarray,
    audio_length: int,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align text tokens to audio frames using CTC logits via DP.

    Args:
        logits: CTC logits, shape (T, vocab)
        text_tokens: Token IDs, shape (N,)
        audio_length: Audio length in samples
        sample_rate: Audio sample rate

    Returns:
        token_positions: Frame positions per token, shape (N,)
        token_masks: Binary mask on frames, shape (num_frames,)
    """
    T, V = logits.shape
    N = len(text_tokens)
    num_frames = int(np.ceil(audio_length / sample_rate * 50))

    if N == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(num_frames, dtype=np.int64)

    # CTC downsampling: wav2vec2 outputs at ~50Hz from 16kHz input
    # Map CTC frames to codec frames (both at 50Hz with 24kHz→480x downsampling)
    ctc_to_codec = np.linspace(0, num_frames - 1, T).astype(np.int64)

    # Log softmax
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True) + 1e-10)

    # Simple greedy alignment: find best frame for each token
    token_scores = np.array([log_probs[:, tok] for tok in text_tokens])  # (N, T)

    # DP: find monotonically increasing positions maximizing total score
    positions = np.zeros(N, dtype=np.int64)
    min_gap = max(1, T // (N + 1))

    for i in range(N):
        start = int(positions[i - 1] + min_gap) if i > 0 else 0
        end = T - (N - i - 1) * min_gap
        if start >= end:
            start = max(0, end - 1)
        best_t = start + np.argmax(token_scores[i, start:end])
        positions[i] = best_t

    # Map CTC positions to codec frame positions (0-indexed)
    codec_positions = ctc_to_codec[positions]

    # Create token masks at 0-indexed positions
    token_masks = np.zeros(num_frames, dtype=np.int64)
    for pos in codec_positions:
        if 0 <= pos < num_frames:
            token_masks[pos] = 1

    # Return 1-indexed positions (matches reference aligner convention:
    # encoder gathers at positions-1, which must land on masked frames)
    return codec_positions + 1, token_masks

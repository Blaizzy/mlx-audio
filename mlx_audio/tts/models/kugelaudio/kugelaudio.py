# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# KugelAudio model ported from https://github.com/Kugelaudio/kugelaudio-open (MIT License)

import re
import time
from pathlib import Path
from typing import Generator, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from ..base import GenerationResult
from ..vibevoice.acoustic_tokenizer import AcousticTokenizer
from ..vibevoice.diffusion_head import DiffusionHead
from ..vibevoice.language_model import Qwen2Model, SpeechConnector
from .config import ModelConfig
from .scheduler import SDEDPMSolverMultistepScheduler

# Special token IDs (repurposed from Qwen2.5 vision tokens)
SPEECH_START_ID = 151652
SPEECH_END_ID = 151653
SPEECH_DIFFUSION_ID = 151654
EOS_TOKEN_ID = 151643
VALID_SPEECH_TOKENS = [
    SPEECH_START_ID,
    SPEECH_END_ID,
    SPEECH_DIFFUSION_ID,
    EOS_TOKEN_ID,
]

# When speech_end wins but speech_diffusion logit is within this margin,
# generate one more latent to avoid cutting off the last syllable.
FINAL_LATENT_LOGIT_MARGIN = 5.0


class Model(nn.Module):
    """KugelAudio text-to-speech model.

    A 7B-parameter TTS model based on Microsoft VibeVoice, fine-tuned on ~200K hours
    of speech data for 24 European languages. Uses a hybrid AR + Diffusion architecture:
    a Qwen2.5 language model generates speech tokens autoregressively, and each
    speech_diffusion token triggers a DPM-Solver diffusion step to produce an acoustic
    latent, which is decoded to audio by a convolutional VAE decoder.

    Supports classifier-free guidance (CFG) for higher quality output.

    Reference: https://huggingface.co/kugelaudio/kugelaudio-0-open
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        decoder_cfg = config.decoder_config
        diffusion_cfg = config.diffusion_head_config
        acoustic_cfg = config.acoustic_tokenizer_config

        # Single unified language model (all layers)
        self.language_model = Qwen2Model(decoder_cfg, use_norm=True)

        # LM head for next-token prediction
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                decoder_cfg.hidden_size, decoder_cfg.vocab_size, bias=False
            )
        else:
            self.lm_head = None

        # Acoustic tokenizer (decoder only — encoder stripped)
        self.acoustic_tokenizer = AcousticTokenizer(acoustic_cfg)

        # Speech connector: vae_dim -> hidden_size
        vae_dim = config.acoustic_vae_dim
        self.acoustic_connector = SpeechConnector(
            vae_dim, decoder_cfg.hidden_size, eps=decoder_cfg.rms_norm_eps
        )

        # Diffusion head for speech latent prediction
        self.prediction_head = DiffusionHead(diffusion_cfg)

        # Noise scheduler (SDE-DPM-Solver++ stochastic variant)
        self.noise_scheduler = SDEDPMSolverMultistepScheduler(
            num_train_timesteps=diffusion_cfg.ddpm_num_steps,
            beta_schedule=diffusion_cfg.ddpm_beta_schedule,
            prediction_type=diffusion_cfg.prediction_type,
        )
        self.ddpm_inference_steps = diffusion_cfg.ddpm_num_inference_steps

        # Scaling factors (loaded from weights)
        self.speech_scaling_factor = mx.array(1.0)
        self.speech_bias_factor = mx.array(0.0)

        # Tokenizer (set by post_load_hook)
        self.tokenizer = None
        self.model_path = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def get_lm_logits(self, hidden_states: mx.array) -> mx.array:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        # Tied weights
        return hidden_states @ self.language_model.embed_tokens.weight.T

    def sample_speech_tokens(
        self,
        condition: mx.array,
        cfg_scale: float = 1.0,
        neg_condition: Optional[mx.array] = None,
    ) -> mx.array:
        """Sample one frame of speech latents via SDE-DPM-Solver++ diffusion.

        Uses classifier-free guidance when cfg_scale > 1.0 and neg_condition is provided.
        """
        vae_dim = self.config.acoustic_vae_dim

        # Cast to float32 for diffusion math
        condition = condition.astype(mx.float32)

        self.noise_scheduler.reset()
        self.noise_scheduler.set_timesteps(self.ddpm_inference_steps)

        if cfg_scale <= 1.0 or neg_condition is None:
            # No CFG — single forward pass
            speech = mx.random.normal((condition.shape[0], vae_dim)).astype(mx.float32)
            prev_x0 = None
            for t in self.noise_scheduler.timesteps:
                t_batch = mx.broadcast_to(t, (speech.shape[0],))
                eps = self.prediction_head(speech, t_batch, condition=condition)
                eps = eps.astype(mx.float32)
                result = self.noise_scheduler.step(eps, t, speech, prev_x0=prev_x0)
                speech = result.prev_sample
                prev_x0 = result.x0_pred
                mx.eval(speech)
            return speech

        # With CFG — batched prediction head, single scheduler step
        neg_condition = neg_condition.astype(mx.float32)
        combined_condition = mx.concatenate([condition, neg_condition], axis=0)
        n = condition.shape[0]

        speech = mx.random.normal((n, vae_dim)).astype(mx.float32)
        prev_x0 = None

        for t in self.noise_scheduler.timesteps:
            # Duplicate speech for both cond/uncond branches through prediction head
            combined_speech = mx.concatenate([speech, speech], axis=0)

            t_batch = mx.broadcast_to(t, (combined_speech.shape[0],))
            eps = self.prediction_head(
                combined_speech, t_batch, condition=combined_condition
            )
            eps = eps.astype(mx.float32)

            # Apply CFG on the (n,) guided result only
            cond_eps, uncond_eps = eps[:n], eps[n:]
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            result = self.noise_scheduler.step(guided_eps, t, speech, prev_x0=prev_x0)
            speech = result.prev_sample
            prev_x0 = result.x0_pred
            mx.eval(speech)

        return speech

    def _build_prompt_tokens(self, text: str) -> list[int]:
        """Build the token sequence for generation."""
        formatted_text = text.strip()
        if not formatted_text.startswith("Speaker"):
            formatted_text = f"Speaker 0: {formatted_text}"

        system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
        text_section = f" Text input:\n {formatted_text}\n Speech output:\n"

        full_text = system_prompt + text_section
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        tokens.append(SPEECH_START_ID)
        return tokens

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,  # pylint: disable=unused-argument
        cfg_scale: float = 3.0,
        max_tokens: int = 2048,
        ddpm_steps: Optional[int] = None,
        verbose: bool = True,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Generator[GenerationResult, None, None]:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded — was post_load_hook called?")

        prev_steps = self.ddpm_inference_steps
        if ddpm_steps is not None:
            self.ddpm_inference_steps = ddpm_steps

        try:
            yield from self._generate_impl(text, cfg_scale, max_tokens, verbose)
        finally:
            self.ddpm_inference_steps = prev_steps

    def _generate_impl(
        self,
        text: str,
        cfg_scale: float,
        max_tokens: int,
        verbose: bool,
    ) -> Generator[GenerationResult, None, None]:
        start_time = time.perf_counter()

        # Build prompt
        prompt_tokens = self._build_prompt_tokens(text)
        input_ids = mx.array([prompt_tokens], dtype=mx.int32)

        # Initial forward pass through language model (full prompt)
        hidden_states, cache = self.language_model(input_ids=input_ids)
        mx.eval(hidden_states)

        # Initialize negative branch for CFG
        neg_cache = None
        neg_hidden = None
        if cfg_scale > 1.0:
            neg_input = mx.array([[SPEECH_START_ID]], dtype=mx.int32)
            neg_hidden, neg_cache = self.language_model(input_ids=neg_input)
            mx.eval(neg_hidden)

        # Collect all speech latents first, then batch-decode at the end
        # This avoids click artifacts from independent chunk decoding
        all_speech_latents = []
        finished = False
        step = 0
        total_tokens = len(prompt_tokens)

        while not finished and step < max_tokens:
            last_hidden = hidden_states[:, -1:, :]
            logits = self.get_lm_logits(last_hidden)[:, 0, :]

            # Constrain to valid speech tokens
            constraint_mask = mx.full(logits.shape, float("-inf"))
            valid_indices = mx.array(VALID_SPEECH_TOKENS)
            constraint_mask[:, valid_indices] = 0.0
            logits = logits + constraint_mask

            next_token = mx.argmax(logits, axis=-1)
            next_token_id = next_token.item()
            total_tokens += 1
            step += 1

            if next_token_id == SPEECH_END_ID or next_token_id == EOS_TOKEN_ID:
                # Generate one final latent if speech_diffusion was a close runner-up
                # This prevents cutting off the last syllable
                diff_logit = logits[0, SPEECH_DIFFUSION_ID].item()
                end_logit = logits[0, next_token_id].item()
                if (
                    all_speech_latents
                    and diff_logit > end_logit - FINAL_LATENT_LOGIT_MARGIN
                ):
                    condition = hidden_states[:, -1, :]
                    neg_condition = (
                        neg_hidden[:, -1, :] if neg_hidden is not None else None
                    )
                    final_latent = self.sample_speech_tokens(
                        condition, cfg_scale=cfg_scale, neg_condition=neg_condition
                    )
                    all_speech_latents.append(final_latent)
                finished = True
                break

            if next_token_id == SPEECH_DIFFUSION_ID:
                condition = hidden_states[:, -1, :]

                # Get negative condition for CFG
                neg_condition = None
                if cfg_scale > 1.0 and neg_hidden is not None:
                    neg_condition = neg_hidden[:, -1, :]

                # Sample speech latents via diffusion
                speech_latents = self.sample_speech_tokens(
                    condition, cfg_scale=cfg_scale, neg_condition=neg_condition
                )
                all_speech_latents.append(speech_latents)

                # Feed speech latent back through connector into LM
                acoustic_embed = self.acoustic_connector(
                    mx.expand_dims(speech_latents, axis=1)
                )

                hidden_states, cache = self.language_model(
                    inputs_embeds=acoustic_embed, cache=cache
                )
                mx.eval(hidden_states)

                # Update negative branch
                if cfg_scale > 1.0 and neg_cache is not None:
                    neg_hidden, neg_cache = self.language_model(
                        inputs_embeds=acoustic_embed, cache=neg_cache
                    )
                    mx.eval(neg_hidden)

                if verbose and step % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"  Step {step}: {len(all_speech_latents)} latents, {elapsed:.1f}s"
                    )

            elif next_token_id == SPEECH_START_ID:
                token_embed = self.language_model.embed_tokens(
                    mx.array([[next_token_id]], dtype=mx.int32)
                )
                hidden_states, cache = self.language_model(
                    inputs_embeds=token_embed, cache=cache
                )
                mx.eval(hidden_states)

        elapsed = time.perf_counter() - start_time

        if not all_speech_latents:
            yield GenerationResult(
                audio=mx.zeros((0,)),
                samples=0,
                sample_rate=self.sample_rate,
                segment_idx=0,
                token_count=total_tokens,
                audio_duration="00:00:00.000",
                real_time_factor=0.0,
                prompt={"tokens": total_tokens, "tokens-per-sec": 0},
                audio_samples={"samples": 0, "samples-per-sec": 0},
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )
            return

        # Batch-decode all latents at once (avoids click artifacts)
        latent_seq = mx.concatenate(
            [mx.expand_dims(lat, axis=1) for lat in all_speech_latents], axis=1
        )  # (B, T, vae_dim)

        # Unscale
        if not mx.isnan(self.speech_scaling_factor).item():
            latent_seq = (
                latent_seq / self.speech_scaling_factor - self.speech_bias_factor
            )

        if verbose:
            print(f"  Decoding {latent_seq.shape[1]} latents to audio...")

        audio_out = self.acoustic_tokenizer.decode(latent_seq)  # (B, 1, samples)
        mx.eval(audio_out)
        audio = audio_out.squeeze()  # flatten to 1D

        # Normalize
        max_val = mx.abs(audio).max()
        if max_val.item() > 1.0:
            audio = audio * (0.95 / max_val)

        n_samples = audio.shape[-1]
        duration_s = n_samples / self.sample_rate
        rtf = elapsed / duration_s if duration_s > 0 else 0

        duration_hours = int(duration_s // 3600)
        remaining_s = duration_s % 3600
        duration_mins = int(remaining_s // 60)
        duration_secs = int(remaining_s % 60)
        duration_ms = int((duration_s % 1) * 1000)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        yield GenerationResult(
            audio=audio,
            samples=n_samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=total_tokens,
            audio_duration=duration_str,
            real_time_factor=round(rtf, 2),
            prompt={
                "tokens": total_tokens,
                "tokens-per-sec": (
                    round(total_tokens / elapsed, 2) if elapsed > 0 else 0
                ),
            },
            audio_samples={
                "samples": n_samples,
                "samples-per-sec": round(n_samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def sanitize(self, weights: dict) -> dict:
        """Convert PyTorch HuggingFace weights to MLX format."""
        new_weights = {}

        # Get target shapes for transpose decisions
        target_shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}

        for k, v in weights.items():
            # Skip encoder/semantic weights (not needed for inference)
            if any(
                skip in k
                for skip in [
                    "semantic_tokenizer",
                    "semantic_connector",
                    "acoustic_tokenizer.encoder.",
                    "rotary_emb.inv_freq",
                ]
            ):
                continue

            new_key = k

            # Strip "model." prefix (HF weights have it, our model doesn't)
            if new_key.startswith("model."):
                new_key = new_key[6:]

            # Fix Sequential layer indexing for diffusion head
            # "prediction_head.t_embedder.mlp.0.weight" -> "prediction_head.t_embedder.mlp.layers.0.weight"
            new_key = re.sub(
                r"(t_embedder\.mlp|adaLN_modulation)\.(\d+)\.",
                r"\1.layers.\2.",
                new_key,
            )

            # Check if target key exists
            if new_key not in target_shapes:
                # Try without any prefix modification
                if k in target_shapes:
                    new_key = k
                elif k.startswith("lm_head."):
                    new_key = k  # lm_head has no "model." prefix in HF
                    if new_key not in target_shapes:
                        continue
                else:
                    # Preserve quantization metadata (scales, biases)
                    # even if not in model parameters
                    if any(
                        suffix in new_key
                        for suffix in [".scales", ".biases", ".group_size", ".bits"]
                    ):
                        new_weights[new_key] = v
                    continue

            target_shape = target_shapes.get(new_key)
            if target_shape is None:
                continue

            # Transpose based on dimensions
            if v.ndim == 2 and target_shape is not None and len(target_shape) == 2:
                # Linear layer: PyTorch (out, in) -> MLX (out, in) but check shape match
                if v.shape != tuple(target_shape):
                    v = v.T
            elif v.ndim == 3:
                # Conv1d: PyTorch (C_out, C_in, K) -> MLX (C_out, K, C_in)
                if "convtr" in new_key or "conv_transpose" in new_key:
                    # ConvTranspose1d: PyTorch (C_in, C_out, K) -> MLX (C_out, K, C_in)
                    v = v.transpose(1, 2, 0)
                else:
                    v = v.swapaxes(1, 2)

            new_weights[new_key] = v

        return new_weights

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        """Load tokenizer after weights are loaded."""
        model.model_path = str(model_path)
        # Derive base tokenizer name from the Qwen2 decoder config vocab size.
        # Avoids "model of type kugelaudio" warning from transformers AutoConfig.
        qwen_model = "Qwen/Qwen2.5-7B"
        vocab = model.config.decoder_config.vocab_size
        if vocab <= 151936:
            qwen_model = "Qwen/Qwen2.5-1.5B"
        model.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model, trust_remote_code=False
        )
        return model

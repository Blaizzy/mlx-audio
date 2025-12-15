# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import os
import time
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from mlx_audio.tts.models.base import GenerationResult

from .acoustic_tokenizer import AcousticTokenizer
from .config import ModelConfig, Qwen2DecoderConfig
from .diffusion_head import DiffusionHead
from .language_model import BinaryClassifier, Qwen2Model, SpeechConnector
from .scheduler import DPMSolverMultistepScheduler

# Constants from original implementation
TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6


class Model(nn.Module):
    """VibeVoice streaming TTS model.

    This model generates speech from text using a Qwen2-based language model
    backbone with a diffusion-based prediction head.

    Architecture:
        - language_model: Lower transformer layers for text encoding
        - tts_language_model: Upper transformer layers for TTS
        - acoustic_tokenizer: VAE decoder for latents -> audio
        - prediction_head: Diffusion model for speech latent prediction
        - tts_eos_classifier: Binary classifier for end-of-speech detection
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Calculate layer split
        decoder_config = config.decoder_config
        tts_layers = config.tts_backbone_num_hidden_layers
        lm_layers = decoder_config.num_hidden_layers - tts_layers

        # Create configs for split models
        lm_config = Qwen2DecoderConfig(
            model_type=decoder_config.model_type,
            hidden_size=decoder_config.hidden_size,
            intermediate_size=decoder_config.intermediate_size,
            num_attention_heads=decoder_config.num_attention_heads,
            num_key_value_heads=decoder_config.num_key_value_heads,
            num_hidden_layers=lm_layers,
            rms_norm_eps=decoder_config.rms_norm_eps,
            vocab_size=decoder_config.vocab_size,
            max_position_embeddings=decoder_config.max_position_embeddings,
            rope_theta=decoder_config.rope_theta,
            head_dim=decoder_config.head_dim,
        )

        tts_lm_config = Qwen2DecoderConfig(
            model_type=decoder_config.model_type,
            hidden_size=decoder_config.hidden_size,
            intermediate_size=decoder_config.intermediate_size,
            num_attention_heads=decoder_config.num_attention_heads,
            num_key_value_heads=decoder_config.num_key_value_heads,
            num_hidden_layers=tts_layers,
            rms_norm_eps=decoder_config.rms_norm_eps,
            vocab_size=decoder_config.vocab_size,  # Reuse embeddings
            max_position_embeddings=decoder_config.max_position_embeddings,
            rope_theta=decoder_config.rope_theta,
            head_dim=decoder_config.head_dim,
        )

        # Language models
        # Base LM doesn't have final norm (it continues into tts_language_model)
        self.language_model = Qwen2Model(lm_config, use_norm=False)
        self.tts_language_model = Qwen2Model(tts_lm_config, use_norm=True)

        # TTS input type embeddings (0=speech, 1=text)
        self.tts_input_types = nn.Embedding(2, decoder_config.hidden_size)

        # Acoustic tokenizer (VAE decoder)
        self.acoustic_tokenizer = AcousticTokenizer(config.acoustic_tokenizer_config)

        # Speech connector
        self.acoustic_connector = SpeechConnector(
            input_dim=config.acoustic_vae_dim,
            output_dim=decoder_config.hidden_size,
        )

        # Diffusion head
        self.prediction_head = DiffusionHead(config.diffusion_head_config)

        # TTS EOS classifier
        self.tts_eos_classifier = BinaryClassifier(decoder_config.hidden_size)

        # Noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
        )

        # Scaling factors (will be loaded from weights)
        self.speech_scaling_factor = mx.array(1.0)
        self.speech_bias_factor = mx.array(0.0)

        # Inference settings
        self.ddpm_inference_steps = (
            config.diffusion_head_config.ddpm_num_inference_steps
        )

        # Tokenizer placeholder
        self.tokenizer = None

        # Optional voice cache state (loaded via load_voice_cache)
        self._voice_path: Optional[str] = None
        self._voice_lm_hidden: Optional[mx.array] = None
        self._voice_tts_hidden: Optional[mx.array] = None
        self._voice_neg_tts_hidden: Optional[mx.array] = None
        self._voice_lm_cache: Optional[list] = None
        self._voice_tts_cache: Optional[list] = None
        self._voice_neg_tts_cache: Optional[list] = None
        self._voice_neg_lm_cache: Optional[list] = None

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self.config.sample_rate

    def load_voice(self, voice: str) -> None:
        """Load a VibeVoice voice-cache (.safetensors) for conditioning.


        Expected keys (per layer):
          - lm_hidden
          - lm_key_{i}, lm_value_{i}
          - tts_lm_hidden
          - tts_lm_key_{i}, tts_lm_value_{i}
          - neg_tts_lm_hidden
          - neg_lm_key_{i}, neg_lm_value_{i}   (optional for our inference)
          - neg_tts_lm_key_{i}, neg_tts_lm_value_{i}
        """
        voice_path = Path(self.config.model_path) / f"voices/{voice}.safetensors"

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice cache not found: {voice_path}")

        tensors = mx.load(str(voice_path))

        lm_layers = self.language_model.config.num_hidden_layers
        tts_layers = self.tts_language_model.config.num_hidden_layers

        def _mx(name: str) -> mx.array:
            if name not in tensors:
                raise KeyError(f"Voice cache missing key: {name}")
            return mx.array(tensors[name])

        def _load_kv(prefix: str, i: int) -> Tuple[mx.array, mx.array]:
            k = _mx(f"{prefix}_key_{i}")
            v = _mx(f"{prefix}_value_{i}")
            # Swift caches are stored as (B, kv_heads, seq, head_dim).
            # Our attention cache expects (B, seq, kv_heads, head_dim).
            if k.ndim == 4:
                k = mx.transpose(k, (0, 2, 1, 3))
            if v.ndim == 4:
                v = mx.transpose(v, (0, 2, 1, 3))
            return (k, v)

        # Store caches/hidden states for generation
        self._voice_path = str(voice_path)
        self._voice_lm_hidden = _mx("lm_hidden")
        self._voice_tts_hidden = _mx("tts_lm_hidden")
        self._voice_neg_tts_hidden = _mx("neg_tts_lm_hidden")

        self._voice_lm_cache = [_load_kv("lm", i) for i in range(lm_layers)]
        self._voice_tts_cache = [_load_kv("tts_lm", i) for i in range(tts_layers)]
        self._voice_neg_tts_cache = [
            _load_kv("neg_tts_lm", i) for i in range(tts_layers)
        ]

        if all(
            f"neg_lm_key_{i}" in tensors and f"neg_lm_value_{i}" in tensors
            for i in range(lm_layers)
        ):
            self._voice_neg_lm_cache = [_load_kv("neg_lm", i) for i in range(lm_layers)]
        else:
            self._voice_neg_lm_cache = None

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.language_model.embed_tokens

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights for loading from HuggingFace format."""
        import re

        from mlx.utils import tree_flatten

        new_weights = {}
        curr_shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}

        def transform_key(key: str) -> str:
            """Transform HuggingFace key to MLX key format."""
            # Remove "model." prefix
            if key.startswith("model."):
                key = key[6:]

            # Prediction head transformations
            # t_embedder.mlp.0 -> t_embedder.mlp.layers.0
            key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.", r".t_embedder.mlp.layers.\1.", key
            )
            # Handle weight at end: t_embedder.mlp.0.weight -> t_embedder.mlp.layers.0.weight
            key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.weight$",
                r".t_embedder.mlp.layers.\1.weight",
                key,
            )

            # adaLN_modulation.1 -> adaLN_modulation.layers.1
            key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.", r".adaLN_modulation.layers.\1.", key
            )
            key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.weight$",
                r".adaLN_modulation.layers.\1.weight",
                key,
            )

            return key

        for k, v in weights.items():
            new_key = transform_key(k)

            # Handle scaling factors specially
            if "speech_scaling_factor" in new_key:
                new_weights["speech_scaling_factor"] = v
                continue
            if "speech_bias_factor" in new_key:
                new_weights["speech_bias_factor"] = v
                continue

            # Skip rotary embedding inv_freq (computed at init)
            if "rotary_emb.inv_freq" in new_key:
                continue

            # Check if key exists in model
            if new_key not in curr_shapes:
                # Debug: uncomment to see missing keys
                # print(f"Warning: Key {new_key} (from {k}) not found in model")
                continue

            target_shape = curr_shapes[new_key]

            # Handle shape mismatches
            if v.shape == target_shape:
                new_weights[new_key] = v
            elif len(v.shape) == 2 and v.T.shape == target_shape:
                # Transpose Linear weights (PyTorch vs MLX layout)
                new_weights[new_key] = v.T
            elif len(v.shape) == 3:
                # Check if it's a transposed conv weight
                is_convtr = "convtr" in new_key

                if is_convtr:
                    # ConvTranspose1d: PyTorch (C_in, C_out, K) -> MLX (C_out, K, C_in)
                    # Need to transpose (0, 1, 2) -> (1, 2, 0)
                    if (
                        v.shape[1] == target_shape[0]
                        and v.shape[2] == target_shape[1]
                        and v.shape[0] == target_shape[2]
                    ):
                        new_weights[new_key] = mx.transpose(v, (1, 2, 0))
                    else:
                        new_weights[new_key] = v
                else:
                    # Conv1d weights: PyTorch (C_out, C_in, K) -> MLX (C_out, K, C_in)
                    if (
                        v.shape[0] == target_shape[0]
                        and v.shape[1] == target_shape[2]
                        and v.shape[2] == target_shape[1]
                    ):
                        # Swap last two dimensions
                        new_weights[new_key] = mx.transpose(v, (0, 2, 1))
                    else:
                        new_weights[new_key] = v
            else:
                # Keep as is - might be a different kind of mismatch
                new_weights[new_key] = v

        return new_weights

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Hook called after model weights are loaded."""
        import json

        if model.tokenizer is None:
            # Try to load tokenizer from preprocessor_config
            preprocessor_config_path = model_path / "preprocessor_config.json"
            tokenizer_name = "Qwen/Qwen2.5-0.5B"  # Default

            if preprocessor_config_path.exists():
                with open(preprocessor_config_path, encoding="utf-8") as f:
                    preprocessor_config = json.load(f)
                    tokenizer_name = preprocessor_config.get(
                        "language_model_pretrained_name", tokenizer_name
                    )

            model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return model

    def sample_speech_tokens(
        self,
        condition: mx.array,
        neg_condition: mx.array,
        cfg_scale: float = 3.0,
        ddpm_steps: Optional[int] = None,
    ) -> mx.array:
        """Sample speech latents using diffusion with classifier-free guidance.

        Args:
            condition: Positive conditioning, shape (B, hidden_size)
            neg_condition: Negative conditioning, shape (B, hidden_size)
            cfg_scale: Classifier-free guidance scale

        Returns:
            Sampled speech latents, shape (B, acoustic_vae_dim)
        """
        # Use float32 for diffusion math; this tends to reduce gritty artifacts.
        condition = condition.astype(mx.float32)
        neg_condition = neg_condition.astype(mx.float32)

        # Reset scheduler for new generation
        self.noise_scheduler.reset()
        self.noise_scheduler.set_timesteps(ddpm_steps or self.ddpm_inference_steps)

        # Concatenate conditions for batched prediction
        condition_combined = mx.concatenate([condition, neg_condition], axis=0)

        # Initialize noise
        batch_size = condition.shape[0]
        latent_dim = self.config.acoustic_vae_dim
        speech = mx.random.normal((batch_size, latent_dim), dtype=mx.float32)

        prev_x0 = None

        # Get timesteps as list
        timesteps_list = self.noise_scheduler.timesteps.tolist()

        for _, t_val in enumerate(timesteps_list):
            # Create timestep array for both positive and negative
            t_float = float(t_val)
            timesteps = mx.array([t_float, t_float], dtype=mx.float32)

            # Duplicate speech for batched CFG prediction
            combined_speech = mx.concatenate([speech, speech], axis=0)

            # Predict v/epsilon
            eps = self.prediction_head(
                combined_speech, timesteps, condition=condition_combined
            )

            # Apply CFG
            cond_eps = eps[:batch_size]
            uncond_eps = eps[batch_size:]
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            # Duplicate for scheduler (it expects same batch size as input)
            full_eps = mx.concatenate([guided_eps, guided_eps], axis=0)
            full_speech = mx.concatenate([speech, speech], axis=0)

            # Scheduler step with multi-order support
            output = self.noise_scheduler.step(
                full_eps,
                t_val,
                full_speech,
                prev_x0=prev_x0,
            )

            # Extract just the first half (positive conditioning result)
            speech = output.prev_sample[:batch_size]
            prev_x0 = (
                output.x0_pred[:batch_size] if output.x0_pred is not None else None
            )

        return speech

    def generate(
        self,
        text: str,
        max_tokens: int = 512,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        voice: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        Args:
            text: Input text to synthesize
            max_tokens: Maximum number of tokens to generate
            cfg_scale: Classifier-free guidance scale
            ddpm_steps: Override diffusion inference steps (higher = better quality, slower)
            voice: Optional path to a `.safetensors` voice cache for conditioning
            verbose: Whether to show progress

        Yields:
            GenerationResult containing audio and metadata
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        start_time = time.perf_counter()

        # Optional: prime caches/hidden states from a voice cache (recommended for coherence)
        if voice is not None:
            # Only reload if different
            if not hasattr(self, "_voice_path") or str(voice) != getattr(
                self, "_voice_path"
            ):
                self.load_voice(voice)

        # Tokenize input
        # NOTE: Reference implementations (Microsoft + Swift) do not add special tokens here.
        text_token_ids = self.tokenizer.encode(
            text.strip() + "\n", add_special_tokens=False
        )
        input_ids = mx.array([text_token_ids], dtype=mx.int32)  # (1, L_text)

        batch_size = 1
        seq_len = input_ids.shape[1]

        # If we have a voice cache, start from its KV caches and hidden states.
        use_voice_cache = hasattr(self, "_voice_lm_cache") and hasattr(
            self, "_voice_tts_cache"
        )

        if use_voice_cache:
            # Start from cached context
            lm_cache = self._voice_lm_cache
            tts_cache = self._voice_tts_cache
            tts_hidden = self._voice_tts_hidden
            neg_hidden = self._voice_neg_tts_hidden
            neg_cache = self._voice_neg_tts_cache
        else:
            lm_cache = None
            tts_cache = None
            tts_hidden = None

            # Initialize negative condition (unconditional) for TTS LM
            neg_hidden = None
            neg_cache = None

            cfg_scale = 1.5

        # Audio generation loop
        # IMPORTANT: we must decode with full temporal context.
        # Decoding one latent at a time resets the convolutional decoder state
        # and produces incoherent audio. Instead, collect latents and decode once.
        speech_latents = []
        finished = False
        step = 0  # speech tokens generated
        total_speech_tokens = 0

        # Stream text in windows, interleaving speech generation.
        # This matches the upstream algorithm and is important for coherence.
        text_window_index = 0
        text_pos = 0

        while not finished and step < max_tokens:

            # 1) Prefill next text window (if any)
            if text_pos < seq_len:
                cur_text_ids = input_ids[
                    :, text_pos : min(seq_len, text_pos + TTS_TEXT_WINDOW_SIZE)
                ]
                cur_window = cur_text_ids.shape[1]
                text_pos += cur_window
                text_window_index += 1

                # Base LM forward on text window
                text_embeds = self.language_model.embed_tokens(cur_text_ids)
                lm_out, lm_cache = self.language_model(
                    inputs_embeds=text_embeds, cache=lm_cache
                )

                # TTS LM forward on text window (type=1)
                text_type = mx.ones((batch_size, cur_window), dtype=mx.int32)
                type_embed = self.tts_input_types(text_type)
                tts_in = lm_out + type_embed
                tts_out, tts_cache = self.tts_language_model(
                    inputs_embeds=tts_in, cache=tts_cache
                )

                # Accumulate hidden states for conditioning
                if tts_hidden is None:
                    tts_hidden = tts_out
                else:
                    tts_hidden = mx.concatenate([tts_hidden, tts_out], axis=1)

                # Negative path:
                # In the reference implementation, negative TTS LM is primed from voice cache and
                # then advanced only on speech tokens (not text tokens). If we do NOT have voice
                # cache, keep a shape-aligned unconditional stream (zeros + type embed).
                if neg_hidden is None or not use_voice_cache:
                    neg_embed = mx.zeros(
                        (batch_size, cur_window, self.config.decoder_config.hidden_size)
                    )
                    neg_type_embed = self.tts_input_types(
                        mx.ones((batch_size, cur_window), dtype=mx.int32)
                    )
                    neg_in = neg_embed + neg_type_embed
                    neg_out, neg_cache = self.tts_language_model(
                        inputs_embeds=neg_in, cache=neg_cache
                    )
                    if neg_hidden is None:
                        neg_hidden = neg_out
                    else:
                        neg_hidden = mx.concatenate([neg_hidden, neg_out], axis=1)

            # Safety: must have conditioning now
            if tts_hidden is None or neg_hidden is None:
                break

            # 2) Generate a speech window
            for _ in range(TTS_SPEECH_WINDOW_SIZE):
                positive_condition = tts_hidden[:, -1, :]
                negative_condition = neg_hidden[:, -1, :]

                # Sample speech latents
                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                    ddpm_steps=ddpm_steps,
                )
                speech_latent = mx.expand_dims(speech_latent, 1)  # (B, 1, D)

                # Collect latents for decoding at the end (keeps conv context)
                speech_latents.append(speech_latent)

                # Embed speech for next step
                acoustic_embed = self.acoustic_connector(speech_latent)

                # Add type embedding for speech (type=0)
                type_embed = self.tts_input_types(
                    mx.zeros((batch_size, 1), dtype=mx.int32)
                )
                tts_input = acoustic_embed + type_embed

                # Forward through TTS LM
                tts_out, tts_cache = self.tts_language_model(
                    inputs_embeds=tts_input,
                    cache=tts_cache,
                )
                tts_hidden = mx.concatenate([tts_hidden, tts_out], axis=1)

                # Forward through negative path (also with speech type embedding)
                neg_type_embed = self.tts_input_types(
                    mx.zeros((batch_size, 1), dtype=mx.int32)
                )
                neg_input = acoustic_embed + neg_type_embed
                neg_out, neg_cache = self.tts_language_model(
                    inputs_embeds=neg_input,
                    cache=neg_cache,
                )
                neg_hidden = mx.concatenate([neg_hidden, neg_out], axis=1)

                # Check for EOS
                eos_logits = mx.sigmoid(self.tts_eos_classifier(tts_out[:, -1, :]))
                if eos_logits[0].item() > 0.5:
                    finished = True
                    break

                step += 1
                total_speech_tokens += 1

                if step >= max_tokens:
                    finished = True
                    break

        # Decode full latent sequence at once for coherent audio
        if speech_latents:
            # (B, T_speech, D)
            speech_latent_seq = mx.concatenate(speech_latents, axis=1)
            scaled_latents = (
                speech_latent_seq / self.speech_scaling_factor - self.speech_bias_factor
            )
            # (B, 1, T_audio)
            audio = self.acoustic_tokenizer.decode(scaled_latents)
            final_audio = audio.squeeze(1).squeeze(0)  # Remove batch and channel dims
        else:
            final_audio = mx.array([])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Calculate statistics
        samples = final_audio.shape[0] if final_audio.size > 0 else 0
        audio_duration_seconds = samples / self.sample_rate if samples > 0 else 0

        # Format duration
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=final_audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(input_ids[0]),
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": len(input_ids[0]),
                "tokens-per-sec": (
                    round(len(input_ids[0]) / elapsed_time, 2)
                    if elapsed_time > 0
                    else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import json
import math
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.base import GenerationResult

from .config import ModelConfig
from .dit import DiTModel
from .encoders import AudioTokenDetokenizer, AudioTokenizer, ConditionEncoder
from .lm import ACEStepLM, LMConfig
from .modules import make_cache


class Model(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._sample_rate = config.sample_rate
        self.dtype = mx.bfloat16

        # Main diffusion components
        self.decoder = DiTModel(config)
        self.encoder = ConditionEncoder(config)
        self.tokenizer = AudioTokenizer(config)
        self.detokenizer = AudioTokenDetokenizer(config)

        # Null condition embedding for classifier-free guidance
        self.null_condition_emb = mx.random.normal((1, 1, config.hidden_size))

        # External components (loaded via post_load_hook)
        self.vae = None
        self.text_encoder = None
        self.silence_latent = None

        # Base model path (set when loading from pretrained)
        self._model_path = None

        # Optional 5Hz Language Model (lazy loaded)
        self._lm = None
        self._lm_config = None

    @property
    def sample_rate(self) -> int:
        """Return audio sample rate."""
        return self._sample_rate

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "ace_step"

    @classmethod
    def from_pretrained(cls, model_path: str, dtype: mx.Dtype = mx.bfloat16) -> "Model":
        """Load a pretrained ACE-Step model.

        This is the recommended way to load ACE-Step models as it handles
        the model's specific directory structure.

        Args:
            model_path: Path to the model directory or HuggingFace repo ID
            dtype: Data type for model weights

        Returns:
            Fully loaded Model with VAE, text encoder, and all components
        """
        from huggingface_hub import snapshot_download

        # Download if needed
        if not Path(model_path).exists():
            model_path = snapshot_download(model_path)

        model_path = Path(model_path)

        # Load config from the turbo subdirectory
        config_path = model_path / "acestep-v15-turbo" / "config.json"
        if not config_path.exists():
            # Try root config
            config_path = model_path / "config.json"

        with open(config_path) as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)
        model = cls(config)
        model.dtype = dtype

        # Load weights from the turbo subdirectory
        weights_path = model_path / "acestep-v15-turbo" / "model.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(weights, strict=False)
        else:
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        # Store model path for LM loading
        model._model_path = str(model_path)

        # Load VAE, text encoder, and silence latent
        model = cls.post_load_hook(model, model_path)

        return model

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Hook called after model weights are loaded.

        Loads additional components: VAE, text encoder, silence latent.

        Args:
            model: The loaded model instance
            model_path: Path to the model directory

        Returns:
            Model with all components loaded
        """
        from .text_encoder import TextEncoder
        from .vae import AutoencoderOobleck

        model_path = Path(model_path)

        # Load VAE decoder
        vae_config_path = model_path / "vae" / "config.json"
        if vae_config_path.exists():
            with open(vae_config_path) as f:
                vae_config = json.load(f)

            model.vae = AutoencoderOobleck(
                audio_channels=vae_config.get("audio_channels", 2),
                channel_multiples=vae_config.get("channel_multiples", [1, 2, 4, 8, 16]),
                decoder_channels=vae_config.get("decoder_channels", 128),
                decoder_input_channels=vae_config.get("decoder_input_channels", 64),
                downsampling_ratios=vae_config.get(
                    "downsampling_ratios", [2, 4, 4, 6, 10]
                ),
                sampling_rate=vae_config.get("sampling_rate", 48000),
            )

            weights_path = model_path / "vae" / "diffusion_pytorch_model.safetensors"
            if weights_path.exists():
                weights = mx.load(str(weights_path))
                model.vae.load_weights(weights, strict=False)

            model._sample_rate = vae_config.get("sampling_rate", 48000)

        # Load silence latent
        silence_path = model_path / "acestep-v15-turbo" / "silence_latent.pt"
        if silence_path.exists():
            import torch

            silence_pt = torch.load(silence_path, map_location="cpu", weights_only=True)
            silence_pt = silence_pt.transpose(1, 2)  # [1, 64, T] -> [1, T, 64]
            model.silence_latent = mx.array(silence_pt.numpy())
        else:
            model.silence_latent = mx.zeros((1, 3000, 64))

        # Load text encoder
        qwen_path = model_path / "Qwen3-Embedding-0.6B"
        if qwen_path.exists():
            try:
                model.text_encoder = TextEncoder(str(qwen_path))
            except Exception as e:
                print(f"Warning: Could not load text encoder: {e}")
                model.text_encoder = None

        return model

    def _format_prompt(
        self,
        caption: str,
        duration: float = 30.0,
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        timesignature: Optional[str] = None,
    ) -> str:
        """Format prompt using the official SFT template."""
        instruction = "Fill the audio semantic mask based on the given conditions:"

        bpm_str = str(bpm) if bpm is not None else "N/A"
        keyscale_str = keyscale if keyscale else "N/A"
        timesig_str = timesignature if timesignature else "N/A"
        duration_str = f"{int(duration)} seconds"

        metas = (
            f"- bpm: {bpm_str}\n"
            f"- timesignature: {timesig_str}\n"
            f"- keyscale: {keyscale_str}\n"
            f"- duration: {duration_str}\n"
        )

        return f"""# Instruction
{instruction}

# Caption
{caption}

# Metas
{metas}<|endoftext|>
"""

    def _format_lyrics(self, lyrics: str, language: str = "unknown") -> str:
        """Format lyrics text with language header."""
        return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    def _prepare_text_embeddings(
        self,
        text: str,
        max_length: int = 256,
        duration: float = 30.0,
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        timesignature: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare text embeddings using Qwen3-Embedding."""
        formatted_text = self._format_prompt(
            caption=text,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
        )

        if self.text_encoder is not None:
            text_hidden, text_mask = self.text_encoder.encode(
                formatted_text, max_length=max_length
            )
            return text_hidden.astype(self.dtype), text_mask.astype(self.dtype)
        else:
            text_len = min(len(formatted_text.split()) * 2, max_length)
            text_len = max(text_len, 10)
            text_hidden = mx.random.normal((1, text_len, self.config.text_hidden_dim))
            text_mask = mx.ones((1, text_len))
            return text_hidden.astype(self.dtype), text_mask.astype(self.dtype)

    def _prepare_lyric_embeddings(
        self,
        lyrics: str,
        max_length: int = 2048,
        language: str = "unknown",
    ) -> Tuple[mx.array, mx.array]:
        """Prepare lyric embeddings using embed_tokens only."""
        if not lyrics:
            lyrics = "[instrumental]"

        formatted_lyrics = self._format_lyrics(lyrics, language)

        if self.text_encoder is not None:
            lyric_hidden, lyric_mask = self.text_encoder.embed_tokens(
                formatted_lyrics, max_length=max_length
            )
            return lyric_hidden.astype(self.dtype), lyric_mask.astype(self.dtype)
        else:
            lyric_len = min(len(formatted_lyrics.split()) * 3, max_length)
            lyric_len = max(lyric_len, 10)
            lyric_hidden = mx.random.normal((1, lyric_len, self.config.text_hidden_dim))
            lyric_mask = mx.ones((1, lyric_len))
            return lyric_hidden.astype(self.dtype), lyric_mask.astype(self.dtype)

    def _prepare_timbre(self, batch_size: int = 1) -> Tuple[mx.array, mx.array]:
        """Prepare timbre embeddings using silence latent."""
        if self.silence_latent is None:
            self.silence_latent = mx.zeros((1, 3000, 64))

        timbre_len = min(self.silence_latent.shape[1], self.config.timbre_fix_frame)
        timbre_hidden = self.silence_latent[
            :, :timbre_len, : self.config.timbre_hidden_dim
        ]
        timbre_hidden = mx.broadcast_to(
            timbre_hidden, (batch_size, timbre_len, self.config.timbre_hidden_dim)
        )

        order_mask = mx.arange(batch_size, dtype=mx.int32)
        return timbre_hidden.astype(self.dtype), order_mask

    def _get_or_load_lm(self, model_size: str = "0.6B") -> ACEStepLM:
        """Get or lazily load the 5Hz Language Model.

        Args:
            model_size: Model size ("0.6B" or "4B")
                - 0.6B: Fastest, smallest (default)
                - 4B: Best quality, slower

        Returns:
            Loaded ACEStepLM instance
        """
        # Check if we need to reload with different config
        if self._lm is not None and self._lm_config is not None:
            if self._lm_config.model_size == model_size:
                return self._lm

        # Create new config and LM (pass base_model_path for bundled models)
        self._lm_config = LMConfig(
            model_size=model_size,
            base_model_path=self._model_path,
        )
        self._lm = ACEStepLM(self._lm_config)
        self._lm.load()

        return self._lm

    def _offload_lm(self) -> None:
        """Offload the LM to free memory."""
        if self._lm is not None:
            self._lm.offload()
            self._lm = None
            self._lm_config = None
            mx.clear_cache()

    def _generate_lm_hints(
        self,
        caption: str,
        lyrics: str,
        duration: int,
        language: str = "en",
        target_len: int = 750,
        seed: Optional[int] = None,
        model_size: str = "0.6B",
        verbose: bool = True,
    ) -> Optional[mx.array]:
        """Generate LM hints using the 5Hz Language Model.

        Args:
            caption: Music description/prompt
            lyrics: Song lyrics
            duration: Duration in seconds
            language: Language code
            target_len: Target length in 25Hz frames
            seed: Random seed
            model_size: LM model size
            verbose: Whether to print progress

        Returns:
            LM hints of shape [1, target_len, audio_dim] or None if generation fails
        """
        if verbose:
            print(f"Generating LM hints with {model_size} model...")

        lm = self._get_or_load_lm(model_size)

        # Generate audio codes
        code_string, metadata = lm.generate_audio_codes(
            caption=caption,
            lyrics=lyrics,
            duration=duration,
            language=language,
            seed=seed,
        )

        if verbose:
            code_count = len(lm.parse_audio_codes(code_string))
            print(f"Generated {code_count} audio codes")
            if metadata:
                print(f"Metadata: {metadata}")

        if not code_string:
            if verbose:
                print("Warning: No audio codes generated, falling back to silence")
            return None

        # Decode to 25Hz latents
        lm_hints = lm.decode_codes_to_latents(
            code_string=code_string,
            quantizer=self.tokenizer.quantizer,
            detokenizer=self.detokenizer,
            target_len=target_len,
        )

        # Offload LM to free memory before diffusion
        if verbose:
            print("Offloading LM to free memory...")
        self._offload_lm()

        return lm_hints

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert PyTorch weights to MLX format.

        Args:
            weights: Dictionary of PyTorch weights

        Returns:
            Dictionary of MLX-compatible weights
        """
        sanitized = {}

        for key, value in weights.items():
            new_key = key

            # Handle Conv1d weights: PyTorch [out, in, K] -> MLX [out, K, in]
            if "proj_in.1.weight" in key:
                # Conv1d: PyTorch [out_ch, in_ch, K] -> MLX [out_ch, K, in_ch]
                if len(value.shape) == 3:
                    value = value.transpose(0, 2, 1)
                new_key = key.replace(".1.", "_")

            # Handle ConvTranspose1d weights: PyTorch [in, out, K] -> MLX [out, K, in]
            elif "proj_out.1.weight" in key:
                # ConvTranspose1d: PyTorch [in_ch, out_ch, K] -> MLX [out_ch, K, in_ch]
                if len(value.shape) == 3:
                    value = value.transpose(1, 2, 0)
                new_key = key.replace(".1.", "_")

            # Handle Conv1d/ConvTranspose1d bias
            elif "proj_in.1.bias" in key or "proj_out.1.bias" in key:
                new_key = key.replace(".1.", "_")

            # Handle scale_shift_table - ensure correct shape
            if "scale_shift_table" in key:
                # Should be [1, 6, hidden_size] or [1, 2, hidden_size]
                if len(value.shape) == 2:
                    value = value[None, :, :]

            sanitized[new_key] = value

        return sanitized

    def load_weights(
        self, weights: Union[Dict[str, mx.array], list], strict: bool = True
    ) -> Tuple[List[str], List[str]]:
        """Load weights into the model.

        Args:
            weights: Dictionary of weights or list of (key, value) tuples
            strict: Whether to require all weights to be loaded

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Handle list input
        if isinstance(weights, list):
            weights = dict(weights)

        # Sanitize weights
        weights = self.sanitize(weights)

        # Use mlx's update function for loading
        from mlx.utils import tree_unflatten

        # Convert flat dict to nested structure
        nested_weights = tree_unflatten(list(weights.items()))

        # Get current parameters for comparison
        current_params = dict(nn.utils.tree_flatten(self.parameters()))

        missing_keys = []
        unexpected_keys = []

        # Check for missing and unexpected keys
        weight_keys = set(weights.keys())
        param_keys = set(current_params.keys())

        for key in weight_keys - param_keys:
            unexpected_keys.append(key)

        for key in param_keys - weight_keys:
            missing_keys.append(key)

        # Load weights that match
        matched_weights = {k: v for k, v in weights.items() if k in param_keys}

        if matched_weights:
            nested_matched = tree_unflatten(list(matched_weights.items()))
            self.update(nested_matched)

        if strict and (missing_keys or unexpected_keys):
            msg = []
            if missing_keys:
                msg.append(
                    f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
                )
            if unexpected_keys:
                msg.append(
                    f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
                )
            # Don't raise, just warn for now
            print(f"Warning: {'; '.join(msg)}")

        return missing_keys, unexpected_keys

    def prepare_noise(
        self,
        context_latents: mx.array,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> mx.array:
        """Prepare noise tensor for generation.

        Args:
            context_latents: Context latents to determine noise shape
            seed: Random seed (int, list of ints, or None)

        Returns:
            Noise tensor
        """
        batch_size = context_latents.shape[0]
        src_latents_shape = (
            batch_size,
            context_latents.shape[1],
            context_latents.shape[-1] // 2,
        )

        if seed is None:
            noise = mx.random.normal(src_latents_shape)
        elif isinstance(seed, list):
            # Generate noise for each sample with different seeds
            noise_list = []
            for i, s in enumerate(seed):
                if s is None or s < 0:
                    noise_i = mx.random.normal(
                        (1, src_latents_shape[1], src_latents_shape[2])
                    )
                else:
                    mx.random.seed(int(s))
                    noise_i = mx.random.normal(
                        (1, src_latents_shape[1], src_latents_shape[2])
                    )
                noise_list.append(noise_i)
            noise = mx.concatenate(noise_list, axis=0)
        else:
            mx.random.seed(int(seed))
            noise = mx.random.normal(src_latents_shape)

        return noise.astype(context_latents.dtype)

    def get_x0_from_noise(self, zt: mx.array, vt: mx.array, t: mx.array) -> mx.array:
        """Compute clean sample from noisy sample and velocity.

        Args:
            zt: Noisy sample
            vt: Predicted velocity
            t: Timestep

        Returns:
            Clean sample estimate
        """
        return zt - vt * t[:, None, None]

    def renoise(
        self,
        x: mx.array,
        t: Union[float, mx.array],
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        """Add noise to sample at timestep t.

        Args:
            x: Clean sample
            t: Timestep
            noise: Optional noise (if None, will be sampled)

        Returns:
            Noisy sample
        """
        if noise is None:
            noise = mx.random.normal(x.shape)

        if isinstance(t, mx.array) and t.ndim != x.ndim:
            t = t[:, None, None]

        xt = t * noise + (1 - t) * x
        return xt

    def prepare_condition(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_acoustic_hidden_states_packed: mx.array,
        refer_audio_order_mask: mx.array,
        hidden_states: mx.array,
        attention_mask: mx.array,
        silence_latent: mx.array,
        src_latents: mx.array,
        chunk_masks: mx.array,
        is_covers: mx.array,
        lm_hints_25hz: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Prepare conditioning inputs.

        Args:
            text_hidden_states: Text embeddings
            text_attention_mask: Text attention mask
            lyric_hidden_states: Lyric embeddings
            lyric_attention_mask: Lyric attention mask
            refer_audio_acoustic_hidden_states_packed: Reference audio features
            refer_audio_order_mask: Reference audio order mask
            hidden_states: Input hidden states
            attention_mask: Attention mask
            silence_latent: Silence latent for padding
            src_latents: Source latents
            chunk_masks: Chunk masks
            is_covers: Cover song flags
            lm_hints_25hz: Optional pre-computed LM hints at 25Hz rate

        Returns:
            Tuple of (encoder_hidden_states, encoder_attention_mask, context_latents)
        """
        dtype = hidden_states.dtype

        # Encode conditions
        encoder_hidden_states, encoder_attention_mask = self.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # Compute LM hints - either from precomputed or by tokenizing
        if lm_hints_25hz is not None:
            # Use precomputed LM hints (e.g., from 5Hz LM audio codes)
            lm_hints_25Hz = lm_hints_25hz[:, : src_latents.shape[1], :].astype(dtype)
        else:
            # Tokenize and detokenize to get LM hints
            lm_hints_5Hz, _ = self.tokenizer.tokenize(hidden_states)
            lm_hints_25Hz = self.detokenizer(lm_hints_5Hz)
            lm_hints_25Hz = lm_hints_25Hz[:, : src_latents.shape[1], :]

        # LM hints ONLY replace src_latents for cover songs (is_covers > 0)
        # For text-to-music (is_covers = 0), keep original src_latents (silence)
        src_latents = mx.where(is_covers[:, None, None] > 0, lm_hints_25Hz, src_latents)

        # Concatenate source latents with chunk masks
        context_latents = mx.concatenate(
            [src_latents, chunk_masks.astype(dtype)], axis=-1
        )

        return encoder_hidden_states, encoder_attention_mask, context_latents

    def generate_audio(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_acoustic_hidden_states_packed: mx.array,
        refer_audio_order_mask: mx.array,
        src_latents: mx.array,
        chunk_masks: mx.array,
        is_covers: mx.array,
        silence_latent: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        seed: Optional[int] = None,
        fix_nfe: int = 8,
        infer_method: str = "ode",
        shift: float = 3.0,
        guidance_scale: float = 15.0,
        guidance_interval: float = 0.5,
        omega_scale: float = 10.0,
        cfg_type: str = "apg",
        lm_hints_25hz: Optional[mx.array] = None,
    ) -> Dict:
        """Generate audio latents.

        Args:
            text_hidden_states: Text embeddings
            text_attention_mask: Text attention mask
            lyric_hidden_states: Lyric embeddings
            lyric_attention_mask: Lyric attention mask
            refer_audio_acoustic_hidden_states_packed: Reference audio features
            refer_audio_order_mask: Reference audio order mask
            src_latents: Source latents
            chunk_masks: Chunk masks
            is_covers: Cover song flags
            silence_latent: Silence latent for padding
            attention_mask: Attention mask
            seed: Random seed
            fix_nfe: Number of function evaluations (steps)
            infer_method: Inference method ('ode' or 'sde')
            shift: Timestep schedule shift
            guidance_scale: Classifier-free guidance scale (default 15.0)
            guidance_interval: Fraction of steps where guidance is applied (0.5 = middle 50%)
            omega_scale: Granularity scale for variance control (default 10.0)
            cfg_type: CFG type ('cfg' for standard, 'apg' for Adaptive Projected Gradient)
            lm_hints_25hz: Optional pre-computed LM hints at 25Hz rate

        Returns:
            Dictionary with 'target_latents' and 'time_costs'
        """
        # Pre-defined timestep schedules for turbo model (fix_nfe=8)
        # These are the valid timesteps from the PyTorch turbo implementation
        SHIFT_TIMESTEPS = {
            1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
            2.0: [
                1.0,
                0.9333333333333333,
                0.8571428571428571,
                0.7692307692307693,
                0.6666666666666666,
                0.5454545454545454,
                0.4,
                0.2222222222222222,
            ],
            3.0: [
                1.0,
                0.9545454545454546,
                0.9,
                0.8333333333333334,
                0.75,
                0.6428571428571429,
                0.5,
                0.3,
            ],
        }

        # Get timestep schedule based on shift (round to nearest valid shift)
        valid_shifts = [1.0, 2.0, 3.0]
        nearest_shift = min(valid_shifts, key=lambda x: abs(x - shift))
        t_schedule_list = SHIFT_TIMESTEPS[nearest_shift]

        # If fix_nfe differs from 8, interpolate or truncate the schedule
        if fix_nfe != 8:
            import numpy as np

            # Generate custom schedule using shift formula
            sigma_max = 1.0
            sigma_min = 0.001
            timesteps = np.linspace(sigma_max, sigma_min, fix_nfe).astype(np.float64)
            sigmas = shift * timesteps / (1 + (shift - 1) * timesteps)
            t_schedule_list = list(sigmas)

        if attention_mask is None:
            attention_mask = mx.ones(
                (src_latents.shape[0], src_latents.shape[1]),
                dtype=src_latents.dtype,
            )

        time_costs = {}
        start_time = time.time()
        total_start_time = start_time

        # Prepare conditions
        encoder_hidden_states, encoder_attention_mask, context_latents = (
            self.prepare_condition(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
                hidden_states=src_latents,
                attention_mask=attention_mask,
                silence_latent=silence_latent,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                lm_hints_25hz=lm_hints_25hz,
            )
        )

        # Prepare null (unconditional) embeddings for CFG
        do_cfg = guidance_scale != 1.0 and guidance_scale != 0.0
        if do_cfg:
            # Encode zeros through the encoder (same as PyTorch)
            null_cond, _ = self.encoder(
                text_hidden_states=mx.zeros_like(text_hidden_states),
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=mx.zeros_like(lyric_hidden_states),
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=mx.zeros_like(
                    refer_audio_acoustic_hidden_states_packed
                ),
                refer_audio_order_mask=refer_audio_order_mask,
            )

        end_time = time.time()
        time_costs["encoder_time_cost"] = end_time - start_time
        start_time = end_time

        # Prepare noise
        noise = self.prepare_noise(context_latents, seed)
        batch_size = context_latents.shape[0]
        dtype = context_latents.dtype

        # Timestep schedule for turbo model
        num_steps = len(t_schedule_list)

        # Calculate guidance interval bounds
        # guidance_interval=0.5 means guidance applied from 25% to 75% of steps
        guidance_start = int(num_steps * (1 - guidance_interval) / 2)
        guidance_end = int(num_steps * (1 + guidance_interval) / 2)

        # APG momentum buffer (running average for momentum-based guidance)
        apg_momentum = 0.0
        apg_momentum_coef = -0.75  # Momentum coefficient from PyTorch reference

        xt = noise

        # Cross-attention caches: K,V computed on first step, reused for all subsequent steps
        # Each cache auto-populates when first accessed, then returns cached values
        num_layers = len(self.decoder.layers)
        cond_cache = make_cache(num_layers)
        uncond_cache = make_cache(num_layers) if do_cfg else None

        for step_idx in range(num_steps):
            current_sigma = t_schedule_list[step_idx]
            t_curr = mx.full((batch_size,), current_sigma, dtype=dtype)

            # Predict velocity with conditions
            # Cache auto-populates on first call, reuses on subsequent calls
            vt_cond = self.decoder(
                hidden_states=xt,
                timestep=t_curr,
                timestep_r=t_curr,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                cache=cond_cache,
            )

            # Check if we should apply guidance at this step
            apply_guidance = do_cfg and guidance_start <= step_idx < guidance_end

            if apply_guidance:
                # Predict velocity without conditions (null)
                vt_uncond = self.decoder(
                    hidden_states=xt,
                    timestep=t_curr,
                    timestep_r=t_curr,
                    attention_mask=attention_mask,
                    encoder_hidden_states=null_cond,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    cache=uncond_cache,
                )

                if cfg_type == "apg":
                    # Adaptive Projected Gradient (APG) guidance
                    # Projects diff onto orthogonal component of vt_cond
                    # PyTorch base model uses dims=[1] (time dimension) for [B, T, C]
                    diff = vt_cond - vt_uncond

                    # Apply momentum
                    apg_momentum = apg_momentum_coef * apg_momentum + diff
                    diff = apg_momentum

                    # Norm thresholding over time dimension (axis=1)
                    norm_threshold = 2.5
                    diff_norm = mx.linalg.norm(diff, axis=1, keepdims=True)
                    scale_factor = mx.minimum(
                        mx.ones_like(diff_norm), norm_threshold / (diff_norm + 1e-8)
                    )
                    diff = diff * scale_factor

                    # Project over time dimension (axis=1) like PyTorch
                    # Use float32 for numerical stability
                    diff_f32 = diff.astype(mx.float32)
                    vt_cond_f32 = vt_cond.astype(mx.float32)

                    # Normalize vt_cond over time dimension (axis=1)
                    vt_cond_normalized = vt_cond_f32 / (
                        mx.linalg.norm(vt_cond_f32, axis=1, keepdims=True) + 1e-8
                    )

                    # Parallel component: projection of diff onto vt_cond direction
                    diff_parallel = (
                        mx.sum(diff_f32 * vt_cond_normalized, axis=1, keepdims=True)
                        * vt_cond_normalized
                    )

                    # Orthogonal component
                    diff_orthogonal = diff_f32 - diff_parallel

                    # APG uses only the orthogonal component (eta=0 by default)
                    normalized_update = diff_orthogonal.astype(vt_cond.dtype)

                    # Apply APG formula
                    vt = vt_cond + (guidance_scale - 1) * normalized_update
                else:
                    # Standard CFG formula: v = v_uncond + scale * (v_cond - v_uncond)
                    vt = vt_uncond + guidance_scale * (vt_cond - vt_uncond)
            else:
                vt = vt_cond

            # Update x_t using Euler integration
            # Turbo model uses simple Euler: xt = xt - vt * dt
            next_t = (
                t_schedule_list[step_idx + 1]
                if step_idx + 1 < len(t_schedule_list)
                else 0.0
            )

            # On final step, directly compute x0 from noise
            if step_idx == num_steps - 1:
                xt = self.get_x0_from_noise(xt, vt, t_curr)
            elif infer_method == "sde":
                # Stochastic: predict clean, then renoise
                pred_clean = self.get_x0_from_noise(xt, vt, t_curr)
                xt = self.renoise(pred_clean, next_t)
            else:  # ode
                # Turbo model Euler: dx/dt = -v, so x_{t+1} = x_t - v_t * dt
                dt = current_sigma - next_t
                xt = xt - vt * dt

            mx.eval(xt)

        end_time = time.time()
        time_costs["diffusion_time_cost"] = end_time - start_time
        time_costs["diffusion_per_step_time_cost"] = (
            time_costs["diffusion_time_cost"] / num_steps
        )
        time_costs["total_time_cost"] = end_time - total_start_time

        return {
            "target_latents": xt,
            "time_costs": time_costs,
        }

    def generate(
        self,
        text: str,
        lyrics: str = "",
        voice: Optional[str] = None,
        duration: float = 30.0,
        seed: Optional[int] = None,
        num_steps: int = 8,
        shift: float = 3.0,
        guidance_scale: float = 1.0,
        guidance_interval: float = 0.5,
        omega_scale: float = 10.0,
        cfg_type: str = "apg",
        vocal_language: str = "unknown",
        verbose: bool = True,
        # LM parameters
        use_lm: bool = False,
        lm_model_size: str = "0.6B",
        lm_temperature: float = 0.8,
        lm_top_p: float = 0.95,
        lm_precomputed_hints: Optional[mx.array] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate music from text prompt.

        Args:
            text: Text description/prompt
            lyrics: Optional lyrics (use "" for instrumental)
            voice: Optional voice/timbre reference (not yet implemented)
            duration: Target duration in seconds
            seed: Random seed for reproducibility
            num_steps: Number of diffusion steps (8 for turbo, higher for quality)
            shift: Timestep schedule shift (1, 2, or 3)
            guidance_scale: CFG scale (1.0 = no guidance, turbo model default; >1 for non-turbo models)
            guidance_interval: Fraction of steps where guidance is applied (0.5 recommended)
            omega_scale: Granularity scale for variance control
            cfg_type: CFG type ('cfg' for standard, 'apg' for Adaptive Projected Gradient)
            vocal_language: Language code for vocals (e.g., "en", "zh")
            verbose: Whether to print progress
            use_lm: Whether to use the 5Hz LM for better quality
            lm_model_size: LM model size ("0.6B" or "4B"). 0.6B is fastest, 4B is highest quality
            lm_temperature: LM sampling temperature
            lm_top_p: LM top-p sampling
            lm_precomputed_hints: Pre-computed LM hints (skips LM generation if provided)

        Yields:
            GenerationResult objects
        """
        if self.vae is None:
            raise RuntimeError(
                "VAE not loaded. Use Model.post_load_hook() after loading weights, "
                "or use mlx_audio.tts.load() to load the full model."
            )

        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"ACE-Step Music Generation")
            print(f"{'='*60}")
            print(f"Prompt: {text[:100]}...")
            print(f"Duration: {duration}s")
            print(f"Steps: {num_steps}, Shift: {shift}")
            print(
                f"CFG: {guidance_scale}, Type: {cfg_type}, Interval: {guidance_interval}"
            )
            if use_lm:
                print(f"5Hz LM: {lm_model_size}")
            if seed is not None:
                print(f"Seed: {seed}")
            print(f"{'='*60}\n")

        # Set seed
        if seed is not None:
            mx.random.seed(seed)

        # Calculate latent length (25Hz latent rate)
        latent_rate = 25
        latent_len = int(duration * latent_rate)

        # Ensure divisible by pool_window_size
        pool_window = self.config.pool_window_size
        if latent_len % pool_window != 0:
            latent_len = (latent_len // pool_window + 1) * pool_window

        if verbose:
            print(f"Generating {latent_len} latent frames...")

        # Prepare inputs
        text_hidden, text_mask = self._prepare_text_embeddings(text, duration=duration)
        lyric_hidden, lyric_mask = self._prepare_lyric_embeddings(
            lyrics, language=vocal_language
        )
        timbre_hidden, timbre_order = self._prepare_timbre()

        # Prepare source latents from silence_latent
        audio_dim = self.config.audio_acoustic_hidden_dim
        src_latents = self.silence_latent[:, :latent_len, :].astype(self.dtype)
        if src_latents.shape[1] < latent_len:
            pad_len = latent_len - src_latents.shape[1]
            padding = mx.broadcast_to(src_latents[:, -1:, :], (1, pad_len, audio_dim))
            src_latents = mx.concatenate([src_latents, padding], axis=1)

        # Chunk masks (all ones for full generation)
        chunk_masks = mx.ones((1, latent_len, audio_dim), dtype=self.dtype)

        # is_covers flag (False for text-to-music)
        # Note: LM hints are designed for cover songs, not text-to-music enhancement
        is_covers = mx.zeros((1,), dtype=self.dtype)

        # LM hints support (experimental - for cover song use cases)
        lm_hints = None
        if lm_precomputed_hints is not None:
            lm_hints = lm_precomputed_hints.astype(self.dtype)
            is_covers = mx.ones((1,), dtype=self.dtype)
            if verbose:
                print("Using pre-computed LM hints (cover mode)")
        elif use_lm:
            if verbose:
                print("Note: 5Hz LM is designed for cover songs, not text-to-music.")
                print("For best text-to-music quality, use_lm=False is recommended.")

        if verbose:
            print("Running diffusion...")

        # Run diffusion
        diffusion_start = time.time()

        result = self.generate_audio(
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=timbre_hidden,
            refer_audio_order_mask=timbre_order,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=self.silence_latent.astype(self.dtype),
            seed=seed,
            fix_nfe=num_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            omega_scale=omega_scale,
            cfg_type=cfg_type,
            lm_hints_25hz=lm_hints,
        )

        target_latents = result["target_latents"]
        mx.eval(target_latents)

        diffusion_time = time.time() - diffusion_start
        if verbose:
            print(f"Diffusion completed in {diffusion_time:.2f}s")
            print(f"Latent shape: {target_latents.shape}")

        # Decode with VAE
        if verbose:
            print("Decoding audio...")

        decode_start = time.time()
        latents_f32 = target_latents.astype(mx.float32)
        audio = self.vae.decode(latents_f32)
        mx.eval(audio)
        decode_time = time.time() - decode_start

        if verbose:
            print(f"Decode completed in {decode_time:.2f}s")
            print(f"Audio shape: {audio.shape}")

        # Format output: [batch, channels, samples] -> [channels, samples]
        audio = audio[0]
        audio = mx.clip(audio, -1.0, 1.0)
        num_samples = audio.shape[-1]
        actual_duration = num_samples / self.sample_rate

        total_time = time.time() - start_time
        rtf = total_time / actual_duration if actual_duration > 0 else 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation Complete!")
            print(f"  Duration: {actual_duration:.2f}s")
            print(f"  Samples: {num_samples:,}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  RTF: {rtf:.2f}x")
            print(f"{'='*60}\n")

        yield GenerationResult(
            audio=audio,
            sample_rate=self.sample_rate,
            samples=num_samples,
            segment_idx=0,
            token_count=latent_len,
            audio_samples=num_samples,
            audio_duration=f"{int(actual_duration // 60):02d}:{int(actual_duration % 60):02d}.{int((actual_duration % 1) * 1000):03d}",
            real_time_factor=rtf,
            prompt={"text": text, "lyrics": lyrics},
            processing_time_seconds=total_time,
            peak_memory_usage=0.0,
            is_streaming_chunk=False,
            is_final_chunk=True,
        )


def test_model():
    """Test model instantiation."""
    config = ModelConfig()
    model = Model(config)
    print(f"Model created with config: {config.model_type}")
    print(f"Sample rate: {model.sample_rate}")
    return model


if __name__ == "__main__":
    test_model()

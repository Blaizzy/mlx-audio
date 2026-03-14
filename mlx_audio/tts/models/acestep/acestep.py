import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union, List, Tuple
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .config import AceStepConfig
from .dit import MLXDiTDecoder
from .vae import MLXAutoEncoderOobleck
from .conditioner import MLXAceStepConditionEncoder
from .generate_utils import mlx_generate_diffusion

logger = logging.getLogger(__name__)

# Basic instruction format used by ACE-Step
DEFAULT_DIT_INSTRUCTION = "Generate the full track based on the audio context."

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

class AceStepTTAModel(nn.Module):
    """
    ACE-Step: Text-To-Audio (Music and SFX generation) model.
    Generates high-quality music/audio using a Diffusion Transformer (DiT).
    """
    def __init__(self, config: AceStepConfig):
        super().__init__()
        self.config = config
        
        # Initialize Encoder and DiT Decoder
        self.encoder = MLXAceStepConditionEncoder(config.dit_config)
        self.dit = MLXDiTDecoder(
            hidden_size=config.dit_config.hidden_size,
            intermediate_size=config.dit_config.intermediate_size,
            num_hidden_layers=config.dit_config.num_hidden_layers,
            num_attention_heads=config.dit_config.num_attention_heads,
            num_key_value_heads=config.dit_config.num_key_value_heads,
            head_dim=config.dit_config.head_dim,
            rms_norm_eps=config.dit_config.rms_norm_eps,
            attention_bias=config.dit_config.attention_bias,
            in_channels=config.dit_config.in_channels,
            audio_acoustic_hidden_dim=config.dit_config.audio_acoustic_hidden_dim,
            patch_size=config.dit_config.patch_size,
            sliding_window=config.dit_config.sliding_window,
            layer_types=config.dit_config.layer_types,
            rope_theta=config.dit_config.rope_theta,
            max_position_embeddings=config.dit_config.max_position_embeddings,
        )
        
        # Initialize VAE
        self.vae = MLXAutoEncoderOobleck(
            encoder_hidden_size=config.vae_config.encoder_hidden_size,
            downsampling_ratios=config.vae_config.downsampling_ratios,
            channel_multiples=config.vae_config.channel_multiples,
            decoder_channels=config.vae_config.decoder_channels,
            decoder_input_channels=config.vae_config.decoder_input_channels,
            audio_channels=config.vae_config.audio_channels,
        )
        
        self.lm = None
        self.tokenizer = None
        self._silence_latent = None

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        path = Path(model_path)
        
        if not path.exists():
            path = Path(snapshot_download(model_path))
            
        # Try to load config.json
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = AceStepConfig.from_dict(config_dict)
        else:
            config = AceStepConfig()
            
        model = cls(config)
        
        # Load DiT and Encoder weights combined
        dit_weights_path = path / "model.safetensors"
        if dit_weights_path.exists():
            # In our converted weights, they are prefixed with 'dit.' and 'encoder.'
            model.load_weights(str(dit_weights_path), strict=True)
            
        # Load VAE weights
        vae_weights_path = path / "vae/diffusion_pytorch_model.safetensors"
        if vae_weights_path.exists():
            model.vae.load_weights(str(vae_weights_path), strict=True)
        else:
            local_vae = path / "vae.safetensors"
            if local_vae.exists():
                model.vae.load_weights(str(local_vae), strict=True)
                
        # Silence Latent (converted to numpy during convert.py)
        silence_path = path / "silence_latent.npy"
        if silence_path.exists():
            model._silence_latent = mx.array(np.load(str(silence_path)))
        
        # Load Text LLM if requested
        lm_repo = kwargs.get("lm_repo", config.lm_repo)
        if lm_repo and kwargs.get("load_lm", True):
            try:
                from mlx_lm import load as load_lm
                model.lm, model.tokenizer = load_lm(lm_repo)
                logger.info(f"Loaded ACE-Step LLM from {lm_repo}")
            except ImportError:
                logger.warning("mlx_lm not installed. Text conditioning will fail.")
                
        mx.eval(model.parameters())
        return model

    def _format_lyrics(self, lyrics: str, language: str) -> str:
        if lyrics:
            return f"[{language}]\n{lyrics}"
        return f"[{language}]\n"

    def _prepare_conditioning(self, text: str, lyrics: str = "", metadata: str = "", language: str = "en") -> Tuple[mx.array, mx.array]:
        """
        Prepare text and lyric conditioning embeddings using the LLM encoder.
        """
        if self.lm is None:
            raise ValueError("LLM not loaded. Cannot process text prompt.")
            
        # Text Prompt setup
        instruction = DEFAULT_DIT_INSTRUCTION
        formatted_caption = f"Local: {text}\nMask Control: true"
        text_prompt = SFT_GEN_PROMPT.format(instruction, formatted_caption, metadata)
        
        text_input_ids = self.tokenizer.encode(text_prompt)
        text_mx = mx.array([text_input_ids])
        
        # Extract last hidden states from the LLM 
        text_hidden_states = self.lm.model(text_mx)
        text_attention_mask = mx.ones_like(text_mx)
        
        # Lyrics setup
        lyrics_text = self._format_lyrics(lyrics, language)
        lyric_input_ids = self.tokenizer.encode(lyrics_text)
        lyric_mx = mx.array([lyric_input_ids])
        lyric_attention_mask = mx.ones_like(lyric_mx)
        
        # For lyrics, ACE-Step just embeds them using the raw embedding table
        if hasattr(self.lm.model, "embed_tokens"):
            lyric_hidden_states = self.lm.model.embed_tokens(lyric_mx)
        else:
            lyric_hidden_states = self.lm.model(lyric_mx)
            
        # Pack via internal ConditionEncoder
        encoder_hidden_states, encoder_attention_mask = self.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
        )
            
        return encoder_hidden_states, encoder_attention_mask

    def generate(self, text: str, duration: float = 10.0, lyrics: str = "", 
                 reference_audio: Optional[mx.array] = None, **kwargs) -> Generator[mx.array, None, None]:
        """
        Generate audio/music from a text prompt.
        
        Args:
            text: Text description of the music/audio.
            duration: Target duration in seconds (max 600s).
            lyrics: Optional lyrics for the song.
            reference_audio: Optional reference audio array for timbre cloning.
            
        Yields:
            Audio PCM frames as `mx.array`.
        """
        metadata = kwargs.get("metadata", "")
        language = kwargs.get("language", "en")
        
        encoder_hidden_states, encoder_attention_mask = self._prepare_conditioning(text, lyrics, metadata, language)
        
        frame_rate = 25 # ACE-Step internal DiT frame rate is 25Hz
        num_frames = int(duration * frame_rate)
        
        in_channels = self.config.dit_config.in_channels
        bsz = 1
        src_latents_shape = (bsz, in_channels, num_frames)
        
        if reference_audio is not None:
            raise NotImplementedError("Timbre cloning not yet fully supported.")
        else:
            if self._silence_latent is not None:
                ctx = self._silence_latent[:, :750, :]
                context_latents = mx.transpose(ctx, (0, 2, 1))
                # Pad to match sequence length
                if context_latents.shape[2] < num_frames:
                    # Repeat or pad
                    pass # Usually context is fixed length for silence
            else:
                context_latents = mx.zeros((bsz, in_channels, min(num_frames, 750)))
        
        logger.info(f"Running MLX DiT Diffusion for {num_frames} frames ({duration}s)...")
        diffusion_output = mlx_generate_diffusion(
            mlx_decoder=self.dit,
            encoder_hidden_states_np=np.array(encoder_hidden_states),
            context_latents_np=np.array(context_latents),
            src_latents_shape=src_latents_shape,
            infer_steps=kwargs.get("steps", 50),
            guidance_scale=kwargs.get("guidance_scale", 4.5),
        )
        
        target_latents_np = diffusion_output["target_latents"]
        target_latents = mx.array(target_latents_np)
        
        target_latents_nlc = mx.transpose(target_latents, (0, 2, 1))
        
        logger.info("Decoding audio with VAE...")
        audio = self.vae.decode(target_latents_nlc)
        mx.eval(audio)
        
        yield audio

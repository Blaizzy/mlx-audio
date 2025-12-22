# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from scipy import signal

from mlx_audio.tts.models.base import GenerationResult

from .models.s3gen import S3GEN_SIL, S3GEN_SR, S3Gen
from .models.s3tokenizer import S3TokenizerV2, log_mel_spectrogram
from .models.t3 import T3, T3Cond, T3Config
from .models.voice_encoder import VoiceEncoder

logger = logging.getLogger(__name__)

# Constants
S3_SR = 16000  # S3Tokenizer sample rate
REPO_ID = "ResembleAI/chatterbox-turbo"


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy's polyphase resampling."""
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down, padtype="edge")


def _load_audio(file_path: str, target_sr: int) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(file_path)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        audio = _resample_audio(audio, sr, target_sr)
    return audio.astype(np.float32)


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset.
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (
            """, '"'),
        (""",
            '"',
        ),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen.
    """

    t3: T3Cond
    gen: dict

    def save(self, fpath: Path):
        """Save conditionals to file."""
        import pickle

        with open(fpath, "wb") as f:
            pickle.dump({"t3": self.t3, "gen": self.gen}, f)

    @classmethod
    def load(cls, fpath: Path) -> "Conditionals":
        """Load conditionals from file."""
        import pickle

        with open(fpath, "rb") as f:
            data = pickle.load(f)
        return cls(data["t3"], data["gen"])


class ChatterboxTurboTTS(nn.Module):
    """
    MLX implementation of Chatterbox Turbo TTS.
    Optimized for Apple Silicon.
    """

    ENC_COND_LEN = 15 * S3_SR  # 15 seconds for encoder conditioning
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds for decoder conditioning

    def __init__(
        self,
        config_or_t3: Union[dict, T3] = None,
        s3gen: S3Gen = None,
        ve: VoiceEncoder = None,
        tokenizer=None,  # HuggingFace tokenizer
        s3tokenizer: S3TokenizerV2 = None,  # Speech tokenizer for conditioning
        conds: Optional[Conditionals] = None,
        local_path: Optional[str] = None,
    ):
        super().__init__()
        self.sr = S3GEN_SR  # Output sample rate

        # Check if first argument is a config dict (from load_model)
        if config_or_t3 is None or isinstance(config_or_t3, dict):
            # Initialize from config
            self.config = config_or_t3 or {}
            hp = T3Config.turbo()
            self.t3 = T3(hp)
            self.s3gen = S3Gen(meanflow=True)
            self.ve = VoiceEncoder()
        else:
            # Initialize with individual components
            self.config = {}
            self.t3 = config_or_t3
            self.s3gen = s3gen if s3gen is not None else S3Gen(meanflow=True)
            self.ve = ve if ve is not None else VoiceEncoder()

        self.tokenizer = tokenizer
        # S3 speech tokenizer for reference audio tokenization
        self._s3tokenizer = s3tokenizer or S3TokenizerV2("speech_tokenizer_v2_25hz")
        self.conds = conds
        self.local_path = local_path

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.sr

    def sanitize(self, weights: dict) -> dict:
        """
        Sanitize PyTorch weights for MLX.

        Routes weights to the appropriate component based on prefix:
        - ve.* -> VoiceEncoder weights
        - t3.* -> T3 model weights
        - s3gen.* -> S3Gen weights

        Args:
            weights: Dictionary of weight name -> array

        Returns:
            Sanitized weights dictionary
        """
        new_weights = {}

        # Separate weights by component prefix
        ve_weights = {}
        t3_weights = {}
        s3gen_weights = {}
        other_weights = {}

        for key, value in weights.items():
            if key.startswith("ve."):
                ve_weights[key[3:]] = value
            elif key.startswith("t3."):
                t3_weights[key[3:]] = value
            elif key.startswith("s3gen."):
                s3gen_weights[key[6:]] = value
            else:
                other_weights[key] = value

        # Sanitize each component's weights if they have sanitize methods
        if ve_weights:
            if hasattr(self.ve, "sanitize"):
                ve_sanitized = self.ve.sanitize(ve_weights)
            else:
                ve_sanitized = ve_weights
            for k, v in ve_sanitized.items():
                new_weights[f"ve.{k}"] = v

        if t3_weights:
            if hasattr(self.t3, "sanitize"):
                t3_sanitized = self.t3.sanitize(t3_weights)
            else:
                t3_sanitized = t3_weights
            for k, v in t3_sanitized.items():
                new_weights[f"t3.{k}"] = v

        if s3gen_weights:
            if hasattr(self.s3gen, "sanitize"):
                s3gen_sanitized = self.s3gen.sanitize(s3gen_weights)
            else:
                s3gen_sanitized = s3gen_weights
            for k, v in s3gen_sanitized.items():
                new_weights[f"s3gen.{k}"] = v

        # Add other weights as-is
        new_weights.update(other_weights)

        return new_weights

    def load_weights(self, weights, strict: bool = True):
        """
        Load weights into the model.

        Uses strict=False by default for components because ChatterboxTurbo has
        several non-checkpoint parameters that are generated during initialization.

        Args:
            weights: List of (key, value) tuples or dict
            strict: If False, ignore missing/extra keys. Default True for
                    compatibility with utils.load_model().
        """
        if isinstance(weights, dict):
            weights = list(weights.items())

        # Split weights by component prefix
        ve_weights = []
        t3_weights = []
        s3gen_weights = []
        other_weights = []

        for k, v in weights:
            if k.startswith("ve."):
                ve_weights.append((k[3:], v))
            elif k.startswith("t3."):
                t3_weights.append((k[3:], v))
            elif k.startswith("s3gen."):
                s3gen_weights.append((k[6:], v))
            elif k.startswith("gen."):
                # Skip gen.* keys - these are conditionals, not model weights
                continue
            else:
                other_weights.append((k, v))

        # Load each component with strict=False to handle non-checkpoint params
        if ve_weights:
            logger.info(f"Loading {len(ve_weights)} VE weights")
            self.ve.load_weights(ve_weights, strict=False)

        if t3_weights:
            logger.info(f"Loading {len(t3_weights)} T3 weights")
            self.t3.load_weights(t3_weights, strict=False)

        if s3gen_weights:
            logger.info(f"Loading {len(s3gen_weights)} S3Gen weights")
            self.s3gen.load_weights(s3gen_weights, strict=False)

        # Warn about unrecognized weights
        if other_weights and strict:
            unrecognized = [k for k, _ in other_weights]
            logger.warning(f"Unrecognized weight keys: {unrecognized}")

        # Evaluate all parameters
        mx.eval(
            self.ve.parameters(),
            self.t3.parameters(),
            self.s3gen.parameters(),
        )
        logger.info("Weights loaded successfully")

    @staticmethod
    def post_load_hook(
        model: "ChatterboxTurboTTS", model_path: Path
    ) -> "ChatterboxTurboTTS":
        """
        Post-load hook called by load_model to initialize tokenizer and conditionals.

        Args:
            model: The loaded model instance
            model_path: Path to the model directory

        Returns:
            The model with tokenizer and conditionals initialized
        """
        model.local_path = str(model_path)

        # Load text tokenizer
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.tokenizer = tokenizer
            logger.info("Loaded text tokenizer")
        except Exception as e:
            logger.warning(f"Could not load text tokenizer: {e}")

        # Load S3 speech tokenizer weights
        try:
            from huggingface_hub import hf_hub_download

            s3tok_repo = "mlx-community/S3TokenizerV2"
            s3tok_weights_path = hf_hub_download(
                repo_id=s3tok_repo,
                filename="model.safetensors",
            )
            s3tok_weights = mx.load(s3tok_weights_path)
            if hasattr(model._s3tokenizer, "sanitize"):
                s3tok_weights = model._s3tokenizer.sanitize(s3tok_weights)
            model._s3tokenizer.load_weights(list(s3tok_weights.items()), strict=False)
            logger.info("Loaded S3 speech tokenizer weights")
        except Exception as e:
            logger.warning(f"Could not load S3 speech tokenizer: {e}")

        # Load pre-computed conditionals (prefer safetensors, fallback to .pt)
        builtin_voice_safetensors = model_path / "conds.safetensors"

        if builtin_voice_safetensors.exists():
            try:
                conds_data = mx.load(str(builtin_voice_safetensors))

                speaker_emb = conds_data.get("t3.speaker_emb")
                if speaker_emb is None:
                    speaker_emb = mx.zeros((1, 256))

                cond_tokens = conds_data.get("t3.cond_prompt_speech_tokens")

                t3_cond = T3Cond(
                    speaker_emb=speaker_emb,
                    cond_prompt_speech_tokens=cond_tokens,
                )

                gen_mlx = {}
                for k, v in conds_data.items():
                    if k.startswith("gen."):
                        gen_mlx[k.replace("gen.", "")] = v

                model._conds = Conditionals(t3_cond, gen_mlx)
                logger.info("Loaded pre-computed conditionals from safetensors")

            except Exception as e:
                logger.warning(f"Failed to load conds.safetensors: {e}")

        else:
            raise FileNotFoundError("conds.safetensors not found")
        return model

    @classmethod
    def from_local(
        cls, ckpt_dir: Union[str, Path], device: str = "cpu"
    ) -> "ChatterboxTurboTTS":
        """
        Load model from local checkpoint directory.

        Args:
            ckpt_dir: Path to checkpoint directory
            device: Device to use (ignored in MLX, always uses Metal)

        Returns:
            ChatterboxTurboTTS instance
        """
        ckpt_dir = Path(ckpt_dir)

        # Load Voice Encoder
        ve = VoiceEncoder()

        # Create T3 config for Turbo
        hp = T3Config.turbo()

        # Create T3 model
        t3 = T3(hp)

        # Create S3Gen
        s3gen = S3Gen(meanflow=True)

        # Try to load converted weights from model.safetensors
        model_weights_path = ckpt_dir / "model.safetensors"
        if model_weights_path.exists():
            logger.info(f"Loading converted weights from {model_weights_path}")
            weights = mx.load(str(model_weights_path))

            # Split weights by prefix and load into each model
            ve_weights = {
                k.replace("ve.", ""): v
                for k, v in weights.items()
                if k.startswith("ve.")
            }
            t3_weights = {
                k.replace("t3.", ""): v
                for k, v in weights.items()
                if k.startswith("t3.")
            }
            s3gen_weights = {
                k.replace("s3gen.", ""): v
                for k, v in weights.items()
                if k.startswith("s3gen.")
            }

            # Debug: Print expected vs loaded keys for VE
            from mlx.utils import tree_flatten

            ve_param_keys = [k for k, _ in tree_flatten(ve.parameters())]
            print(f"VE model expects these parameter keys: {ve_param_keys[:10]}...")
            print(f"VE weights from file: {list(ve_weights.keys())[:10]}...")

            if ve_weights:
                logger.info(f"Loading {len(ve_weights)} VE weights")
                try:
                    ve.load_weights(list(ve_weights.items()), strict=True)
                    logger.info("VE weights loaded successfully with strict=True")
                except Exception as e:
                    logger.warning(f"VE strict loading failed: {e}")
                    logger.info("Falling back to strict=False")
                    ve.load_weights(list(ve_weights.items()), strict=False)

            if t3_weights:
                logger.info(f"Loading {len(t3_weights)} T3 weights")
                try:
                    t3.load_weights(list(t3_weights.items()), strict=True)
                    logger.info("T3 weights loaded successfully with strict=True")
                except Exception as e:
                    logger.warning(f"T3 strict loading failed: {e}")
                    logger.info("Falling back to strict=False")
                    t3.load_weights(list(t3_weights.items()), strict=False)

            if s3gen_weights:
                logger.info(f"Loading {len(s3gen_weights)} S3Gen weights")
                # S3Gen has some parameters generated at init (not from weights):
                # - encoder.embed.pos_enc.pe, encoder.up_embed.pos_enc.pe (positional encodings)
                # - mel2wav.stft_window (STFT window from scipy)
                # - trim_fade (fade buffer)
                init_generated_params = {
                    "encoder.embed.pos_enc.pe",
                    "encoder.up_embed.pos_enc.pe",
                    "mel2wav.stft_window",
                    "trim_fade",
                }

                # Get all S3Gen parameter keys
                s3gen_param_keys = set(k for k, _ in tree_flatten(s3gen.parameters()))
                loadable_param_keys = s3gen_param_keys - init_generated_params

                # Find matching weights (weights that exist in model's loadable params)
                matching_weights = [
                    (k, v) for k, v in s3gen_weights.items() if k in loadable_param_keys
                ]

                # Check for any weights in file that don't match model
                unmatched_weights = set(s3gen_weights.keys()) - s3gen_param_keys
                if unmatched_weights:
                    logger.debug(
                        f"Weights in file not in model: {len(unmatched_weights)}"
                    )

                # Check for loadable params that don't have weights
                missing_weights = loadable_param_keys - set(s3gen_weights.keys())
                if missing_weights:
                    logger.warning(f"Model params without weights: {missing_weights}")

                logger.info(
                    f"Loading {len(matching_weights)} S3Gen weights (excluding {len(init_generated_params)} init-generated params)"
                )

                if matching_weights:
                    # Load with strict=False since we're intentionally excluding init-generated params
                    s3gen.load_weights(matching_weights, strict=False)

                    # Verify all expected weights were loaded
                    if len(matching_weights) == len(loadable_param_keys):
                        logger.info(
                            "S3Gen weights loaded successfully (all loadable params matched)"
                        )
                    else:
                        logger.warning(
                            f"S3Gen loaded {len(matching_weights)}/{len(loadable_param_keys)} loadable params"
                        )
                else:
                    logger.warning(
                        "No matching S3Gen weights found - model may not work correctly"
                    )

            mx.eval(ve.parameters(), t3.parameters(), s3gen.parameters())
            logger.info("Weights loaded successfully")
        else:
            logger.warning(f"No converted weights found at {model_weights_path}")
            logger.warning(
                "Run convert_weights.py first to convert PyTorch weights to MLX format"
            )

        # Load tokenizer
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            tokenizer = None

        # Load pre-computed conditionals (prefer safetensors, fallback to .pt)
        conds = None
        builtin_voice_safetensors = ckpt_dir / "conds.safetensors"
        builtin_voice_pt = ckpt_dir / "conds.pt"

        if builtin_voice_safetensors.exists():
            try:
                # Load from safetensors (pure MLX, no PyTorch dependency)
                conds_data = mx.load(str(builtin_voice_safetensors))

                # Extract t3 conditionals
                speaker_emb = conds_data.get("t3.speaker_emb")
                if speaker_emb is None:
                    speaker_emb = mx.zeros((1, 256))

                cond_tokens = conds_data.get("t3.cond_prompt_speech_tokens")

                t3_cond = T3Cond(
                    speaker_emb=speaker_emb,
                    cond_prompt_speech_tokens=cond_tokens,
                )

                # Extract gen conditionals
                gen_mlx = {}
                for k, v in conds_data.items():
                    if k.startswith("gen."):
                        gen_mlx[k.replace("gen.", "")] = v

                conds = Conditionals(t3_cond, gen_mlx)
                logger.info("Loaded pre-computed conditionals from safetensors")

            except Exception as e:
                logger.warning(f"Failed to load conds.safetensors: {e}")

        elif builtin_voice_pt.exists():
            try:
                import torch

                conds_data = torch.load(
                    builtin_voice_pt, map_location="cpu", weights_only=True
                )

                # Convert to MLX arrays
                t3_cond_dict = conds_data.get("t3", {})
                gen_dict = conds_data.get("gen", {})

                # Helper to convert PyTorch tensor to numpy (handles requires_grad)
                def to_numpy(t):
                    if hasattr(t, "detach"):
                        return t.detach().cpu().numpy()
                    elif hasattr(t, "numpy"):
                        return t.numpy()
                    return np.array(t)

                # Convert tensors to MLX arrays
                speaker_emb = t3_cond_dict.get("speaker_emb")
                if speaker_emb is not None:
                    speaker_emb = mx.array(to_numpy(speaker_emb))
                else:
                    speaker_emb = mx.array(np.zeros((1, 256), dtype=np.float32))

                cond_tokens = t3_cond_dict.get("cond_prompt_speech_tokens")
                if cond_tokens is not None:
                    cond_tokens = mx.array(to_numpy(cond_tokens).astype(np.int32))

                t3_cond = T3Cond(
                    speaker_emb=speaker_emb,
                    cond_prompt_speech_tokens=cond_tokens,
                )

                gen_mlx = {}
                for k, v in gen_dict.items():
                    if hasattr(v, "detach") or hasattr(v, "numpy"):
                        gen_mlx[k] = mx.array(to_numpy(v))
                    elif isinstance(v, (int, float)):
                        gen_mlx[k] = v

                conds = Conditionals(t3_cond, gen_mlx)
                logger.info("Loaded pre-computed conditionals from .pt file")
            except Exception as e:
                logger.warning(f"Could not load conditionals: {e}")

        return cls(t3, s3gen, ve, tokenizer, conds=conds, local_path=str(ckpt_dir))

    @classmethod
    def from_pretrained(
        cls, device: str = "cpu", weights_path: str = None
    ) -> "ChatterboxTurboTTS":
        """
        Load model from HuggingFace Hub.

        Args:
            device: Device to use (ignored in MLX)
            weights_path: Optional path to converted model.safetensors

        Returns:
            ChatterboxTurboTTS instance
        """
        try:
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                repo_id=REPO_ID,
                token=os.getenv("HF_TOKEN") or True,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
            )

            # If weights_path provided, always copy to ensure latest version is used
            if weights_path:
                import shutil

                dest = Path(local_path) / "model.safetensors"
                # Always copy to ensure we use the latest converted weights
                shutil.copy(weights_path, dest)
                logger.info(f"Copied converted weights to {dest}")

            return cls.from_local(local_path, device)

        except ImportError:
            raise ImportError(
                "Please install huggingface_hub: pip install huggingface_hub"
            )

    def norm_loudness(
        self, wav: np.ndarray, sr: int, target_lufs: float = -27
    ) -> np.ndarray:
        """Normalize audio loudness."""
        try:
            import pyloudnorm as ln

            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            logger.warning(f"Error in norm_loudness, skipping: {e}")
        return wav

    def _extract_conditionals(
        self,
        ref_wav_24k: np.ndarray,
        ref_wav_16k: np.ndarray,
    ) -> tuple:
        """
        Extract all conditioning using pure MLX (S3Gen embeddings + T3 tokens).

        Args:
            ref_wav_24k: Reference audio at 24kHz (for S3Gen mel/decoder)
            ref_wav_16k: Reference audio at 16kHz (for S3Tokenizer)

        Returns:
            Tuple of (s3gen_ref_dict, t3_cond_prompt_tokens)
        """
        s3gen_ref_dict = {}
        t3_cond_prompt_tokens = None

        if self._s3tokenizer is not None:
            # --- S3Gen tokens (from 10s audio at 16kHz) ---
            # Trim to decoder conditioning length
            ref_16k_for_s3gen = ref_wav_16k[: int(self.DEC_COND_LEN * S3_SR / S3GEN_SR)]
            s3gen_mel = log_mel_spectrogram(mx.array(ref_16k_for_s3gen))
            s3gen_mel = mx.expand_dims(s3gen_mel, 0)  # Add batch dim
            s3gen_mel_len = mx.array([s3gen_mel.shape[2]])
            s3gen_tokens, s3gen_token_lens = self._s3tokenizer(s3gen_mel, s3gen_mel_len)

            # Get S3Gen embeddings with tokens
            s3gen_ref_dict = self.s3gen.embed_ref(
                ref_wav=mx.array(ref_wav_24k)[None, :],
                ref_sr=S3GEN_SR,
                ref_speech_tokens=s3gen_tokens,
                ref_speech_token_lens=s3gen_token_lens,
            )

            # --- T3 conditioning tokens (from encoder cond length audio) ---
            ref_16k_for_t3 = ref_wav_16k[: self.ENC_COND_LEN]
            t3_mel = log_mel_spectrogram(mx.array(ref_16k_for_t3))
            t3_mel = mx.expand_dims(t3_mel, 0)
            t3_mel_len = mx.array([t3_mel.shape[2]])
            t3_tokens, _ = self._s3tokenizer(t3_mel, t3_mel_len)

            # Limit T3 tokens to prompt length
            plen = self.t3.hp.speech_cond_prompt_len
            t3_cond_prompt_tokens = t3_tokens[:, :plen]

            logger.info("Extracted conditionals using MLX S3Tokenizer")
        else:
            logger.warning("S3Tokenizer not available - using fallback")
            # Fallback: use S3Gen's embed_ref without tokens
            s3gen_ref_dict = self.s3gen.embed_ref(
                ref_wav=mx.array(ref_wav_24k),
                ref_sr=S3GEN_SR,
            )
            # Zero tokens fallback
            plen = self.t3.hp.speech_cond_prompt_len
            if plen:
                t3_cond_prompt_tokens = mx.zeros((1, plen), dtype=mx.int32)

        return s3gen_ref_dict, t3_cond_prompt_tokens

    def prepare_conditionals(
        self,
        ref_audio: Union[str, mx.array, np.ndarray],
        sample_rate: Optional[int] = None,
        exaggeration: float = 0.5,
        norm_loudness: bool = True,
    ):
        """
        Prepare conditioning from a reference audio file or array.

        Args:
            ref_audio: Path to reference audio file or audio array (should be > 5 seconds)
            sample_rate: Sample rate of audio array (required if ref_audio is array)
            exaggeration: Emotion exaggeration factor (not used in Turbo)
            norm_loudness: Whether to normalize loudness
        """
        # Handle string path vs array input
        if isinstance(ref_audio, str):
            # Load reference audio at 24kHz for S3Gen
            ref_wav_24k = _load_audio(ref_audio, S3GEN_SR)
        else:
            # Convert mx.array to numpy if needed
            if isinstance(ref_audio, mx.array):
                ref_wav_24k = np.array(ref_audio)
            else:
                ref_wav_24k = np.asarray(ref_audio)

            # Resample to S3GEN_SR if sample_rate provided and different
            input_sr = sample_rate if sample_rate is not None else S3GEN_SR
            if input_sr != S3GEN_SR:
                ref_wav_24k = _resample_audio(ref_wav_24k, input_sr, S3GEN_SR)

        assert (
            len(ref_wav_24k) / S3GEN_SR > 5.0
        ), "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            ref_wav_24k = self.norm_loudness(ref_wav_24k, S3GEN_SR)

        # Resample to 16kHz for S3Tokenizer and voice encoder
        ref_wav_16k = _resample_audio(ref_wav_24k, S3GEN_SR, S3_SR)

        # Trim 24kHz audio to decoder conditioning length
        ref_wav_24k_trimmed = ref_wav_24k[: self.DEC_COND_LEN]

        # Extract S3Gen embeddings and T3 tokens using MLX
        s3gen_ref_dict, t3_cond_prompt_tokens = self._extract_conditionals(
            ref_wav_24k_trimmed, ref_wav_16k
        )

        # Get voice encoder speaker embedding (use full 16kHz audio)
        ve_embed = self.ve.embeds_from_wavs(
            [ref_wav_16k[: self.ENC_COND_LEN]], sample_rate=S3_SR
        )
        ve_embed = mx.array(np.mean(ve_embed, axis=0, keepdims=True))

        # Create T3 conditioning
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=(
                mx.array([[[exaggeration]]]) if self.t3.hp.emotion_adv else None
            ),
        )

        self._conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: str,
        repetition_penalty: float = 1.2,
        min_p: float = 0.0,
        top_p: float = 0.95,
        ref_audio: Optional[Union[str, mx.array, np.ndarray]] = None,
        sample_rate: Optional[int] = None,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
        stream: bool = False,
        streaming_interval: float = 2.0,
        split_pattern: Optional[str] = r"(?<=[.!?])\s+",
        max_tokens: int = 800,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            repetition_penalty: Penalty for repeating tokens
            min_p: Minimum probability threshold (not used in Turbo)
            top_p: Nucleus sampling threshold
            ref_audio: Optional reference audio for voice cloning (path or array)
            sample_rate: Sample rate of audio array (required if ref_audio is array)
            exaggeration: Emotion exaggeration (not used in Turbo)
            cfg_weight: Classifier-free guidance weight (not used in Turbo)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            norm_loudness: Whether to normalize output loudness
            stream: Whether to stream audio chunks as they're generated
            streaming_interval: Time interval in seconds for streaming chunks
            split_pattern: Regex pattern to split long text into chunks (default: sentence boundaries)
            max_tokens: Maximum tokens per chunk to maintain quality (default: 900)

        Yields:
            GenerationResult with generated waveform and metrics
        """
        import re

        # If streaming is enabled, delegate to stream_generate
        if stream:
            # Convert streaming_interval (seconds) to chunk_size (tokens)
            # Each token represents ~40ms of audio (25Hz token rate)
            chunk_size = max(10, int(streaming_interval * 25))
            yield from self.stream_generate(
                text=text,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                ref_audio=ref_audio,
                sample_rate=sample_rate,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_k=top_k,
                norm_loudness=norm_loudness,
                chunk_size=chunk_size,
                split_pattern=split_pattern,
                max_tokens=max_tokens,
                **kwargs,
            )
            return

        # Prepare conditionals if audio prompt provided
        if ref_audio is not None:
            self.prepare_conditionals(
                ref_audio,
                sample_rate=sample_rate,
                exaggeration=exaggeration,
                norm_loudness=norm_loudness,
            )
        else:
            assert (
                self._conds is not None
            ), "Please `prepare_conditionals` first or specify `ref_audio`"

        # Warn about unsupported parameters
        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning(
                "CFG, min_p and exaggeration are not supported by Turbo version and will be ignored."
            )

        # Normalize text
        text = punc_norm(text)

        # Split text into chunks at sentence boundaries to keep speech tokens under limit
        # Estimate ~8 tokens per text token, so max_tokens/8 ≈ 112 text tokens
        # With ~4 chars per token, that's ~450 chars per chunk
        max_chars_per_chunk = (max_tokens // 8) * 4

        if split_pattern:
            sentences = re.split(split_pattern, text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if (
                    current_chunk
                    and len(current_chunk) + len(sentence) + 1 > max_chars_per_chunk
                ):
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [text]

        # Filter empty chunks
        chunks = [c for c in chunks if c.strip()]

        if not chunks:
            chunks = [text]

        start_time = time.time()
        total_token_count = 0
        total_samples = 0
        segment_idx = 0

        # Clear any accumulated cache from previous generations
        mx.clear_cache()

        for chunk in chunks:
            # Tokenize chunk
            if self.tokenizer is not None:
                text_tokens = self.tokenizer(
                    chunk, return_tensors="np", padding=True, truncation=True
                )
                text_tokens = mx.array(text_tokens.input_ids)
            else:
                logger.warning("No tokenizer available, using simple fallback")
                text_tokens = mx.array([[ord(c) for c in chunk[:512]]])

            chunk_token_count = text_tokens.shape[1]
            total_token_count += chunk_token_count

            # Generate speech tokens with T3
            speech_tokens = self.t3.inference_turbo(
                t3_cond=self._conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_gen_len=max_tokens,
            )

            # Clear cache after T3 inference
            mx.clear_cache()

            # Remove OOV tokens and add silence
            speech_tokens = speech_tokens.reshape(-1)
            token_data = np.array(speech_tokens)
            mask = np.where(token_data < 6561)[0]
            speech_tokens = mx.array(token_data[mask], dtype=mx.int32)
            silence = mx.array([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=mx.int32)
            speech_tokens = mx.concatenate([speech_tokens, silence])
            speech_tokens = speech_tokens[None, :]  # Add batch dimension

            # Generate waveform with S3Gen
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self._conds.gen,
                n_cfm_timesteps=2,  # Turbo uses 2 steps
            )

            # Flatten to 1D if needed
            if wav.ndim == 2:
                wav = wav.squeeze(0)

            samples = wav.shape[0]
            total_samples += samples

            # Calculate timing and metrics
            processing_time = time.time() - start_time
            audio_duration_seconds = samples / self.sample_rate

            # Format duration
            duration_hours = int(audio_duration_seconds // 3600)
            duration_mins = int((audio_duration_seconds % 3600) // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            # Calculate real-time factor based on total audio generated
            total_audio_duration = total_samples / self.sample_rate
            rtf = (
                processing_time / total_audio_duration
                if total_audio_duration > 0
                else 0
            )

            yield GenerationResult(
                audio=wav,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=chunk_token_count,
                audio_duration=duration_str,
                real_time_factor=round(rtf, 2),
                prompt={
                    "tokens": chunk_token_count,
                    "tokens-per-sec": (
                        round(total_token_count / processing_time, 2)
                        if processing_time > 0
                        else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(total_samples / processing_time, 2)
                        if processing_time > 0
                        else 0
                    ),
                },
                processing_time_seconds=processing_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            segment_idx += 1

            # Clear cache between chunks
            mx.clear_cache()

    def stream_generate(
        self,
        text: str,
        repetition_penalty: float = 1.2,
        min_p: float = 0.0,
        top_p: float = 0.95,
        ref_audio: Optional[Union[str, mx.array, np.ndarray]] = None,
        sample_rate: Optional[int] = None,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
        chunk_size: int = 40,
        split_pattern: Optional[str] = r"(?<=[.!?])\s+",
        max_tokens: int = 800,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """
        Stream generate speech from text, yielding audio chunks as they're generated.

        This method generates audio in a streaming fashion, yielding audio chunks
        as soon as they're ready. This allows for lower latency audio playback.

        Args:
            text: Input text to synthesize
            repetition_penalty: Penalty for repeating tokens
            min_p: Minimum probability threshold (not used in Turbo)
            top_p: Nucleus sampling threshold
            ref_audio: Optional reference audio for voice cloning (path or array)
            sample_rate: Sample rate of audio array (required if ref_audio is array)
            exaggeration: Emotion exaggeration (not used in Turbo)
            cfg_weight: Classifier-free guidance weight (not used in Turbo)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            norm_loudness: Whether to normalize output loudness
            chunk_size: Number of tokens to accumulate before converting to audio
            split_pattern: Regex pattern to split long text into chunks (default: sentence boundaries)
            max_tokens: Maximum tokens per text chunk to maintain quality (default: 800)

        Yields:
            GenerationResult with generated audio chunks and metrics
        """
        import re

        # Prepare conditionals if audio prompt provided
        if ref_audio is not None:
            self.prepare_conditionals(
                ref_audio,
                sample_rate=sample_rate,
                exaggeration=exaggeration,
                norm_loudness=norm_loudness,
            )
        else:
            assert (
                self._conds is not None
            ), "Please `prepare_conditionals` first or specify `ref_audio`"

        # Warn about unsupported parameters
        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning(
                "CFG, min_p and exaggeration are not supported by Turbo version and will be ignored."
            )

        # Normalize text
        text = punc_norm(text)

        # Split text into chunks at sentence boundaries to keep speech tokens under limit
        max_chars_per_chunk = (max_tokens // 8) * 4

        if split_pattern:
            sentences = re.split(split_pattern, text)
            text_chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if (
                    current_chunk
                    and len(current_chunk) + len(sentence) + 1 > max_chars_per_chunk
                ):
                    text_chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            if current_chunk:
                text_chunks.append(current_chunk.strip())
        else:
            text_chunks = [text]

        # Filter empty chunks
        text_chunks = [c for c in text_chunks if c.strip()]
        if not text_chunks:
            text_chunks = [text]

        start_time = time.time()
        segment_idx = 0
        total_token_count = 0
        global_total_samples = 0

        # Clear any accumulated cache from previous generations
        mx.clear_cache()

        # Process each text chunk
        for text_chunk_idx, text_chunk in enumerate(text_chunks):
            is_last_text_chunk = text_chunk_idx == len(text_chunks) - 1

            # Tokenize chunk
            if self.tokenizer is not None:
                text_tokens = self.tokenizer(
                    text_chunk, return_tensors="np", padding=True, truncation=True
                )
                text_tokens = mx.array(text_tokens.input_ids)
            else:
                logger.warning("No tokenizer available, using simple fallback")
                text_tokens = mx.array([[ord(c) for c in text_chunk[:512]]])

            chunk_token_count = text_tokens.shape[1]
            total_token_count += chunk_token_count

            # Streaming generation - use pre-allocated buffer (in-place approach)
            accumulated_tokens = mx.zeros((1, max_tokens), dtype=mx.int32)
            num_tokens = 0
            prev_audio_samples = 0
            total_samples = 0
            cached_noise = None  # Cache noise for consistent audio across chunks

            # Generate speech tokens in chunks
            for token_chunk, is_final in self.t3.inference_turbo_stream(
                t3_cond=self._conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
                max_gen_len=max_tokens,
            ):
                # In-place update of pre-allocated buffer
                chunk_len = token_chunk.shape[1]
                mx.eval(token_chunk)

                # Expand buffer if needed
                if num_tokens + chunk_len > max_tokens:
                    max_tokens = max_tokens * 2
                    new_buffer = mx.zeros((1, max_tokens), dtype=mx.int32)
                    new_buffer[:, :num_tokens] = accumulated_tokens[:, :num_tokens]
                    accumulated_tokens = new_buffer

                accumulated_tokens[:, num_tokens : num_tokens + chunk_len] = token_chunk
                num_tokens += chunk_len

                # Remove OOV tokens (convert to numpy for masking)
                token_data = np.array(accumulated_tokens[0, :num_tokens])
                mask = np.where(token_data < 6561)[0]
                valid_tokens = mx.array(token_data[mask], dtype=mx.int32)

                # Add silence tokens for final chunk of this text chunk
                if is_final:
                    silence = mx.array(
                        [S3GEN_SIL, S3GEN_SIL, S3GEN_SIL, S3GEN_SIL, S3GEN_SIL],
                        dtype=mx.int32,
                    )
                    valid_tokens = mx.concatenate([valid_tokens, silence])

                valid_tokens = valid_tokens[None, :]  # Add batch dimension

                # Convert tokens to audio using streaming with cached noise
                try:
                    new_audio, total_samples, cached_noise = (
                        self.s3gen.inference_stream(
                            speech_tokens=valid_tokens,
                            ref_dict=self._conds.gen,
                            n_cfm_timesteps=2,  # Turbo uses 2 steps
                            prev_audio_samples=prev_audio_samples,
                            is_final=is_final,
                            cached_noise=cached_noise,
                        )
                    )
                    mx.eval(new_audio)
                    if cached_noise is not None:
                        mx.eval(cached_noise)

                    # Flatten to 1D if needed
                    if new_audio.ndim == 2:
                        new_audio = new_audio.squeeze(0)

                    mx.clear_cache()

                    # Update global sample count
                    global_total_samples += (
                        new_audio.shape[0]
                        if new_audio.ndim == 1
                        else new_audio.shape[1]
                    )

                    # Only yield if we have new audio
                    audio_len = (
                        new_audio.shape[0]
                        if new_audio.ndim == 1
                        else new_audio.shape[1]
                    )
                    if audio_len > 0:
                        # Flatten to 1D if needed
                        if new_audio.ndim == 2:
                            wav = new_audio.squeeze(0)
                        else:
                            wav = new_audio

                        # Calculate timing and metrics
                        current_time = time.time()
                        processing_time = current_time - start_time
                        samples = wav.shape[0]
                        audio_duration_seconds = samples / self.sample_rate

                        # Format duration
                        duration_hours = int(audio_duration_seconds // 3600)
                        duration_mins = int((audio_duration_seconds % 3600) // 60)
                        duration_secs = int(audio_duration_seconds % 60)
                        duration_ms = int((audio_duration_seconds % 1) * 1000)
                        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

                        # Calculate real-time factor
                        total_audio_duration = global_total_samples / self.sample_rate
                        rtf = (
                            processing_time / total_audio_duration
                            if total_audio_duration > 0
                            else 0
                        )

                        yield GenerationResult(
                            audio=wav,
                            samples=samples,
                            sample_rate=self.sample_rate,
                            segment_idx=segment_idx,
                            token_count=chunk_token_count,
                            audio_duration=duration_str,
                            real_time_factor=round(rtf, 2),
                            prompt={
                                "tokens": total_token_count,
                                "tokens-per-sec": (
                                    round(total_token_count / processing_time, 2)
                                    if processing_time > 0
                                    else 0
                                ),
                            },
                            audio_samples={
                                "samples": samples,
                                "samples-per-sec": (
                                    round(global_total_samples / processing_time, 2)
                                    if processing_time > 0
                                    else 0
                                ),
                            },
                            processing_time_seconds=processing_time,
                            peak_memory_usage=mx.get_peak_memory() / 1e9,
                        )

                        segment_idx += 1
                        prev_audio_samples = total_samples

                except Exception as e:
                    logger.warning(f"Error generating audio chunk: {e}")
                    continue

            # Clear cache between text chunks
            mx.clear_cache()

        # Clear cache after generation
        mx.clear_cache()

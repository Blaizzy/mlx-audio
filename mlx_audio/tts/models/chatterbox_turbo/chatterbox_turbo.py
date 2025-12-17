# Copyright (c) 2025 Resemble AI
# MIT License
# MLX port of ChatterboxTurboTTS

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .models.s3gen import S3GEN_SIL, S3GEN_SR, S3Gen
from .models.t3 import T3, T3Cond, T3Config
from .models.voice_encoder import VoiceEncoder

logger = logging.getLogger(__name__)

# Constants
S3_SR = 16000  # S3Tokenizer sample rate
REPO_ID = "ResembleAI/chatterbox-turbo"


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
            """, "\""),
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


class ChatterboxTurboTTS:
    """
    MLX implementation of Chatterbox Turbo TTS.
    Optimized for Apple Silicon.
    """

    ENC_COND_LEN = 15 * S3_SR  # 15 seconds for encoder conditioning
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds for decoder conditioning

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer,  # HuggingFace tokenizer
        conds: Optional[Conditionals] = None,
        local_path: Optional[str] = None,
    ):
        self.sr = S3GEN_SR  # Output sample rate
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.conds = conds
        self.local_path = local_path

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

    def _extract_pytorch_conditionals(
        self, wav_fpath: str, norm_loudness: bool = True
    ) -> tuple:
        """
        Extract all conditioning using PyTorch (S3Gen embeddings + T3 tokens).
        This matches the original PyTorch tts_turbo.prepare_conditionals behavior.

        Args:
            wav_fpath: Path to reference audio
            norm_loudness: Whether to normalize loudness

        Returns:
            Tuple of (s3gen_ref_dict, t3_cond_prompt_tokens) or (None, None) on failure
        """
        try:
            import sys

            import torch
            from safetensors.torch import load_file

            # Add PyTorch chatterbox to path
            pytorch_path = str(Path(__file__).parent.parent / "chatterbox" / "src")
            if pytorch_path not in sys.path:
                sys.path.insert(0, pytorch_path)

            from chatterbox.models.s3gen import S3Gen as S3GenPT

            # Initialize PyTorch S3Gen
            s3gen_pt = S3GenPT()

            # Load weights
            weights_path = Path(self.local_path) / "s3gen_meanflow.safetensors"
            if weights_path.exists():
                state_dict = load_file(str(weights_path))
                s3gen_pt.load_state_dict(state_dict, strict=False)

            s3gen_pt.eval()

            # Load and process audio at 24kHz for S3Gen
            s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)

            if norm_loudness:
                s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, S3GEN_SR)

            # Resample to 16kHz for tokenizer
            ref_16k_wav = librosa.resample(
                s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR
            )

            # Trim to conditioning lengths
            s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
            ref_16k_wav = ref_16k_wav[: self.ENC_COND_LEN]

            with torch.no_grad():
                # Get S3Gen embeddings
                ref_dict = s3gen_pt.embed_ref(s3gen_ref_wav, S3GEN_SR)

                # Get T3 conditioning tokens using S3Gen's tokenizer (matches PyTorch exactly)
                plen = self.t3.hp.speech_cond_prompt_len
                t3_cond_prompt_tokens = None
                if plen and s3gen_pt.tokenizer is not None:
                    t3_cond_prompt_tokens, _ = s3gen_pt.tokenizer.forward(
                        [ref_16k_wav], max_len=plen
                    )
                    t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Convert S3Gen dict to MLX arrays
            mlx_ref_dict = {}
            for k, v in ref_dict.items():
                if v is not None and torch.is_tensor(v):
                    mlx_ref_dict[k] = mx.array(v.cpu().numpy())
                elif v is not None:
                    mlx_ref_dict[k] = v
                else:
                    mlx_ref_dict[k] = None

            # Convert T3 tokens to MLX
            mlx_t3_tokens = None
            if t3_cond_prompt_tokens is not None:
                mlx_t3_tokens = mx.array(t3_cond_prompt_tokens.cpu().numpy())

            logger.info("Extracted all conditionals using PyTorch")
            return mlx_ref_dict, mlx_t3_tokens

        except Exception as e:
            logger.warning(f"Failed to extract conditionals with PyTorch: {e}")
            import traceback

            traceback.print_exc()
            return None, None

    def prepare_conditionals(
        self,
        wav_fpath: str,
        exaggeration: float = 0.5,
        norm_loudness: bool = True,
    ):
        """
        Prepare conditioning from a reference audio file.

        Args:
            wav_fpath: Path to reference audio file (should be > 5 seconds)
            exaggeration: Emotion exaggeration factor (not used in Turbo)
            norm_loudness: Whether to normalize loudness
        """
        # Load reference audio at 24kHz for S3Gen
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert (
            len(s3gen_ref_wav) / S3GEN_SR > 5.0
        ), "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, S3GEN_SR)

        # Resample to 16kHz for voice encoder
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        # Trim to conditioning length
        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]

        # Try to extract all conditionals using PyTorch (for better quality)
        s3gen_ref_dict, t3_cond_prompt_tokens = self._extract_pytorch_conditionals(
            wav_fpath, norm_loudness
        )

        # Fallback if PyTorch extraction failed
        if s3gen_ref_dict is None:
            logger.warning(
                "PyTorch extraction failed, using MLX fallback (may have lower quality)"
            )
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

        # Fallback for T3 tokens
        plen = self.t3.hp.speech_cond_prompt_len
        if plen and t3_cond_prompt_tokens is None:
            logger.warning(
                "Using zero tokens for T3 conditioning - audio quality may be poor"
            )
            t3_cond_prompt_tokens = mx.zeros((1, plen), dtype=mx.int32)

        # Get voice encoder speaker embedding
        ve_embed = self.ve.embeds_from_wavs(
            [ref_16k_wav[: self.ENC_COND_LEN]], sample_rate=S3_SR
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

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: str,
        repetition_penalty: float = 1.2,
        min_p: float = 0.0,
        top_p: float = 0.95,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
    ) -> mx.array:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            repetition_penalty: Penalty for repeating tokens
            min_p: Minimum probability threshold (not used in Turbo)
            top_p: Nucleus sampling threshold
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration (not used in Turbo)
            cfg_weight: Classifier-free guidance weight (not used in Turbo)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            norm_loudness: Whether to normalize output loudness

        Returns:
            Generated waveform as MLX array (1, T)
        """
        # Prepare conditionals if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(
                audio_prompt_path,
                exaggeration=exaggeration,
                norm_loudness=norm_loudness,
            )
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Warn about unsupported parameters
        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning(
                "CFG, min_p and exaggeration are not supported by Turbo version and will be ignored."
            )

        # Normalize and tokenize text
        text = punc_norm(text)

        if self.tokenizer is not None:
            text_tokens = self.tokenizer(
                text, return_tensors="np", padding=True, truncation=True
            )
            text_tokens = mx.array(text_tokens.input_ids)
        else:
            # Fallback: simple character-level tokenization (for testing)
            logger.warning("No tokenizer available, using simple fallback")
            text_tokens = mx.array([[ord(c) for c in text[:512]]])

        # Generate speech tokens with T3
        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence
        speech_tokens = speech_tokens.reshape(-1)
        mask = np.where(speech_tokens < 6561)[0].tolist()
        speech_tokens = speech_tokens[mask]
        silence = mx.array([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=mx.int32)
        speech_tokens = mx.concatenate([speech_tokens, silence])
        speech_tokens = speech_tokens[None, :]  # Add batch dimension

        # Generate waveform with S3Gen
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,  # Turbo uses 2 steps
        )

        # Post-process
        wav = wav[0]  # Remove batch dimension
        wav_np = np.array(wav)

        # # Normalize loudness
        # if norm_loudness:
        #     wav_np = self.norm_loudness(wav_np, self.sr)

        return mx.array(wav_np)[None, :]  # (1, T)

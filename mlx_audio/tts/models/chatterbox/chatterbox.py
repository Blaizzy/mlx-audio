# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Optional, Union

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy import signal

from ..base import GenerationResult
from .config import ModelConfig
from .s3gen import S3Token2Wav
from .s3tokenizer import S3TokenizerV2, log_mel_spectrogram
from .t3 import T3
from .t3.cond_enc import T3Cond
from .tokenizer import MTLTokenizer
from .voice_encoder import VoiceEncoder

# Constants
S3_SR = 16000  # Sample rate for speech tokenizer
S3GEN_SR = 24000  # Sample rate for vocoder
SPEECH_VOCAB_SIZE = 6561  # Size of speech token vocabulary

# Special tokens
SOT = "[START]"
EOT = "[STOP]"
SPACE = "[SPACE]"


def resample_audio(audio: mx.array, orig_sr: int, target_sr: int) -> mx.array:
    """
    Resample audio to a target sample rate using scipy.

    Args:
        audio: Audio waveform as MLX array (samples,) or (channels, samples)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio as MLX array
    """
    if orig_sr == target_sr:
        return audio

    # Convert to numpy for scipy
    import numpy as np

    audio_np = np.array(audio)

    # Calculate resampling factors
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    # Resample
    resampled = signal.resample_poly(audio_np, up, down, padtype="edge")

    return mx.array(resampled)


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
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
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


def drop_invalid_tokens(x: mx.array) -> mx.array:
    """
    Drop SOS and EOS tokens, extracting only the speech content between them.

    This matches the original PyTorch implementation which slices between
    the start-of-speech (SOS=6561) and end-of-speech (EOS=6562) markers.
    """
    SOS = SPEECH_VOCAB_SIZE  # 6561
    EOS = SPEECH_VOCAB_SIZE + 1  # 6562

    assert len(x.shape) == 1 or (
        len(x.shape) == 2 and x.shape[0] == 1
    ), "only batch size of one allowed for now"

    x_flat = x.flatten()

    # Find SOS position using argmax (returns first True index)
    sos_mask = x_flat == SOS
    s = 0
    if mx.any(sos_mask):
        s = int(mx.argmax(sos_mask)) + 1  # Start after SOS

    # Find EOS position using argmax
    eos_mask = x_flat == EOS
    e = x_flat.shape[0]
    if mx.any(eos_mask):
        e = int(mx.argmax(eos_mask))  # End before EOS

    return x_flat[s:e]


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen.

    T3 conditionals:
        - speaker_emb
        - cond_prompt_speech_tokens
        - emotion_adv

    S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """

    t3: T3Cond
    gen: dict


class Model(nn.Module):
    """
    Chatterbox Text-to-Speech model.

    This integrates:
    - T3: LLaMA-based text-to-speech-token generator
    - S3Gen: Flow matching decoder with HiFi-GAN vocoder
    - VoiceEncoder: Speaker embedding extractor
    - S3Tokenizer: Speech tokenizer for reference audio
    - Text Tokenizer: Text to token ID conversion

    Usage:
        model = Model.from_pretrained("path/to/weights")
        audio = model.generate("Hello world!", audio_prompt_path="reference.wav")
    """

    ENC_COND_LEN = 6 * S3_SR  # 6 seconds at 16kHz for encoder
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds at 24kHz for decoder

    def __init__(
        self,
        config_or_t3: Union[ModelConfig, T3] = None,
        s3gen: Optional[S3Token2Wav] = None,
        ve: Optional[VoiceEncoder] = None,
        conds: Optional[Conditionals] = None,
    ):
        super().__init__()
        self.sr = S3GEN_SR  # sample rate of synthesized audio

        # Check if first argument is a config
        if config_or_t3 is None or isinstance(config_or_t3, ModelConfig):
            # Initialize from config
            config = config_or_t3 or ModelConfig()
            self.config = config
            self.t3 = T3(config.t3_config)
            self.s3gen = S3Token2Wav()
            self.ve = VoiceEncoder()
            self._conds = None
        else:
            # Initialize with individual components
            self.config = None
            self.t3 = config_or_t3
            self.s3gen = s3gen
            self.ve = ve
            self._conds = conds

        # S3 tokenizer for speech token extraction (initialized lazily or during load_weights)
        self._s3_tokenizer = S3TokenizerV2("speech_tokenizer_v2_25hz")
        # Text tokenizer (initialized during load_weights if model_path is available)
        self.tokenizer = None
        self.mtl_tokenizer = None

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize PyTorch weights for MLX.

        This routes weights to the appropriate component's sanitize method
        based on the weight key prefix. S3Tokenizer weights are loaded
        separately from mlx-community/S3TokenizerV2.

        Handles two cases:
        1. Pre-prefixed weights (from MLX converted models):
           - ve.* -> VoiceEncoder weights
           - t3.* -> T3 model weights
           - s3gen.* -> S3Gen (S3Token2Wav) weights

        2. Original PyTorch weights (without prefix):
           - Infers component from key names

        Each component's sanitize method handles:
        - Conv1d/Conv2d weight transposition
        - LSTM weight renaming (for VoiceEncoder)
        - Perceiver attention weight renaming (for T3)
        - ConvTranspose weight transposition (for S3Gen)
        """
        new_weights = {}

        # Separate weights by component (S3Tokenizer loaded separately)
        ve_weights = {}
        t3_weights = {}
        s3gen_weights = {}
        other_weights = {}

        for key, value in weights.items():
            if key.startswith("ve."):
                # Remove prefix for component sanitize
                ve_weights[key[3:]] = value
            elif key.startswith("t3."):
                t3_weights[key[3:]] = value
            elif key.startswith("s3gen."):
                s3gen_weights[key[6:]] = value
            else:
                # If no prefix, infer from key names
                # VoiceEncoder keys: lstm.*, similarity_*, proj.*
                if (
                    key.startswith("lstm.")
                    or key.startswith("similarity")
                    or key == "proj.weight"
                    or key == "proj.bias"
                ):
                    ve_weights[key] = value
                # T3 keys: tfmr.*, text_emb.*, speech_emb.*, text_head.*, speech_head.*,
                #          perceiver.*, cond_emb.*, prompt_pos_emb.*, cond_enc.*, text_pos_emb.*, speech_pos_emb.*
                elif (
                    key.startswith("tfmr.")
                    or key.startswith("text_emb.")
                    or key.startswith("speech_emb.")
                    or key.startswith("text_head.")
                    or key.startswith("speech_head.")
                    or key.startswith("perceiver.")
                    or key.startswith("cond_emb.")
                    or key.startswith("prompt_pos_emb.")
                    or key.startswith("cond_enc.")
                    or key.startswith("text_pos_emb.")
                    or key.startswith("speech_pos_emb.")
                ):
                    t3_weights[key] = value
                # S3Gen keys: flow.*, mel2wav.*, speaker_encoder.*, f0_predictor.*
                elif (
                    key.startswith("flow.")
                    or key.startswith("mel2wav.")
                    or key.startswith("speaker_encoder.")
                    or key.startswith("f0_predictor.")
                ):
                    s3gen_weights[key] = value
                else:
                    other_weights[key] = value

        # Sanitize each component's weights
        if ve_weights:
            ve_sanitized = self.ve.sanitize(ve_weights)
            for k, v in ve_sanitized.items():
                new_weights[f"ve.{k}"] = v

        if t3_weights:
            t3_sanitized = self.t3.sanitize(t3_weights)
            for k, v in t3_sanitized.items():
                new_weights[f"t3.{k}"] = v

        if s3gen_weights:
            s3gen_sanitized = self.s3gen.sanitize(s3gen_weights)
            for k, v in s3gen_sanitized.items():
                new_weights[f"s3gen.{k}"] = v

        # Add other weights as-is
        new_weights.update(other_weights)

        return new_weights

    def load_weights(
        self,
        weights,
        strict: bool = True,
    ):
        """
        Load weights into the model.

        Uses strict=False by default for components because Chatterbox has
        several non-checkpoint parameters (rand_noise, pos_enc.pe, stft_window,
        trim_fade) that are generated during initialization.

        Args:
            weights: List of (key, value) tuples or dict
            strict: If False, ignore missing/extra keys. Default True for
                    compatibility with utils.load_model().
        """
        if isinstance(weights, dict):
            weights = list(weights.items())

        # Split weights by component prefix (S3Tokenizer loaded separately)
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
            self.ve.load_weights(ve_weights, strict=False)
        if t3_weights:
            self.t3.load_weights(t3_weights, strict=False)
        if s3gen_weights:
            self.s3gen.load_weights(s3gen_weights, strict=False)

        # Handle any remaining weights at the top level
        if other_weights and strict:
            raise ValueError(
                f"Unrecognized weight keys: {[k for k, v in other_weights]}"
            )

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: Union[str, Path],
        s3_tokenizer_repo: str = "mlx-community/S3TokenizerV2",
    ) -> "Model":
        """
        Load a pretrained Chatterbox model from a checkpoint directory.

        Expects the standard mlx-audio format: a single model.safetensors file
        with component prefixes (ve.*, t3.*, s3gen.*). S3Tokenizer weights are loaded
        separately from a shared repository.

        Automatically handles quantized weights if config.json contains quantization info.

        Use scripts/convert.py to convert from original PyTorch weights.

        Args:
            ckpt_dir: Path to checkpoint directory containing model.safetensors
                and tokenizer.json.
            s3_tokenizer_repo: Hugging Face repo for S3Tokenizer weights.
                Default: "mlx-community/S3TokenizerV2"

        Returns:
            Initialized ChatterboxTTS model
        """
        import json

        from huggingface_hub import snapshot_download

        ckpt_dir = Path(ckpt_dir)

        # Download from Hub if path doesn't exist locally
        if not ckpt_dir.exists():
            print(f"Downloading {ckpt_dir} from Hugging Face...")
            ckpt_dir = Path(
                snapshot_download(
                    repo_id=str(ckpt_dir),
                    allow_patterns=[
                        "model.safetensors",
                        "tokenizer.json",
                        "config.json",
                    ],
                )
            )

        # Initialize models with default config
        model = cls()

        print("Loading MLX weights...")

        combined_path = ckpt_dir / "model.safetensors"
        if not combined_path.exists():
            raise FileNotFoundError(
                f"model.safetensors not found in {ckpt_dir}. "
                "Use scripts/convert.py to convert from PyTorch weights."
            )

        # Load config to check for quantization
        config_path = ckpt_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        all_weights = mx.load(str(combined_path))

        # Split weights by prefix (S3Tokenizer loaded separately)
        ve_weights = {k[3:]: v for k, v in all_weights.items() if k.startswith("ve.")}
        t3_weights = {k[3:]: v for k, v in all_weights.items() if k.startswith("t3.")}
        s3gen_weights = {
            k[6:]: v for k, v in all_weights.items() if k.startswith("s3gen.")
        }

        # Handle quantization if present in config
        if quantization := config.get("quantization"):
            print(f"Detected {quantization['bits']}-bit quantization...")

            # Quantize the model before loading weights
            # This converts Linear layers to QuantizedLinear so they can accept packed weights
            def should_quantize(path, module):
                """Check if this module should be quantized based on weights."""
                if not hasattr(module, "to_quantized"):
                    return False
                # Check if we have quantized weights (scales) for this path
                # Need to check in the appropriate weight dict based on prefix
                if path.startswith("t3."):
                    weight_path = path[3:]  # Remove "t3." prefix
                    return f"{weight_path}.scales" in t3_weights
                return False

            nn.quantize(
                model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                class_predicate=should_quantize,
            )

        def load_component_weights(component, weights, strict: bool = True):
            """Load weights dict into component."""
            if hasattr(component, "sanitize"):
                weights = component.sanitize(weights)
            component.load_weights(list(weights.items()), strict=strict)

        if ve_weights:
            load_component_weights(model.ve, ve_weights, strict=True)
        if t3_weights:
            load_component_weights(model.t3, t3_weights, strict=False)
        if s3gen_weights:
            load_component_weights(model.s3gen, s3gen_weights, strict=False)

        # Load S3Tokenizer from separate repo
        print(f"Loading S3Tokenizer from {s3_tokenizer_repo}...")
        s3tok_dir = Path(
            snapshot_download(
                repo_id=s3_tokenizer_repo,
                allow_patterns=["model.safetensors", "config.json"],
            )
        )
        s3tok_path = s3tok_dir / "model.safetensors"
        if not s3tok_path.exists():
            raise FileNotFoundError(
                f"model.safetensors not found in {s3_tokenizer_repo}. "
                "S3Tokenizer weights are required for Chatterbox."
            )
        s3tok_weights = mx.load(str(s3tok_path))
        model._s3_tokenizer = S3TokenizerV2("speech_tokenizer_v2_25hz")
        load_component_weights(model._s3_tokenizer, s3tok_weights, strict=False)

        # Initialize text tokenizer (check config for multilingual setting)
        tokenizer_path = ckpt_dir / "tokenizer.json"
        config = None
        if tokenizer_path.exists():
            try:
                import json

                from .tokenizer import EnTokenizer, MTLTokenizer

                # Check if multilingual model from config.json
                config_path = ckpt_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)

                if config and config.get("multilingual", False):
                    model.mtl_tokenizer = MTLTokenizer(tokenizer_path)
                    print("Loaded multilingual tokenizer (MTLTokenizer)")

                model.tokenizer = EnTokenizer(tokenizer_path)
                print("Loaded English tokenizer (EnTokenizer)")
            except ImportError:
                print("Warning: tokenizers library not available")
                model.tokenizer = None
        else:
            print("Warning: tokenizer.json not found")
            model.tokenizer = None

        # Set to eval mode for inference (important for BatchNorm)
        model.eval()
        print("Model loaded successfully!")
        return model

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        """
        Post-load hook called by load_model to initialize tokenizer and conditionals.

        Args:
            model: The loaded model instance
            model_path: Path to the model directory

        Returns:
            The model with tokenizer and conditionals initialized
        """
        # Load text tokenizer (check config for multilingual setting)
        tokenizer_path = model_path / "tokenizer.json"
        config = None
        if tokenizer_path.exists():
            try:
                import json

                from .tokenizer import EnTokenizer, MTLTokenizer

                # Check if multilingual model from config..json
                config_path = model_path / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)

                if config and config.get("multilingual", False):
                    model.mtl_tokenizer = MTLTokenizer(tokenizer_path)
                    print("Loaded multilingual tokenizer (MTLTokenizer)")

                model.tokenizer = EnTokenizer(tokenizer_path)
                print("Loaded English tokenizer (EnTokenizer)")
            except ImportError:
                print("Warning: tokenizers library not available")
                model.tokenizer = None
        else:
            print(f"Warning: tokenizer.json not found at {tokenizer_path}")
            model.tokenizer = None

        # Load S3Tokenizer from separate repo
        from huggingface_hub import snapshot_download

        s3_tokenizer_repo = "mlx-community/S3TokenizerV2"
        print(f"Loading S3Tokenizer from {s3_tokenizer_repo}...")
        s3tok_dir = Path(
            snapshot_download(
                repo_id=s3_tokenizer_repo,
                allow_patterns=["model.safetensors", "config.json"],
            )
        )
        s3tok_path = s3tok_dir / "model.safetensors"
        if s3tok_path.exists():
            s3tok_weights = mx.load(str(s3tok_path))
            model._s3_tokenizer = S3TokenizerV2("speech_tokenizer_v2_25hz")
            if hasattr(model._s3_tokenizer, "sanitize"):
                s3tok_weights = model._s3_tokenizer.sanitize(s3tok_weights)
            model._s3_tokenizer.load_weights(list(s3tok_weights.items()), strict=False)
            print("Loaded S3Tokenizer weights")
        else:
            print(f"Warning: S3Tokenizer weights not found at {s3tok_path}")

        # Load pre-computed conditionals from conds.safetensors
        conds_path = model_path / "conds.safetensors"

        if conds_path.exists():
            conds_data = mx.load(str(conds_path))
            print("Loaded pre-computed conditionals from conds.safetensors")

            # Extract T3 conditionals
            speaker_emb = conds_data.get("t3.speaker_emb")
            if speaker_emb is None:
                speaker_emb = mx.zeros((1, 256))

            cond_tokens = conds_data.get("t3.cond_prompt_speech_tokens")
            emotion_adv = conds_data.get("t3.emotion_adv")
            if emotion_adv is None:
                emotion_adv = mx.ones((1, 1, 1)) * 0.5

            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_tokens,
                emotion_adv=emotion_adv,
            )

            # Extract gen conditionals
            gen_dict = {}
            for k, v in conds_data.items():
                if k.startswith("gen."):
                    gen_dict[k.replace("gen.", "")] = v

            # Compute prompt_feat_len if missing
            if "prompt_feat_len" not in gen_dict and "prompt_feat" in gen_dict:
                prompt_feat = gen_dict["prompt_feat"]
                gen_dict["prompt_feat_len"] = mx.array([prompt_feat.shape[1]])

            model._conds = Conditionals(t3_cond, gen_dict)
        else:
            print("Warning: conds.safetensors not found - ref_audio will be required")
            model._conds = None

        return model

    def prepare_conditionals(
        self,
        ref_wav: Union[str, mx.array, np.ndarray],
        ref_sr: int,
        exaggeration: float = 0.5,
    ) -> Conditionals:
        """
        Prepare conditioning from a reference audio clip.

        Args:
            ref_wav: Reference waveform (samples,) or (1, samples)
            ref_sr: Reference sample rate
            exaggeration: Emotion exaggeration factor (0-1)

        Returns:
            Conditionals object with T3 and S3Gen conditioning

        Note:
            Following the original PyTorch implementation:
            - S3Gen uses up to 10s of audio (DEC_COND_LEN at 24kHz)
            - T3 encoder uses up to 6s of audio (ENC_COND_LEN at 16kHz)
            - S3Gen tokens are computed from 24kHz audio resampled to 16kHz
            - T3 tokens are computed from 16kHz audio (resampled from original)
        """
        # Ensure 1D waveform
        if isinstance(ref_wav, str):
            ref_wav, ref_sr = librosa.load(ref_wav, sr=S3GEN_SR)
            ref_wav = mx.array(ref_wav)

        if ref_wav.ndim == 2:
            ref_wav = ref_wav.squeeze(0)

        # Resample to 24kHz for S3Gen (mel extraction and base for speaker embedding)
        if ref_sr != S3GEN_SR:
            ref_wav_24k = resample_audio(ref_wav, ref_sr, S3GEN_SR)
        else:
            ref_wav_24k = ref_wav
        # Truncate to decoder conditioning length (10s)
        ref_wav_24k = ref_wav_24k[: self.DEC_COND_LEN]

        # Resample 24kHz to 16kHz for S3Gen tokenization and speaker embedding
        # This matches the original which resamples from 24kHz
        ref_wav_16k_from_24k = resample_audio(ref_wav_24k, S3GEN_SR, S3_SR)

        # Resample original to 16kHz for T3 encoder conditioning and VE embedding
        if ref_sr != S3_SR:
            ref_wav_16k_full = resample_audio(ref_wav, ref_sr, S3_SR)
        else:
            ref_wav_16k_full = ref_wav
        # Truncate to encoder conditioning length (6s) for T3 tokens only
        ref_wav_16k = ref_wav_16k_full[: self.ENC_COND_LEN]

        # Get S3Gen reference embeddings
        s3gen_ref_dict = {}
        t3_cond_prompt_tokens = None

        if self._s3_tokenizer is not None:
            # --- S3Gen tokens (from 10s audio, resampled 24k->16k) ---
            s3gen_mel = log_mel_spectrogram(ref_wav_16k_from_24k)
            s3gen_mel = mx.expand_dims(s3gen_mel, 0)  # Add batch dim
            s3gen_mel_len = mx.array([s3gen_mel.shape[2]])
            s3gen_tokens, s3gen_token_lens = self._s3_tokenizer(
                s3gen_mel, s3gen_mel_len
            )

            # Get S3Gen embeddings (tokens may be truncated by embed_ref to match mel)
            s3gen_ref_dict = self.s3gen.embed_ref(
                ref_wav=mx.expand_dims(ref_wav_24k, 0),
                ref_sr=S3GEN_SR,
                ref_speech_tokens=s3gen_tokens,
                ref_speech_token_lens=s3gen_token_lens,
            )

            # --- T3 conditioning tokens (from 6s audio, limited to plen) ---
            t3_mel = log_mel_spectrogram(ref_wav_16k)
            t3_mel = mx.expand_dims(t3_mel, 0)
            t3_mel_len = mx.array([t3_mel.shape[2]])
            t3_tokens, t3_token_lens = self._s3_tokenizer(t3_mel, t3_mel_len)

            # Limit T3 tokens to prompt length
            plen = self.t3.hp.speech_cond_prompt_len if hasattr(self.t3, "hp") else 150
            t3_cond_prompt_tokens = t3_tokens[:, :plen]

        # Voice encoder speaker embedding (from full 16kHz audio, not truncated)
        ve_embed = self.ve.embeds_from_wavs([ref_wav_16k_full], sample_rate=S3_SR)
        ve_embed = mx.mean(ve_embed, axis=0, keepdims=True)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=mx.ones((1, 1, 1)) * exaggeration,
        )

        return Conditionals(t3_cond, s3gen_ref_dict)

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return S3GEN_SR

    def generate(
        self,
        text: str,
        audio_prompt: Optional[mx.array] = None,
        audio_prompt_sr: Optional[int] = None,
        conds: Optional[Conditionals] = None,
        exaggeration: float = 0.1,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        max_new_tokens: int = 1000,
        # Standard mlx_audio.tts.generate parameters (mapped to Chatterbox params)
        ref_audio: Optional[Union[str, mx.array, np.ndarray]] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "en",
        max_tokens: int = None,
        verbose: bool = True,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            audio_prompt: Reference audio for voice cloning (optional)
            audio_prompt_sr: Sample rate of audio prompt
            conds: Pre-computed conditionals (optional)
            exaggeration: Emotion exaggeration factor (0-1)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeated tokens
            min_p: Minimum probability threshold
            top_p: Top-p (nucleus) sampling threshold
            max_new_tokens: Maximum number of tokens to generate
            ref_audio: Alias for audio_prompt (for mlx_audio.tts.generate compatibility)
            voice: Ignored (Chatterbox uses reference audio for voice cloning)
            speed: Ignored (Chatterbox doesn't support speed adjustment)
            lang_code: Ignored (Chatterbox is English-only)
            max_tokens: Alias for max_new_tokens
            verbose: Whether to print verbose output
            stream: Ignored (Chatterbox doesn't support streaming)
            streaming_interval: Ignored

        Yields:
            GenerationResult with generated audio waveform
        """
        start_time = time.time()

        # Handle parameter aliases for mlx_audio.tts.generate compatibility
        if ref_audio is not None and audio_prompt is None:
            audio_prompt = ref_audio
            audio_prompt_sr = (
                self.sample_rate
            )  # Assume ref_audio is already at correct sample rate
        if max_tokens is not None and max_new_tokens == 1000:
            max_new_tokens = max_tokens

        # Prepare conditionals if needed
        if conds is None:
            if audio_prompt is not None and audio_prompt_sr is not None:
                conds = self.prepare_conditionals(
                    audio_prompt, audio_prompt_sr, exaggeration
                )
            elif self._conds is not None:
                conds = self._conds
            else:
                raise ValueError(
                    "No conditionals available. Either provide audio_prompt/audio_prompt_sr "
                    "for voice cloning, or ensure conds.safetensors is in the model directory."
                )

        # Update exaggeration if needed
        if exaggeration != float(conds.t3.emotion_adv[0, 0, 0]):
            conds.t3.emotion_adv = mx.ones((1, 1, 1)) * exaggeration

        # Normalize and tokenize text
        text = punc_norm(text)

        try:
            if lang_code == "en":
                print("Using English tokenizer")
                text_tokens = self.tokenizer.text_to_tokens(text)
            elif isinstance(self.mtl_tokenizer, MTLTokenizer):
                print("Using multilingual tokenizer")
                text_tokens = self.mtl_tokenizer.text_to_tokens(
                    text, language_id=lang_code
                )
            else:
                if self.tokenizer is None and self.mtl_tokenizer is None:
                    raise ValueError(
                        "Text tokenizer or multilingual tokenizer not initialized.\n"
                        "Load model with from_pretrained() or set model.tokenizer manually.\n"
                    )
                else:
                    raise ValueError(
                        "Invalid language code. Supported languages: "
                        "ar (Arabic), da (Danish), de (German), el (Greek), en (English), "
                        "es (Spanish), fi (Finnish), fr (French), he (Hebrew), hi (Hindi), "
                        "it (Italian), ja (Japanese), ko (Korean), ms (Malay), nl (Dutch), "
                        "no (Norwegian), pl (Polish), pt (Portuguese), ru (Russian), "
                        "sv (Swedish), sw (Swahili), tr (Turkish), zh (Chinese)"
                    )

        except Exception as e:
            print(f"Error tokenizing text: {e}")
            raise e

        token_count = text_tokens.shape[1]

        if cfg_weight > 0.0:
            text_tokens = mx.concatenate([text_tokens, text_tokens], axis=0)

        # Add start/end tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token

        # Pad with start token
        sot_tokens = mx.full((text_tokens.shape[0], 1), sot, dtype=mx.int32)
        eot_tokens = mx.full((text_tokens.shape[0], 1), eot, dtype=mx.int32)
        text_tokens = mx.concatenate([sot_tokens, text_tokens, eot_tokens], axis=1)

        # Clear any accumulated cache from previous generations
        mx.clear_cache()

        # Generate speech tokens with T3
        speech_tokens = self.t3.inference(
            t3_cond=conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

        # Clear cache after T3 inference to free memory before S3Gen
        mx.clear_cache()

        # Extract conditional batch (first in CFG pair)
        speech_tokens = speech_tokens[0:1]

        # Drop invalid tokens
        speech_tokens = drop_invalid_tokens(speech_tokens)
        # Filter out tokens >= SPEECH_VOCAB_SIZE
        mask = speech_tokens < SPEECH_VOCAB_SIZE
        # Use argsort to get indices of valid tokens (preserves order via stable sort)
        valid_count = int(mx.sum(mask.astype(mx.int32)))
        sorted_indices = mx.argsort(-mask.astype(mx.int32))
        valid_indices = sorted_indices[:valid_count]
        speech_tokens = mx.take(speech_tokens, valid_indices)

        # Reshape for S3Gen
        speech_tokens = mx.expand_dims(speech_tokens, 0)

        # Generate waveform with S3Gen
        wav = self.s3gen(
            speech_tokens=speech_tokens,
            ref_dict=conds.gen,
            finalize=True,
        )

        # Flatten to 1D if needed
        if wav.ndim == 2:
            wav = wav.squeeze(0)

        # Calculate timing and metrics
        processing_time = time.time() - start_time
        samples = wav.shape[0]
        audio_duration_seconds = samples / self.sample_rate

        # Format duration as HH:MM:SS.mmm
        duration_hours = int(audio_duration_seconds // 3600)
        duration_mins = int((audio_duration_seconds % 3600) // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        # Calculate real-time factor
        rtf = (
            processing_time / audio_duration_seconds
            if audio_duration_seconds > 0
            else 0
        )

        yield GenerationResult(
            audio=wav,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=round(rtf, 2),
            prompt={
                "tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / processing_time, 2)
                    if processing_time > 0
                    else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / processing_time, 2) if processing_time > 0 else 0
                ),
            },
            processing_time_seconds=processing_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

        # Clear cache after generation
        mx.clear_cache()

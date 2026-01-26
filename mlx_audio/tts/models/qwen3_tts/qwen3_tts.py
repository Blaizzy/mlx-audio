# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import json
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.dsp import mel_filters, stft
from mlx_audio.tts.models.base import GenerationResult

from .config import (
    ModelConfig,
    Qwen3TTSTokenizerConfig,
    Qwen3TTSTokenizerDecoderConfig,
    Qwen3TTSTokenizerEncoderConfig,
)
from .speaker_encoder import Qwen3TTSSpeakerEncoder
from .speech_tokenizer import Qwen3TTSSpeechTokenizer
from .talker import Qwen3TTSTalkerForConditionalGeneration, RMSNorm


def mel_spectrogram(
    audio: mx.array,
    n_fft: int = 1024,
    num_mels: int = 128,
    sample_rate: int = 24000,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax: float = 12000.0,
) -> mx.array:
    """Compute mel spectrogram from audio waveform."""
    if audio.ndim == 1:
        audio = audio[None, :]

    batch_size, _ = audio.shape

    # Get mel filterbank from shared DSP module (cached)
    mel_basis = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        f_min=fmin,
        f_max=fmax,
        norm="slaney",
        mel_scale="slaney",
    )

    # Compute STFT for each sample in batch
    mels = []
    padding = n_fft // 2
    for i in range(batch_size):
        # Manual reflect padding to match PyTorch reference (center=False with manual pad)
        sample = audio[i]
        left_pad = sample[1 : padding + 1][::-1]
        right_pad = sample[-(padding + 1) : -1][::-1]
        sample = mx.concatenate([left_pad, sample, right_pad])

        spec = stft(
            sample,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window="hann",
            center=False,
            pad_mode="reflect",
        )
        # Get magnitude spectrum (with epsilon for numerical stability)
        spec_mag = mx.sqrt(mx.abs(spec) ** 2 + 1e-9)

        # Apply mel filterbank: spec_mag is [frames, n_fft//2+1], mel_basis is [n_mels, n_fft//2+1]
        mel = mx.matmul(spec_mag, mel_basis.T)

        # Log scale
        mel = mx.log(mx.clip(mel, 1e-5, None))
        mels.append(mel)

    return mx.stack(mels, axis=0)  # [batch, frames, n_mels]


def check_array_shape_qwen3(arr: mx.array) -> bool:
    """Check if Conv1d weights are already in MLX format.

    MLX format: (out_channels, kernel_size, in_channels)
    PyTorch format: (out_channels, in_channels, kernel_size)
    """
    shape = arr.shape
    if len(shape) != 3:
        return False

    out_channels, dim2, dim3 = shape

    if dim2 == 1:
        # Pattern: (out, 1, dim3)
        if dim3 > 64:
            # dim3 is large, likely in_channels -> MLX format (out, kernel=1, in)
            return True
        else:
            # dim3 is small, likely kernel -> PyTorch format (out, in=1, kernel)
            return False
    elif dim3 == 1:
        # Pattern: (out, dim2, 1)
        if dim2 > 64:
            # dim2 is large, likely in_channels -> PyTorch format (out, in, kernel=1)
            return False
        else:
            # dim2 is small, likely kernel -> MLX format (out, kernel, in=1)
            return True

    # General heuristic: kernel_size < in_channels is more common
    # So if middle dimension is smaller, it's likely already MLX format
    if dim2 < dim3:
        return True
    else:
        return False


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class Model(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._sample_rate = config.sample_rate

        # Main talker model
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config.talker_config)

        # Speaker encoder (only for base models that support voice cloning)
        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        # Speech tokenizer (loaded separately)
        self.speech_tokenizer = None

        # Text tokenizer (loaded in post_load_hook)
        self.tokenizer = None

        # Generation config
        self.generate_config = None

        # Supported speakers and languages from config
        self.supported_speakers = (
            list(config.talker_config.spk_id.keys())
            if config.talker_config.spk_id
            else []
        )
        self.supported_languages = ["auto"]
        if config.talker_config.codec_language_id:
            for lang_id in config.talker_config.codec_language_id.keys():
                if "dialect" not in lang_id:
                    self.supported_languages.append(lang_id)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_type(self) -> str:
        return "qwen3_tts"

    def load_speech_tokenizer(self, speech_tokenizer: Qwen3TTSSpeechTokenizer):
        """Load the speech tokenizer model."""
        self.speech_tokenizer = speech_tokenizer

    def load_generate_config(self, generate_config: dict):
        """Load generation configuration."""
        self.generate_config = generate_config

    def get_supported_speakers(self) -> List[str]:
        """Get list of supported speaker names."""
        return self.supported_speakers

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return self.supported_languages

    def model_quant_predicate(self, path: str, module) -> bool:

        skip_patterns = [
            "codec_embedding",
            "text_embedding",
            "speech_tokenizer",
            "speaker_encoder",
        ]
        return not any(pattern in path for pattern in skip_patterns)

    def extract_speaker_embedding(
        self,
        audio: mx.array,
        sr: int = 24000,
    ) -> mx.array:
        """Extract speaker embedding from reference audio.

        Args:
            audio: Audio waveform [samples]
            sr: Sample rate (must be 24000)

        Returns:
            Speaker embedding [1, enc_dim]
        """
        if sr != 24000:
            raise ValueError(
                "Only 24kHz audio is supported for speaker embedding extraction"
            )

        if self.speaker_encoder is None:
            raise ValueError("Speaker encoder not available for this model type")

        # Compute mel spectrogram
        mels = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=128,
            sample_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        )  # [batch, time, mels]

        # Extract embedding
        speaker_embedding = self.speaker_encoder(mels)
        return speaker_embedding

    def _prepare_generation_inputs(
        self,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Prepare inputs for generation.

        Args:
            text: Text to synthesize
            language: Language code
            speaker: Speaker name (for CustomVoice/Base models)
            ref_audio: Reference audio for voice cloning
            ref_text: Reference text for voice cloning
            instruct: Instruction text for voice style (for VoiceDesign/CustomVoice models)

        Returns:
            input_embeds: Input embeddings for the talker
            trailing_text_hidden: Remaining text embeddings
            tts_pad_embed: Padding embedding
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        config = self.config.talker_config

        # Tokenize text with chat template
        chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = mx.array(self.tokenizer.encode(chat_text))[None, :]

        # Get text embeddings
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(input_ids)
        )

        # TTS special tokens
        tts_tokens = mx.array(
            [
                [
                    self.config.tts_bos_token_id,
                    self.config.tts_eos_token_id,
                    self.config.tts_pad_token_id,
                ]
            ]
        )
        tts_embeds = self.talker.text_projection(
            self.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        # Speaker embedding
        speaker_embed = None
        if ref_audio is not None and self.speaker_encoder is not None:
            speaker_embed = self.extract_speaker_embedding(ref_audio)
        elif speaker and speaker.lower() in (config.spk_id or {}):
            spk_ids = mx.array([[config.spk_id[speaker.lower()]]])  # [1, 1]
            speaker_embed = self.talker.get_input_embeddings()(
                spk_ids
            )  # [1, 1, hidden]

        # Language ID
        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            if language.lower() in config.codec_language_id:
                language_id = config.codec_language_id[language.lower()]

        # Check for dialect override
        if (
            language.lower() in ["chinese", "auto"]
            and speaker
            and speaker.lower() in (config.spk_is_dialect or {})
            and config.spk_is_dialect[speaker.lower()]
        ):
            dialect = config.spk_is_dialect[speaker.lower()]
            if dialect in config.codec_language_id:
                language_id = config.codec_language_id[dialect]

        # Build codec prefix
        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_embed = self.talker.get_input_embeddings()(mx.array([codec_prefill]))

        codec_embed_suffix = self.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )

        if speaker_embed is not None:
            codec_embed = mx.concatenate(
                [
                    codec_embed,
                    speaker_embed.reshape(1, 1, -1),
                    codec_embed_suffix,
                ],
                axis=1,
            )
        else:
            codec_embed = mx.concatenate([codec_embed, codec_embed_suffix], axis=1)

        # Instruct embedding (for VoiceDesign/CustomVoice models)
        instruct_embed = None
        if instruct:
            instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_ids = mx.array(self.tokenizer.encode(instruct_text))[None, :]
            instruct_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(instruct_ids)
            )

        # Role embedding (first 3 tokens: <|im_start|>assistant\n)
        role_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(input_ids[:, :3])
        )

        # Combine embeddings
        # tts_pad * (codec_len - 2) + tts_bos
        pad_count = codec_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(
            tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1])
        )
        combined_embed = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined_embed = combined_embed + codec_embed[:, :-1, :]

        # Full input embedding
        # If instruct is provided, prepend it
        if instruct_embed is not None:
            input_embeds = mx.concatenate(
                [instruct_embed, role_embed, combined_embed], axis=1
            )
        else:
            input_embeds = mx.concatenate([role_embed, combined_embed], axis=1)

        # Add first text token
        first_text_embed = (
            self.talker.text_projection(
                self.talker.get_text_embeddings()(input_ids[:, 3:4])
            )
            + codec_embed[:, -1:, :]
        )
        input_embeds = mx.concatenate([input_embeds, first_text_embed], axis=1)

        # Trailing text (rest of the text)
        trailing_text_hidden = mx.concatenate(
            [
                self.talker.text_projection(
                    self.talker.get_text_embeddings()(input_ids[:, 4:-5])
                ),
                tts_eos_embed,
            ],
            axis=1,
        )

        return input_embeds, trailing_text_hidden, tts_pad_embed

    def _prepare_icl_generation_inputs(
        self,
        text: str,
        ref_audio: mx.array,
        ref_text: str,
        language: str = "auto",
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Prepare inputs for ICL (In-Context Learning) voice cloning.

        Matches the official Qwen3-TTS generate_icl_prompt structure:
        1. text_embed = text_projection(text_embeddings(ref_text_tokens + target_text_tokens)) + eos
        2. codec_embed = codec_bos + sum_of_all_codebook_embeddings(ref_codes)
        3. Streaming overlay: text[:codec_len] + codec if text longer, else padded text + codec

        Args:
            text: Target text to synthesize
            ref_audio: Reference audio waveform [samples]
            ref_text: Transcript of the reference audio
            language: Language code

        Returns:
            input_embeds: Input embeddings for prefill
            trailing_text_hidden: Remaining text embeddings for generation
            tts_pad_embed: Padding embedding
            ref_codes: Reference codes [1, num_quantizers, ref_time]
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        config = self.config.talker_config

        # 1. Encode reference audio -> ref_codes [1, 16, ref_time]
        audio_for_spk = ref_audio  # Save original shape for speaker embedding
        if ref_audio.ndim == 1:
            ref_audio = ref_audio[None, None, :]  # [1, 1, samples]
        elif ref_audio.ndim == 2:
            ref_audio = ref_audio[None, :]  # [1, 1, samples]
        ref_codes = self.speech_tokenizer.encode(ref_audio)  # [1, 16, ref_time]
        mx.eval(ref_codes)
        ref_time = ref_codes.shape[2]

        # 2. Tokenize ref_text and target_text separately
        # ref_text format: <|im_start|>assistant\n{ref_text}<|im_end|>\n
        ref_chat = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_ids = mx.array(self.tokenizer.encode(ref_chat))[None, :]
        # Pure ref text tokens: skip first 3 (role) and last 2 (<|im_end|>\n)
        ref_text_ids = ref_ids[:, 3:-2]

        # target_text format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        target_chat = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        target_ids = mx.array(self.tokenizer.encode(target_chat))[None, :]
        # Pure target text tokens: skip first 3 (role) and last 5 (trailing template)
        text_ids = target_ids[:, 3:-5]

        # 3. TTS special tokens
        tts_tokens = mx.array(
            [
                [
                    self.config.tts_bos_token_id,
                    self.config.tts_eos_token_id,
                    self.config.tts_pad_token_id,
                ]
            ]
        )
        tts_embeds = self.talker.text_projection(
            self.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        # 4. Build text_embed: text_projection(text_embeddings(ref_tokens + target_tokens)) + eos
        combined_text_ids = mx.concatenate([ref_text_ids, text_ids], axis=1)
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(combined_text_ids)
        )
        text_embed = mx.concatenate([text_embed, tts_eos_embed], axis=1)
        text_lens = text_embed.shape[1]

        # 5. Build codec_embed: codec_bos + sum_of_all_codebook_embeddings(ref_codes)
        # ref_codes shape: [1, 16, ref_time]
        first_cb_codes = ref_codes[:, 0, :]  # [1, ref_time]
        ref_codec_embed = self.talker.get_input_embeddings()(first_cb_codes)
        for i in range(config.num_code_groups - 1):
            cb_codes = ref_codes[:, i + 1, :]
            ref_codec_embed = (
                ref_codec_embed
                + self.talker.code_predictor.codec_embedding[i](cb_codes)
            )

        # Prepend codec_bos
        codec_bos_embed = self.talker.get_input_embeddings()(
            mx.array([[config.codec_bos_id]])
        )
        codec_embed_icl = mx.concatenate(
            [codec_bos_embed, ref_codec_embed], axis=1
        )  # [1, ref_time+1, hidden]
        codec_lens = codec_embed_icl.shape[1]

        # 6. Non-streaming mode overlay (matching official Qwen3-TTS non_streaming_mode=True)
        # All text first (overlaid with codec_pad), then all codec (overlaid with tts_pad).
        # This preserves full text context in the prefill, which is critical when
        # codec_lens > text_lens (long references).
        codec_pad_embed = self.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id]])
        )
        text_with_codec_pad = text_embed + mx.broadcast_to(
            codec_pad_embed, (1, text_lens, codec_pad_embed.shape[-1])
        )
        codec_with_text_pad = codec_embed_icl + mx.broadcast_to(
            tts_pad_embed, (1, codec_lens, tts_pad_embed.shape[-1])
        )
        icl_input_embed = mx.concatenate(
            [text_with_codec_pad, codec_with_text_pad], axis=1
        )
        trailing_text_hidden = tts_pad_embed

        # 7. Language ID
        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            if language.lower() in config.codec_language_id:
                language_id = config.codec_language_id[language.lower()]

        # 8. Speaker embedding (ICL still uses x-vector)
        speaker_embed = None
        if self.speaker_encoder is not None:
            speaker_embed = self.extract_speaker_embedding(audio_for_spk)

        # 9. Build codec prefix (think/nothink + speaker + pad + bos)
        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_prefix_embed = self.talker.get_input_embeddings()(
            mx.array([codec_prefill])
        )
        codec_prefix_suffix = self.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )

        if speaker_embed is not None:
            codec_prefix_embed = mx.concatenate(
                [
                    codec_prefix_embed,
                    speaker_embed.reshape(1, 1, -1),
                    codec_prefix_suffix,
                ],
                axis=1,
            )
        else:
            codec_prefix_embed = mx.concatenate(
                [codec_prefix_embed, codec_prefix_suffix], axis=1
            )

        # 10. Role embedding (first 3 tokens: <|im_start|>assistant\n)
        role_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(target_ids[:, :3])
        )

        # 11. Build pad/bos prefix (text side overlaid with codec prefix[:-1])
        pad_count = codec_prefix_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(
            tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1])
        )
        combined_prefix = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined_prefix = combined_prefix + codec_prefix_embed[:, :-1, :]

        # 12. Full input_embeds: role + codec_prefix + icl_embed
        input_embeds = mx.concatenate(
            [role_embed, combined_prefix, icl_input_embed], axis=1
        )

        return input_embeds, trailing_text_hidden, tts_pad_embed, ref_codes

    def _sample_token(
        self,
        logits: mx.array,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        generated_tokens: Optional[List[int]] = None,
        suppress_tokens: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
    ) -> mx.array:
        """Sample next token from logits."""
        logits = logits[:, -1, :]  # Get last position

        # Suppress invalid tokens (set to -inf)
        if suppress_tokens:
            logits_np = np.array(logits.astype(mx.float32))
            logits_np[:, suppress_tokens] = float("-inf")
            logits = mx.array(logits_np)

        # Apply repetition penalty (matches HuggingFace RepetitionPenaltyLogitsProcessor)
        if generated_tokens and repetition_penalty != 1.0:
            logits_np = np.array(logits.astype(mx.float32))
            for token in set(generated_tokens):
                if token < logits_np.shape[-1]:
                    score = logits_np[:, token]
                    # Negative scores: multiply by penalty (more negative = less likely)
                    # Positive scores: divide by penalty (smaller = less likely)
                    logits_np[:, token] = np.where(
                        score < 0,
                        score * repetition_penalty,
                        score / repetition_penalty,
                    )
            logits = mx.array(logits_np)

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            return mx.argmax(logits, axis=-1, keepdims=True)

        # Save EOS logit before filtering (to preserve it as a valid candidate)
        eos_logit = None
        if eos_token_id is not None and eos_token_id < logits.shape[-1]:
            eos_logit = logits[:, eos_token_id : eos_token_id + 1]

        # Top-k filtering
        if top_k > 0 and top_k < logits.shape[-1]:
            top_k_vals = mx.topk(logits, k=top_k, axis=-1)
            threshold = top_k_vals[:, -1:]
            logits = mx.where(logits < threshold, float("-inf"), logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
            # Remove tokens with cumulative prob above threshold
            sorted_mask = cumulative_probs > top_p
            # Shift right to keep at least one token
            sorted_mask = mx.concatenate(
                [
                    mx.zeros((sorted_mask.shape[0], 1), dtype=mx.bool_),
                    sorted_mask[:, :-1],
                ],
                axis=-1,
            )
            threshold = mx.take_along_axis(
                sorted_logits,
                mx.argmax(sorted_mask.astype(mx.int32), axis=-1, keepdims=True),
                axis=-1,
            )
            logits = mx.where(logits < threshold, float("-inf"), logits)

        # Restore EOS logit after filtering so it's always a valid candidate
        if eos_logit is not None:
            logits_np = np.array(logits.astype(mx.float32))
            logits_np[:, eos_token_id] = np.array(
                eos_logit.astype(mx.float32)
            ).flatten()
            logits = mx.array(logits_np)

        # Sample from categorical distribution (takes logits directly)
        return mx.random.categorical(logits)[:, None]

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        temperature: float = 0.9,
        speed: float = 1.0,
        lang_code: str = "auto",
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        split_pattern: str = "\n",
        max_tokens: int = 4096,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate audio from text.

        Automatically routes to the appropriate generation method based on model type:
        - voice_design: Uses generate_voice_design() with instruct as voice description
        - custom_voice: Uses generate_custom_voice() with voice as speaker and optional instruct
        - base: Uses standard generation with voice as speaker

        Args:
            text: Input text to synthesize
            voice: Speaker name (for multi-speaker models, e.g., 'Chelsie', 'Ethan')
            instruct: Instruction for emotion/style (CustomVoice) or voice description (VoiceDesign)
            temperature: Sampling temperature
            speed: Speech speed factor (not directly supported yet)
            lang_code: Language code (auto, chinese, english, etc.)
            ref_audio: Reference audio for voice cloning
            ref_text: Reference text for voice cloning
            split_pattern: Pattern to split text into segments
            max_tokens: Maximum tokens per segment
            verbose: Print verbose output
            stream: Enable streaming output
            streaming_interval: Interval for streaming chunks (seconds)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Repetition penalty

        Yields:
            GenerationResult objects with generated audio
        """
        # Route to appropriate method based on model type
        tts_model_type = getattr(self.config, "tts_model_type", "base")

        if tts_model_type == "voice_design":
            if not instruct:
                raise ValueError(
                    "VoiceDesign model requires 'instruct' to describe the voice "
                    "(e.g., 'A cheerful young female voice with high pitch')"
                )
            yield from self.generate_voice_design(
                text=text,
                instruct=instruct,
                language=lang_code,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                verbose=verbose,
            )
            return

        if tts_model_type == "custom_voice":
            if not voice:
                raise ValueError(
                    "CustomVoice model requires 'voice' (speaker name) "
                    "(e.g., 'Chelsie', 'Ethan', 'Vivian')"
                )
            yield from self.generate_custom_voice(
                text=text,
                speaker=voice,
                language=lang_code,
                instruct=instruct,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                verbose=verbose,
            )
            return

        # Base model generation
        if self.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")

        # Check if we should use ICL mode
        use_icl = (
            ref_audio is not None
            and ref_text is not None
            and self.speech_tokenizer.has_encoder
        )

        if use_icl:
            # ICL mode needs stronger repetition penalty to prevent code
            # degeneration with long reference audio prefills
            icl_rep_penalty = max(repetition_penalty, 1.5)
            yield from self._generate_icl(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=lang_code,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=icl_rep_penalty,
                verbose=verbose,
            )
            return

        # Split text into segments
        if split_pattern:
            segments = [s.strip() for s in text.split(split_pattern) if s.strip()]
        else:
            segments = [text]

        total_samples = 0
        total_tokens = 0

        for segment_idx, segment_text in enumerate(segments):
            start_time = time.time()

            if verbose:
                print(
                    f"Processing segment {segment_idx + 1}/{len(segments)}: {segment_text[:50]}..."
                )

            # Prepare inputs
            input_embeds, trailing_text_hidden, tts_pad_embed = (
                self._prepare_generation_inputs(
                    segment_text,
                    language=lang_code,
                    speaker=voice,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            )

            # Initialize cache using mlx_lm's KVCache
            cache = self.talker.make_cache()
            generated_codes = []
            config = self.config.talker_config
            eos_token_id = config.codec_eos_token_id
            trailing_idx = 0

            # Suppress special tokens [vocab_size-1024, vocab_size) except EOS
            suppress_tokens = [
                i
                for i in range(config.vocab_size - 1024, config.vocab_size)
                if i != eos_token_id
            ]

            for step in range(max_tokens):
                # Forward pass through talker
                logits, hidden = self.talker(
                    input_embeds,
                    cache=cache,
                )

                # Sample first codebook token (with special token suppression)
                next_token = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=(
                        [int(c[0, 0]) for c in generated_codes]
                        if generated_codes
                        else None
                    ),
                    suppress_tokens=suppress_tokens,
                    eos_token_id=eos_token_id,
                )

                # Check for EOS
                if int(next_token[0, 0]) == eos_token_id:
                    break

                # Generate remaining codebook tokens with code predictor
                code_tokens = [next_token]
                code_hidden = hidden[:, -1:, :]
                code_cache = self.talker.code_predictor.make_cache()

                for code_idx in range(config.num_code_groups - 1):
                    if code_idx == 0:
                        # Prefill: concatenate [hidden_state, code_0_embed] as sequence
                        # This matches PyTorch where inputs_embeds.shape[1] > 1
                        code_0_embed = self.talker.get_input_embeddings()(next_token)
                        code_input = mx.concatenate(
                            [code_hidden, code_0_embed], axis=1
                        )  # [1, 2, hidden]
                    else:
                        # Generation: just pass embedding of previous code token
                        # The KV cache provides context from previous positions
                        code_embed = self.talker.code_predictor.codec_embedding[
                            code_idx - 1
                        ](code_tokens[-1])
                        code_input = code_embed  # [1, 1, hidden]

                    # Code predictor forward
                    code_logits, code_cache, _ = self.talker.code_predictor(
                        code_input,
                        cache=code_cache,
                        generation_step=code_idx,
                    )

                    # Sample
                    next_code = self._sample_token(
                        code_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    code_tokens.append(next_code)

                # Stack all codebook tokens
                all_codes = mx.concatenate(code_tokens, axis=1)  # [1, num_code_groups]
                generated_codes.append(all_codes)

                # Prepare next input
                # Add trailing text if available
                if trailing_idx < trailing_text_hidden.shape[1]:
                    text_embed = trailing_text_hidden[
                        :, trailing_idx : trailing_idx + 1, :
                    ]
                    trailing_idx += 1
                else:
                    text_embed = tts_pad_embed

                # Codec embedding for next step
                codec_embed = self.talker.get_input_embeddings()(next_token)
                for i, code in enumerate(code_tokens[1:]):
                    codec_embed = (
                        codec_embed
                        + self.talker.code_predictor.codec_embedding[i](code)
                    )

                input_embeds = text_embed + codec_embed

                mx.eval(input_embeds)

                if verbose and step % 100 == 0:
                    print(f"  Step {step}, generated {len(generated_codes)} tokens")

            if not generated_codes:
                if verbose:
                    print(f"  No codes generated for segment {segment_idx}")
                continue

            # Stack all generated codes
            codes = mx.stack(generated_codes, axis=1)  # [1, seq_len, num_code_groups]

            if verbose:
                print(f"  Decoding {codes.shape[1]} tokens to audio...")

            # Decode to audio
            audio, audio_lengths = self.speech_tokenizer.decode(codes)
            audio = audio[0]  # Remove batch dim

            # Trim to valid length
            valid_len = int(audio_lengths[0])
            if valid_len > 0 and valid_len < audio.shape[0]:
                audio = audio[:valid_len]

            mx.eval(audio)

            elapsed_time = time.time() - start_time
            samples = audio.shape[0]
            token_count = len(generated_codes)

            total_samples += samples
            total_tokens += token_count

            duration_seconds = samples / self.sample_rate
            rtf = duration_seconds / elapsed_time if elapsed_time > 0 else 0

            yield GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=token_count,
                audio_duration=format_duration(duration_seconds),
                real_time_factor=rtf,
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": (
                        token_count / elapsed_time if elapsed_time > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        samples / elapsed_time if elapsed_time > 0 else 0
                    ),
                },
                processing_time_seconds=elapsed_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            # Clear cache between segments

            mx.clear_cache()

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 4096,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        verbose: bool = False,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech with the CustomVoice model using a predefined speaker.

        This method is for CustomVoice model variants (e.g., Qwen3-TTS-12Hz-*-CustomVoice).
        It uses predefined speaker voices with optional emotion/style instructions.

        Args:
            text: Text to synthesize
            speaker: Speaker name (e.g., 'Vivian', 'Ryan'). Use get_supported_speakers() to list available.
            language: Language code ('auto', 'chinese', 'english', etc.)
            instruct: Optional instruction for emotion/style (e.g., '用特别愤怒的语气说', 'Very happy.')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Repetition penalty
            verbose: Print verbose output

        Yields:
            GenerationResult objects with generated audio

        Example:
            >>> results = list(model.generate_custom_voice(
            ...     text="Hello, how are you?",
            ...     speaker="Vivian",
            ...     language="English",
            ...     instruct="Very happy and excited."
            ... ))
        """
        if self.config.tts_model_type != "custom_voice":
            raise ValueError(
                f"Model type '{self.config.tts_model_type}' does not support generate_custom_voice. "
                "Please use a CustomVoice model (e.g., Qwen/Qwen3-TTS-12Hz-*-CustomVoice)."
            )

        # Validate speaker
        if speaker.lower() not in [s.lower() for s in self.supported_speakers]:
            raise ValueError(
                f"Speaker '{speaker}' not supported. Available: {self.supported_speakers}"
            )

        # For 0.6B models, instruct is not supported
        if self.config.tts_model_size == "0b6":
            instruct = None

        yield from self._generate_with_instruct(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "auto",
        temperature: float = 0.9,
        max_tokens: int = 4096,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        verbose: bool = False,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech with the VoiceDesign model using natural language voice description.

        This method is for VoiceDesign model variants (e.g., Qwen3-TTS-12Hz-*-VoiceDesign).
        The voice characteristics are entirely defined by the instruction text.

        Args:
            text: Text to synthesize
            instruct: Voice description (e.g., '体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显')
            language: Language code ('auto', 'chinese', 'english', etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Repetition penalty
            verbose: Print verbose output

        Yields:
            GenerationResult objects with generated audio

        Example:
            >>> results = list(model.generate_voice_design(
            ...     text="哥哥，你回来啦！",
            ...     instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、卖萌的听觉效果。",
            ...     language="Chinese"
            ... ))
        """
        if self.config.tts_model_type != "voice_design":
            raise ValueError(
                f"Model type '{self.config.tts_model_type}' does not support generate_voice_design. "
                "Please use a VoiceDesign model (e.g., Qwen/Qwen3-TTS-12Hz-*-VoiceDesign)."
            )

        yield from self._generate_with_instruct(
            text=text,
            speaker=None,  # No speaker for VoiceDesign
            language=language,
            instruct=instruct,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )

    def _generate_icl(
        self,
        text: str,
        ref_audio: mx.array,
        ref_text: str,
        language: str = "auto",
        temperature: float = 0.9,
        max_tokens: int = 4096,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.5,
        verbose: bool = False,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech using ICL (In-Context Learning) voice cloning.

        Encodes reference audio through the speech tokenizer encoder, uses the
        encoded codes as context for generation, then prepends them to the
        generated codes for decoding.
        """
        start_time = time.time()

        if verbose:
            print(f"ICL generation: {text[:50]}...")
            print(f"  Reference text: {ref_text[:50]}...")

        # Prepare ICL inputs
        input_embeds, trailing_text_hidden, tts_pad_embed, ref_codes = (
            self._prepare_icl_generation_inputs(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
            )
        )

        if verbose:
            print(
                f"  Encoded reference: {ref_codes.shape[2]} frames, "
                f"input sequence: {input_embeds.shape[1]} tokens"
            )

        # Cap max_tokens based on target text length to prevent runaway generation
        # when reference audio is long and EOS logit is suppressed by top-k.
        # At 12.5 Hz codec rate, ~3-5 codec tokens per text token is typical speech.
        # Factor of 6 gives ~50% margin for slow speech / pauses.
        target_token_count = len(self.tokenizer.encode(text))
        effective_max_tokens = min(max_tokens, max(75, target_token_count * 6))
        if verbose and effective_max_tokens < max_tokens:
            print(
                f"  Effective max_tokens: {effective_max_tokens} "
                f"(based on {target_token_count} text tokens)"
            )

        # Initialize cache
        cache = self.talker.make_cache()
        generated_codes = []
        config = self.config.talker_config
        eos_token_id = config.codec_eos_token_id
        suppress_tokens = [
            i
            for i in range(config.vocab_size - 1024, config.vocab_size)
            if i != eos_token_id
        ]
        trailing_idx = 0

        for step in range(effective_max_tokens):
            # Forward pass through talker
            logits, hidden = self.talker(input_embeds, cache=cache)

            # Sample first codebook token
            next_token = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=(
                    [int(c[0, 0]) for c in generated_codes] if generated_codes else None
                ),
                suppress_tokens=suppress_tokens,
                eos_token_id=eos_token_id,
            )

            # Check for EOS
            if int(next_token[0, 0]) == eos_token_id:
                break

            # Generate remaining codebook tokens with code predictor
            code_tokens = [next_token]
            code_hidden = hidden[:, -1:, :]
            code_cache = self.talker.code_predictor.make_cache()

            for code_idx in range(config.num_code_groups - 1):
                if code_idx == 0:
                    code_0_embed = self.talker.get_input_embeddings()(next_token)
                    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                else:
                    code_embed = self.talker.code_predictor.codec_embedding[
                        code_idx - 1
                    ](code_tokens[-1])
                    code_input = code_embed

                code_logits, code_cache, _ = self.talker.code_predictor(
                    code_input,
                    cache=code_cache,
                    generation_step=code_idx,
                )

                next_code = self._sample_token(
                    code_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                code_tokens.append(next_code)

            # Stack all codebook tokens
            all_codes = mx.concatenate(code_tokens, axis=1)
            generated_codes.append(all_codes)

            # Prepare next input
            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed

            codec_embed = self.talker.get_input_embeddings()(next_token)
            for i, code in enumerate(code_tokens[1:]):
                codec_embed = codec_embed + self.talker.code_predictor.codec_embedding[
                    i
                ](code)

            input_embeds = text_embed + codec_embed
            mx.eval(input_embeds)

            if verbose and step % 100 == 0:
                print(f"  Step {step}, generated {len(generated_codes)} tokens")

        if not generated_codes:
            if verbose:
                print("  No codes generated")
            return

        # Stack generated codes
        gen_codes = mx.stack(generated_codes, axis=1)  # [1, gen_len, num_code_groups]

        # Prepend reference codes to generated codes for decoding
        # ref_codes: [1, 16, ref_time] -> [1, ref_time, 16]
        ref_codes_t = mx.transpose(ref_codes, (0, 2, 1))
        # Combine: [1, ref_time + gen_len, 16]
        full_codes = mx.concatenate([ref_codes_t, gen_codes], axis=1)

        ref_len = ref_codes.shape[2]
        total_len = full_codes.shape[1]

        if verbose:
            print(
                f"  Decoding {total_len} tokens "
                f"({ref_len} ref + {gen_codes.shape[1]} gen) to audio..."
            )

        # Decode full codes to audio
        audio, audio_lengths = self.speech_tokenizer.decode(full_codes)
        audio = audio[0]  # Remove batch dim

        # Trim to valid length
        valid_len = int(audio_lengths[0])
        if valid_len > 0 and valid_len < audio.shape[0]:
            audio = audio[:valid_len]

        # Remove the reference audio portion using proportional trimming
        # (matches official implementation)
        cut = int(ref_len / max(total_len, 1) * audio.shape[0])
        if cut > 0 and cut < audio.shape[0]:
            audio = audio[cut:]

        mx.eval(audio)

        elapsed_time = time.time() - start_time
        samples = audio.shape[0]
        token_count = len(generated_codes)

        duration_seconds = samples / self.sample_rate
        rtf = duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=format_duration(duration_seconds),
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": (
                    token_count / elapsed_time if elapsed_time > 0 else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (samples / elapsed_time if elapsed_time > 0 else 0),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

        mx.clear_cache()

    def _generate_with_instruct(
        self,
        text: str,
        speaker: Optional[str],
        language: str,
        instruct: Optional[str],
        temperature: float,
        max_tokens: int,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        verbose: bool,
    ) -> Generator[GenerationResult, None, None]:
        """Internal method for generation with instruct support."""
        if self.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")

        start_time = time.time()

        if verbose:
            print(f"Generating: {text[:50]}...")
            if instruct:
                print(f"  Instruct: {instruct[:50]}...")

        # Prepare inputs with instruct
        input_embeds, trailing_text_hidden, tts_pad_embed = (
            self._prepare_generation_inputs(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
        )

        # Initialize cache
        cache = self.talker.make_cache()
        generated_codes = []
        config = self.config.talker_config
        eos_token_id = config.codec_eos_token_id
        suppress_tokens = [
            i
            for i in range(config.vocab_size - 1024, config.vocab_size)
            if i != eos_token_id
        ]
        trailing_idx = 0

        for step in range(max_tokens):
            # Forward pass through talker
            logits, hidden = self.talker(input_embeds, cache=cache)

            # Sample first codebook token
            next_token = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=(
                    [int(c[0, 0]) for c in generated_codes] if generated_codes else None
                ),
                suppress_tokens=suppress_tokens,
                eos_token_id=eos_token_id,
            )

            # Check for EOS
            if int(next_token[0, 0]) == eos_token_id:
                break

            # Generate remaining codebook tokens with code predictor
            code_tokens = [next_token]
            code_hidden = hidden[:, -1:, :]
            code_cache = self.talker.code_predictor.make_cache()

            for code_idx in range(config.num_code_groups - 1):
                if code_idx == 0:
                    code_0_embed = self.talker.get_input_embeddings()(next_token)
                    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                else:
                    code_embed = self.talker.code_predictor.codec_embedding[
                        code_idx - 1
                    ](code_tokens[-1])
                    code_input = code_embed

                code_logits, code_cache, _ = self.talker.code_predictor(
                    code_input,
                    cache=code_cache,
                    generation_step=code_idx,
                )

                next_code = self._sample_token(
                    code_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                code_tokens.append(next_code)

            # Stack all codebook tokens
            all_codes = mx.concatenate(code_tokens, axis=1)
            generated_codes.append(all_codes)

            # Prepare next input
            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed

            codec_embed = self.talker.get_input_embeddings()(next_token)
            for i, code in enumerate(code_tokens[1:]):
                codec_embed = codec_embed + self.talker.code_predictor.codec_embedding[
                    i
                ](code)

            input_embeds = text_embed + codec_embed
            mx.eval(input_embeds)

            if verbose and step % 100 == 0:
                print(f"  Step {step}, generated {len(generated_codes)} tokens")

        if not generated_codes:
            if verbose:
                print("  No codes generated")
            return

        # Stack all generated codes
        codes = mx.stack(generated_codes, axis=1)

        if verbose:
            print(f"  Decoding {codes.shape[1]} tokens to audio...")

        # Decode to audio
        audio, audio_lengths = self.speech_tokenizer.decode(codes)
        audio = audio[0]

        # Trim to valid length
        valid_len = int(audio_lengths[0])
        if valid_len > 0 and valid_len < audio.shape[0]:
            audio = audio[:valid_len]

        mx.eval(audio)

        elapsed_time = time.time() - start_time
        samples = audio.shape[0]
        token_count = len(generated_codes)

        duration_seconds = samples / self.sample_rate
        rtf = duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=format_duration(duration_seconds),
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": token_count / elapsed_time if elapsed_time > 0 else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples / elapsed_time if elapsed_time > 0 else 0,
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

        mx.clear_cache()

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "Model":
        """Load model from pretrained weights.

        Args:
            path: Local path or Hugging Face repo ID (e.g., 'Qwen/Qwen3-TTS-0.6B-Base')
        """
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        path = Path(path)

        # Download from Hugging Face if not a local path
        if not path.exists():
            print(f"Downloading model from Hugging Face: {path}...")
            path = Path(
                snapshot_download(
                    repo_id=str(path),
                    allow_patterns=[
                        "*.safetensors",
                        "*.json",
                        "*.txt",
                        "*.model",
                        "speech_tokenizer/*",
                    ],
                )
            )
            print(f"Model downloaded to: {path}")

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)
        model = cls(config)

        # Load weights - use PyTorch as intermediate for bfloat16 support
        weights = {}
        weight_files = list(path.glob("*.safetensors"))
        if len(weight_files) == 0:
            raise FileNotFoundError(f"No safetensors found in {path}")

        for wf in weight_files:
            weights.update(mx.load(str(wf)))

        # Sanitize and load
        weights = model.sanitize(weights)

        model.load_weights(list(weights.items()))

        # Call post_load_hook to initialize tokenizer and speech tokenizer
        model = cls.post_load_hook(model, path)

        return model

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Initialize tokenizer and other resources after weight loading."""
        try:
            from transformers import AutoTokenizer

            model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

        # Load speech tokenizer if available
        speech_tokenizer_path = model_path / "speech_tokenizer"
        if speech_tokenizer_path.exists():
            try:
                with open(speech_tokenizer_path / "config.json") as f:
                    tokenizer_config_dict = json.load(f)

                # Build tokenizer config (filter unknown fields)
                from .config import filter_dict_for_dataclass

                decoder_config = None
                encoder_config = None

                if "decoder_config" in tokenizer_config_dict:
                    filtered = filter_dict_for_dataclass(
                        Qwen3TTSTokenizerDecoderConfig,
                        tokenizer_config_dict["decoder_config"],
                    )
                    decoder_config = Qwen3TTSTokenizerDecoderConfig(**filtered)
                if "encoder_config" in tokenizer_config_dict:
                    filtered = filter_dict_for_dataclass(
                        Qwen3TTSTokenizerEncoderConfig,
                        tokenizer_config_dict["encoder_config"],
                    )
                    encoder_config = Qwen3TTSTokenizerEncoderConfig(**filtered)

                tokenizer_config = Qwen3TTSTokenizerConfig(
                    encoder_config=encoder_config,
                    decoder_config=decoder_config,
                )

                # Copy top-level config values
                for k, v in tokenizer_config_dict.items():
                    if k not in ("decoder_config", "encoder_config") and hasattr(
                        tokenizer_config, k
                    ):
                        setattr(tokenizer_config, k, v)

                speech_tokenizer = Qwen3TTSSpeechTokenizer(tokenizer_config)

                # Load speech tokenizer weights

                tokenizer_weights = {}
                for wf in speech_tokenizer_path.glob("*.safetensors"):
                    tokenizer_weights.update(mx.load(str(wf)))

                if tokenizer_weights:
                    tokenizer_weights = Qwen3TTSSpeechTokenizer.sanitize(
                        tokenizer_weights
                    )
                    speech_tokenizer.load_weights(
                        list(tokenizer_weights.items()), strict=False
                    )
                    mx.eval(speech_tokenizer.parameters())
                    speech_tokenizer.eval()

                    # Initialize encoder codebooks (compute _embedding and _c2)
                    if speech_tokenizer.encoder_model is not None:
                        quantizer = speech_tokenizer.encoder_model.quantizer
                        for layer in quantizer.rvq_first.vq.layers:
                            layer.codebook.update_in_place()
                        for layer in quantizer.rvq_rest.vq.layers:
                            layer.codebook.update_in_place()
                        print("  Initialized encoder codebooks")

                model.load_speech_tokenizer(speech_tokenizer)
                print(f"Loaded speech tokenizer from {speech_tokenizer_path}")
            except Exception as e:
                print(f"Warning: Could not load speech tokenizer: {e}")
                import traceback

                traceback.print_exc()

        # Load generation config
        gen_config_path = model_path / "generation_config.json"
        if gen_config_path.exists():
            with open(gen_config_path) as f:
                model.load_generate_config(json.load(f))

        return model

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from PyTorch to MLX format."""
        sanitized = {}

        for k, v in weights.items():
            new_key = k

            # Skip position_ids (not used in inference)
            if "position_ids" in k:
                continue

            # Handle Conv1d weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
            # This covers:
            # - All conv patterns: .conv.weight, conv1.weight, conv2.weight, etc.
            # - speaker_encoder.fc.weight (which is also a Conv1d)
            # - speech_tokenizer decoder convolutions
            is_conv_weight = (
                "conv" in k or "speaker_encoder.fc" in k
            ) and "weight" in k
            if is_conv_weight and len(v.shape) == 3:
                v = v if check_array_shape_qwen3(v) else mx.transpose(v, (0, 2, 1))
            sanitized[new_key] = v

        return sanitized

"""AlignAtt streaming transcription for Whisper.

This module implements low-latency streaming transcription using the AlignAtt
algorithm, which monitors cross-attention weights to determine when decoded
tokens are stable enough to emit.

Reference: https://arxiv.org/abs/2211.00895 (SimulMT with AlignAtt)
"""

from dataclasses import dataclass
from typing import List

import mlx.core as mx


@dataclass
class StreamingConfig:
    """Configuration for AlignAtt streaming transcription.

    Attributes:
        frame_threshold: Number of frames from audio end before emitting tokens.
            Lower = faster but may cut words. Default 25 (~0.5s at 50 tokens/sec).
        min_chunk_duration: Minimum audio duration (seconds) before processing.
        emit_partial: Whether to emit partial (uncommitted) results.
    """

    frame_threshold: int = 25
    min_chunk_duration: float = 0.5
    emit_partial: bool = True


@dataclass
class StreamingResult:
    """Result from streaming transcription.

    Attributes:
        text: Decoded text for this emission.
        tokens: Token IDs that were decoded.
        is_final: True if this is a final (committed) result, False if partial.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        progress: Progress from 0.0 to 1.0 (percentage of audio processed).
        audio_position: Current position in audio (seconds).
        audio_duration: Total audio duration (seconds).
    """

    text: str
    tokens: List[int]
    is_final: bool
    start_time: float
    end_time: float
    progress: float = 0.0
    audio_position: float = 0.0
    audio_duration: float = 0.0


def get_most_attended_frame(cross_qk: List[mx.array], alignment_heads: mx.array) -> int:
    """Extract the most attended audio frame from cross-attention weights.

    Uses the same alignment heads as word-level timestamps (timing.py).

    Args:
        cross_qk: List of cross-attention weights per layer.
                  Each element shape: [batch, n_heads, seq_len, audio_frames]
        alignment_heads: Array of [layer, head] pairs to use for alignment.

    Returns:
        Frame index that receives highest average attention for the last token.
    """
    weights = mx.stack(
        [cross_qk[layer][0, head, -1, :] for layer, head in alignment_heads.tolist()]
    )

    avg_attention = weights.mean(axis=0)
    most_attended = int(mx.argmax(avg_attention).item())

    return most_attended


def should_emit(
    most_attended_frame: int, content_frames: int, config: StreamingConfig
) -> bool:
    """Determine if tokens should be emitted based on AlignAtt algorithm.

    The AlignAtt insight: when cross-attention focuses near the end of
    available audio, the model has "seen enough" to emit stable tokens.

    Args:
        most_attended_frame: Frame index with highest attention for last token.
        content_frames: Total number of audio frames available.
        config: Streaming configuration with frame_threshold.

    Returns:
        True if tokens should be emitted (attention near audio end).
    """
    distance_to_end = content_frames - most_attended_frame
    return distance_to_end <= config.frame_threshold


class StreamingDecoder:
    """AlignAtt-based streaming decoder for Whisper.

    This decoder accumulates audio over time and monitors cross-attention weights
    during token generation to determine when tokens are stable enough to emit,
    enabling low-latency streaming transcription.

    The key insight is that we accumulate mel spectrograms across chunks rather
    than processing each chunk independently. This allows the model to maintain
    context as more audio arrives.
    """

    def __init__(
        self,
        model,
        config: StreamingConfig = None,
        language: str = None,
        task: str = "transcribe",
    ):
        """Initialize streaming decoder.

        Args:
            model: Loaded Whisper model instance.
            config: Streaming configuration. Uses defaults if None.
            language: Language code (e.g., 'en'). Auto-detected if None.
            task: "transcribe" or "translate" (to English).
        """
        from mlx_audio.stt.models.whisper.decoding import (
            Inference,
            SuppressBlank,
            SuppressTokens,
            get_suppress_tokens,
        )
        from mlx_audio.stt.models.whisper.tokenizer import get_tokenizer

        self.model = model
        self.config = config or StreamingConfig()
        self.inference = Inference(model)
        self.tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language or "en",
            task=task,
        )
        self._emitted_tokens = []
        self._pending_tokens = []
        self._accumulated_mel = None

        # Use sot_sequence with notimestamps for text-only output
        self._sot_sequence = self.tokenizer.sot_sequence_including_notimestamps

        # Use shared helper for suppress tokens
        suppress_tokens = set(get_suppress_tokens(self.tokenizer))
        # Don't suppress notimestamps since we need it in sot sequence
        suppress_tokens.discard(self.tokenizer.no_timestamps)

        # Initialize logit filters
        self._sample_begin = len(self._sot_sequence)
        self._logit_filters = [
            SuppressBlank(self.tokenizer, self._sample_begin, model.dims.n_vocab),
            SuppressTokens(list(suppress_tokens), model.dims.n_vocab),
        ]

    def reset(self):
        """Reset decoder state for new transcription session."""
        self.inference.reset()
        self._emitted_tokens = []
        self._pending_tokens = []
        self._accumulated_mel = None

    def decode_chunk(self, mel: mx.array, is_last: bool = False) -> StreamingResult:
        """Decode an audio chunk using AlignAtt streaming.

        Audio is accumulated across chunks to maintain context. The AlignAtt
        algorithm determines when tokens are stable enough to emit based on
        cross-attention weights.

        Args:
            mel: Mel spectrogram of audio chunk, shape [n_frames, n_mels].
            is_last: True if this is the final chunk (emit all remaining).

        Returns:
            StreamingResult with decoded text and metadata.
        """
        from mlx_audio.stt.models.whisper.audio import (
            N_FRAMES,
            TOKENS_PER_SECOND,
            pad_or_trim,
        )

        # Accumulate mel spectrogram across chunks
        if self._accumulated_mel is None:
            self._accumulated_mel = mel
        else:
            self._accumulated_mel = mx.concatenate([self._accumulated_mel, mel], axis=0)

        # Trim to max 30 seconds if accumulated too much (sliding window)
        if self._accumulated_mel.shape[0] > N_FRAMES:
            self._accumulated_mel = self._accumulated_mel[-N_FRAMES:]

        # Reset KV cache for fresh decoding of accumulated audio
        self.inference.reset()

        # Pad mel to required size for encoder
        mel_padded = pad_or_trim(self._accumulated_mel, N_FRAMES, axis=-2)

        # Add batch dimension if needed
        if mel_padded.ndim == 2:
            mel_padded = mel_padded[None, :]

        # Encode audio
        audio_features = self.model.encoder(mel_padded.astype(self.model.dtype))
        content_frames = self._accumulated_mel.shape[0] // 2  # Encoder has stride 2

        # Initialize tokens with SOT sequence (including notimestamps)
        tokens = mx.array([list(self._sot_sequence)])

        # Decode loop with AlignAtt
        # First iteration: pass all initial tokens; subsequent: only last token
        first_iteration = True
        for _ in range(self.model.dims.n_text_ctx // 2):
            inputs = tokens if first_iteration else tokens[:, -1:]
            first_iteration = False

            logits, cross_qk = self.inference.logits_with_cross_qk(
                inputs,
                audio_features,
            )

            # Apply logit filters to suppress non-speech and special tokens
            filtered_logits = logits[:, -1]
            for logit_filter in self._logit_filters:
                filtered_logits = logit_filter.apply(filtered_logits[None, :], tokens)[
                    0
                ]

            # Get next token from filtered logits
            next_token = int(mx.argmax(filtered_logits, axis=-1).item())

            if next_token == self.tokenizer.eot:
                break

            tokens = mx.concatenate([tokens, mx.array([[next_token]])], axis=-1)

            # Check AlignAtt condition (only if we have alignment heads)
            if (
                hasattr(self.model, "alignment_heads")
                and self.model.alignment_heads is not None
            ):
                most_attended = get_most_attended_frame(
                    cross_qk, self.model.alignment_heads
                )
                threshold = 4 if is_last else self.config.frame_threshold
                if should_emit(
                    most_attended,
                    content_frames,
                    StreamingConfig(frame_threshold=threshold),
                ):
                    break

        # Extract text tokens (excluding special tokens)
        text_tokens = [
            t
            for t in tokens[0].tolist()
            if t < self.tokenizer.eot and t not in self.tokenizer.sot_sequence
        ]

        # Calculate what's new since last emission
        new_tokens = text_tokens[len(self._emitted_tokens) :]
        self._emitted_tokens = text_tokens

        emit_start_time = (
            len(self._emitted_tokens) - len(new_tokens)
        ) / TOKENS_PER_SECOND
        emit_end_time = len(self._emitted_tokens) / TOKENS_PER_SECOND

        return StreamingResult(
            text=self.tokenizer.decode(new_tokens),
            tokens=new_tokens,
            is_final=is_last,
            start_time=emit_start_time,
            end_time=emit_end_time,
        )

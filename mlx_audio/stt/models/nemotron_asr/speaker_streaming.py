"""Speaker-masked Nemotron ASR using independent cache-aware RNN-T streams.

Follows NeMo's ``SpeakerTaggedASR(masked_asr=True, mask_preencode=False)``:
average diarization probabilities to the encoder stride, threshold, then mask
log-mel features before subsampling. Model weights are shared; caches are not.
This is temporal masking, not separation of simultaneously speaking voices.
"""

from collections import deque
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.nemo.alignment import AlignedToken

from .audio import StreamingLogMelSpectrogram
from .rnnt import GreedyDecoderState
from .streaming import ConformerStreamingState


@dataclass
class SpeakerTranscript:
    """New tokens from one speaker stream, with absolute emission timestamps."""

    speaker: str
    tokens: list[AlignedToken]

    @property
    def text(self):
        return "".join(token.text for token in self.tokens)


def speaker_activity(probs, factor, threshold):
    """Average fine-resolution activity before thresholding, including tail padding."""
    padded = mx.pad(probs, [(0, -probs.shape[0] % factor), (0, 0)])
    return padded.reshape(-1, factor, probs.shape[1]).mean(axis=1) > threshold


def mask_features(mel, activity, factor):
    """NeMo feature masking: suppressed frames are zero in log-mel space.

    Preserve the reference's log-floor replacement for *original* zero features.
    The replacement does not apply to zeros introduced by the speaker mask.
    """
    mask = mx.repeat(activity, factor)[: mel.shape[1]]
    return mx.where(mel == 0, -16.6355, mel * mask[None, :, None])


class SpeakerStreamingSession:
    """Synchronous mono PCM -> speaker-tagged token deltas.

    ``feed`` accepts arbitrary input partitions and runs both models. It waits
    for committed diarization predictions before encoding the corresponding ASR
    chunk. ``feed([], final=True)`` drains both frontends and the final partial
    chunk once. Sessions own all inference state and must have a single caller.

    Cache gating freezes inactive speakers. Its history includes the current
    chunk (two chunks by default), allowing delayed tokens after a turn ends.
    Only buffers/caches are retained; callers collect the returned transcripts.
    Configure the diarization preset before creating sessions and leave it fixed.
    """

    def __init__(
        self,
        model,
        diarization_model,
        *,
        language=None,
        threshold=0.5,
        att_context_size=None,
        cache_gating=True,
        cache_gating_buffer_size=2,
    ):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if (
            not isinstance(cache_gating_buffer_size, int)
            or isinstance(cache_gating_buffer_size, bool)
            or cache_gating_buffer_size < 1
        ):
            raise ValueError("cache_gating_buffer_size must be a positive integer")
        proc = model.preprocessor_config
        diar_config = diarization_model.config
        diar_proc = diar_config.processor_config
        if (proc.sample_rate, proc.hop_length) != (
            diar_proc.sampling_rate,
            diar_proc.hop_length,
        ):
            raise ValueError(
                "ASR and diarization must use the same sample rate and feature hop"
            )
        if diar_config.output_subsampling_factor != 1:
            raise ValueError(
                "Use native diarization probabilities (output_subsampling_factor=1)"
            )
        self.model = model
        self.diarization_model = diarization_model
        self.input_sample_rate = proc.sample_rate
        self.language = language or model.default_language
        self.threshold = threshold
        self.att_context_size = att_context_size or model.default_att_context_size
        self.factor = model.encoder_config.subsampling_factor
        self.chunk_mel = (self.att_context_size[1] + 1) * self.factor
        if self.chunk_mel <= 0:
            raise ValueError(
                "Speaker streaming requires finite nonnegative right context"
            )
        self.num_speakers = diar_config.num_speakers
        self.cache_gating = cache_gating
        self.cache_gating_buffer_size = cache_gating_buffer_size
        self.reset()

    def reset(self):
        """Start a fresh recording, including new arrival-order speaker IDs."""
        self._frontend = StreamingLogMelSpectrogram(self.model.preprocessor_config)
        self._diar_state = self.diarization_model.init_streaming_state()
        self._mel = mx.zeros((1, 0, self.model.preprocessor_config.features))
        self._probs = mx.zeros((0, self.num_speakers))
        self._history = deque(maxlen=self.cache_gating_buffer_size)
        self._encoders = {}
        self._decoders = {}
        self._mel_offset = 0
        self._closed = False

    @property
    def done(self):
        return self._closed

    def feed(self, samples, *, final=False):
        """Return new ``SpeakerTranscript`` deltas; empty means no new tokens."""
        if self._closed:
            raise RuntimeError("speaker streaming session is closed")
        samples = mx.array(samples, dtype=mx.float32)
        if samples.ndim != 1 or not bool(mx.all(mx.isfinite(samples))):
            raise ValueError("expected finite mono PCM samples")
        # Bound temporary state even when the caller supplies an entire file.
        step = self.chunk_mel * self.model.preprocessor_config.hop_length
        updates = []
        for start in range(0, samples.shape[0], step):
            updates.extend(self._feed(samples[start : start + step], final=False))
        if final:
            updates.extend(self._feed(mx.zeros((0,)), final=True))
            self._closed = True
        return updates

    def _feed(self, samples, *, final):
        if final and not self._frontend.total_samples:
            return []
        mel = self._frontend.push(samples, final=final)
        diar, self._diar_state = self.diarization_model.feed(
            samples,
            self._diar_state,
            sample_rate=self.input_sample_rate,
            threshold=self.threshold,
            final=final,
        )
        return self._push_features(mel, diar.speaker_probs, final=final)

    def _push_features(self, mel, probs, *, final=False):
        self._mel = mx.concatenate([self._mel, mel], axis=1)
        self._probs = mx.concatenate([self._probs, probs], axis=0)
        if final:
            # Diarization omits the extra centered STFT frame and fractional hop.
            # They have no committed activity: zero-pad the final mask.
            missing = self._mel.shape[1] - self._probs.shape[0]
            if missing < 0:
                raise ValueError("Diarization timeline extends beyond ASR features")
            self._probs = mx.pad(self._probs, [(0, missing), (0, 0)])
        updates = []
        while self._mel.shape[1]:
            count = min(self.chunk_mel, self._mel.shape[1])
            if not final and (count < self.chunk_mel or self._probs.shape[0] < count):
                break
            last = final and count == self._mel.shape[1]
            activity = speaker_activity(
                self._probs[:count], self.factor, self.threshold
            )
            self._history.append(mx.any(activity, axis=0))
            active = (
                mx.any(mx.stack(list(self._history)), axis=0)
                if self.cache_gating
                else mx.ones((self.num_speakers,), dtype=mx.bool_)
            )
            for speaker in np.flatnonzero(np.array(active)):
                speaker = int(speaker)
                if speaker not in self._encoders:
                    self._encoders[speaker] = ConformerStreamingState(
                        self.model.encoder, att_context_size=self.att_context_size
                    )
                    self._decoders[speaker] = GreedyDecoderState(self.model)
                encoder = self._encoders[speaker]
                masked = mask_features(
                    self._mel[:, :count], activity[:, speaker], self.factor
                )
                start_frame = self._mel_offset // self.factor
                for encoded in encoder.push(masked, final=last):
                    prompted = self.model.apply_prompt(encoded, self.language)
                    encoder.materialize(prompted)
                    tokens = self._decoders[speaker].decode(prompted, start_frame)
                    start_frame += prompted.shape[1]
                    if tokens:
                        updates.append(SpeakerTranscript(f"speaker_{speaker}", tokens))
            self._mel_offset += count
            self._mel = self._mel[:, count:]
            self._probs = self._probs[count:]
        mx.eval(self._mel, self._probs, *self._history)
        return updates

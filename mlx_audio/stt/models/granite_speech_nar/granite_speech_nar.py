"""Granite Speech 4.1 2B NAR — non-autoregressive ASR via CTC + editor.

Single-pass bidirectional decoder. The encoder produces an initial CTC
hypothesis (BPE); the projector turns multi-layer encoder states into audio
tokens; the bidirectional Granite editor takes [audio | hypothesis] and emits
edited token logits, which collapse to the final transcript via a second CTC
pass.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer

from mlx_audio import dsp
from mlx_audio.stt.models.base import STTOutput

from .config import ModelConfig
from .decoding import add_insertion_slots, ctc_collapse_decode
from .editor import GraniteEditor
from .encoder import ConformerEncoder
from .projector import GraniteSpeechNarProjector

# Feature-extractor constants — same as upstream granite_speech_nar processor.
# Audio assumed mono 16 kHz; STFT n_fft=512 win_length=400 hop=160; 80 mel bins
# with HTK scale; per-sample dynamic-range normalization clamped 8 dB below max
# and rescaled to ~[0, 2] by /4 + 1; pairs of mel frames stacked to 160 dims.
SAMPLING_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
LOG_FLOOR_DB = 8.0


# periodic=True matches the upstream processor's 2π·n/N denominator (spectral
# analysis convention). Window is zero-padded to N_FFT and centered around
# WIN_LENGTH to match np.fft.rfft's implicit centering of shorter windows.
# precise=True on the mel filterbank avoids ~5e-6 weight drift from the default
# float32 path that would otherwise perturb the CTC decode on the multilingual
# eval clip.
_WIN_PAD_L = (N_FFT - WIN_LENGTH) // 2
_WINDOW = mx.concatenate(
    [
        mx.zeros((_WIN_PAD_L,)),
        dsp.hanning(WIN_LENGTH, periodic=True),
        mx.zeros((N_FFT - WIN_LENGTH - _WIN_PAD_L,)),
    ]
)
_MEL_T = dsp.mel_filters(SAMPLING_RATE, N_FFT, N_MELS, precise=True).T


def _compute_features(waveform: mx.array) -> mx.array:
    """1-D 16 kHz mono waveform → [T_enc, 160] stacked log-mel features."""
    n_samples = waveform.shape[0]
    pad = N_FFT // 2
    # reflect padding (matches np.pad mode="reflect"): excludes the boundary sample.
    x = mx.concatenate(
        [
            waveform[1 : pad + 1][::-1],
            waveform,
            waveform[-(pad + 1) : -1][::-1],
        ]
    )
    n_frames = 1 + (x.shape[0] - N_FFT) // HOP_LENGTH
    frames = mx.as_strided(x, shape=(n_frames, N_FFT), strides=(HOP_LENGTH, 1))
    spec = mx.fft.rfft(frames * _WINDOW, axis=-1)
    power = mx.abs(spec) ** 2
    mel = power @ _MEL_T
    l = 2 * (n_samples // (2 * HOP_LENGTH))
    mel = mel[:l]
    logmel = mx.log10(mx.maximum(mel, 1e-10))
    logmel = mx.maximum(logmel, mx.max(logmel) - LOG_FLOOR_DB) / 4.0 + 1.0
    return logmel.reshape(l // 2, 2 * N_MELS)


def _load_waveform(audio) -> mx.array:
    """Accept a file path, raw numpy/mlx array, or list — return 1-D float32
    mx.array at 16 kHz mono."""
    if isinstance(audio, mx.array):
        return audio.astype(mx.float32)
    if isinstance(audio, (str, Path)):
        import soundfile as sf

        wav, sr = sf.read(str(audio), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLING_RATE:
            raise ValueError(f"audio must be {SAMPLING_RATE} Hz; got {sr}")
        return mx.array(wav)
    return mx.array(np.asarray(audio, dtype=np.float32))


class Model(nn.Module):
    """Granite Speech NAR end-to-end ASR model. Batch=1."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ConformerEncoder(config.encoder, config.encoder_layer_indices)
        self.projector = GraniteSpeechNarProjector(config.projector)
        self.editor = GraniteEditor(config.text)
        self._tokenizer = None  # populated by post_load_hook

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Drop BatchNorm `num_batches_tracked` (training-only counters).

        IMPORTANT: Conv1d kernels are NOT transposed here because the bundler
        (scripts/bundle.py in the companion repo) feeds this loader from the
        already-transposed standalone converter output. Bundle is MLX-layout
        end-to-end; sanitize just strips training-only buffers.
        """
        return {
            k: v for k, v in weights.items() if not k.endswith("num_batches_tracked")
        }

    @classmethod
    def post_load_hook(cls, model: "Model", model_path) -> "Model":
        """Load tokenizer.json and attach to the model."""
        model._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        return model

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        """Quantize editor weights only; encoder and projector stay at full precision.

        Mirrors the convention from the existing granite_speech (4.0) port: the
        editor dominates memory (1.6B of 2.25B params) and tolerates quantization
        well, while the conformer encoder and Q-Former projector are smaller and
        more sensitive to numerical noise. Conv1d / RMSNorm / LayerNorm /
        BatchNorm modules are skipped automatically by mlx-audio's quantize()
        helper (no `to_quantized` method) regardless of this predicate.
        """
        return p.startswith("editor.")

    # ---- inference ----

    def _transcribe_tokens(self, input_features: mx.array) -> mx.array:
        """Core transcribe path. Returns 1-D int32 mx.array of final token IDs."""
        cfg = self.config
        blank = cfg.blank_token_id
        min_len = cfg.min_edit_sequence_length

        if input_features.ndim == 2:
            input_features = input_features[None]
        input_features = input_features.astype(mx.bfloat16)

        enc_out = self.encoder(input_features)
        bpe_argmax = mx.argmax(enc_out.bpe_logits[0], axis=-1).astype(mx.int32)
        hypothesis_tokens = ctc_collapse_decode(bpe_argmax, blank_id=blank)

        fused = mx.concatenate(enc_out.hidden_states_for_projector, axis=-1)
        audio_embeds = self.projector(fused)
        audio_embeds = audio_embeds / cfg.text.embedding_multiplier

        text_ids = add_insertion_slots(
            hypothesis_tokens, blank_id=blank, min_len=min_len
        )
        text_embeds = self.editor.embed_tokens(text_ids).astype(audio_embeds.dtype)

        audio_len = audio_embeds.shape[1]
        text_len = text_embeds.shape[0]
        flat_embeds = mx.concatenate([audio_embeds[0], text_embeds], axis=0)[None]
        position_ids = mx.arange(audio_len + text_len, dtype=mx.int32)

        logits = self.editor(inputs_embeds=flat_embeds, position_ids=position_ids)
        text_logits = logits[0, audio_len:, :]
        edited_argmax = mx.argmax(text_logits, axis=-1).astype(mx.int32)
        return ctc_collapse_decode(edited_argmax, blank_id=blank)

    def generate(
        self,
        audio,
        *,
        verbose: bool = False,
        generation_stream=None,
        **kwargs,
    ) -> STTOutput:
        """Transcribe a single audio clip. Returns STTOutput with .text.

        Args:
            audio: a file path (str/Path) loadable by soundfile, an mlx.array of
                samples, or a numpy ndarray. Must be 16 kHz mono.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded — call via base_load_model, which invokes "
                "post_load_hook to attach the tokenizer."
            )
        waveform = _load_waveform(audio)
        features = _compute_features(waveform)
        tokens = self._transcribe_tokens(features)
        text = self._tokenizer.decode(
            [int(t) for t in tokens.tolist()],
            skip_special_tokens=True,
        )
        return STTOutput(text=text)

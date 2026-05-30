import io
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.codec.models.s3 import S3TokenizerV2, log_mel_spectrogram
from mlx_audio.tts.models.chatterbox.s3gen.mel import mel_spectrogram
from mlx_audio.utils import get_model_path, load_audio

from .convert import load_campplus_weights, load_flow_weights, load_hift_weights
from .flow import CausalMaskedDiffWithXvec
from .hift import StepAudio2HiFTGenerator
from .speaker import StepAudio2CAMPPlus

STEPAUDIO2_SAMPLE_RATE = 24_000
S3_SAMPLE_RATE = 16_000


class StepAudio2Token2Wav(nn.Module):
    def __init__(
        self,
        flow: Optional[CausalMaskedDiffWithXvec] = None,
        hift: Optional[StepAudio2HiFTGenerator] = None,
        speech_tokenizer: Optional[S3TokenizerV2] = None,
        speaker_encoder: Optional[StepAudio2CAMPPlus] = None,
    ):
        super().__init__()
        self.flow = flow or CausalMaskedDiffWithXvec()
        self.hift = hift or StepAudio2HiFTGenerator()
        self.speech_tokenizer = speech_tokenizer
        self.speaker_encoder = speaker_encoder
        self._prompt_cache: Optional[Dict[str, mx.array]] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = "mlx-community/Step-Audio-2-token2wav",
        *,
        load_speech_tokenizer: bool = True,
        load_speaker_encoder: bool = True,
        revision: Optional[str] = None,
        force_download: bool = False,
    ) -> "StepAudio2Token2Wav":
        model_path = get_model_path(
            str(model_path),
            revision=revision,
            force_download=force_download,
            allow_patterns=["*.safetensors", "*.yaml"],
        )
        model = cls()
        flow_path = _first_existing(model_path, "flow.safetensors", "flow.pt")
        hift_path = _first_existing(model_path, "hift.safetensors", "hift.pt")
        load_flow_weights(model.flow, flow_path, strict=True)
        load_hift_weights(model.hift, hift_path, strict=True)

        if load_speech_tokenizer:
            model.speech_tokenizer = S3TokenizerV2.from_pretrained(
                "speech_tokenizer_v2_25hz"
            )
        if load_speaker_encoder:
            speaker_path = _first_existing_optional(
                model_path, "campplus.safetensors", "campplus.onnx"
            )
            if speaker_path is not None:
                model.speaker_encoder = StepAudio2CAMPPlus()
                load_campplus_weights(model.speaker_encoder, speaker_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        return model

    def _ensure_speech_tokenizer(self) -> S3TokenizerV2:
        if self.speech_tokenizer is None:
            self.speech_tokenizer = S3TokenizerV2.from_pretrained(
                "speech_tokenizer_v2_25hz"
            )
            mx.eval(self.speech_tokenizer.parameters())
        return self.speech_tokenizer

    def prepare_prompt(
        self,
        prompt_wav: str | mx.array,
        *,
        prompt_tokens: Optional[mx.array] = None,
        speaker_embedding: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        audio_16k = load_audio(prompt_wav, sample_rate=S3_SAMPLE_RATE)
        if prompt_tokens is None:
            tokenizer = self._ensure_speech_tokenizer()
            mels = log_mel_spectrogram(audio_16k)
            mels = mx.expand_dims(mels, 0)
            mel_lens = mx.array([mels.shape[-1]], dtype=mx.int32)
            prompt_tokens, prompt_token_lens = tokenizer.quantize(mels, mel_lens)
        else:
            if prompt_tokens.ndim == 1:
                prompt_tokens = mx.expand_dims(prompt_tokens, 0)
            prompt_token_lens = mx.array([prompt_tokens.shape[1]], dtype=mx.int32)

        if speaker_embedding is None:
            if self.speaker_encoder is None:
                raise ValueError(
                    "speaker_embedding is required unless a CAMPPlus speaker_encoder "
                    "is loaded"
                )
            speaker_embedding = self.speaker_encoder.inference(audio_16k)
        elif speaker_embedding.ndim == 1:
            speaker_embedding = mx.expand_dims(speaker_embedding, 0)

        audio_24k = load_audio(prompt_wav, sample_rate=STEPAUDIO2_SAMPLE_RATE)
        prompt_mels = mel_spectrogram(mx.expand_dims(audio_24k, 0))
        prompt_mels = mx.transpose(prompt_mels, (0, 2, 1))
        prompt_mels_lens = mx.array([prompt_mels.shape[1]], dtype=mx.int32)

        target_mel_len = int(prompt_tokens.shape[1] * self.flow.up_rate)
        if prompt_mels.shape[1] < target_mel_len:
            pad_len = target_mel_len - prompt_mels.shape[1]
            tail = mx.broadcast_to(
                prompt_mels[:, -1:, :],
                (prompt_mels.shape[0], pad_len, prompt_mels.shape[2]),
            )
            prompt_mels = mx.concatenate([prompt_mels, tail], axis=1)
        elif prompt_mels.shape[1] > target_mel_len:
            prompt_mels = prompt_mels[:, :target_mel_len, :]

        return {
            "prompt_token": prompt_tokens.astype(mx.int32),
            "prompt_token_len": prompt_token_lens.astype(mx.int32),
            "prompt_feat": prompt_mels,
            "prompt_feat_len": prompt_mels_lens,
            "embedding": speaker_embedding,
        }

    def decode(
        self,
        speech_tokens: mx.array,
        prompt: Dict[str, mx.array],
        *,
        n_timesteps: int = 10,
    ) -> mx.array:
        if speech_tokens.ndim == 1:
            speech_tokens = mx.expand_dims(speech_tokens, 0)
        speech_tokens = speech_tokens.astype(mx.int32)
        speech_token_lens = mx.array([speech_tokens.shape[1]], dtype=mx.int32)
        mel = self.flow.inference(
            speech_tokens,
            speech_token_lens,
            n_timesteps=n_timesteps,
            **prompt,
        )
        wav, _ = self.hift.inference(speech_feat=mel)
        return wav

    def __call__(
        self,
        generated_speech_tokens: mx.array | list[int],
        prompt_wav: str | mx.array,
        *,
        prompt_tokens: Optional[mx.array] = None,
        speaker_embedding: Optional[mx.array] = None,
        n_timesteps: int = 10,
        use_cache: bool = True,
    ) -> mx.array:
        if not isinstance(generated_speech_tokens, mx.array):
            generated_speech_tokens = mx.array(generated_speech_tokens, dtype=mx.int32)
        if not use_cache or self._prompt_cache is None:
            self._prompt_cache = self.prepare_prompt(
                prompt_wav,
                prompt_tokens=prompt_tokens,
                speaker_embedding=speaker_embedding,
            )
        return self.decode(
            generated_speech_tokens,
            self._prompt_cache,
            n_timesteps=n_timesteps,
        )

    def to_wav_bytes(self, wav: mx.array) -> bytes:
        wav_np = np.asarray(wav, dtype=np.float32)
        if wav_np.ndim == 2:
            wav_np = wav_np[0]
        output = io.BytesIO()
        audio_write(output, wav_np, samplerate=STEPAUDIO2_SAMPLE_RATE, format="wav")
        return output.getvalue()


def _first_existing(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {names} exist under {root}")


def _first_existing_optional(root: Path, *names: str) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None

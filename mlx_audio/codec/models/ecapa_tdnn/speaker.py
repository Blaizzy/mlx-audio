"""SpeechBrain VoxCeleb speaker embeddings using the shared ECAPA backbone."""

import math
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from mlx_audio.dsp import hamming, stft

from .config import EcapaTdnnConfig
from .ecapa_tdnn import EcapaTdnnBackbone

DEFAULT_MODEL = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def _speaker_filterbank() -> mx.array:
    """SpeechBrain uses the lower-side bandwidth on both sides of each triangle."""
    mel_max = 2595.0 * math.log10(1.0 + 8000.0 / 700.0)
    mel_points = mx.linspace(0.0, mel_max, 82)
    hz = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    centers = hz[1:-1]
    bandwidths = hz[1:-1] - hz[:-2]
    frequencies = mx.linspace(0.0, 8000.0, 201)
    slopes = (frequencies[:, None] - centers[None, :]) / bandwidths[None, :]
    return mx.maximum(0.0, mx.minimum(slopes + 1.0, 1.0 - slopes))


def speaker_log_mel(audio: mx.array) -> mx.array:
    """Return SpeechBrain 80-bin log-mel features with zero center-padding."""
    spectrum = stft(
        audio,
        n_fft=400,
        hop_length=160,
        win_length=400,
        window=hamming(400, periodic=True),
        pad_mode="constant",
    )
    power = spectrum.real**2 + spectrum.imag**2
    mel = power @ _speaker_filterbank()
    log_mel = 10.0 * mx.log10(mx.maximum(mel, 1e-10))
    return mx.maximum(log_mel, mx.max(log_mel) - 80.0)


class EcapaSpeakerEmbeddingExtractor:
    """Extract unit-length 192D speaker embeddings from mono 16 kHz audio.

    This uses the speaker-verification checkpoint, independently of the
    VoxLingua107 language-identification model. Instantiate with
    ``from_pretrained`` to load converted SpeechBrain VoxCeleb weights.
    """

    sample_rate = SAMPLE_RATE
    embedding_dimension = 192

    def __init__(self):
        self.model = EcapaTdnnBackbone(
            EcapaTdnnConfig(
                input_size=80,
                channels=1024,
                embed_dim=192,
                global_context=True,
                padding_mode="reflect",
                centered_statistics=True,
            )
        )
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
    ) -> "EcapaSpeakerEmbeddingExtractor":
        """Load an MLX safetensors file, local directory, or Hugging Face repo.

        Only ``model.safetensors`` is downloaded; repository Python code is
        never imported. ``revision`` can pin a Hugging Face commit.
        """
        path = Path(model_path).expanduser()
        if path.is_dir():
            path = path / "model.safetensors"
        elif not path.is_file():
            if path.is_absolute():
                raise FileNotFoundError(path)
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(str(model_path), "model.safetensors", revision=revision)
            )
        weights = mx.load(str(path), format="safetensors")
        # The published export nests block0..3 in `embedding_model.blocks`.
        # Keep the shared backbone's parameter layout and its block aliases.
        remapped = {}
        for key, value in weights.items():
            key = key.removeprefix("embedding_model.")
            for index in range(4):
                prefix = f"blocks.block{index}."
                if key.startswith(prefix):
                    key = f"block{index}." + key[len(prefix) :]
                    break
            remapped[key] = value
            for index in range(1, 4):
                prefix = f"block{index}."
                if key.startswith(prefix):
                    remapped[f"blocks.{index - 1}." + key[len(prefix) :]] = value
                    break
        extractor = cls()
        extractor.model.load_weights(list(remapped.items()), strict=True)
        mx.eval(extractor.model.parameters())
        return extractor

    def encode(self, audio, sample_rate: int = SAMPLE_RATE) -> mx.array:
        """Encode a contiguous speech excerpt; compare outputs with cosine similarity."""
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate}")
        audio = mx.array(audio, dtype=mx.float32)
        if audio.ndim != 1 or audio.size < 1600:
            raise ValueError("Expected at least 0.1 seconds of mono audio")
        if not mx.all(mx.isfinite(audio)).item():
            raise ValueError("Audio contains non-finite samples")
        mel = speaker_log_mel(audio)
        mel = mel - mx.mean(mel, axis=0, keepdims=True)
        embedding = self.model(mel[None])[0]
        return embedding / mx.maximum(mx.linalg.norm(embedding), 1e-12)

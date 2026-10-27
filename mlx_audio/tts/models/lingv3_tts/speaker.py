"""Native MLX CAMPPlus inference shared with the StepAudio2 speaker encoder."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.codec.models.stepaudio2.speaker import StepAudio2CAMPPlus
from mlx_audio.dsp import compute_fbank_kaldi
from mlx_audio.utils import load_audio, resample_audio

# Ming Tiny, Ming 0.5B and StepAudio2 ship byte-identical source encoders
# (SHA256 a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73).
# These native weights contain the same parameters with folded batch norms and
# MLX convolution layouts. No ONNX packages are used for loading or conversion.
SPEAKER_REPO = "mlx-community/Step-Audio-2-token2wav"
SPEAKER_REVISION = "e21026b660069e6d1b9ac5c9a51e08ed0201f30e"
SPEAKER_FILE = "campplus.safetensors"
SPEAKER_SUBPATH = Path("speaker_encoder") / SPEAKER_FILE


def resolve_speaker_weights(path):
    path = Path(path)
    if path.is_file():
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(SPEAKER_REPO, SPEAKER_FILE, revision=SPEAKER_REVISION))


class SpeakerEncoder:
    def __init__(self, path):
        path = resolve_speaker_weights(path)
        self.model = StepAudio2CAMPPlus()
        self.model.load_weights(list(mx.load(str(path)).items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())

    def __call__(self, audio, sample_rate=44100):
        if isinstance(audio, (str, Path)):
            waveform = load_audio(str(audio), sample_rate=16000)
        else:
            waveform = np.asarray(audio, dtype=np.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=-1)
            if sample_rate != 16000:
                waveform = resample_audio(waveform, sample_rate, 16000)
        waveform = mx.array(waveform, dtype=mx.float32)
        if (
            waveform.ndim != 1
            or waveform.size < 720
            or not bool(mx.all(mx.isfinite(waveform)))
        ):
            raise ValueError(
                "Speaker references must contain finite mono audio of at least 45 ms"
            )
        features = compute_fbank_kaldi(
            waveform,
            sample_rate=16000,
            win_len=400,
            win_inc=160,
            num_mels=80,
            win_type="povey",
            dither=0.0,
        )
        features = features - features.mean(axis=0, keepdims=True)
        embedding = self.model(features[None]).reshape(-1, 192)
        if not bool(mx.all(mx.isfinite(embedding))):
            raise FloatingPointError("CAMPPlus produced a non-finite speaker embedding")
        return embedding

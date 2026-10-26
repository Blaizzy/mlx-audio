# ECAPA-TDNN speaker embeddings

`EcapaSpeakerEmbeddingExtractor` produces a unit-length 192-dimensional voice
embedding from mono 16 kHz audio. It uses the
[SpeechBrain VoxCeleb speaker-verification model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb),
through its [Apache-2.0 licensed MLX weights](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX).
This is separate from the 256-dimensional VoxLingua107 language-identification
checkpoint used by `mlx_audio.lid`.

```python
import mlx.core as mx
from mlx_audio.codec import EcapaSpeakerEmbeddingExtractor
from mlx_audio.utils import load_audio

extractor = EcapaSpeakerEmbeddingExtractor.from_pretrained(
    "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX",
    revision="e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77",
)
a = extractor.encode(load_audio("speaker_a.wav", sample_rate=16000))
b = extractor.encode(load_audio("speaker_b.wav", sample_rate=16000))
print("Cosine similarity:", (a @ b).item())
```

`from_pretrained` also accepts a local `model.safetensors` file or directory.
Only model data is downloaded; repository Python code is never imported. The
original SpeechBrain `.ckpt` checkpoint is not an MLX checkpoint and cannot be
loaded directly. The extractor validates sample rate, mono shape, finite values,
and a minimum duration of 0.1 seconds. Use several seconds of clean speech from
one speaker for useful matching. Resampling, speech detection, segmentation, and
matching thresholds belong to the application.

The frontend reproduces SpeechBrain's periodic Hamming window, zero-centered
STFT padding, power mel features, symmetric mel bandwidths, 80 dB clipping, and
sentence mean normalization. The shared ECAPA backbone opts into reflection
padding and centered attentive statistics for this checkpoint. Existing LID
padding and statistics defaults remain unchanged.

Run the focused tests with:

```sh
python -m unittest mlx_audio.codec.tests.test_ecapa_speaker
```

An optional pretrained parity test downloads the pinned MLX weights, original
SpeechBrain weights, and a public example recording. It additionally requires
PyTorch and SpeechBrain (validated with SpeechBrain 1.0.3):

```sh
MLX_AUDIO_TEST_ECAPA_SPEAKER=1 python -m unittest mlx_audio.codec.tests.test_ecapa_speaker.TestPretrainedSpeechBrainParity
```

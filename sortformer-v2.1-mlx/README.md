---
library_name: mlx-audio
tags:
- mlx
- speaker-diarization
- speech
- voice-activity-detection
- streaming
- vad
- mlx-audio
base_model: nvidia/diar_streaming_sortformer_4spk-v2.1
---

# mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32

This model was converted to MLX format from [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) using mlx-audio version **0.3.2**.

Refer to the [original model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) for more details on the model.

## Use with mlx-audio

```bash
pip install -U mlx-audio
```

### Converting from NeMo

The original model is distributed as a `.nemo` archive. This repo contains the pre-converted MLX weights.

```bash
python -m mlx_audio.vad.models.sortformer.convert \
    --nemo-path nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --output-dir ./sortformer-v2.1-mlx
```

### Python Example — Streaming Inference (Recommended):

```python
from mlx_audio.vad import load

model = load("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32")

for result in model.generate_stream("meeting.wav", chunk_duration=5.0, verbose=True):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### Python Example — Offline Inference:

```python
from mlx_audio.vad import load

model = load("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32")
result = model.generate("meeting.wav", threshold=0.5, verbose=True)
print(result.text)
```

### Python Example — Real-time Microphone Streaming:

```python
from mlx_audio.vad import load

model = load("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32")
state = model.init_streaming_state()

for chunk in mic_stream():  # your audio source
    result, state = model.feed(chunk, state, sample_rate=16000)
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

## Model Details

- **Architecture**: FastConformer (17 layers) + Transformer Encoder (18 layers) + Sortformer Modules
- **Mel bins**: 128
- **Max speakers**: 4
- **Streaming**: AOSC (Arrival-Order Speaker Cache) compression for intelligent long-range context
- **Input**: 16kHz mono audio
- **Output**: Per-frame speaker activity probabilities

### Key Streaming Features

- **Speaker Cache + FIFO** buffers for long-range and recent context
- **AOSC compression** scores frames by per-speaker log-likelihood ratio, boosting underrepresented speakers
- **Silence profiling** fills cache gaps with running-mean silence embeddings
- **Left/right context** for chunk boundary handling in file mode

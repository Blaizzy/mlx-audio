# Fish Speech

Fish Audio's dual-autoregressive text-to-speech model with reference voice cloning, multi-speaker tags, and long-form batching.

## Voice Cloning

Clone a voice from reference audio by providing the waveform and its transcript:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/fish-audio-s2-pro")
results = list(
    model.generate(
        text="Hello from Fish Speech.",
        ref_audio="sample_audio.wav",
        ref_text="This is what my voice sounds like.",
    )
)

audio = results[0].audio  # mx.array
```

## Fine-Grained Inline Control

S2 Pro enables localized control over speech generation by embedding natural-language instructions directly within the text using `[tag]` syntax. S2 Pro accepts free-form textual descriptions for open-ended expression control at the word level. Examples include:

`[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]` `[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]` `[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]` `[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]` `[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]` `[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]` `[shocked]`

## Multi-Speaker Tags

Fish Speech supports inline speaker tags such as `<|speaker:0|>` and `<|speaker:1|>`:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/fish-audio-s2-pro")
results = list(
    model.generate(
        text=(
            "<|speaker:0|>Welcome everyone. "
            "<|speaker:1|>Thanks, it's good to be here."
        )
    )
)
```

## Long-Form Generation

Long text is batched internally with `chunk_length` while preserving the running conversation context:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/fish-audio-s2-pro")
results = list(
    model.generate(
        text="This is a longer passage that should be split into manageable batches.",
        chunk_length=200,
    )
)
```

Each yielded result is one generated segment with audio, timing, and token statistics.

## Sampling Controls

Fish Speech exposes the main sampling controls used during semantic and residual decoding:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/fish-audio-s2-pro")
results = list(
    model.generate(
        text="A calm and steady delivery.",
        temperature=0.7,
        top_p=0.7,
        top_k=30,
        max_tokens=1024,
        speed=1.0,
    )
)
```

## Available Models

| Model | Description |
|-------|--------|-------------|
| `mlx-community/fish-audio-s2-pro` | Fish Audio S2 Pro |

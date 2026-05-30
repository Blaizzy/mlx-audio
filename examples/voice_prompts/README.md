# Higgs Audio v2 — Sample Voice Prompts

Drop-in reference voices for `HiggsAudioServer.generate(..., reference_audio_path=...)`. Each `.wav` is paired with a `.txt` containing the transcript of that clip (required for stable alignment between the cloned voice and the target text).

| File | Character |
| --- | --- |
| `en_woman.wav` | English, feminine register |
| `en_man.wav` | English, masculine register |
| `en_man_deep.wav` | English, masculine register, lower pitch |

All three were generated via Higgs Audio v2 smart-voice mode (no human recordings), so they're license-clean and can be freely redistributed.

## Usage

```python
from mlx_audio.tts.models.higgs_audio import HiggsAudioServer
from pathlib import Path

voice_dir = Path("examples/voice_prompts")
ref_wav = voice_dir / "en_woman.wav"
ref_txt = (voice_dir / "en_woman.txt").read_text().strip()

server = HiggsAudioServer.from_pretrained(
    model_path="mlx-community/higgs-audio-v2-3B-mlx-q8",
    codec_path="mlx-community/higgs-audio-v2-tokenizer",
)

result = server.generate(
    target_text="Anything you want cloned in the chosen voice.",
    reference_audio_path=str(ref_wav),
    reference_text=ref_txt,
    temperature=0.7,
    top_p=0.95,
    max_new_frames=1200,
    fade_in_ms=30.0,
)
```

For the recommended parameter set, see [`docs/models/tts/higgs_audio.md`](../../docs/models/tts/higgs_audio.md).

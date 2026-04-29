# Svara TTS

Multilingual autoregressive text-to-speech for **19 Indian languages**, in the Orpheus / SNAC family. Based on [`kenpath/svara-tts-v1`](https://huggingface.co/kenpath/svara-tts-v1) — a Llama-3.2-3B fine-tune over Canopy Labs' [`canopylabs/3b-hi-ft-research_release`](https://huggingface.co/canopylabs/3b-hi-ft-research_release) Orpheus base, paired with the [SNAC 24 kHz](https://huggingface.co/hubertsiuzdak/snac_24khz) neural codec.

## Model Variants

| Model | Format | Size | HuggingFace |
|-------|--------|------|-------------|
| `mlx-community/svara-tts-v1-4bit` | MLX 4-bit | ~1.9 GB | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/svara-tts-v1-4bit) |
| `mlx-community/svara-tts-v1-8bit` | MLX 8-bit | ~3.5 GB | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/svara-tts-v1-8bit) |

## Usage

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/svara-tts-v1-4bit \
        --text "नमस्ते, आप कैसे हैं?" \
        --voice "Hindi (Female)" \
        --temperature 0.75 \
        --top_p 0.9
    ```

=== "Python"

    ```python
    import numpy as np
    import soundfile as sf
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/svara-tts-v1-4bit")

    chunks = []
    for result in model.generate(
        text="नमस्ते, आप कैसे हैं? मैं ठीक हूँ।",
        voice="Hindi (Female)",
        temperature=0.75,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        max_tokens=1200,
    ):
        chunks.append(result.audio)

    audio = mx.concatenate(chunks, axis=0)
    sf.write("hello_hi.wav", np.asarray(audio), model.sample_rate)
    ```

## Voices

Voice names follow the form `"<Language Name> (<Gender>)"`:

| Language | Voices |
|----------|--------|
| Hindi | `Hindi (Male)`, `Hindi (Female)` |
| Bengali | `Bengali (Male)`, `Bengali (Female)` |
| Marathi | `Marathi (Male)`, `Marathi (Female)` |
| Telugu | `Telugu (Male)`, `Telugu (Female)` |
| Kannada | `Kannada (Male)`, `Kannada (Female)` |
| Tamil | `Tamil (Male)`, `Tamil (Female)` |
| Malayalam | `Malayalam (Male)`, `Malayalam (Female)` |
| Gujarati | `Gujarati (Male)`, `Gujarati (Female)` |
| Punjabi | `Punjabi (Male)`, `Punjabi (Female)` |
| Assamese | `Assamese (Male)`, `Assamese (Female)` |
| Bhojpuri | `Bhojpuri (Male)`, `Bhojpuri (Female)` |
| Magahi | `Magahi (Male)`, `Magahi (Female)` |
| Maithili | `Maithili (Male)`, `Maithili (Female)` |
| Chhattisgarhi | `Chhattisgarhi (Male)`, `Chhattisgarhi (Female)` |
| Bodo | `Bodo (Male)`, `Bodo (Female)` |
| Dogri | `Dogri (Male)`, `Dogri (Female)` |
| Nepali | `Nepali (Male)`, `Nepali (Female)` |
| Sanskrit | `Sanskrit (Male)`, `Sanskrit (Female)` |
| English (Indian) | `English (Indian) (Male)`, `English (Indian) (Female)` |

**38 voices across 19 languages.**

## Sampling Recommendations

The upstream `svara-tts-inference` repo uses these defaults; they're a good starting point:

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.75 |
| `top_p` | 0.9 |
| `top_k` | 40 |
| `repetition_penalty` | 1.1 |
| `max_tokens` | 1200–2048 |

## Architecture

- **Backbone:** Llama-3.2-3B fine-tuned from Canopy Labs' Orpheus Hindi base.
- **Codec:** SNAC 24 kHz, 3-level hierarchical RVQ, 7 codes per ~10 ms frame.
- **Output:** 24 kHz mono PCM.

Internally, mlx-audio dispatches Svara to the generic [Llama TTS loader](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/llama) (any model whose `config.json` declares `model_type: llama` and uses the SNAC token layout works out of the box). The SNAC codec is auto-loaded from [`mlx-community/snac_24khz`](https://huggingface.co/mlx-community/snac_24khz).

!!! warning "Voice cloning"
    The shared Orpheus Llama loader exposes a `ref_audio` / `ref_text` voice-cloning path. Per the in-repo warning, it is known to be unreliable on Orpheus-family fine-tunes (including Svara) and is best avoided until upstream addresses the issue.

## License

Apache 2.0 — see the [parent model card](https://huggingface.co/kenpath/svara-tts-v1) for full details, training data, and evaluation.

## Links

- [:octicons-link-external-16: Parent model](https://huggingface.co/kenpath/svara-tts-v1) (`kenpath/svara-tts-v1`)
- [:octicons-link-external-16: Orpheus Hindi base](https://huggingface.co/canopylabs/3b-hi-ft-research_release) (`canopylabs/3b-hi-ft-research_release`)
- [:octicons-link-external-16: Reference inference repo](https://github.com/Kenpath/svara-tts-inference) (`Kenpath/svara-tts-inference`)
- [:octicons-mark-github-16: Llama TTS source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/llama)

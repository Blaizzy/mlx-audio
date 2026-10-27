# Maya-2 Native

Maya-2 Native is an open-weight (Apache-2.0) streaming text-to-speech model from [Maya Research](https://mayaresearch.ai)
for 11 Indic languages and English, with 20 trained voices and zero-shot cross-lingual voice cloning.

- **Backbone:** Qwen3-1.7B predicts codebook 0 of each 80 ms Mimi frame from the text prompt.
- **Depth decoder:** a 31-level decoder fills codebooks 1-31.
- **Codec:** Kyutai's Mimi turns the frames into 24 kHz audio.
- **Speaker conditioning:** an ECAPA speaker embedding conditions both the backbone and the depth decoder, so a short
  reference clip can be cloned into any of the 12 languages.

The MLX port reproduces the reference serving engine for one stream:
- **Stopping:** controlled stop (argmax end-of-speech, probability threshold, cumulative hazard,
  text-conditioned minimum length).
- **Sampling:** codebook-0 sampling with a presence penalty.
- **Guards:** a loop guard and a duration cap.
- **Text handling:** the bundled text normaliser for numbers, currency, dates and URLs in every language.

Cloning uses mlx-audio's native Mimi encoder and ECAPA-TDNN speaker encoder; no PyTorch is needed.

## Model Variants

| Model | Format | HuggingFace |
|-------|--------|-------------|
| `aruntemme/maya2-native-mlx-bf16` | BF16 (reference numerics) | [:octicons-link-external-16: Model Card](https://huggingface.co/aruntemme/maya2-native-mlx-bf16) |
| `aruntemme/maya2-native-mlx-8bit` | 8-bit backbone, BF16/FP16 elsewhere (recommended) | [:octicons-link-external-16: Model Card](https://huggingface.co/aruntemme/maya2-native-mlx-8bit) |

Both are converted from [maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native). On an M4 Max:

| Variant | Size | Time per second of audio | First audio when streaming | Memory |
|---|---|---|---|---|
| BF16 | 4.7 GB | 0.42 s | 62 ms | ~6 GB |
| 8-bit | 3.4 GB | 0.37 s | 53 ms | ~5 GB |

The Mimi codec loads from `kyutai/moshiko-pytorch-bf16` and the speaker encoder from
`aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX` on first use.

## Usage

### Basic Generation

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model aruntemme/maya2-native-mlx-8bit \
        --text "வணக்கம்! நாளைக்கு 2 மணிக்கு meeting இருக்கு." \
        --voice Kayal
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("aruntemme/maya2-native-mlx-8bit")

    for result in model.generate(
        text="नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।",
        voice="Ritvik",
    ):
        audio = result.audio  # mx.array, 24 kHz
    ```

The language is inferred from the text's script. `lang_code` (a code like `mr` or a name like `marathi`) picks
between languages that share a script, such as Hindi and Marathi (Devanagari) or Bengali and Assamese, and applies to
Latin-script text. Code-mixed text such as Tamil with English words works as written.

### Voice Cloning

```python
for result in model.generate(
    text="வணக்கம், எப்படி இருக்கீங்க?",
    ref_audio="reference.wav",          # 3-10 s, one clean speaker (get their permission)
    ref_text="<exact transcript of reference.wav>",
    ref_lang="en",                      # the clip's language; the output language can differ
):
    audio = result.audio
```

To reuse a cloned voice, register it once with
`model.register_voice("me", "reference.wav", "<transcript>", lang="en")` and pass `voice="me"`.

### Streaming

```python
for chunk in model.generate(text="Your order has shipped.", voice="Siddhant", stream=True, streaming_interval=0.4):
    play(chunk.audio)
```

Audio is produced in 80 ms frames; `streaming_interval` sets how much is yielded at a time.

## Languages and Voices

**Languages:** Assamese (`as`), Bengali (`bn`), English (`en`), Gujarati (`gu`), Hindi (`hi`), Kannada (`kn`),
Malayalam (`ml`), Marathi (`mr`), Odia (`or`), Punjabi (`pa`), Tamil (`ta`), Telugu (`te`).

**Voices:** Ahana (default), Vaidehi, Malini, Charvi, Jahnavi, Anirban, Nakul, Vasant, Siddhant, Satvik, Abhay,
Nivedita, Ishwari, Ritvik, Vrinda, Sreekar, Kayal, Shaunak, Akshat, Tanushree, and `plain` (no speaker conditioning).
Their gender, age band and the languages each was trained on are listed in the model's `voices.json`.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `temperature`, `top_k`, `top_p` | 0.9, 64, 1.0 | codebook-0 sampling (the reference engine's defaults) |
| `repetition_penalty` | 0.8 | presence penalty over the last 12 codebook-0 codes |
| `max_tokens` | none | cap on audio frames (12.5 per second) |
| `seed` | none | reproducible output for the same text, voice and settings |
| `normalize` | `True` | expand numbers, currency, dates and URLs into words |

## Converting

The release repo has no top-level `config.json`, so conversion has its own entry point:

```bash
python -m mlx_audio.tts.models.maya2.convert --hf-path maya-research/maya2-native --mlx-path maya2-native-mlx-bf16
python -m mlx_audio.tts.models.maya2.convert --mlx-path maya2-native-mlx-8bit --q-bits 8
```

`--q-bits` quantizes the backbone's projections only. The embedding tables, the depth decoder and the heads keep their
precision, because the depth decoder and the speaker paths are precision-sensitive.

## Fidelity

The port was checked against the reference PyTorch implementation component by component:

| Component | Result |
|---|---|
| Prompts | identical token for token for all voices and languages, including cross-lingual clones |
| Backbone | below PyTorch's own bf16 noise |
| Depth logits | fp16 error equal to PyTorch fp16's |
| Text normaliser | identical output in all 12 languages |
| Seeded generations | match an independent MLX port frame for frame |

Speech-recognition character error rates are in the same range as the reference engine across the 8 languages
tested.

## Notes

- **One stream at a time.** The reference engine's continuous batching across many concurrent requests is not
  ported.
- **Seeds aren't portable.** A seed reproduces audio within MLX, but not the CUDA engine's audio, because the random
  number generators differ.
- **Cloned voices sound similar, not identical.** Similarity and accent vary with the clip.

Reference implementation: [MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native).

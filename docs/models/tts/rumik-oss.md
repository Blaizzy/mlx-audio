# rumik-oss 1

rumik-oss 1 is a 3B expressive, multilingual text-to-speech model from [rumik.ai](https://rumik.ai/research/rumik-oss) for 22 Indic languages plus English, in native scripts and romanized forms, with code-switched synthesis. A decoder-only language model emits eight Mimi codec tokens per 80 ms frame and the frozen codec reconstructs 24 kHz audio. Delivery is controlled with a text description (tone, accent, pace) and inline vocalization tags.

The MLX port loads the original Hugging Face repo directly, applies the model's stop predictor, restricts sampling to the audio vocabulary, and streams audio through mlx-audio's native Mimi decoder.

## Model Variants

| Model | Format | HuggingFace |
|-------|--------|-------------|
| `rumik-ai/rumik-oss-1` | BF16 safetensors (original repo) | [:octicons-link-external-16: Model Card](https://huggingface.co/rumik-ai/rumik-oss-1) |
| `rumik-ai/rumik-oss-1-mlx-8bit` | 8-bit MLX quantized (recommended) | [:octicons-link-external-16: Model Card](https://huggingface.co/rumik-ai/rumik-oss-1-mlx-8bit) |
| `rumik-ai/rumik-oss-1-mlx-4bit` | 4-bit MLX quantized (fastest, lower fidelity) | [:octicons-link-external-16: Model Card](https://huggingface.co/rumik-ai/rumik-oss-1-mlx-4bit) |

The Mimi codec weights are fetched from `kyutai/moshiko-pytorch-bf16` on first use (the codec bundled with the original repo is byte-identical). Requires ~7 GB of memory in BF16, ~4 GB at 8-bit, ~2 GB at 4-bit.

## Usage

### Basic Generation

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model rumik-ai/rumik-oss-1-mlx-8bit \
        --text "नमस्ते, आज आपका दिन कैसा रहा?" \
        --voice Ira \
        --instruct "happy, Hindi accent, steady pace"
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("rumik-ai/rumik-oss-1-mlx-8bit")

    for result in model.generate(
        text="नमस्ते, आज आपका दिन कैसा रहा?",
        voice="Ira",
        instruct="happy, Hindi accent, steady pace",
    ):
        audio = result.audio  # mx.array, 24 kHz
    ```

### Streaming

Audio is yielded in chunks as frames are generated; `streaming_interval` sets the chunk length in seconds.

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model rumik-ai/rumik-oss-1-mlx-8bit \
        --text "Good evening, passengers. Boarding will begin shortly." \
        --voice Aisha \
        --instruct "professional, Indian English accent, steady pace" \
        --stream --streaming_interval 0.5 --play
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("rumik-ai/rumik-oss-1-mlx-8bit")

    for chunk in model.generate(
        text="Good evening, passengers. Boarding will begin shortly.",
        voice="Aisha",
        instruct="professional, Indian English accent, steady pace",
        stream=True,
        streaming_interval=0.5,
    ):
        play(chunk.audio)  # chunk.is_final_chunk marks the last one
    ```

### Delivery Controls

The `instruct` argument becomes the `<description="...">` prefix of the prompt. The description can also be written inline in the text, and vocalization tags are placed where they should occur.

```python
model.generate(
    text='<description="excited, Hindi accent, fast pace"> जल्दी आओ, हमारा नाम लिस्ट में है! <laugh> आज घर में जश्न होगा।',
    voice="Siya",
)
```

| Control | Values |
|---------|--------|
| `voice` | `Ira`, `Aisha`, `Siya`, `Zoya` |
| tone | happy, sad, angry, excited, professional |
| accent | Hindi, Telugu, Tamil, Kannada, Bengali, Punjabi, Indian English |
| pace | slow, fast, steady |
| inline tags | `<laugh>`, `<chuckle>`, `<sigh>` |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.8 | Sampling temperature (reference setting) |
| `top_k` | 30 | Top-k sampling (reference setting) |
| `max_tokens` | 2048 | Cap on audio tokens; 100 tokens make one second of speech |
| `min_tokens` | 8 | Tokens generated before the utterance may end |
| `streaming_interval` | 0.5 | Seconds of audio per streamed chunk |

Generation ends when the model's stop predictor fires or `</audio>` is sampled, matching the reference implementation. Utterances longer than about 35 seconds are outside the model's training distribution.

## Performance

Decode speed on an M5 MacBook Air (16 GB), steady state, audio head sliced to the audio vocabulary:

| Variant | Tokens/s | Real-time factor | Peak memory | Next-token agreement vs. BF16 |
|---------|---------:|-----------------:|------------:|------------------------------:|
| BF16 | 21 | 0.21x | 6.9 GB | -- |
| 8-bit | 33-40 | 0.33-0.40x | 3.8 GB | 96% |
| 4-bit | 69 | 0.69x | 2.1 GB | 80% |

The flattened codec formulation runs the 3B backbone once per codebook token, so 100 forward passes produce one second of audio. Streaming delivers first audio in well under a second, but sustained real-time playback on Apple silicon laptops is not yet reached.

## License

rumik-oss 1 weights are released for research and non-commercial use under [CC-BY-NC 4.0 with the Cohere Labs acceptable-use addendum](https://cohere.com/cohere-labs-cc-by-nc-license). The Mimi codec is CC-BY-4.0.

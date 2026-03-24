# MOSS-TTS

This model belongs to the [**MOSS-TTS**](https://github.com/OpenMOSS/MOSS-TTS/tree/main) family and corresponds to the [moss_tts_delay](https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_tts_delay) variant, whose model_typeis set to `moss_tts_delay`.

At a high level, the integration looks like this:

- **Backbone**: a Qwen3-style transformer that generates 32-channel audio codes using a delay-pattern schedule.
- **Codec**: the [MOSS Audio Tokenizer](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer).

## Run MOSS-TTS locally

This integration is intended for local conversion and local inference on Apple Silicon.

### Prerequisites

- A working Python environment (this repo supports uv).
- HuggingFace model weights are available locally. You can download the pre-quantized 8-bit weights from Hugging Face: `mlx-community/MOSS-TTS-8B-8bit`  
(<https://huggingface.co/mlx-community/MOSS-TTS-8B-8bit>), it has been quantized into 8bit already
- Codec weights are available locally. You can download the weights directly from Hugging Face: `OpenMOSS-Team/MOSS-Audio-Tokenizer`  
(<https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer>).

### (Optional) Convert the backbone to MLX 8bit

If you want to perform the conversion yourself, first download the weights from [this official link](https://huggingface.co/OpenMOSS-Team/MOSS-TTS), then run the following command to convert them to an 8-bit quantized MLX version.

```bash
uv run python -m mlx_audio.convert \
  --hf-path <your huggingface weight path> \
  --mlx-path ./moss-tts-8bit \
  --quantize --q-bits 8 --dtype bfloat16
```

This produces a local directory `./moss-tts-8bit/` that MLX-Audio can load via `load_model()`.

### Codec

You can download the weights directly from Hugging Face: `OpenMOSS-Team/MOSS-Audio-Tokenizer`  
(<https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer>). The downloaded files already include the MLX-compatible weights required by MOSS-TTS:

- `config.json`
- `*.safetensors`

By default, MOSS-TTS looks for this directory in the following locations:

1. `<moss_tts_model_weights_dir>/moss-audio-tokenizer-full/`
2. `./moss-audio-tokenizer-full/` (fallback)

Only one valid directory is needed. You may also choose to download the weights directly into either of these paths.

### Generate audio (CLI)

```bash
uv run python -m mlx_audio.tts.generate \
  --model ./moss-tts-8bit \
  --text "Hello from MOSS-TTS" \
  --output_path ./test_moss_output
```

### Generate audio (Python)

```python
from mlx_audio.tts.utils import load_model

model = load_model("./moss-tts-8bit")

for result in model.generate(
    text="你好，这是 MOSS TTS 的测试。",
    max_tokens=300,
):
    audio = result.audio
    sr = result.sample_rate
    print(audio.shape, sr)
    break
```

### Recommended sampling parameters

The defaults below are the officially recommended settings for both the q8 quantized model and the bf16 model. We **strongly recommend using them as is**, as they provide a reliable starting point for most use cases and produce stable, high-quality output out of the box.

MOSS-TTS uses a two-stream sampling scheme in which text tokens and audio tokens are decoded in parallel, with separate sampler controls for each stream.

| Parameter | Default | Applies to | Description |
| -------------------- | ------- | ---------- | ------------------------------------------------------------------ |
| `temperature` | `1.7` | audio | Sampling temperature for audio tokens. Higher values increase diversity. |
| `top_p` | `0.8` | audio | Nucleus (top-p) sampling threshold for audio tokens. |
| `top_k` | `25` | audio | Top-k cutoff for audio token sampling. |
| `text_temperature` | `1.5` | text | Sampling temperature for the parallel text stream. |
| `text_top_p` | `1.0` | text | Nucleus sampling threshold for text tokens (effectively disabled at 1.0). |
| `text_top_k` | `50` | text | Top-k cutoff for text token sampling. |
| `repetition_penalty` | `1.0` | both | Penalty applied to logits for already-generated tokens. Values above `1.0` discourage repetition. |


### User instruction template (`<user_inst>`)

MOSS-TTS uses a structured prompt template to control generation. Every `generate()` call renders the following template internally:

```
<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>
```

You can control these fields via CLI flags or Python API keyword arguments:


| Template field  | CLI flag      | Python kwarg    | Description                                                            |
| --------------- | ------------- | --------------- | ---------------------------------------------------------------------- |
| `text`          | `--text`      | `text`          | The text to synthesize (required)                                      |
| `reference`     | `--ref_audio` | `ref_audio`     | Reference audio for voice cloning; rendered as `<|audio|>` placeholder |
| `instruction`   | `--instruct`  | `instruct`      | Style/emotion instruction (e.g. "用开心的语气")                              |
| `language`      | `--lang_code` | `lang_code`     | Language tag (e.g. "zh", "en")                                         |
| `tokens`        | —             | `tokens`        | Output token budget hint (integer)                                     |
| `quality`       | —             | `quality`       | Quality descriptor (e.g. "high")                                       |
| `sound_event`   | —             | `sound_event`   | Sound event tag                                                        |
| `ambient_sound` | —             | `ambient_sound` | Ambient sound tag                                                      |


If a field is not provided (e.g. `ref_audio`), it defaults to None and is rendered as the string "None" in the template, matching the HuggingFace upstream behavior.

CLI example with instruction and language:

```bash
uv run python -m mlx_audio.tts.generate \
  --model ./moss-tts-8bit \
  --text "今天天气真好！" \
  --instruct "用开心的语气" \
  --lang_code zh \
  --output_path ./test_moss_output
```

Python example with all fields:

```python
for result in model.generate(
    text="Today is a great day!",
    instruct="cheerful and energetic",
    lang_code="en",
    max_tokens=300,
):
    audio = result.audio
    break
```

For finer control (e.g. `tokens`, `quality`, `sound_event`, `ambient_sound`), use the processor directly:

```python
input_ids = model.processor.prepare_generation_input(
    text="你好",
    instruction="轻柔的语气",
    language="zh",
    quality="high",
    tokens=200,
)
```

### Voice cloning

MOSS-TTS voice cloning is MLX-native.

Requirements:

- MOSS-TTS q8 model (recommended): `./moss-tts-8bit`
- Codec with encoder weights available through `codec_path` or default search path
- Reference audio WAV file

CLI example:

```bash
uv run python -m mlx_audio.tts.generate \
  --model ./moss-tts-8bit \
  --text "请用参考音色读这句话。" \
  --ref_audio <your ref audio>.wav \
  --output_path ./test_moss_output_voice_clone
```

If encoder weights are missing, generation raises a clear error indicating codec encoder params are required.

### Conversation-structured input (stateless)

MOSS-TTS now accepts a full conversation directly through `model.generate(conversation=..., mode=...)`, following the same semantics as the HuggingFace upstream which means it is **stateless**. Each request must include the complete conversation history to condition on.

```python
from mlx_audio.tts.models.moss_tts.processor import build_user_message, build_assistant_message

model = load_model("./moss-tts-8bit")

# Build conversation: [user, assistant, user]
conversation = [
    {
        "role": "user",
        "content": build_user_message(text="请自我介绍", language="zh"),
        "audio_codes_list": [],
    },
    build_assistant_message(
        audio_codes_list=[previous_audio_codes],  # mx.array of shape (T, NQ)
    ),
    {
        "role": "user",
        "content": build_user_message(text="再说一遍，声音大一点", language="zh"),
        "audio_codes_list": [],
    },
]

# mode="generation" → last message is user, appends assistant generation prompt
results = list(
    model.generate(
        conversation=conversation,
        mode="generation",
        max_tokens=300,
    )
)

audio = results[0].audio
```

The `mode` parameter controls truncation behavior:

- `"generation"` (default): last message must be `user`; conversation count must be odd.
- `"continuation"`: last message must be `assistant`; conversation count must be even; the last assistant segment is truncated for continuation.

Continuation uses an assistant prefix audio segment together with the matching prefix transcript in the user text. The model then continues in the same voice/style without replaying the prefix audio:

```python
conversation = [
    {
        "role": "user",
        "text": "太阳系八大行星之一。请继续用同样的音色详细介绍地球。",
        "language": "zh",
    },
    {
        "role": "assistant",
        "audio_codes_list": ["./fixtures/ref_audio.wav"],
    },
]

results = list(
    model.generate(
        conversation=conversation,
        mode="continuation",
        max_tokens=300,
    )
)
```

Entries in `audio_codes_list` may be:

- audio file paths (recommended)
- precomputed audio-code matrices (`mx.array` or nested integer lists shaped `[T, NQ]`)
- raw waveform arrays that the codec can encode

For single-turn generation, `model.generate(text=..., ref_audio=...)` still works exactly as before.

### CLI conversation input

You can pass a full conversation as JSON with `--conversation_json`.

When `--conversation_json` is used, do not also pass `--ref_audio` or `--ref_text`; put all audio context directly inside the JSON payload.

`conversation.json`:

```json
[
  {
    "role": "user",
    "text": "But I really can't complain about not having a normal college experience to you. Please continue in the same voice.",
    "reference": ["./fixtures/ref_audio.wav"],
    "language": "en"
  },
  {
    "role": "assistant",
    "audio_codes_list": ["./fixtures/ref_audio.wav"]
  }
]
```

To run continuation generation, use::

```bash
uv run python -m mlx_audio.tts.generate \
  --model ./moss-tts-8bit \
  --conversation_json ./conversation.json \
  --mode continuation \
  --output_path ./test_moss_output_continuation
```

## Serve via OpenAI-compatible API

Start the server:

```bash
uv run python -m mlx_audio.server --host 127.0.0.1 --port 8321
```

Request TTS:

```bash
curl -X POST http://127.0.0.1:8321/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"./moss-tts-8bit","input":"Hello from MOSS"}' \
  --output moss.wav
```

Conversation-style payloads are also supported on the same endpoint. This remains stateless: send the full conversation each time, and make sure the final assistant message includes the prefix audio in `audio_codes_list`.

```bash
curl -X POST http://127.0.0.1:8321/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "./moss-tts-8bit",
    "mode": "continuation",
    "conversation": [
      {
        "role": "user",
        "text": "太阳系八大行星之一。请继续介绍地球。",
        "language": "zh"
      },
      {
        "role": "assistant",
        "audio_codes_list": ["./fixtures/ref_audio.wav"]
      }
    ]
  }' \
  --output moss_continuation.wav
```

## Codec reconstruction check (write reconstructed wav)

The following command runs the codec encode–decode reconstruction quality test and writes the reconstructed WAV file to the given path:

```bash
MOSS_SAVE_CODEC_RECONSTRUCTION=1 \
MOSS_CODEC_RECONSTRUCTION_PATH=<your local audio file> \
uv run pytest -q mlx_audio/tts/tests/test_moss_tts.py -k "TestCodecReconstruction and reconstructs_fixture"
```


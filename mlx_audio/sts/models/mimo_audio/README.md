# MiMo-Audio

MLX implementation of [MiMo-Audio-7B-Instruct](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct), [MiMo-Audio-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Base).

| Checkpoint | Supported tasks |
|---|---|
| Instruct | English/Chinese TTS, style instructions, reference voice prompting, ASR, audio understanding, spoken dialogue, speech-to-text dialogue, text dialogue, multi-turn history |
| Base | Few-shot speech tasks, raw text/audio completion, audio continuation |
| Audio tokenizer | Standalone waveform encoding and decoding with all 20 RVQ codebooks; MiMo uses the first eight |

Generation returns a complete response, with 24 kHz mono audio when the model produces speech. Streaming, live duplex sessions, and server routes are not implemented. Calls process one example at a time.

## Load a model

```python
from mlx_audio.sts import load

model = load("XiaomiMiMo/MiMo-Audio-7B-Instruct", strict=True)
```

Both original checkpoints and converted MLX directories load through the STS loader. Original metadata identifies the architecture as `MiMoAudioModel` under `qwen2`; converted metadata uses `mimo_audio`.

The audio tokenizer loads lazily for audio tasks.

## Text-to-speech

```python
from mlx_audio.audio_io import write

result = model.generate(task="tts", text="Hello from MLX Audio!")
write("hello.wav", result.audio, result.sample_rate)
print(result.text)

# Style control and an optional reference voice.
result = model.generate(
    task="tts",
    text="你好，欢迎回来！",
    instruct="用开心的语气说",
    ref_audio="reference.wav",
)

# Natural language instructions, including what to say and how to say it.
result = model.generate(
    task="tts",
    text="请用温柔的语气说：晚安，祝你有个好梦。",
    read_text_only=False,
)
```

Plain TTS uses a fixed English or Chinese instruction template. Instructed TTS uses the checkpoint's reasoning prompt. `result.text` contains the final response; `result.reasoning` contains optional reasoning. If generation ends during reasoning, the final text is empty and audio may be `None`. `result.raw_text` retains the reasoning markup, with audio/control tokens removed.

## ASR and audio understanding

```python
result = model.generate("speech.wav", task="asr")
print(result.text)

result = model.generate(
    "recording.wav",
    task="audio_understanding",
    text="Describe the sounds in this recording.",
    thinking=True,
)
print(result.text)
```

Audio inputs may be file paths, mono arrays, or time-major arrays shaped `(samples, channels)`. Arrays require `sample_rate`; files supply their own rate. Inputs are downmixed and resampled to 24 kHz.

```python
result = model.generate(waveform, sample_rate=16000, task="asr")
```

## Dialogue

```python
# Spoken input and spoken response.
result = model.generate("question.wav", task="spoken_dialogue")
write("answer.wav", result.audio, result.sample_rate)

# Spoken input and a text response.
result = model.generate("question.wav", task="speech_to_text", thinking=True)

# Text-only conversation does not load the audio tokenizer.
result = model.generate(task="text", text="Tell me a short story.")
```

Multi-turn calls accept explicit messages. Retain returned audio codes to avoid encoding generated speech again:

```python
first = model.generate("question.wav", task="spoken_dialogue")
messages = [
    {"role": "user", "content": "question.wav"},
    {
        "role": "assistant",
        "content": {"text": first.text, "codes": first.audio_codes},
    },
    {"role": "user", "content": "followup.wav"},
]
second = model.generate(task="spoken_dialogue", messages=messages)
```

Assistant speech history can alternatively contain `{"text": ..., "audio": "answer.wav"}`. For text-only history, use string content for both roles. Each call creates fresh caches and packs the supplied history; there is no hidden conversation state. History must end with a user message and fit the context budget. `ref_audio` sets a reference voice for spoken dialogue, and `system_prompt` supplies an instruction before the first user audio, following the reference format.

## Base: few-shot speech tasks and completion

Base is a pretrained completion model. Use examples or explicit segments to establish the task.

```python
base = load("XiaomiMiMo/MiMo-Audio-7B-Base", strict=True)
result = base.generate(
    "new_input.wav",
    task="few_shot",
    instruction="Convert the speaker's voice to match the example output.",
    prompt_examples=[{
        "input_audio": "example_input.wav",
        "output_audio": "example_output.wav",
        "output_transcription": "The words spoken in the example output.",
    }],
    max_new_tokens=512,
)
write("converted_voice.wav", result.audio, result.sample_rate)

result = base.generate(
    task="completion",
    segments=[{"text": "The capital of France is"}],
    temperature=0,
    max_new_tokens=32,
)

# Continue an unfinished audio prefix, with no closing speech boundary.
result = base.generate("prefix.wav", task="continuation", max_new_tokens=128)
```

Raw completion segments accept `{"text": ...}`, `{"audio": ..., "sample_rate": ...}`, or `{"codes": ...}`. Audio segments normally include `<|sosp|>` and `<|eosp|>` boundaries; set `boundaries=False` for an unfinished prefix. Codes are integers shaped `(codebooks, frames)` with at least eight codebooks. Padding IDs are not valid codec codes. Continuation is prompt-dependent and can stop without producing audio.

## Generation settings and results

`max_new_tokens` counts global model positions: each position is one text token or one four-frame speech patch (160 ms). The context limit is 8,192 positions including the prompt. `finish_reason` is `"stop"` or `"length"`; reaching a length limit returns all generated patches.

Text and audio have separate sampling controls: `temperature`, `top_k`, `top_p`, `audio_temperature`, `audio_top_k`, and `audio_top_p`. Set a temperature to zero for greedy decoding. Seed MLX with `mx.random.seed(...)` for repeatable sampling within the same environment. Numerical differences across precision levels and devices can change generated speech codes.

Sampled speech can repeat or add words, including with reference voice prompts. The returned text is the model's text channel and is not a verified transcript of its waveform. Voice similarity and speech quality need evaluation for the intended use.

The result exposes `text`, `reasoning`, `raw_text`, `audio`, `audio_codes`, `sample_rate`, packed `tokens`, `prompt_tokens`, `generation_tokens`, `finish_reason`, and `processing_time`. `audio` is a mono MLX array or `None`; `audio_codes` has shape `(8, frames)` and may be empty.

## CLI

```bash
python -m mlx_audio.sts.generate \
  --model XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --task tts --text "Hello from MLX Audio!" --output-path hello.wav

python -m mlx_audio.sts.generate \
  --model XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --task asr --audio speech.wav --output-path transcript.txt

python -m mlx_audio.sts.generate \
  --model XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --task spoken_dialogue --audio question.wav --output-path answer.wav
```

Use `--audio-tokenizer-path` for a local codec, `--instruct` and `--ref-audio` for TTS voice/style control, `--natural-instruction` for natural-language TTS, and `--thinking` for text reasoning. `--messages`, `--segments`, and `--prompt-examples` accept JSON files using the Python formats above. The CLI saves a text file beside generated audio; text-only tasks write only text. `--stream` is rejected for MiMo.

## Audio tokenizer

See the [standalone codec guide](../../../codec/models/mimo_audio_tokenizer/README.md) for all 20 codebooks and reconstruction examples.

Model cards: [MiMo-Audio-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Base) and [MiMo-Audio-7B-Instruct](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct). Model weights license: MIT. [Reference code](https://github.com/XiaomiMiMo/MiMo-Audio) license: Apache-2.0.

# LFM3-Audio

[LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
supports transcription, text-to-speech, audio questions, and speech-to-speech
chat. Speech output is mono at 24 kHz. The runtime uses MLX and does not require
PyTorch or remote model code.

Supported checkpoints:

| Model | Precision |
| --- | --- |
| [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) | BF16 |
| [LiquidAI/LFM3-Audio-2B-MLX-8bit](https://huggingface.co/LiquidAI/LFM3-Audio-2B-MLX-8bit) | 8-bit |
| [LiquidAI/LFM3-Audio-2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM3-Audio-2B-MLX-4bit) | 4-bit |

```python
from mlx_audio import audio_io
from mlx_audio.sts import load

model = load("LiquidAI/LFM3-Audio-2B-MLX-8bit")

# Transcription
result = model.generate("speech.wav", task="asr", language="English")
print(result.text)

# Text-to-speech
result = model.generate(
    text="Hello there!", task="tts", language="American English", voice="female",
)
audio_io.write("speech.wav", result.audio, result.sample_rate)

# Voice chat
result = model.generate("question.wav", task="sts")
print(result.text)
if result.audio is not None:
    audio_io.write("answer.wav", result.audio, result.sample_rate)

# Audio questions with text-only answers
result = model.generate("recording.wav", task="chat", text="What is happening?")
print(result.text)
```

```sh
python -m mlx_audio.sts.generate \
  --model LiquidAI/LFM3-Audio-2B-MLX-8bit \
  --audio speech.wav --task asr --output-path transcript.wav

python -m mlx_audio.sts.generate \
  --model LiquidAI/LFM3-Audio-2B-MLX-8bit \
  --task tts --text "Hello there!" --output-path speech.wav \
  --gen-kwargs '{"language": "American English", "voice": "female"}'

python -m mlx_audio.sts.generate \
  --model LiquidAI/LFM3-Audio-2B-MLX-8bit \
  --audio question.wav --task sts --output-path answer.wav
```

For text-only results, the CLI writes a `.txt` file beside the requested audio
path. Input files are resampled to 16 kHz; arrays default to 16 kHz and accept
`sample_rate` for other rates.

`generate()` accepts `max_new_tokens` (text tokens and speech frames), `seed`,
`temperature`, `top_k`, `top_p`, `repetition_penalty`, `system_prompt`, and
`generation_mode` (`text`, `audio`, or `interleaved`). Speech sampling defaults
to `audio_temperature=1.0`, `audio_top_k=128`, and
`audio_repetition_penalty=1.1` over `audio_repetition_window=4096` frames.
The voice is selected through the TTS system prompt; `voice` accepts `female`
or `male`. `thinking=True` enables reasoning for `task="chat"`.

For multi-turn conversations, pass `messages` instead of `audio` and `text`.
Audio content blocks use `{"type": "audio", "audio": path_or_array}`. Earlier
generated speech uses `{"type": "audio_out", "codes": result.audio_tokens.T}`.
Pass function schemas through `tools` for spoken tool requests. Low-level
`model.stream_slots(**model.processor.prepare(messages))` yields text tokens
and speech code frames as they are generated.

Speech supports Arabic, Danish, German, Greek, English (British and American),
Finnish, French, Hebrew, Italian, Japanese, Korean, Dutch, Polish, Portuguese,
Spanish, Swedish, Turkish, and Chinese. The model context is 32K tokens.
Interleaved voice chat is English only. Generation handles one conversation
per call. LiveASR, streaming waveform input/output, waveform speech history,
and voice cloning are not supported.

Weights use the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/main/LICENSE).
The adapted reference implementation uses [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

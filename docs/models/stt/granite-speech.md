---
title: Granite Speech
---

# Granite Speech

IBM's Granite Speech combines an audio encoder with a language-model decoder. MLX Audio supports the original speech checkpoint and the Plus variant through the same model package.

| Checkpoint | Tasks | Languages |
| --- | --- | --- |
| [Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | Transcription, speech translation, keyword biasing | EN, FR, DE, ES, PT, JA |
| [Granite Speech 4.1 2B Plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | Transcription, speaker attribution, word timestamps, keyword biasing | EN, FR, DE, ES, PT |

The examples below use IBM's native Plus checkpoint, which loads directly. Local MLX-converted and quantized checkpoints use the same API.

## Python

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-speech-4.1-2b-plus")

# Plain transcription is the default.
result = model.generate("audio.wav")
print(result.text)

# Speaker attribution returns speaker IDs and text for each detected turn.
result = model.generate("meeting.wav", task="saa")
for segment in result.segments:
    print(segment["speaker_id"], segment["text"])

# Word timestamps are a separate task from speaker attribution.
result = model.generate("audio.wav", task="timestamps", max_tokens=8192)
for segment in result.segments:
    for word in segment["words"]:
        print(word["word"], word["start"], word["end"])
```

| Option | Behavior |
| --- | --- |
| `task="asr"` | Plain transcription; permits a custom `prompt` |
| `task="saa"` | Speaker-attributed text; requires a Plus checkpoint |
| `task="timestamps"` | Word timings in seconds; requires a Plus checkpoint |
| `word_timestamps=True` | Alias for timestamp mode on Plus checkpoints, ignored by 4.0; cannot be combined with `task="saa"` |
| `hotwords=["Acme", "QFormer"]` | Adds keyword hints to the task prompt |
| `system_prompt="..."` | Replaces the system turn that Plus checkpoints send by default |

Rich tasks use canonical prompts to select their output format, so `prompt=` cannot override `saa` or `timestamps`. If the model returns plain or malformed text instead of the requested tags, `generate()` raises `StructuredTranscriptError` (importable from `mlx_audio.stt.models.granite_speech.granite_speech`) with the model output on its `raw_text` attribute. The original 4.0 checkpoint also accepts `language="fr"` (or another supported target language) for translation.

## CLI

```bash
mlx_audio.stt.generate \
  --model ibm-granite/granite-speech-4.1-2b-plus \
  --audio meeting.wav \
  --output-path transcript \
  --format json \
  --gen-kwargs '{"task": "saa", "hotwords": ["Acme", "QFormer"]}'
```

For subtitles, use `--format srt` or `--format vtt` with `--gen-kwargs '{"task": "timestamps"}'`; the file holds one cue for the whole utterance followed by one cue per word. Speaker-only segments have no timestamps, so use JSON to preserve their speaker labels.

The CLI passes `--language en` by default, which selects the translation prompt for `task="asr"`. To use the checkpoint's plain transcription prompt instead, add `"language": null` to `--gen-kwargs`.

## Streaming

```python
for chunk in model.generate("audio.wav", stream=True):
    print(chunk.text, end="", flush=True)
```

Streaming yields decoder text after the supplied recording has been encoded; it does not ingest live audio. With `task="saa"` or `task="timestamps"` the stream carries the raw `[Speaker N]:` or `[T:N]` tags. Call `generate()` without `stream=True` to receive parsed segments.

## Audio and output limits

The [Plus model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) specifies up to nine minutes for ASR or speaker attribution and 3.5 minutes for word timestamps. Timestamp tags need more output tokens than plain text, so raise `max_tokens` for long recordings.

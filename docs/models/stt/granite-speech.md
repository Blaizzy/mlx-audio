---
title: Granite Speech
---

# Granite Speech

IBM's Granite Speech combines an audio encoder with a language-model decoder. MLX Audio supports the original speech checkpoint, Granite Speech 5.0 2B, and the Plus variant through the same model package.

| Checkpoint | Tasks | Languages |
| --- | --- | --- |
| [Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | Transcription, speech translation | EN, FR, DE, ES, PT, JA |
| [Granite Speech 5.0 2B](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) | Transcription, speech translation, keyword biasing, punctuation, language hints | EN, FR, DE, ES, PT, JA; also translates English to IT and Mandarin |
| [Granite Speech 4.1 2B Plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | Transcription, speaker attribution, word timestamps | EN, FR, DE, ES, PT |

IBM's native checkpoints load directly. Checkpoints quantized with `mlx_audio.convert` use the same API.

## Granite Speech 5.0 2B

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-speech-5.0-2b")
result = model.generate("audio.wav", punctuate=True, hotwords=["Granite", "IBM"])
print(result.text)

# source_language hints the spoken language; language requests translation.
result = model.generate("japanese.wav", source_language="ja")
result = model.generate("audio.wav", language="fr", punctuate=True)
```

`source_language` and `language` cannot be combined. Keyword lists can contain
up to 2,000 terms. A custom `prompt` replaces the default instruction; language
hints and keywords are still added, while `punctuate` only affects default
instructions. Streaming accepts the same options. Speaker attribution and word
timestamps require the Plus checkpoint.

```bash
mlx_audio.stt.generate \
  --model ibm-granite/granite-speech-5.0-2b \
  --audio audio.wav --output-path transcript \
  --gen-kwargs '{"language": null, "punctuate": true}'
```

The CLI defaults to translation to English; `"language": null` selects
transcription in the spoken language. IBM recommends decoding segments of
about two minutes or less. Files are resampled to mono 16 kHz; raw arrays must
already contain mono 16 kHz samples.

### Speculative decoding

```python
result = model.generate("audio.wav", speculative=True, num_draft_tokens=16)
print(result.text)
print(result.speculative_stats)
```

The model's BPE CTC head drafts tokens from the same encoder pass. The LLM
verifies blocks against its greedy predictions, accepts their matching prefix,
and rolls back rejected cache entries before continuing. This uses argmax
verification rather than IBM's approximate confidence-threshold shortcuts.
It requires `temperature=0`, no repetition penalty, and `out_llm.safetensors`;
conversion preserves this sidecar. `model.supports_speculative` reports its
availability. Streaming reports decoding statistics in the final chunk.
Batched BF16 evaluation can change argmax decisions at near ties, so literal
transcript equality is not guaranteed; the benchmark records equality and word
errors when references are available.

This mode is disabled by default. It works best for transcription; translation
can run slower because CTC drafts the spoken language. Use
`--gen-kwargs '{"language": null, "speculative": true}'` from the CLI.
The `examples/benchmark_granite_speculative.py` script compares warmed,
synchronized latency and transcript equality across draft sizes and checkpoints.

## Plus: rich transcription

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
| `system_prompt="..."` | Replaces the system turn that Plus checkpoints send by default |

Rich tasks use canonical prompts to select their output format, so `prompt=` cannot override `saa` or `timestamps`. If the model returns plain or malformed text instead of the requested tags, `generate()` raises `StructuredTranscriptError` (importable from `mlx_audio.stt.models.granite_speech.granite_speech`) with the model output on its `raw_text` attribute. The original 4.0 checkpoint also accepts `language="fr"` (or another supported target language) for translation.

## CLI

```bash
mlx_audio.stt.generate \
  --model ibm-granite/granite-speech-4.1-2b-plus \
  --audio meeting.wav \
  --output-path transcript \
  --format json \
  --gen-kwargs '{"task": "saa"}'
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

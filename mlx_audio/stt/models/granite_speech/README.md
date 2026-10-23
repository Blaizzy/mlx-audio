# Granite Speech

MLX implementation of IBM's Granite Speech, a speech-to-text model that combines a CTC Conformer encoder with a Granite LLM decoder via a BLIP-2 QFormer projector. Supports ASR (transcription), AST (speech translation), and, with the plus checkpoint, speaker-attributed ASR and word-level timestamps.

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [ibm-granite/granite-4.0-1b-speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | ~1B | Speech recognition and translation |
| [ibm-granite/granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) | ~2B | Multilingual ASR, translation, punctuation, language hints, and keyword biasing |
| [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | ~2B | Rich transcription: speaker attribution, word timestamps (EN, FR, DE, ES, PT) |

**Supported Languages:** English, French, German, Spanish, Portuguese, Japanese (plus checkpoint: no Japanese)

## Granite Speech 5.0

Load IBM's checkpoint directly; no conversion is required. It uses the same
Conformer/QFormer/Granite architecture with a 348-character encoder head and
YaRN position scaling to 16,384 tokens. The encoder BPE head shipped in
`out_llm.safetensors` enables optional speculative decoding.

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-speech-5.0-2b")
print(model.generate("audio.wav", punctuate=True).text)

# Hint the spoken language without translating it.
result = model.generate("japanese.wav", source_language="ja")

# Bias transcription toward domain vocabulary (up to 2,000 keywords).
result = model.generate("audio.wav", hotwords=["Granite", "IBM"])

# language selects the translation target, including Italian and Mandarin.
result = model.generate("audio.wav", language="fr", punctuate=True)
```

`source_language` selects transcription and cannot be combined with `language`.
`prompt` replaces the default instruction; language hints and keywords are still
added when supplied. `punctuate` selects a punctuated default instruction and
does not rewrite a custom `prompt`. These options also work with `stream=True`.
The 5.0 checkpoint supports ASR and translation; speaker attribution and word
timestamps require the Plus checkpoint.

```bash
mlx_audio.stt.generate \
  --model ibm-granite/granite-speech-5.0-2b \
  --audio audio.wav --output-path transcript \
  --gen-kwargs '{"language": null, "punctuate": true, "hotwords": ["Granite", "IBM"]}'
```

The CLI defaults to `--language en`, which requests translation to English.
Use `"language": null` in `--gen-kwargs` for transcription in the spoken language.
IBM recommends segments of about two minutes or less; split longer recordings
before decoding. The 5.0 translation targets include English, French, German,
Spanish, Portuguese, Japanese, Italian, and Mandarin.

### Speculative decoding

```python
model = load("ibm-granite/granite-speech-5.0-2b")
print(model.supports_speculative)
result = model.generate("audio.wav", speculative=True, num_draft_tokens=16)
print(result.text)
print(result.speculative_stats)
```

The existing encoder supplies a BPE CTC draft using posterior-weighted pooling.
The LLM checks blocks of draft tokens in parallel, accepts only the prefix that
matches its greedy predictions, and corrects the first mismatch. Rejected KV
cache entries are discarded, and drafting resumes when the output aligns with
the CTC transcript again. The encoder runs once. Unlike the confidence-threshold
shortcuts in [IBM's notebook](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/speculative_decoding_bpe.ipynb),
each emitted token is verified by the LLM's argmax.

Batched and single-token BF16 kernels can produce different logits at near
ties, so literal transcript equality is not guaranteed. The benchmark records
text equality and, when reference transcripts are supplied, word errors.

This opt-in mode requires `temperature=0`, no repetition penalty, and the
checkpoint's `out_llm.safetensors`. The BPE head loads lazily on first use and is
preserved by conversion. Older converted checkpoints that omitted it must be
reconverted. Streaming accepts the same options and includes statistics on its
final chunk. Statistics report draft length, proposed/accepted tokens,
verification and autoregressive steps, and encoder/projector, draft, and prefill
times in seconds.

The CTC draft transcribes the spoken language. Translation and other prompts
that substantially change the text can reject most proposals and run slower.
Speculative decoding stays disabled by default.

```bash
mlx_audio.stt.generate \
  --model ibm-granite/granite-speech-5.0-2b \
  --audio audio.wav --output-path transcript \
  --gen-kwargs '{"language": null, "speculative": true, "num_draft_tokens": 16}'
```

For synchronized comparisons with warmup, rotating configuration order, transcript
equality checks, and JSON results:

```bash
python examples/benchmark_granite_speculative.py \
  --model ibm-granite/granite-speech-5.0-2b \
  --audio short.wav long.wav --draft-tokens 4 8 16 32 \
  --repeats 5 --output benchmark.json
```

#### Local benchmark

Apple M5 Max, 128 GB unified memory, MLX 0.32.4. Times below are medians of five
timed calls after warmup, with both the default and decoding GPU streams
synchronized. Each call includes feature extraction, encoding, drafting,
prefill, decoding, and text conversion; audio I/O and weight loading are excluded.
The speculative columns use 32-token proposals and the default ASR prompt.

| Audio duration | BF16 standard | BF16 speculative | Speedup | 8-bit standard | 8-bit speculative | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| 5.9 s | 0.241 s | 0.090 s | 2.67x | 0.215 s | 0.097 s | 2.23x |
| 12.5 s | 0.372 s | 0.125 s | 2.99x | 0.321 s | 0.137 s | 2.34x |
| 24.9 s, English/French | 0.830 s | 0.230 s | 3.61x | 0.680 s | 0.244 s | 2.78x |
| 29.4 s | 0.894 s | 0.453 s | 1.98x | 0.733 s | 0.500 s | 1.47x |
| 62.5 s, joined utterances | 1.859 s | 0.923 s | 2.01x | 1.470 s | 1.054 s | 1.39x |
| 109.8 s, joined utterances | 3.168 s | 2.879 s | 1.10x | 2.512 s | 2.518 s | 1.00x |
| Sum of medians | 7.364 s | 4.699 s | 1.57x | 5.931 s | 4.551 s | 1.30x |

Inputs are three utterances from the
[LibriSpeech test fixture](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy),
concatenations of five and ten distinct utterances from that fixture, and IBM's
bundled multilingual sample. The concatenations overlap the shorter inputs.
All 8-bit transcripts matched standard decoding. BF16 changed one proper-name
spelling on the 62.5-second input; reference word errors stayed at 33/522 (6.32%)
for both modes and precisions. These overlapping reference counts serve as a
local regression check.

[Full timings and transcripts](../../../../benchmarks/granite_speech_5_2b_speculative.json)
and the [draft-size sweep](../../../../benchmarks/granite_speech_5_2b_draft_sweep.json)
are saved with the benchmark script. Gains depend on draft agreement; keep
speculation optional for translation and prompts that change transcription style.
On the 24.9-second sample, punctuation produced 1.94x BF16 and 1.39x 8-bit
speedups, while French translation took about 14–15% longer with speculation.
The [punctuation](../../../../benchmarks/granite_speech_5_2b_punctuation.json)
and [translation](../../../../benchmarks/granite_speech_5_2b_translation.json)
runs use the same five-repeat protocol and 32-token proposals.

## CLI Usage

```bash
# Basic transcription
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output

# Verbose output with timing info
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --verbose

# Streaming output
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --stream

# Translate to French using language flag
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --language fr

# Translate using full language name
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --language Portuguese

# Output formats: txt, srt, vtt, json
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --format json
```

## Python Usage

### ASR (Transcription)

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

# Basic transcription (default prompt)
result = model.generate("audio.wav")
print(result.text)

# With custom prompt
result = model.generate("audio.wav", prompt="Translate the speech to text.")
print(result.text)
```

### AST (Speech Translation)

Use the `language` parameter to translate speech. Accepts full names or codes (`fr`, `de`, `es`, `pt`, `ja`):

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

# Translate speech to French (using language code)
result = model.generate("audio.wav", language="fr")
print(result.text)

# Translate speech to Spanish (using full name)
result = model.generate("audio.wav", language="Spanish")
print(result.text)

# Translate speech to Portuguese
result = model.generate("audio.wav", language="pt")
print(result.text)

# Or use a custom prompt directly
result = model.generate("audio.wav", prompt="Translate the speech to German.")
print(result.text)
```

> **Note:** If the model receives an unfamiliar prompt, it falls back to transcription as the default mode.

### Rich Transcription (granite-speech-4.1-2b-plus)

The plus checkpoint selects its mode through the prompt; the `task` parameter picks the right one. Audio limits from the model card: up to 9 minutes for `asr`/`saa`, up to 3.5 minutes for `timestamps`. Timestamps mode emits roughly one tag per word, so budget `max_tokens` accordingly. The model card describes unpunctuated, lowercase output, but generated text can include punctuation and casing; the implementation preserves it.

The canonical `saa` and `timestamps` prompts define their output schemas and cannot be replaced with `prompt=`; use `task="asr"` when supplying a custom instruction. If the checkpoint returns plain or malformed text instead of the requested tags, `generate()` raises `StructuredTranscriptError`; the model output is available on its `raw_text` attribute. Requesting `saa` or `timestamps` from a 4.0/4.1 checkpoint raises `UnsupportedTranscriptionTask`. Both exceptions are importable from `mlx_audio.stt.models.granite_speech.granite_speech`.

Plus checkpoints send the system turn from the model card by default; pass `system_prompt=` to replace it.

IBM updated the plus weights and configuration in place on October 21, 2026 (v2610). Both the original and updated checkpoints load, but MLX conversions or quantizations made from the original contain the original weights; reconvert from the current source to use the update.

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-speech-4.1-2b-plus")

# Speaker-attributed ASR: [Speaker N]: tags, parsed into segments
result = model.generate("meeting.wav", task="saa")
for seg in result.segments:
    print(f"Speaker {seg['speaker_id']}: {seg['text']}")

# Word-level timestamps: [T:N] tags, parsed into at most one segment with word timings
result = model.generate("audio.wav", task="timestamps", max_tokens=8192)
for seg in result.segments:
    for word in seg["words"]:
        print(f"{word['word']}\t{word['start']:.2f}-{word['end']:.2f}s")
```

From the CLI, `task` goes through `--gen-kwargs`:

```bash
mlx_audio.stt.generate --model ibm-granite/granite-speech-4.1-2b-plus \
  --audio meeting.wav --output-path output --format json \
  --gen-kwargs '{"task": "saa"}'
```

Speaker-attributed segments carry no timing, so save them as JSON; SRT and VTT output needs `task="timestamps"`.

### Streaming

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

for text in model.generate("audio.wav", stream=True):
    print(text, end="", flush=True)
```

For `saa` and `timestamps`, the stream carries the raw tagged text; call `generate()` without `stream=True` to get parsed segments.

### Generation Parameters

```python
result = model.generate(
    "audio.wav",
    max_tokens=4096,
    temperature=0.0,       # 0 = greedy decoding
    top_p=1.0,
    top_k=0,
    repetition_penalty=None,
    prompt="Translate the speech to text.",
    prefill_step_size=2048,
    verbose=True,          # print timing info
)
```

## Architecture

- **Encoder**: CTC Conformer (16 layers, 1024 hidden dim, Shaw's relative positional embeddings, block-wise attention with context_size=200)
- **Projector**: BLIP-2 QFormer (2 layers, windowed cross-attention with window_size=15, downsample_rate=5)
- **Decoder**: Granite LLM (40 layers, 2048 hidden dim, GQA with 16/4 heads, RoPE, SwiGLU MLP)
- Audio input: any sample rate, 80-bin mel spectrogram with pair stacking (160-dim input)

## Audio Input

Audio files are resampled to mono **16 kHz**. Raw arrays must already contain
mono 16 kHz samples. Supported input types:

- File path (WAV, FLAC, MP3, etc.)
- NumPy array (raw waveform)
- MLX array (raw waveform)

## Output Format

```python
STTOutput(
    text="Full transcription text",
    segments=[],
    prompt_tokens=154,
    generation_tokens=42,
    total_tokens=196,
    total_time=0.95,
    prompt_tps=162.1,
    generation_tps=44.2,
)
```

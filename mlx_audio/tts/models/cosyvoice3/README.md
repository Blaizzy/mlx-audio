# CosyVoice3 (MLX)

Port of Alibaba **CosyVoice3** to `mlx-audio`. All model components are
implemented; `llm.pt` / `flow.pt` / `hift.pt` have each been verified to
load `strict`ly (or `strict=False` modulo documented non-persistent buffers)
against the real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint, and a
real `generate()` call against that checkpoint has been run end-to-end
(see Status). What remains is absolute numerical-parity validation against
a captured PyTorch reference run and streaming support.

## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("mlx-community/CosyVoice3-0.5B")  # or a local checkpoint dir

result = next(model.generate(
    text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    ref_audio="examples/cosyvoice3/zero_shot_prompt.wav",
    ref_text="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
))
audio_write("output.wav", result.audio, result.sample_rate)

# English zero-shot
result = next(model.generate(
    text="Hello, this is a cloned voice speaking English text.",
    ref_audio="examples/voice_prompts/en_woman.wav",
    ref_text="You are a helpful assistant.<|endofprompt|>The radio quietly played a familiar song. Outside, rain tapped against the window in a steady rhythm. Coffee cooled slowly in a ceramic mug. Somewhere down the hall, a door clicked shut.",
))
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --ref_audio examples/cosyvoice3/zero_shot_prompt.wav \
  --ref_text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
  --file_prefix zero_shot

# English zero-shot
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --text "Hello, this is a cloned voice speaking English text." \
  --ref_audio examples/voice_prompts/en_woman.wav \
  --ref_text "You are a helpful assistant.<|endofprompt|>The radio quietly played a familiar song. Outside, rain tapped against the window in a steady rhythm. Coffee cooled slowly in a ceramic mug. Somewhere down the hall, a door clicked shut." \
  --file_prefix zero_shot_en
```

`ref_audio` (a zero-shot voice-cloning prompt) is required — CosyVoice3 has no
built-in preset speakers in this port. `ref_text` is optional; when omitted,
`generate.py` transcribes `ref_audio` automatically before calling the model
(the transcription is NOT auto-wrapped with `<|endofprompt|>`, so add it
yourself first if you need it — see below).

The concatenation of `prompt_text` (`ref_text`) and `text` fed to the LLM
must contain the `<|endofprompt|>` token (id 151646) — `CosyVoice3LM`
requires it (mirrors the reference's `assert 151646 in text`) and
`llm.py::inference` raises `ValueError` if it's missing. In zero-shot mode,
put it inside `ref_text` (e.g. `"You are a helpful assistant.<|endofprompt|>那份意外的惊喜..."`).

### Cross-lingual voice cloning

When `ref_audio`'s language differs from `text`'s, plain zero-shot mode
still feeds the LLM `ref_text` + the reference's own speech tokens, which
biases the first generated tokens towards `ref_audio`'s language (audible
as a wrong-language opening). Pass `cross_lingual=True` (`--cross_lingual`
on the CLI) to withhold both from the LLM call — the reference's timbre is
still cloned via the flow/HiFT stages. In this mode, embed the
`<|endofprompt|>` marker directly in `text` instead of `ref_text` (`ref_text`
is not sent to the LLM at all), matching the reference's
`inference_cross_lingual` usage:

```bash
# Chinese reference, English target
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --text "You are a helpful assistant.<|endofprompt|>Hello, this is a cloned voice." \
  --ref_audio examples/cosyvoice3/cross_lingual_prompt.wav \
  --cross_lingual \
  --file_prefix cross_lingual_zh2en

# English reference, Chinese target
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --text "You are a helpful assistant.<|endofprompt|>收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --ref_audio examples/voice_prompts/en_woman.wav \
  --cross_lingual \
  --file_prefix cross_lingual_en2zh
```

### Instruct-based generation (instruct2)

Port of ``CosyVoice2.inference_instruct2``. Feeds an instruction (e.g. language
or style prompt) as the LLM prompt_text while withholding the reference audio's
speech tokens from the LLM — the flow/HiFT stages still condition on
``ref_audio`` for timbre cloning.

Must contain ``<|endofprompt|>`` in the instruction text (or in
``ref_text`` for zero-shot fallback). Mutually exclusive with
``cross_lingual``.

```python
# Chinese instruct2
result = next(model.generate(
    text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    ref_audio="examples/cosyvoice3/zero_shot_prompt.wav",
    instruct="You are a helpful assistant. 用四川话说这句话<|endofprompt|>",
))

# English instruct2
result = next(model.generate(
    text="Hello, this is a cloned voice.",
    ref_audio="examples/voice_prompts/en_woman.wav",
    instruct="You are a helpful assistant. Speak in a happy and excited tone.<|endofprompt|>",
))
```

```bash
# CLI
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  --ref_audio examples/cosyvoice3/zero_shot_prompt.wav \
  --instruct "You are a helpful assistant. 用四川话说这句话<|endofprompt|>" \
  --file_prefix instruct2

# English instruct2
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --text "Hello, this is a cloned voice." \
  --ref_audio examples/voice_prompts/en_woman.wav \
  --instruct "You are a helpful assistant. Speak in a happy and excited tone.<|endofprompt|>" \
  --file_prefix instruct2_en
```

### Fine-grained control tokens

The frontend registers the same special tokens as the reference
``CosyVoice3Tokenizer``, including vocal events (``[breath]``, ``[laughter]``,
``[cough]``, etc.), emphasis markers (``<strong>``, ``</strong>``), ARPABET
phonemes (``[AH0]``), and pinyin tokens (``[nǐ]``, ``[hǎo]``) for
pronunciation hotfix. Use them directly in the input text:

```
[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点
高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评
```

### Speaker presets (spk2info)

Pre-extract and cache the acoustic prompt (speech tokens, mel, speaker
embedding) for a reference audio, then reuse it by name on subsequent
synthesis calls — skipping a full pass through the speech tokenizer,
mel extractor, and speaker encoder.

Each cached speaker stores four tensors:

| Field | Shape | Description |
|---|---|---|
| ``prompt_text`` | `(1, N)` | Tokenized reference text (includes `<|endofprompt|>`) |
| ``prompt_speech_token`` | `(1, M)` | Speech-tokenizer output from reference audio |
| ``prompt_feat`` | `(1, 80, T)` | Mel spectrogram of reference audio (24kHz) |
| ``embedding`` | `(1, 192)` | CAMPPlus speaker x-vector (voice timbre) |

A single speaker preset is ~33 KB on disk.

#### In-memory workflow

```python
model = load("mlx-community/CosyVoice3-0.5B")

# cache one or more speakers
model.add_zero_shot_spk(
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    "examples/cosyvoice3/zero_shot_prompt.wav",
    "alice",
)
model.add_zero_shot_spk(
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    "examples/cosyvoice3/zero_shot_prompt.wav",
    "bob",
)

# reuse — no speech/mel/speaker extraction needed
result = next(model.generate(
    "收到好友从远方寄来的生日礼物，那份意外的惊喜...",
    spk_id="alice",
))

# list cached speakers
print(model.list_spks())  # ['alice', 'bob']
```

#### Persist to disk (safetensors)

``save_spkinfo`` flattens ``spk2info`` into a single safetensors file where
each tensor key is ``{spk_id}.{field}`` (e.g. ``alice.embedding``).
``load_spkinfo`` reverses the process, merging the loaded entries back into
the in-memory ``spk2info`` dict.

```python
# ---- session 1: create and save ----
model.add_zero_shot_spk(
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    "examples/cosyvoice3/zero_shot_prompt.wav", "alice",
)
model.add_zero_shot_spk(
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    "examples/cosyvoice3/zero_shot_prompt.wav", "bob",
)
model.add_zero_shot_spk(
    "The radio quietly played a familiar song. Outside, rain tapped against "
    "the window in a steady rhythm. Coffee cooled slowly in a ceramic mug. "
    "Somewhere down the hall, a door clicked shut.<|endofprompt|>",
    "examples/voice_prompts/en_woman.wav", "en_speaker",
)
model.save_spkinfo("my_speakers.safetensors")   # ~99 KB for 3 speakers

# ---- session 2: load and reuse (no ref audio needed) ----
model2 = load("mlx-community/CosyVoice3-0.5B")
model2.load_spkinfo("my_speakers.safetensors")  # returns 2
print(model2.list_spks())                       # ['alice', 'bob']

result = next(model2.generate("你好世界", spk_id="alice"))
#               ↑ no ref_audio / ref_text needed
```

When the same ``spk_id`` already exists, ``load_spkinfo`` overwrites it.
Unknown keys are silently merged, so a single file can carry an arbitrary
number of speakers.

#### CLI

```bash
# Create and persist a speaker preset (no generation)
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --ref_audio examples/cosyvoice3/zero_shot_prompt.wav \
  --ref_text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
  --add_spk alice \
  --save_spkinfo my_speakers.safetensors \
  --file_prefix cached_alice

# Load presets and generate (no ref audio needed)
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --load_spkinfo my_speakers.safetensors \
  --text "你好世界" \
  --spk_id alice \
  --file_prefix loaded_alice

# Combine: add a speaker, save, and generate in one call
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --ref_audio examples/cosyvoice3/zero_shot_prompt.wav \
  --ref_text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
  --add_spk bob \
  --save_spkinfo my_speakers.safetensors \
  --text "你好世界" \
  --spk_id bob \
  --file_prefix combined_bob

# English speaker preset
python -m mlx_audio.tts.generate \
  --model mlx-community/CosyVoice3-0.5B \
  --no-strict \
  --ref_audio examples/voice_prompts/en_woman.wav \
  --ref_text "You are a helpful assistant.<|endofprompt|>The radio quietly played a familiar song. Outside, rain tapped against the window in a steady rhythm. Coffee cooled slowly in a ceramic mug. Somewhere down the hall, a door clicked shut." \
  --add_spk en_speaker \
  --save_spkinfo my_speakers.safetensors \
  --file_prefix cached_en
```

### Text normalization

By default (``text_frontend=True``), input text is normalised before synthesis,
mirroring the reference ``CosyVoiceFrontEnd.text_normalize``:

- **Chinese**: blank removal between CJK characters, corner-mark / bracket
  cleanup, period normalisation, optional wetext TN
- **English**: optional wetext TN, Arabic-numeral spelling (via ``inflect``)

Set ``text_frontend=False`` to skip normalisation (e.g. when text already
contains SSML-style control tokens).

```python
# skip text_normalize for pre-tokenized / control-token text
result = next(model.generate(
    text="[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点",
    ref_audio="examples/cosyvoice3/zero_shot_prompt.wav",
    ref_text="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    text_frontend=False,
))
```

### Converting a raw checkpoint

The real checkpoint ships as three PyTorch `.pt` files (no safetensors) plus
an HF-format Qwen2 dir and onnx assets. Convert it into the flat
`config.json` + `*.safetensors` (+ tokenizer/onnx files) layout
`mlx_audio.tts.load()` expects:

```bash
python -m mlx_audio.tts.models.cosyvoice3.convert \
  --torch-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --out /path/to/mlx-cosyvoice3
```

Then `load("/path/to/mlx-cosyvoice3", strict=False)` — `strict=False` is
required because `flow.pt`/`hift.pt` are each missing a few non-persistent
buffers not present in the checkpoint (see Status). Requires `pip install
onnx` (for the CAMPPlus speaker-encoder weight extraction; `onnxruntime` is
already a project dependency and is used at inference time for the speech
tokenizer).

## Architecture

```
text --(LLM: Qwen2)--> speech tokens --(Flow: PreLookahead + DiT CFM)--> mel --(HiFT)--> waveform
```

Note: unlike CosyVoice2 (and stepaudio2), the v3 **flow has NO conformer
encoder** — the token path is `input_embedding → PreLookaheadLayer →
repeat_interleave(x2) → DiT flow-matching decoder`.

| Stage | File | Status | Notes / reuse |
|-------|------|--------|---------------|
| LLM (speech-token AR) | `llm.py` | ✅ verified vs. real `llm.pt` (strict) | mlx_lm Qwen2 via `input_embeddings` + KV cache; `sampling.py` ras_sampling |
| Flow (token→mel) | `flow.py` | ✅ verified vs. real `flow.pt` (strict) | PreLookaheadLayer + repeat_interleave + DiT; no encoder |
| **DiT estimator (v3-only)** | `dit.py` | ✅ numerically verified vs. real `flow.pt` (max abs mel diff 0.0026 after 10 Euler steps, same inputs) | F5-TTS style: RoPE + AdaLayerNormZero(6-way) + ConvNeXtV2 |
| Flow matching (CFM) | `flow_matching.py` | ✅ implemented | Euler ODE + CFG + cosine schedule |
| HiFT vocoder | `hift.py` | ✅ verified vs. real `hift.pt` (strict=False, minus `rand_ini`/`stft_window`) | from-scratch causal port: `CausalConv1d`/`CausalConv1dDownSample`/`CausalConv1dUpsample`, causal NSF source, `CausalConvRNNF0Predictor` |
| Speech tokenizer | `tokenizer.py`, `frontend.py` | ✅ real `speech_tokenizer_v3.onnx` via onnxruntime | MLX `S3TokenizerV2` port is v2's 6-layer encoder — architecturally incompatible with v3's 12-layer checkpoint, so the real onnx file is run directly (matches upstream's own production path); `ModelConfig.use_onnx_speech_tokenizer=True` by default |
| Frontend | `frontend.py` | ✅ implemented | Qwen AutoTokenizer (registers `<\|endofprompt\|>` etc. via `add_special_tokens`, matching `CosyVoice3Tokenizer`) + CAMPPlus xvector + chatterbox mel |
| Weight conversion | `convert.py` | ✅ verified vs. real checkpoints; `convert_cosyvoice3_assets` drives end-to-end conversion | key-mapping bugs found & fixed for all three `.pt` files (see Status) |

## Status

- **Tail "snake-like" hiss mitigated with a post-hoc guard (2026-07-28)**:
  a small fraction of `generate()` calls produced an audible hissing burst
  in the final ~100ms of output (e.g. one reported case: -28.1dB RMS / 29.7%
  of tail energy above 4kHz, versus a clean tail's -60 to -110dB RMS /
  under ~5-15% HF). LLM speech-token sampling (`ras_sampling`) is
  stochastic, so the exact same `text`/`ref_audio`/`ref_text` can decode
  cleanly on one run and hiss on another. Root cause narrowed to LLM
  sampling rather than the flow/HiFT stages: HiFT's decode is fully
  deterministic given a fixed mel (`CausalSourceModuleHnNSF`'s unseeded
  noise term is computed but discarded, never reaching the waveform), and
  the CFM flow's `rand_noise` uses a fixed `mx.random.key(0)` — neither
  varies run-to-run on identical tokens. Over 100+ fresh LLM samples across
  both the low-level frontend path and the full `generate()` CLI path, none
  reproduced a tail above -62dB, confirming the bad token sequence is rare
  bad luck in autoregressive sampling, not a deterministic bug reachable on
  demand. Rather than chase the exact trigger, added
  `CosyVoice3._tame_tail_hiss` — called on every `generate()` output right
  after HiFT decode — which inspects the last 100ms of audio and, only when
  it is BOTH audible (RMS > -45dB) AND abnormally bright (>20% of FFT
  energy above 4kHz, several times a normal tail's ratio), fades the last
  60ms to silence. Verified against the originally reported hiss file
  (dropped from -28.1dB to -120dB) and against synthetic clean-tail audio
  (left untouched, bit-identical) to confirm no false-positive fades on
  normal speech.
- **"分段感/突变噪音" (segmentation-like abrupt-noise glitch) root-caused and
  fixed (2026-07-26)**: generated mel had audible glitches concentrated in
  specific time regions. Earlier cross-implementation listening tests
  (feeding the *same* real ground-truth tokens through the reference
  PyTorch `flow.inference()` + `hift.inference()`) had wrongly suggested this
  was an inherent characteristic of the checkpoint at low sampling-step
  counts, not an MLX-specific bug — that conclusion was based on audio
  *auditioning*, not numerical diffing, and was wrong. A follow-up direct
  numerical diff (same tokens/prompt-mel/speaker-embedding/noise fed to both
  MLX and a real-checkpoint-loaded PyTorch `CausalMaskedDiffWithDiT`,
  comparing raw mel tensors step-by-step through the Euler solver) found the
  two implementations diverge starting at the *first* DiT forward pass
  (before any accumulation), isolating the bug to `dit.py`'s `Attention`:
  the reference `x_transformers.apply_rotary_pos_emb` rotates q/k **before**
  splitting into attention heads, and only rotates the first `rot_dim`
  (=`dim_head`=64) channels of the flat 1024-dim (`heads=16 × dim_head=64`)
  vector — so only **head 0** ever receives a positional rotation; heads
  1–15 pass through completely unrotated (`t_unrotated` in x_transformers).
  This looks like a bug but the checkpoint was trained under exactly this
  behavior. The MLX port instead split into heads first and rotated *every*
  head independently, injecting position information into 15 heads that
  were never trained to have it — the resulting corruption compounds across
  DiT's 22 layers and surfaces as time-localized glitches in the final mel.
  Fixed `apply_rotary_pos_emb`/`Attention.__call__` to rotate the flat q/k
  before the head split, matching `x_transformers` exactly. Verified via a
  step-by-step Euler-solver diff against the real `flow.pt` (same
  tokens/prompt/embedding/noise): max abs mel diff dropped from **4.17**
  (severely diverging) to **0.0026** (float32-precision parity) after 10
  steps. Confirmed by direct listening: the glitch is gone.
- **Pitch-elevation bug fully resolved (2026-07-26)**: three independent
  real bugs were found and fixed across two sessions; combined, they take
  the end-to-end median-F0 error (10-seed, same-text self-clone vs. the real
  recording, via `librosa.yin`) from ~30% high down to **median +1.5%
  / mean +2.6% / std 7.3%** — i.e. no residual systematic bias, only
  seed-to-seed flow-matching/CFG sampling variance (comparable in magnitude
  to HiFT's own NSF-noise floor, measured separately; see below).
  1. `frontend.py::extract_prompt_feat` used
     `chatterbox.s3gen.mel.mel_spectrogram`'s default `fmax=8000`, but the
     real checkpoint's config (`cosyvoice3.yaml`'s `feat_extractor`/
     `mel_spec_transform1`) specifies `fmax: null` (full Nyquist, 12000Hz for
     24kHz audio) — HiFT's `f0_predictor`/decoder were trained on full-band
     mel, so feeding them 8kHz-band-limited mel confused the pitch
     reconstruction. Fixed by passing `fmax=None` explicitly. Verified by
     running the same mel through the *reference PyTorch* `CausalHiFTGenerator`
     with the real `hift.pt` weights: `fmax=8000` → 311Hz (biased),
     `fmax=None` → 283.6Hz (matches the real 282Hz) — proves the bug was in
     the mel fed to HiFT, not an MLX-specific HiFT/NSF porting bug.
  2. `dit.py`'s `RotaryEmbedding`/`_rotate_half` used the half-split
     ("LLaMA"/GPT-NeoX, dims `(i, i+D/2)` paired) RoPE convention, but the
     reference `x_transformers.RotaryEmbedding` the checkpoint was trained
     under uses the *interleaved* ("GPT-J", dims `(2i, 2i+1)` paired)
     convention. Since `to_q`/`to_k` are loaded verbatim from the PyTorch
     checkpoint, a mismatched pairing convention rotates the *same* learned
     weights incorrectly, corrupting the relative-position structure. Fixed
     to match x_transformers exactly (verified against its source line by
     line; float32 max error 2.38e-7). This flips the flow-model-attributable
     pitch bias sign in isolated self-reconstruction tests (bypassing the
     LLM) from +6.1% to -4.1%.
  3. The CAMPPlus speaker encoder (`frontend.py::load_speaker_encoder`) was
     **never switched to eval mode**, so its `DenseLayer`'s
     `BatchNorm(affine=False)` normalized using the current batch's own
     statistics rather than the trained running stats. `base_load_model`
     calls `model.eval()` *before* `post_load_hook` constructs the frontend
     (and its `speaker_encoder` submodule), so this module sits outside the
     `Module` parameter tree at eval-call time and never gets switched out of
     training mode on its own. With a `(batch=1, "time"=1)` input at the
     final dense layer, the batch/time-reduced variance is exactly 0, so
     `(x - mean) * rsqrt(0 + eps)` evaluates to 0 for every element — the
     entire 192-dim speaker embedding was silently zeroed. This means
     zero-shot voice cloning never received any real speaker/timbre/pitch-
     range signal from the reference audio. Fixed by calling
     `self.speaker_encoder.eval()` right after loading its weights. The
     CAMPPlus MLX port's own numerics were cross-validated against the
     original `campplus.onnx` (identical `kaldi_fbank` input, cosine
     similarity 0.999998, max abs diff 0.01) — confirming the zero-output was
     purely this eval-mode bug, not a deeper porting error.
  - Measurement methodology notes (for anyone extending this investigation):
    comparing F0 across *different-text* utterances is invalid — always use
    same-text or literal-same-target-region comparisons. HiFT's own NSF
    source injects unseeded random noise (`mx.random.normal`/`uniform` in
    `SourceModuleHnNSF`) even for identical mel input, so any bias
    measurement needs a same-seed "real mel round-tripped through HiFT"
    control subtracted out to isolate the flow model's own contribution.
- **Real `generate()` verified end-to-end (2026-07-25)** against the real
  checkpoint (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) and
  `/Users/admin/CosyVoice/asset/zero_shot_prompt.wav`, using
  `convert_cosyvoice3_assets` (see `convert.py`) to produce a local
  mlx-audio model directory, then `mlx_audio.tts.load(dir, strict=False)`
  + `model.generate(...)`. Produced ~8.8s of audio for a 218-token
  generation (RTF 0.28 on CPU/Metal), no NaNs, no full-signal clipping, and
  an amplitude envelope with speech-like pauses/bursts (not silence or
  noise). Three real bugs were found and fixed along the way:
  - `CosyVoice3.post_load_hook` never `return`ed `model` — `base_load_model`
    does `model = Model.post_load_hook(model, model_path)`, so every load
    silently produced `model = None` after weights loaded successfully.
  - `frontend.py` hardcoded the MLX `S3TokenizerV2` "speech_tokenizer_v2_25hz"
    checkpoint (6-layer encoder) for the *v3* speech tokenizer, which is
    architecturally different (12-layer encoder, per the `s3tokenizer`
    PyPI package's `ModelConfigV3`) — those weights don't match. Fixed by
    running the real `speech_tokenizer_v3.onnx` directly via onnxruntime by
    default (`ModelConfig.use_onnx_speech_tokenizer=True`), the same
    approach the reference PyTorch implementation uses in production
    (`cosyvoice/utils/onnx.py::SpeechTokenExtractor`).
  - `frontend.py::encode_text` never registered `<|endofprompt|>` (id
    151646) as a special token — `AutoTokenizer.from_pretrained` alone
    splits it into 7 sub-word pieces. Fixed by calling
    `tokenizer.add_special_tokens(...)` at load time, matching the
    reference `CosyVoice3Tokenizer`.
  - Conversion detail: `convert_flow_weights`'s Conv1d transpose is not
    idempotent, so `convert_cosyvoice3_assets` saves *raw* (unsanitized)
    torch weights — `base_load_model` calls `model.sanitize()` itself at
    load time; sanitizing twice would double-transpose shapes.
- **Weight conversion verified against the real checkpoint**
  (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, confirmed 2026-07-24):
  - `llm.pt`: `load_weights(strict=True)` — 0 missing/extra/mismatched keys.
    Fixed: `tie_word_embeddings=True` drops `llm.model.lm_head.weight`
    (bit-identical to `embed_tokens.weight`; mlx_lm's tied Qwen2 has no
    separate `lm_head` param).
  - `flow.pt`: `load_weights(strict=False)` — only `decoder.rand_noise`
    missing (non-persistent buffer, re-seeded per process; not in the
    checkpoint). Fixed: PyTorch `nn.Sequential` index gaps around
    GELU/Mish/SiLU (`ff.ff.0.0.*`/`ff.ff.2.*`, `conv_pos_embed.conv{1,2}.0.*`,
    `time_mlp.2.*`) collapsed to this module's flat parameter-list indices;
    `rotary_embed.inv_freq` dropped (recomputed on demand, verified
    bit-for-bit identical).
  - `hift.pt`: `load_weights(strict=False)` — only `m_source.l_sin_gen.rand_ini`
    and `stft_window` missing (both non-persistent buffers). Fixed by
    rewriting `hift.py` from scratch as the causal generator (see below);
    ran a real forward pass with the loaded weights (mel → waveform, no
    NaNs, correctly clipped to `[-0.99, 0.99]`).
- **Not yet validated**: absolute numerical parity against a captured
  PyTorch reference run (layer-by-layer tensor diffing). Speech tokenizer
  correctness now rests on the real onnx graph (not an MLX port), so this
  is lower-risk than before, but audio *quality* (does it sound like the
  cloned speaker, is prosody correct) hasn't been human-evaluated yet.

### HiFT causal rewrite (2026-07-24)

The previous `hift.py` reused the non-causal `chatterbox.s3gen.hifigan.HiFTGenerator`
(`ConvTranspose1d` upsampling, symmetric-padding `Conv1d`), which is a
**different architecture** from the real v3 checkpoint's
`CausalHiFTGenerator` — the checkpoint's `ups.*` conv weights are laid out
like plain `Conv1d` (out, in, k), not `ConvTranspose1d` (in, out, k), and
`f0_predictor` uses a 5-layer causal conv stack, not `ConvRNNF0Predictor`.
`hift.py` is now a standalone causal implementation:
  - `CausalConv1d` (left/right causal padding, `causal_padding = dilation *
    (kernel_size - 1)`), `CausalConv1dDownSample` (strided, left-padded,
    `causal_padding = stride - 1`), `CausalConv1dUpsample` (nearest-neighbor
    upsample + left-padded plain conv, replacing `ConvTranspose1d`,
    `causal_padding = kernel_size - 1`).
  - `ResBlock` uses `CausalConv1d(causal_type='left')` throughout.
  - `CausalConvRNNF0Predictor`: `condnet[0]` right-context (kernel 4),
    `condnet[1:]` left-context (kernel 3); checkpoint's
    `condnet.{0,2,4,6,8}` (PyTorch `nn.Sequential` interleaving `ELU`)
    collapse to flat indices `{0,1,2,3,4}`.
  - `CausalSineGen`: downsample-by-`1/upsample_scale` (linear) → cumsum →
    nearest-neighbor upsample (vs. linear in the non-causal path), so no
    future sample leaks backwards.
  - DSP primitives unchanged from the reference between causal/non-causal
    generators (Hann window, pure-MLX STFT/ISTFT, Snake activation) are
    reused directly from `chatterbox.s3gen.hifigan`.
  - Only the whole-utterance (`finalize=True`) path is implemented; chunked
    streaming with left-context caches (`finalize=False` in the reference)
    is a TODO (see Remaining work).

### Remaining work

1. Numerical alignment: DiT's RoPE mismatch is now fixed and verified
   (max abs mel diff 0.0026 vs. real `flow.pt` after 10 Euler steps, see
   Status) — HiFT and the LLM still have no direct tensor-value diff against
   a captured PyTorch reference run.
2. Native MLX port of `speech_tokenizer_v3.onnx` weights (currently run via
   onnxruntime directly — correct, but not "pure MLX"). `s3tokenizer`'s
   `onnx2torch_v3` (PyPI package) already does the graph-name mapping;
   `S3TokenizerV2` in `codec/models/s3/model_v2.py` can be instantiated with
   `n_audio_layer=12` to match v3's architecture — only the weight-loading
   path needs writing.
3. Human-evaluate audio quality (voice similarity, prosody) beyond the
   automated sanity checks (no NaNs/clipping, speech-like envelope) already
   run.
4. Streaming: causal chunk masks in DiT (`static_chunk_size`) + chunked
   causal-cache inference in HiFT (`conv_pre_look_right`,
   `CausalConv1d`/`CausalConv1dUpsample`/`CausalConv1dDownSample`'s
   `cache` arguments, `finalize=False`).
5. Text normalization (zh/en) if inputs aren't pre-normalized.

## Reference

- PyTorch source (upstream): `https://github.com/FunAudioLLM/CosyVoice`
- Local checkout used for this port: `/Users/admin/CosyVoice/cosyvoice`
- Config: `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`
  (DiT `dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2`)
- Weights (ModelScope): `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` — `llm.pt`,
  `flow.pt`, `hift.pt`, a HF-format Qwen2 dir (`CosyVoice-BlankEN/`),
  `speech_tokenizer_v3.onnx`, `campplus.onnx`, `cosyvoice3.yaml`.
  Local copy: `/Users/admin/.cache/modelscope/models/FunAudioLLM--Fun-CosyVoice3-0.5B-2512`.
- Key classes: `CosyVoice3LM`, `CausalMaskedDiffWithDiT`, `CausalHiFTGenerator`


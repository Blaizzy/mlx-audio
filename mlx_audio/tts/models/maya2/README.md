# Maya2 Native

[Maya Research's Maya2 Native](https://huggingface.co/maya-research/maya2-native)
generates 24 kHz mono speech with 20 named voices, streaming, and voice cloning.

Supported language codes: `as`, `bn`, `en`, `gu`, `hi`, `kn`, `ml`, `mr`, `or`,
`pa`, `ta`, `te`.

## Generate speech

```python
from mlx_audio.audio_io import write
from mlx_audio.tts.utils import load

model = load("maya-research/maya2-native")
result = next(model.generate(
    "नमस्ते, आप कैसे हैं?", voice="Ahana", lang_code="hi"
))
write("speech.wav", result.audio, result.sample_rate)
```

```bash
python -m mlx_audio.tts.generate \
    --model maya-research/maya2-native \
    --text "Hello from Maya." --voice Siddhant --lang_code en
```

Voice names are case-insensitive. `model.voices` lists all 20; the default is
`Ahana`. Pass only the speech text, without adding a voice name to it.
Additional audio models download automatically on first use.

## Streaming

```python
for chunk in model.generate(
    "Hello from Maya.", voice="Siddhant", lang_code="en",
    stream=True, streaming_interval=0.32,
):
    play(chunk.audio, chunk.sample_rate)  # Supply your playback function.
```

`streaming_interval` controls chunk duration in seconds. The final chunk has
`is_final_chunk=True`. Close the generator if you stop playback early. Use one
active generation per model instance.

## Voice cloning

```python
result = next(model.generate(
    "வணக்கம், எப்படி இருக்கிறீர்கள்?",
    lang_code="ta",
    ref_audio="reference.wav",
    ref_text="The exact transcript of the reference recording.",
    ref_lang="en",
))
```

Use a clean 3–10 second reference from one speaker, with permission to clone it.
Audio files are converted to mono at 24 kHz automatically; array inputs must
already use that format. Provide the exact reference transcript in `ref_text`.
`ref_lang` specifies the reference language and defaults to `lang_code`.
The reference and output languages can differ. Omit `voice` when supplying
reference audio.

The default `conditioning="icl_spk"` uses both the reference speech and speaker
identity. `"icl"` uses the reference speech alone. `"spk"` uses speaker identity
alone and does not require `ref_text`.

## Text normalization

Text normalization is enabled by default in all 12 languages. It expands numbers,
decimals, currency amounts, dates, times, percentages, and common identifiers
such as phone numbers, account numbers, OTPs, email addresses, and URLs.
`lang_code` selects the language used for these spoken forms.

```python
result = next(model.generate(
    "Your OTP is 482913 and your balance is ₹95.41.", lang_code="en"
))

result = next(model.generate(
    "Text already prepared for speech.", lang_code="en", normalize=False
))
```

Numeric dates use day/month/year; ISO `YYYY-MM-DD` and English or native month
names are also supported. Two-digit years `00`–`50` mean 2000–2050, and `51`–`99`
mean 1951–1999. Invalid calendar dates and times are left unchanged.

Leading zeros and long digit strings are read digit by digit. Label a short
identifier with a word such as `PIN`, `OTP`, or `account`; an unlabelled six-digit
value is read as a quantity. Inline tags such as `[laughs]` and `[pause 0.5s]` are
preserved. Reference transcripts supplied in `ref_text` are kept as written.

## Generation options

- `max_tokens` limits output length in 80 ms units; for example, `100` allows up
  to eight seconds of speech. The default is `1500` (two minutes).
- `temperature` defaults to `0.9` and `top_k` to `64`. Set `temperature=0` and
  `depth_temperature=0` for greedy decoding.
- Speed is fixed at `1.0`; use punctuation to guide pacing.

Split long passages into shorter requests.

## Licenses and attribution

Maya Research's [Maya2 weights](https://huggingface.co/maya-research/maya2-native)
and [reference implementation](https://github.com/MayaResearch/maya2-native) are
Apache-2.0.
Normalization rules and pronunciation tables are adapted from Maya Research's
Apache-2.0 normalizer. Assamese number data includes
[AI4Bhārat's indic-numtowords](https://github.com/AI4Bharat/indic-numtowords),
Copyright (c) 2022 AI4Bhārat, under the
[MIT license](https://github.com/AI4Bharat/indic-numtowords/blob/master/LICENSE).
[Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) is Apache-2.0.
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) codec weights are CC-BY-4.0;
additional codec weights are available from
[kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16).
[SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
and its [MLX export](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX)
are Apache-2.0. Preserve the respective notices and attribution when distributing
these components. MLX-Audio retains its repository license.

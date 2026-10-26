# Nemotron Transcribe: six audio tasks

This example runs transcription, speaker attribution, speech translation, audio
questions, spoken instructions, and rich transcription. It creates short,
**synthetic** audio inputs with [Phonon English](https://huggingface.co/gradium/phonon-31m-en)
through MLX Audio, sends the audio to Nemotron, and saves
the model's actual text responses. Source text is used to synthesize the audio;
it is not passed to Nemotron as a transcript.

The optional spoken reply uses a separate text-to-speech stage. Nemotron itself
produces text.

## Run all examples

From the repository root, install MLX Audio and the Phonon loader's `requests`
dependency, then run the demo:

```bash
python -m pip install -e '.[stt]' requests

python examples/nemotron_transcribe/run.py \
  --model "$HOME/Models/Nemotron-3.5-Transcribe-30B-A4B-EA-0.1.0" \
  --tts-model gradium/phonon-31m-en \
  --tts-revision 92741c5b07d085ff840e26fea8ce43110b87852f \
  --output-dir "$HOME/Models/nemotron-transcribe-phonon-examples" \
  --quantize --speak-answer
```

`--quantize` converts the original checkpoint's language model to 4-bit in
memory. Omit it for BF16 or an already quantized checkpoint. Original-checkpoint
loading occurs before quantization, so allow memory for the full model.
`--speak-answer` creates `spoken_answer.wav` with Phonon; omit
it for text output only.

Phonon generates the input recordings first, at 24 kHz. Nemotron's file loader
resamples them to 16 kHz. The two-speaker example joins the voices after
resampling, with 0.5 seconds of silence between turns. Phonon generates the final
spoken reply from Nemotron's returned text.

The default voices are Zoey and Easton from the official
[Phonon voice catalogue](https://huggingface.co/gradium/phonon-voices), using pinned
voice embeddings. Set `--voice`, `--second-voice`, or `--rich-voice` to another
local embedding or `hf://` URL. `--seed` controls Phonon sampling (default `0`);
the report records each input's seed and the resolved TTS checkpoint path.

The script writes the audio inputs, one `.txt` per task, and `results.json`
containing the input provenance, exact user prompts, generation options, and
model outputs. Rerunning replaces these generated demo files. The timings in
that report are single example runs, not a performance benchmark.

To supply your own recordings, pass `--audio`, `--conversation-audio`,
`--spoken-audio`, and `--rich-audio`. Set `--question` to match the content of your
`--audio`, and optionally set `--language` to the desired translation language.
Missing inputs are synthesized automatically using Phonon. Supplying all four
inputs and omitting `--speak-answer` skips loading the TTS model.

## 1. Transcription

The announcement says: "Tomorrow's design review starts at ten thirty in room
four. Please bring the updated prototype."

```python
result = model.generate("announcement.wav", task="asr")
print(result.text)
```

Recorded output:

```text
Tomorrow's design review starts at ten thirty in room four. Please bring the updated prototype.
```

## 2. Speaker attribution

The two-voice recording contains the announcement followed by another voice
saying: "I will bring the prototype and the latest test results."

```python
result = model.generate("conversation.wav", task="saa")
print(result.text)
print(result.segments)  # Speaker IDs and text; no timestamps.
```

Recorded output, with line breaks added between speakers:

```text
<spk:0> Tomorrow's design review starts at 10:30 in room 4. Please bring the updated prototype.
<spk:1> I will bring the prototype and the latest test results.
<spk:0> Okay.
```

Speaker IDs distinguish voices within the recording; they do not identify people
by name.

This run added a final "Okay." that is absent from the Phonon synthesis text.
Plain transcription of each individual clip and of the joined recording did
not include it. The example preserves that extra output so the result is not
mistaken for a perfect transcript.

## 3. Speech translation

Use the same English announcement as the audio input:

```python
result = model.generate("announcement.wav", task="translate", language="German")
print(result.text)
```

Recorded output:

```text
Die Designprüfung für morgen beginnt um 10:30 Uhr im Raum 4. Bitte bringen Sie den aktualisierten Prototyp mit.
```

## 4. Questions about audio

Attach the announcement and a written question in the same request:

```python
result = model.generate(
    "announcement.wav",
    prompt="What time is the design review, and what should attendees bring?",
)
print(result.text)
```

Recorded output:

```text
The design review starts at ten thirty in room four, and attendees should bring the updated prototype.
```

## 5. Spoken instructions

The audio itself asks: "What is twelve multiplied by seven? Answer with just the
number." No text version of that question is sent to the model. `task="qa"`
supplies only the audio placeholder as the user prompt. The system instruction
selects answering rather than transcription:

```python
result = model.generate(
    "spoken_request.wav",
    task="qa",
    system_prompt=(
        "You are a voice-controlled assistant. Treat the speech in the provided "
        "audio as the user's instruction and carry out that request. Respond to "
        "the spoken request rather than merely transcribing it, unless "
        "transcription is explicitly requested. Give a direct, accurate answer "
        "to the spoken request."
    ),
)
print(result.text)
```

Recorded output:

```text
84
```

With `--speak-answer`, the runner sends this returned text to Phonon and saves
the resulting speech separately as `spoken_answer.wav`.

## 6. Rich transcription

Phonon speaks "We have wonderful news! The project is finished!" using Zoey.
Rich mode requests event and delivery tags:

```python
result = model.generate("rich_speech.wav", task="rich")
print(result.text)
```

Recorded output:

```text
We have wonderful news! The project is finished!
```

The Phonon sample produced a plain transcript with no event or style tags. Rich
mode can emit `<sty:stt:...>` and `<sty:end:...>` around a speaking-style span, or
non-speech event tokens such as `<ev:laughter>`, but it does not force tags onto
every recording. Use `--rich-audio` with an expressive recording to examine those
outputs. This sample does not validate laughter or style detection.

## Validation scope

The recorded examples were run with the original v0.1.0 checkpoint, 4-bit
language-model quantization, greedy decoding, and MLX 0.32.4 on an M3 Ultra.
Phonon used revision `92741c5b07d085ff840e26fea8ce43110b87852f`, the pinned Zoey and
Easton embeddings, and base seed `0`. All six tasks and the Phonon spoken reply
ran end to end. A follow-up transcription of the spoken reply returned "eighty
four." The extra speaker-attributed word and missing rich tags are recorded
above. These are demonstrations using synthetic voices, not an accuracy
evaluation.

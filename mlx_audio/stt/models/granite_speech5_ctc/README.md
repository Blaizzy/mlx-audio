# Granite Speech 5.0 TurboCTC

MLX support for IBM's encoder-only Granite Speech 5.0 TurboCTC models. These
models perform English speech recognition with a Conformer encoder and greedy
CTC decoding; they do not include the language-model translation and prompting
features of earlier Granite Speech releases.

## Models

| Model | License | Tokenizer |
|---|---|---|
| [ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | Apache 2.0 | BPE |
| [ibm-granite/granite-speech-5.0-470m-turboctc-nc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc) | CC-BY-NC-SA-4.0 | SentencePiece-derived BPE |

The `-nc` checkpoint is restricted to non-commercial use. Review its license
before using it.

## Python

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-speech-5.0-470m-turboctc")
result = model.generate("audio.wav")
print(result.text)
```

## CLI

```bash
python -m mlx_audio.stt.generate \
  --model ibm-granite/granite-speech-5.0-470m-turboctc \
  --audio audio.wav \
  --output-path transcript
```

Audio files are resampled to mono 16 kHz. Raw NumPy and MLX arrays are assumed
to already contain mono 16 kHz samples.

## Architecture

- 80-bin log-mel features concatenated with deltas and stacked in pairs
- 16 Conformer blocks with 128-frame block-local attention
- stride-2 temporal subsampling in blocks 0 and 1 (8x total with frame stacking)
- self-conditioned CTC at the midpoint and a tied 16,384-token CTC head

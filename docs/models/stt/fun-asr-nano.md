---
title: Fun-ASR-Nano
---

# Fun-ASR-Nano

Fun-ASR-Nano-2512 is FunAudioLLM's compact ASR model with a SenseVoice-style audio encoder, Qwen3-0.6B text decoder, hotword prompting, and language-conditioned transcription.

## Convert

The upstream release ships PyTorch weights in `model.pt`, so convert it with the model-specific converter:

```bash
python -m mlx_audio.stt.models.fun_asr_nano.convert \
  --hf-path FunAudioLLM/Fun-ASR-Nano-2512 \
  --mlx-path Fun-ASR-Nano-2512-mlx
```

The converted directory is directly uploadable to Hugging Face as `mlx-community/Fun-ASR-Nano-2512`.

## Usage

=== "Python"

    ```python
    from mlx_audio.stt.utils import load_model

    model = load_model("mlx-community/Fun-ASR-Nano-2512")
    result = model.generate("audio.wav", language="zh", hotwords=["开放时间"])
    print(result.text)
    ```

=== "CLI"

    ```bash
    mlx_audio.stt.generate \
      --model mlx-community/Fun-ASR-Nano-2512 \
      --audio audio.wav \
      --output-path transcript \
      --language zh \
      --context "开放时间, 地址"
    ```

## Options

`language` accepts ISO hints for the current Nano checkpoint: `zh`, Chinese dialect codes that map to the Chinese prompt (`yue`, `wuu`, `nan`, `hak`, `gan`, `hsn`, `cjy`), `en`, and `ja`. Pass `None` or `auto` to omit the language constraint.

`hotwords` accepts a list of domain terms to include in the upstream prompt.
The shared CLI and server `context` string is supported as an alias, so the same
contextual biasing works through `mlx_audio.stt.generate --context ...` and
`POST /v1/audio/transcriptions`. Non-empty `hotwords` and `context` values are
mutually exclusive.

`itn=False` asks the model to skip inverse text normalization.

## Boundaries

The MLX implementation is transcription-only. The public `FunAudioLLM/Fun-ASR-Nano-2512` checkpoint does not include timestamp head weights, so this model does not emit word timestamps.

VAD and diarization remain external, so split or diarize audio before calling this model when those features are required.

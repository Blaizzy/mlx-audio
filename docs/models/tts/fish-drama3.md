# Fish Audio Drama 3

Fish Audio's multi-speaker text-to-speech model for scenes and long-form content. It pairs a 35B MoE backbone (3B active per token) with a small audio tower that predicts the ten codec codebooks, and decodes with the Fish S1 DAC codec at 44.1 kHz. Based on [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3).

## Converting the checkpoint

The release stores its routed experts as block FP8, so convert it once into an MLX checkpoint:

```bash
hf download fishaudio/drama-3 --local-dir drama-3
python -m mlx_audio.tts.models.fish_drama3.convert \
    --src drama-3 --dst drama-3-mlx
```

By default every weight stays in bfloat16, so the numerics match the reference implementation; that comes to about 67 GB. `--expert-bits`, `--dense-bits` and `--tower-bits` quantize the routed experts, the dense projections and the audio tower. Reading `codec.pth` needs PyTorch.

[`mlx-community/fish-audio-drama-3-4bit`](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit) is converted with `--expert-bits 4 --dense-bits 8 --tower-bits 8`, about 22 GB. On an M3 Max it generates a frame (46 ms of audio) in about 22 ms, roughly twice real time, and peaks at about 26 GB of memory. The bfloat16 checkpoint runs at about 1.4 times real time and peaks at about 74 GB. [`mlx-community/fish-audio-drama-3-8bit`](https://huggingface.co/mlx-community/fish-audio-drama-3-8bit) is converted with `--expert-bits 8` alone, so everything but the routed experts stays in bfloat16: about 42 GB, peaking at about 49 GB, which fits a 64 GB Mac.

### Choosing expert bits

The routed experts hold most of the weights, so `--expert-bits` sets the size. The sizes below also pass `--dense-bits 8 --tower-bits 8`. Keep the dense projections at 8 bits or more: quantizing them to 3 bits as well made 9 of 20 test clips fail.

| `--expert-bits` | Checkpoint | Quality vs 8-bit experts |
|---|---|---|
| 4 | ~22 GB | Same |
| 3 | ~18.5 GB | Same; peaks at about 22.5–24 GB, so it fits a 32 GB Mac ([converted checkpoint](https://huggingface.co/mlx-community/fish-audio-drama-3-3bit)) |
| 2 | ~14.5 GB | Not recommended: dropped the end of a multi-speaker dialogue |

Measured on 20 voice-cloned English clips (Whisper WER, speaker similarity, UTMOS) and a blind listening test.

Without `--dense-bits` and `--tower-bits`, the dense projections and the audio tower stay in bfloat16. With 4-bit experts that brings the codebook heads slightly closer to the PyTorch reference, but needs about 2.3 GB more peak memory and runs about a quarter slower on an M3 Max.

Three more options give finer control. They only change how the checkpoint is written; the result loads like any other.

- `--head-bits` quantizes the tower's nine codebook heads separately from the rest of the tower. It defaults to `--tower-bits`; `0` keeps the heads in bfloat16.
- `--expert-group-size` sets the experts' quantization group size: 32, 64 (the default) or 128.
- `--expert-bits-map` takes a JSON file that maps layer index to expert bits, such as `{"38": 4, "39": 4}`. Layers it does not list use `--expert-bits`.

For example, [`fcmeyer/drama-3-MLX-3bit-HQ`](https://huggingface.co/fcmeyer/drama-3-MLX-3bit-HQ) uses `--expert-bits 3 --dense-bits 8 --tower-bits 8 --head-bits 0` with a map that keeps the experts of 15 layers at 4 bits.

## Usage

=== "CLI"

    ```bash
    python -m mlx_audio.tts.generate \
        --model mlx-community/fish-audio-drama-3-4bit \
        --text "[sighs] Fine. If we miss it, it's on you." \
        --ref_audio reference.wav \
        --ref_text "Transcript of the reference clip."
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/fish-audio-drama-3-4bit")

    for result in model.generate(
        text="[sighs] Fine. If we miss it, it's on you.",
        ref_audio="reference.wav",
        ref_text="Transcript of the reference clip.",
    ):
        print(result.audio_duration)
    ```

The output copies the bandwidth of the reference, so use references recorded at 44.1 kHz or higher; a 16 kHz clip gives duller speech. Without a reference the model picks a voice itself. Delivery cues such as `[whispers]` or `[laughs]` go inline where they should happen. Sampling arguments left unset use SGLang-Omni's defaults, in Python, the CLI and the server: temperature 1.0 and top-p 0.9, with no top-k cap on the main token and a cap of 30 on the codebook heads. A top-k cap on the main token makes takes without a reference fall silent more often. A `top_k` you pass caps the main token, and the codebook heads at most 30.

## Multiple speakers

Speaker tags `<|speaker:0|>` to `<|speaker:15|>` switch characters inside one text. Untagged text is speaker 0. To give each speaker a voice, pass one reference per speaker and start each transcript with that speaker's tag:

```python
results = model.generate(
    text="<|speaker:0|> Are you coming? <|speaker:1|> Give me a minute.",
    ref_audio=["anna.wav", "ben.wav"],
    ref_text=[
        "<|speaker:0|> Transcript of Anna's clip.",
        "<|speaker:1|> Transcript of Ben's clip.",
    ],
)
```

Long text is split into chunks of about 300 bytes with at most three speakers each. Every chunk sees the first and the previous chunk as context, so voices stay consistent. The encoded references and their prompt prefix are cached for the four most recent voices.

## Streaming

With `stream=True`, audio is yielded every `streaming_interval` seconds while the frames are sampled. `first_chunk_frames` makes the first piece smaller for a faster first audio:

```python
from mlx_audio.tts.audio_player import AudioPlayer

player = AudioPlayer(sample_rate=model.sample_rate)
for result in model.generate(
    text="Streaming speech from Drama 3.",
    ref_audio="reference.wav",
    ref_text="Transcript of the reference clip.",
    stream=True,
    streaming_interval=0.32,
    first_chunk_frames=4,
):
    player.queue_audio(result.audio)
player.wait_for_drain()
```

With a cached voice and `first_chunk_frames=4`, the first audio arrives in under 200 ms on an M3 Max.

## License

The Drama 3 weights are released under the Fish Audio Community License. Check its terms before commercial use or hosting.

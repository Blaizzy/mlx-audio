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

[`mlx-community/fish-audio-drama-3-4bit`](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit) is converted with `--expert-bits 4 --dense-bits 8 --tower-bits 8`, about 22 GB. On an M3 Max it generates a frame (46 ms of audio) in about 22 ms, roughly twice real time, and peaks at about 26 GB of memory. The bfloat16 checkpoint runs at about 1.4 times real time and peaks at about 74 GB.

### Choosing expert bits

The routed experts hold most of the weights, so `--expert-bits` sets the size. Keep the dense projections at 8 bits or more: quantizing them to 3 bits as well made 9 of 20 test clips fail.

| `--expert-bits` | Checkpoint | Quality vs 8-bit experts |
|---|---|---|
| 4 | ~22 GB | Same |
| 3 | ~18.5 GB | Same; peaks at about 22.5–24 GB, so it fits a 32 GB Mac ([converted checkpoint](https://huggingface.co/Nav8ha/drama-3-mlx-mixed-3bit)) |
| 2 | ~14.5 GB | Not recommended: dropped the end of a multi-speaker dialogue |

Measured on 20 voice-cloned English clips (Whisper WER, speaker similarity, UTMOS) and a blind listening test.

Without `--dense-bits` and `--tower-bits`, the dense projections and the audio tower stay in bfloat16. With 4-bit experts that brings the codebook heads slightly closer to the PyTorch reference, but needs about 2.3 GB more peak memory and runs about a quarter slower on an M3 Max.

Three more options give finer control. They only change how the checkpoint is written; the result loads like any other.

- `--head-bits` quantizes the tower's nine codebook heads separately from the rest of the tower. It defaults to `--tower-bits`; `0` keeps the heads in bfloat16.
- `--expert-group-size` sets the experts' quantization group size: 32, 64 (the default) or 128.
- `--expert-bits-map` takes a JSON file that maps layer index to expert bits, such as `{"38": 4, "39": 4}`. Layers it does not list use `--expert-bits`.

For example, [`fcmeyer/drama-3-MLX-3bit-HQ`](https://huggingface.co/fcmeyer/drama-3-MLX-3bit-HQ) uses `--expert-bits 3 --dense-bits 8 --tower-bits 8 --head-bits 0` with a map that keeps the experts of 15 layers at 4 bits.

### Comparing quantized checkpoints

WER barely separates the quantized checkpoints, because it sits near Whisper's floor for all of them. Teacher-forced KL to the bfloat16 model does separate them. To measure it, run each checkpoint on the same bfloat16-generated sequences and compare its next-token distributions with the bfloat16 ones. The table compares the published checkpoints and a few stock recipes, all run on one harness:

| Recipe | Checkpoint | Weights | Main KL | Head KL | Failures | Peak memory | Speed |
|---|---|---|---|---|---|---|---|
| (none) | bfloat16 reference | 70.8 GB | 0 | 0 | 4.0% | 84.9 GB | 2.0× |
| `--expert-bits 8` | | 40.6 GB | 0.0046 | 0.0025 | 6.2% | 54.7 GB | 2.1× |
| `--expert-bits 6` | | 32.5 GB | 0.0062 | 0.0032 | 7.2% | 46.6 GB | 2.0× |
| `--expert-bits 5 --dense-bits 8 --tower-bits 8 --head-bits 0` | [fcmeyer/drama-3-MLX-5bit](https://huggingface.co/fcmeyer/drama-3-MLX-5bit) | 26.3 GB | 0.0094 | 0.0059 | 3.8% | 40.4 GB | 2.8× |
| `--expert-bits 4` | [fcmeyer/drama-3-MLX-4bit-HQ](https://huggingface.co/fcmeyer/drama-3-MLX-4bit-HQ) | 24.5 GB | 0.0190 | 0.0082 | 4.6% | 38.6 GB | 2.3× |
| `--expert-bits 4 --dense-bits 8 --tower-bits 8` | [mlx-community/fish-audio-drama-3-4bit](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit) | 22.2 GB | 0.0204 | 0.0101 | 4.4% | 36.3 GB | 3.0× |
| `--expert-bits 3 --dense-bits 8 --tower-bits 8 --head-bits 0` with a 15-layer `--expert-bits-map` | [fcmeyer/drama-3-MLX-3bit-HQ](https://huggingface.co/fcmeyer/drama-3-MLX-3bit-HQ) | 19.7 GB | 0.0352 | 0.0146 | 3.6% | 33.8 GB | 2.7× |
| `--expert-bits 3 --dense-bits 8 --tower-bits 8` | [Nav8ha/drama-3-mlx-mixed-3bit](https://huggingface.co/Nav8ha/drama-3-mlx-mixed-3bit) | 18.2 GB | 0.0660 | 0.0271 | 3.0% | 32.2 GB | 3.1× |

- **Main KL** compares the distributions for the semantic token, and **Head KL** averages the comparison over the nine codebook heads. Lower is closer to bfloat16. The KLs are averaged over 64 sequences, and every pair of neighbouring rows differs with a paired 95% confidence interval that excludes zero.
- **Failures** counts free-running clips that stall, ramble to the length limit or drop text, out of 500 (100 prompts × 5 seeds, in six languages, with dialogue and cues). This happens at every precision, bfloat16 included, and the differences between rows are within noise.
- **Peak memory** and **Speed** (audio seconds per second) were measured over the whole run of 500 clips on an M5 Max. A single generation peaks lower.
- **Sampling:** temperature 1.0, top-p 0.9, top-k 30, no reference voice.

Keeping the codebook heads in bfloat16 lowers head KL at little cost, and so does keeping the more sensitive late layers' experts at 4 bits. The harness and per-clip results are in the [`eval/`](https://huggingface.co/fcmeyer/drama-3-MLX-3bit-HQ/tree/main/eval) folder of the checkpoints above.

## Usage

=== "CLI"

    ```bash
    python -m mlx_audio.tts.generate \
        --model mlx-community/fish-audio-drama-3-4bit \
        --text "[sighs] Fine. If we miss it, it's on you." \
        --ref_audio reference.wav \
        --ref_text "Transcript of the reference clip." \
        --temperature 0.7 --top_p 0.7 --top_k 30
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

The output copies the bandwidth of the reference, so use references recorded at 44.1 kHz or higher; a 16 kHz clip gives duller speech. Without a reference the model picks a voice itself. Delivery cues such as `[whispers]` or `[laughs]` go inline where they should happen. In Python, sampling arguments left unset use the reference recipe (temperature 0.7, top-p 0.7, top-k 30); the CLI has its own defaults, so pass them explicitly there.

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

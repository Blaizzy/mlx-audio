# MiMo-Audio-Tokenizer

Encoder, residual vector quantizer, decoder, and Transformer-Vocos vocoder for [XiaomiMiMo/MiMo-Audio-Tokenizer](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer).

```python
from mlx_audio.codec import MiMoAudioTokenizer
from mlx_audio.audio_io import write

codec = MiMoAudioTokenizer.from_pretrained("XiaomiMiMo/MiMo-Audio-Tokenizer")
codes = codec.encode("input.wav")             # (20, frames), integers
reconstructed = codec.decode(codes)           # mono MLX array
write("reconstructed.wav", reconstructed, codec.sample_rate)

# MiMo-Audio consumes the first eight codebooks.
speech_codes = codec.encode("input.wav", num_quantizers=8)
speech = codec.decode(speech_codes)
```

The codec uses 24 kHz mono audio at approximately 25 code frames per second. All 20 codebooks are supported: the first two have 1,024 entries each and the remaining 18 have 128 entries each. Any nonempty prefix of codebooks can be decoded. Codebook padding IDs used by the language model must not be passed to the codec.

`encode` accepts an audio path or a waveform array. Arrays require `sample_rate` and use shape `(samples,)` or `(samples, channels)`; multichannel audio is averaged to mono, then resampled. `encode_mels` accepts the model's log magnitude mel features shaped `(frames, 128)` for callers that already have the exact frontend output.

The frontend uses a 960-sample periodic Hann window, a 240-sample hop, 128 HTK mel bands without filter normalization, reflect centering, magnitude power 1, and a `1e-7` log floor. Encoding uses up to 6,000 mel frames per segment; decoding uses up to 1,500 codec frames per segment. Segment boundaries follow the reference offline processing and may affect boundary audio. There is no streaming cache.

Decoded length is `frames * 960` samples. Use `decode(codes, length=original_length)` to trim framing padding when reconstructing a known waveform; use the original length at 24 kHz. Very short nonempty clips are extended enough to compute the frontend safely.

Original and converted MLX checkpoints both load directly. The default inference dtype is BF16, with RVQ computation in FP32, matching the reference wrapper's casting order. Use `dtype=None` to retain stored precision or `dtype=mx.float32` for full-precision validation. The standalone codec is not quantized.

Model card: [MiMo-Audio-Tokenizer](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer). Model weights license: MIT. [Reference code](https://github.com/XiaomiMiMo/MiMo-Audio) license: Apache-2.0.

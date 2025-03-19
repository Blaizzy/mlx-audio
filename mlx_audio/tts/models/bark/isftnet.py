import mlx.core as mx
import mlx.nn as nn


# Loads to torch Encodec model
def codec_decode(codec: nn.Module, fine_tokens: mx.array):
    arr = fine_tokens.astype(mx.int32)[None]
    emb = codec.quantizer.decode(arr)
    out = codec.decoder(emb).astype(mx.float32)
    audio_arr = mx.squeeze(out, -1)
    del arr, emb, out
    return audio_arr

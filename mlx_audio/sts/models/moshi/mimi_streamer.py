import mlx.core as mx
import numpy as np


class StreamTokenizer:
    def __init__(self, mimi_model):
        self.model = mimi_model
        from mlx_audio.codec.models.mimi.mimi import MimiStreamingDecoder

        self.decoder = MimiStreamingDecoder(self.model)
        self.encode_buffer = []
        self.decode_buffer = []

    def decode(self, tokens_np):
        # tokens_np is expected to be [codebooks]
        # model expects [1, codebooks, 1]
        tokens_mx = mx.array(tokens_np).reshape(1, -1, 1)
        # the model expects float32/int inputs?
        pcm = self.decoder.decode_frames(tokens_mx)
        mx.eval(pcm)

        # pcm is [batch, channels, time]
        pcm_np = np.array(pcm)[0, 0, :]
        self.decode_buffer.append(pcm_np)

    def get_decoded(self):
        if len(self.decode_buffer) > 0:
            return self.decode_buffer.pop(0)
        return None

    def encode(self, pcm_np):
        pass

    def get_encoded(self):
        pass

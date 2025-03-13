from encodec import EncodecModel
import mlx.core as mx
import torch
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings


def _load_codec_model(device):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    return model


# Loads to torch Encodec model
def codec_decode(codec: nn.Module, fine_tokens: mx.array):
    arr = torch.from_dlpack(fine_tokens.astype(mx.int32))
    arr = arr.to("cpu")
    arr = arr.transpose(0, 1)
    emb = codec.quantizer.decode(arr)
    out = codec.decoder(emb)
    audio_arr = out.detach().cpu().numpy().squeeze()
    del arr, emb, out
    return audio_arr

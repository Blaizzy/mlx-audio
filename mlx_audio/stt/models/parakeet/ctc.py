from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ConvASRDecoderArgs:
    feat_in: int
    num_classes: int
    vocabulary: list[str]


@dataclass
class AuxCTCArgs:
    decoder: ConvASRDecoderArgs


class ConvASRDecoder(nn.Module):
    def __init__(self, args: ConvASRDecoderArgs):
        super().__init__()

        args.num_classes = (
            len(args.vocabulary) if args.num_classes <= 0 else args.num_classes
        ) + 1

        self.decoder_layers = [
            nn.Conv1d(args.feat_in, args.num_classes, kernel_size=1, bias=True)
        ]

        self.temperature = 1.0  # change manually if desired

    def __call__(self, x: mx.array) -> mx.array:
        return nn.log_softmax(self.decoder_layers[0](x) / self.temperature)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import yaml
from huggingface_hub import snapshot_download

from mlx_audio.utils import hanning, istft

from ..encodec import Encodec
from .mel import log_mel_spectrogram


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio: mx.array, **kwargs):
        return log_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            padding=0,
        )


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()

        if encodec_model == "encodec_24khz":
            encodec, preprocessor = Encodec.from_pretrained(
                "mlx-community/encodec-24khz-float32"
            )
        elif encodec_model == "encodec_48khz":
            encodec, preprocessor = Encodec.from_pretrained(
                "mlx-community/encodec-48khz-float32"
            )
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )

        self.encodec = encodec
        self.preprocessor = preprocessor
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            bandwidth=max(bandwidths)
        )
        self.codebook_weights = mx.concatenate(
            [vq.codebook.embed for vq in self.encodec.quantizer.layers[: self.num_q]]
        )
        self.bandwidths = bandwidths

    def get_encodec_codes(self, audio: mx.array, bandwidth_id: int) -> mx.array:
        features, mask = self.preprocessor(audio)

        if isinstance(bandwidth_id, mx.array):
            bandwidth_id = int(bandwidth_id.flatten().tolist()[0])
        elif isinstance(bandwidth_id, list):
            bandwidth_id = bandwidth_id[0]

        codes, _ = self.encodec.encode(
            features, mask, bandwidth=self.bandwidths[bandwidth_id]
        )
        return mx.reshape(codes, (codes.shape[-2], 1, codes.shape[-1]))

    def get_features_from_codes(self, codes: mx.array) -> mx.array:
        offsets = mx.arange(
            0,
            self.encodec.quantizer.codebook_size * codes.shape[0],
            self.encodec.quantizer.codebook_size,
        )
        embeddings_idxs = codes + mx.reshape(offsets, (offsets.shape[0], 1, 1))
        embeddings = self.codebook_weights[embeddings_idxs]
        features = mx.sum(embeddings, axis=0)
        return features

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")

        codes = self.get_encodec_codes(audio, bandwidth_id=bandwidth_id)
        return self.get_features_from_codes(codes)


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.out(x).swapaxes(1, 2)
        mag, p = x.split(2, axis=1)
        mag = mx.exp(mag)
        mag = mx.clip(mag, None, 1e2)
        x = mx.cos(p)
        y = mx.sin(p)
        S = mag * (x + 1j * y)
        audio = istft(
            S.squeeze(0),
            window=hanning(self.n_fft),
            hop_length=self.hop_length,
            win_length=self.n_fft,
        )
        return audio


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()

        # depthwise conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            layer_scale_init_value * mx.ones(dim)
            if layer_scale_init_value > 0
            else None
        )

    def __call__(
        self, x: mx.array, cond_embedding_id: Optional[mx.array] = None
    ) -> mx.array:
        residual = x

        x = self.dwconv(x)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim

        self.scale = nn.Linear(num_embeddings, embedding_dim)
        self.shift = nn.Linear(num_embeddings, embedding_dim)
        self.scale.weight = mx.ones(self.scale.weight.shape)
        self.shift.weight = mx.zeros(self.shift.weight.shape)

    def __call__(self, x: mx.array, cond_embedding: mx.array) -> mx.array:
        scale = self.scale(cond_embedding)
        shift = self.shift(cond_embedding)
        x = mx.fast.layer_norm(x, weight=None, bias=None, eps=self.eps)
        x = x * scale[:, None, :] + shift[:, None, :]
        return x


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = [
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                adanorm_num_embeddings=adanorm_num_embeddings,
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6, bias=bias)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        bandwidth_id = kwargs.get("bandwidth_id", None)

        # Transpose if the input is not in the correct shape
        if x.shape[-1] != self.input_channels:
            x = x.transpose(0, 2, 1)

        x = self.embed(x)

        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x, bandwidth_id)

        else:
            x = self.norm(x)

        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x)
        return x


class Vocos(nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: VocosBackbone,
        head: ISTFTHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config: dict) -> Vocos:
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        config = SimpleNamespace(**config)

        if "MelSpectrogramFeatures" in config.feature_extractor["class_path"]:
            feature_extractor_init_args = config.feature_extractor["init_args"]
            feature_extractor = MelSpectrogramFeatures(**feature_extractor_init_args)
        elif "EncodecFeatures" in config.feature_extractor["class_path"]:
            feature_extractor = EncodecFeatures(**config.feature_extractor["init_args"])
        backbone = VocosBackbone(**config.backbone["init_args"])
        head = ISTFTHead(**config.head["init_args"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(cls, path_or_repo: str) -> Vocos:
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """

        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.yaml", "*.safetensors"],
                )
            )

        model_path = path / "model.safetensors"
        with open(model_path, "rb") as f:
            weights = mx.load(f)

        config_path = path / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = cls.from_hparams(config)

        # remove unused weights
        try:
            del weights["feature_extractor.mel_spec.spectrogram.window"]
            del weights["head.istft.window"]
        except KeyError:
            pass

        # transpose weights as needed
        new_weights = {}
        for k, v in weights.items():
            basename, pname = k.rsplit(".", 1)
            if "backbone.embed" in basename and pname == "weight":
                new_weights[k] = v.moveaxis(1, 2)
            elif "dwconv" in basename and pname == "weight":
                new_weights[k] = v.moveaxis(1, 2)
            else:
                new_weights[k] = v

        # use strict = False to avoid the encodec weights
        model.load_weights(list(new_weights.items()), strict=False)
        model.eval()

        return model

    def __call__(self, audio_input: mx.array, **kwargs: Any) -> mx.array:
        features = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    def get_encodec_codes(self, audio_input: mx.array, bandwidth_id: int) -> mx.array:
        if not isinstance(self.feature_extractor, EncodecFeatures):
            raise ValueError("This model does not support getting encodec codes.")

        return self.feature_extractor.get_encodec_codes(audio_input, bandwidth_id)

    def decode(self, features_input: mx.array, **kwargs: Any) -> mx.array:
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def decode_from_codes(self, codes: mx.array, **kwargs: Any) -> mx.array:
        features = self.feature_extractor.get_features_from_codes(codes)
        audio_output = self.decode(features, **kwargs)
        return audio_output

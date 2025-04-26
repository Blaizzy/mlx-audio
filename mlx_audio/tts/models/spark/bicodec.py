from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn
import torch
from omegaconf import DictConfig
from safetensors.torch import load_file

from mlx_audio.codec.models.descript.nn.quantize import ResidualVectorQuantize
from mlx_audio.tts.models.spark.modules.encoder_decoder.feat_decoder import Decoder
from mlx_audio.tts.models.spark.modules.encoder_decoder.feat_encoder import Encoder
from mlx_audio.tts.models.spark.modules.encoder_decoder.wave_generator import (
    WaveGenerator,
)
from mlx_audio.tts.models.spark.modules.speaker.speaker_encoder import SpeakerEncoder
from mlx_audio.tts.models.spark.utils.file import load_config
from mlx_audio.tts.utils import get_model_path


class BiCodec(nn.Module):
    """
    BiCodec model for speech synthesis, incorporating a speaker encoder, feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        **kwargs,
    ) -> None:
        """
        Initializes the BiCodec model with the required components.

        Args:
            mel_params (dict): Parameters for the mel-spectrogram transformer.
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            quantizer (nn.Module): Quantizer module.
            speaker_encoder (nn.Module): Speaker encoder module.
            prenet (nn.Module): Prenet network.
            postnet (nn.Module): Postnet network.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.init_mel_transformer(mel_params)

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path, **kwargs) -> "BiCodec":
        """
        Loads the model from a checkpoint.

        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.

        Returns:
            BiCodec: The initialized BiCodec model.
        """
        ckpt_path = f"{model_dir}/model.safetensors"
        config = load_config(f"{model_dir}/config.yaml")["audio_tokenizer"]
        mel_params = config["mel_params"]
        encoder = Encoder(**config["encoder"])
        quantizer = ResidualVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])
        speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        model = cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
        )

        weights = load_file(ckpt_path)
        model.load_weights(weights, strict=False)

        model.eval()

        return model

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.

        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = mx.array(batch["feat"])
        mel = self.mel_transformer(torch.from_dlpack(batch["ref_wav"])).squeeze(1)
        mel = mx.array(mel)

        z = self.encoder(feat.transpose(0, 2, 1))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel.transpose(0, 2, 1))

        conditions = d_vector
        with_speaker_loss = False

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions[:, None]
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "d_vector": d_vector,
            "audios": batch["wav"][:, None],
            "with_speaker_loss": with_speaker_loss,
        }

    def tokenize(self, batch: Dict[str, Any]):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = mx.array(batch["feat"])
        mel = self.mel_transformer(mx.array(batch["ref_wav"])).squeeze(1)

        z = self.encoder(feat.transpose(0, 2, 1))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(0, 2, 1))

        return semantic_tokens, global_tokens

    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        semantic_tokens = mx.array(semantic_tokens)
        global_tokens = mx.array(global_tokens)
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector[:, None]
        wav_recon = self.decoder(x)

        return torch.from_numpy(wav_recon)

    def init_mel_transformer(self, config: Dict[str, Any]):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )


if __name__ == "__main__":
    model_path = get_model_path("SparkAudio/Spark-TTS-0.5B")
    model = BiCodec.load_from_checkpoint(model_path)
    wav = mx.random.normal((1, 16000), dtype=mx.float32)
    mel = model.mel_transformer(wav)
    print(mel.shape)

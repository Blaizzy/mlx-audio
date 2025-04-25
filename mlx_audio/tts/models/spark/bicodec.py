

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
from safetensors.torch import load_file

from .utils.file import load_config
from .modules.speaker.speaker_encoder import SpeakerEncoder
from .modules.encoder_decoder.feat_encoder import Encoder
from .modules.encoder_decoder.feat_decoder import Decoder
from .modules.encoder_decoder.wave_generator import WaveGenerator
from .modules.factorized_vector_quantize import FactorizedVectorQuantize


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
        **kwargs
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
        ckpt_path = f'{model_dir}/model.safetensors'
        config = load_config(f'{model_dir}/config.yaml')['audio_tokenizer']
        mel_params = config["mel_params"]
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
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

        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        for key in missing_keys:
            print(f"Missing tensor: {key}")
        for key in unexpected_keys:
            print(f"Unexpected tensor: {key}")

        model.eval()
        model.remove_weight_norm()

        return model

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.

        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)

        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel.transpose(1, 2))

        conditions = d_vector
        with_speaker_loss = False

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "d_vector": d_vector,
            "audios": batch["wav"].unsqueeze(1),
            "with_speaker_loss": with_speaker_loss,
        }

    @torch.no_grad()
    def tokenize(self, batch: Dict[str, Any]):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)

        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))

        return semantic_tokens, global_tokens

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

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

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)

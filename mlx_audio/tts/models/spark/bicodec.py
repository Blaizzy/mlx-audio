from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from omegaconf import DictConfig
from safetensors.torch import load_file

from mlx_audio.tts.models.spark.modules.encoder_decoder.feat_decoder import Decoder
from mlx_audio.tts.models.spark.modules.encoder_decoder.feat_encoder import Encoder
from mlx_audio.tts.models.spark.modules.encoder_decoder.wave_generator import (
    WaveGenerator,
)
from mlx_audio.tts.models.spark.modules.residual import FactorizedVectorQuantize
from mlx_audio.tts.models.spark.modules.speaker.speaker_encoder import SpeakerEncoder
from mlx_audio.tts.models.spark.utils.file import load_config
from mlx_audio.tts.utils import get_model_path
from mlx_audio.utils import hanning, mel_filters, stft


def mel_spectrogram(
    audio: mx.array,
    sample_rate: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    f_min: int = 10,
    f_max: Optional[int] = None,
    hop_length: int = 320,
    win_length: int = 640,
    padding: int = 0,
):
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)
    if padding > 0:
        audio = mx.pad(audio, (0, padding))
    window = hanning(win_length + 1)[:-1]
    freqs = stft(
        audio, window=window, win_length=win_length, hop_length=hop_length, n_fft=n_fft
    )
    magnitudes = freqs.abs()
    filters = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm="slaney",
        mel_scale="slaney",
    )
    mel_spec = magnitudes @ filters.T
    return mx.expand_dims(mel_spec, axis=0)


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
        self.mel_params = mel_params

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

        weights = load_file(ckpt_path)

        # Convert PyTorch weights to MLX arrays and sanitize
        weights = {
            k: mx.array(v) for k, v in weights.items() if "num_batches_tracked" not in k
        }

        for module in [encoder, decoder, quantizer, speaker_encoder]:
            if hasattr(module, "sanitize"):
                weights = module.sanitize(weights)

        model.load_weights(list(weights.items()), strict=True)

        return model

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.

        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        ref_wav = batch["ref_wav"]
        mel = self.get_mel_spectrogram(ref_wav)
        z = self.encoder(feat.transpose(0, 2, 1))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel)

        conditions = d_vector
        with_speaker_loss = False

        # Ensure conditions is an integer type for embedding lookup
        # The error shows that the embedding layer expects integral indices
        if isinstance(conditions, mx.array) and conditions.dtype == mx.float32:
            # Convert to integer type if needed for the embedding layer
            # or ensure it's properly formatted for the prenet
            conditions = conditions.astype(mx.int32)

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions[..., None]
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
        feat = batch["feat"]
        ref_wav = mx.array(batch["ref_wav"])
        mel = self.get_mel_spectrogram(ref_wav)
        z = self.encoder(feat.transpose(0, 2, 1))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel)

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

        z_q = self.quantizer.detokenize(semantic_tokens.transpose(0, 1)).transpose(
            0, 2, 1
        )
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector[..., None]
        wav_recon = self.decoder(x)

        return wav_recon  # Return MLX array directly

    def get_mel_spectrogram(self, wav):
        mels = []
        for i in range(wav.shape[0]):
            audio_sample = mx.squeeze(wav[i])
            mel = mel_spectrogram(
                audio=audio_sample,
                sample_rate=self.mel_params["sample_rate"],
                n_mels=self.mel_params["num_mels"],
                n_fft=self.mel_params["n_fft"],
                hop_length=self.mel_params["hop_length"],
                win_length=self.mel_params["win_length"],
                f_min=self.mel_params["mel_fmin"],
                f_max=self.mel_params["mel_fmax"],
            )
            mels.append(mel)
        return mx.concatenate(mels, axis=0)


if __name__ == "__main__":

    model_path = get_model_path("SparkAudio/Spark-TTS-0.5B")

    model = BiCodec.load_from_checkpoint(model_path / "BiCodec")
    model.eval()

    # Generate random inputs for testing
    duration = 0.96
    x = mx.random.normal((20, 1, int(duration * 16000)), dtype=mx.float32)
    feat = mx.random.normal((20, int(duration * 50), 1024), dtype=mx.float32)
    inputs = {"feat": feat, "wav": x, "ref_wav": x}

    # Forward pass
    outputs = model(inputs)
    semantic_tokens, global_tokens = model.tokenize(inputs)

    wav_recon = model.detokenize(semantic_tokens, global_tokens)

    print(outputs["recons"].shape)
    print(wav_recon.shape)

    if np.allclose(outputs["recons"], wav_recon):
        print("Test successful")
    else:
        print("Test failed")

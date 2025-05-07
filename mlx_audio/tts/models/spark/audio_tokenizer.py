from pathlib import Path
from typing import Any, Dict, Tuple

import mlx.core as mx
import numpy as np
import torch
from transformers import Wav2Vec2Model

from .bicodec import BiCodec
from .utils.audio import load_audio
from .utils.file import load_config

from mlx_audio.stt.models.wav2vec.feature_extractor import Wav2Vec2FeatureExtractor

class BiCodecTokenizer:
    """BiCodec tokenizer for handling audio input and tokenization."""

    def __init__(self, model_dir: Path, device: torch.device = None, **kwargs):
        super().__init__()
        """
        Args:
            model_dir: Path to the model directory.
            device: Device to run the model on (default is GPU if available).
        """
        self.device = device
        self.model_dir = model_dir
        self.config = load_config(f"{model_dir}/config.yaml")
        self._initialize_model()

    def _initialize_model(self):
        """Load and initialize the BiCodec model and Wav2Vec2 feature extractor."""
        self.model = BiCodec.load_from_checkpoint(f"{self.model_dir}/BiCodec")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        ).to(self.device)
        self.feature_extractor.config.output_hidden_states = True

    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Get reference audio clip for speaker embedding."""
        ref_segment_length = (
            int(self.config["sample_rate"] * self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    def process_audio(self, wav_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """load auido and get reference audio from wav path"""
        wav = load_audio(
            wav_path,
            sampling_rate=self.config["sample_rate"],
            volume_normalize=self.config["volume_normalize"],
        )

        wav_ref = self.get_ref_clip(wav)

        wav_ref = torch.from_numpy(wav_ref).unsqueeze(0).float()
        return wav, wav_ref

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="mx",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs)
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    def tokenize_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """tokenize the batch of audio

        Args:
            batch:
                wavs (List[np.ndarray]): batch of audio
                ref_wavs (torch.Tensor): reference audio. shape: (batch_size, seq_len)

        Returns:
            semantic_tokens: semantic tokens. shape: (batch_size, seq_len, latent_dim)
            global_tokens: global tokens. shape: (batch_size, seq_len, global_dim)
        """
        feats = self.extract_wav2vec2_features(batch["wav"])
        batch["feat"] = feats

        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def tokenize(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize the audio"""
        wav, ref_wav = self.process_audio(audio_path)
        feat = self.extract_wav2vec2_features(wav)
        batch = {
            "wav": torch.from_numpy(wav).unsqueeze(0).float().to(self.device),
            "ref_wav": ref_wav.to(self.device),
            "feat": feat.to(self.device),
        }

        # convert to mlx array
        batch["wav"] = mx.array(batch["wav"])
        batch["ref_wav"] = mx.array(batch["ref_wav"])
        batch["feat"] = mx.array(batch["feat"])
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.array:
        """detokenize the tokens to waveform

        Args:
            global_tokens: global tokens. shape: (batch_size, global_dim)
            semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)

        Returns:
            wav_rec: waveform. shape: (batch_size, seq_len) for batch or (seq_len,) for single
        """
        global_tokens = global_tokens.unsqueeze(1)

        # convert to mlx array
        global_tokens = mx.array(global_tokens)
        semantic_tokens = mx.array(semantic_tokens)
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        return wav_rec.squeeze()

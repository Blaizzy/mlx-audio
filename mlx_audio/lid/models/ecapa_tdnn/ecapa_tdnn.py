"""ECAPA-TDNN model for spoken language identification (107 languages).

Based on the SpeechBrain ``speechbrain/lang-id-voxlingua107-ecapa`` model,
trained on VoxLingua107 dataset. Input is raw 16 kHz mono audio; output is
a probability distribution over 107 languages.
"""

from typing import Dict, List, Tuple, cast

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig

from .config import ModelConfig
from .mel import compute_mel_spectrogram

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class DNNLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w(x)


class DNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = DNNLinear(in_dim, out_dim)
        self.norm = nn.BatchNorm(out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.norm(self.linear(x)))


class DNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block_0 = DNNBlock(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.block_0(x)


class ClassifierLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w(x)


class EcapaClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.BatchNorm(config.embedding_dim)
        self.DNN = DNN(config.embedding_dim, config.classifier_hidden_dim)
        self.out = ClassifierLinear(config.classifier_hidden_dim, config.num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.squeeze(x, axis=1)
        out = self.norm(out)
        out = self.DNN(out)
        out = self.out(out)
        return mx.log(mx.softmax(out, axis=-1) + 1e-10)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class EcapaTdnn(nn.Module):
    """ECAPA-TDNN for spoken language identification.

    Architecture:
        Mel spectrogram → EcapaTdnnEmbedding → EcapaClassifier → log-probs

    Args:
        config: ``ModelConfig`` with ECAPA-TDNN hyper-parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        backbone_config = EcapaTdnnConfig(
            input_size=config.n_mels,
            channels=config.channels,
            embed_dim=config.embedding_dim,
            kernel_sizes=config.kernel_sizes,
            dilations=config.dilations,
            attention_channels=config.attention_channels,
            res2net_scale=config.res2net_scale,
            se_channels=config.se_channels,
            global_context=True,
        )
        self.embedding_model = EcapaTdnnBackbone(backbone_config)
        self.classifier = EcapaClassifier(config)

        self.id2label: Dict[int, str] = {}
        if config.id2label:
            for k, v in config.id2label.items():
                idx = int(k)
                lang = v.split(":")[0].strip()
                self.id2label[idx] = lang

    def __call__(self, mel_features: mx.array) -> mx.array:
        """Forward pass: mel features → log-probabilities.

        Args:
            mel_features: ``[batch, time, n_mels]`` mel spectrogram.

        Returns:
            Log-probabilities ``[batch, num_classes]``.
        """
        embeddings = self.embedding_model(mel_features)
        embeddings = mx.expand_dims(embeddings, axis=1)
        return self.classifier(embeddings)

    def predict(
        self,
        audio: mx.array,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Predict language from raw 16 kHz mono audio.

        Computes SpeechBrain-compatible mel spectrogram internally.

        Args:
            audio: Raw waveform, shape ``(T,)``.
            top_k: Number of top predictions to return.

        Returns:
            List of ``(language_code, probability)`` tuples, sorted by
            probability descending.
        """
        mel = compute_mel_spectrogram(audio)
        log_probs = self(mel)
        probs = mx.exp(log_probs)
        mx.eval(probs)

        sorted_indices = cast(List[int], mx.argsort(-probs[0]).tolist())
        indexed = [(idx, float(probs[0, idx].item())) for idx in sorted_indices]

        id2label = self.id2label or {}
        return [
            (id2label.get(idx, f"LABEL_{idx}"), prob) for idx, prob in indexed[:top_k]
        ]

    def sanitize(self, weights):
        """Remap SpeechBrain checkpoint keys to MLX model structure.

        Handles:
        - Dropping ``num_batches_tracked`` keys
        - Remapping top-level block indices: ``blocks.0.`` → ``block0.``
        - Flattening SpeechBrain double-nesting: ``.conv.conv.`` → ``.conv.``
        - SE block conv wrappers, ASP BN norm, FC conv flattening
        """
        sanitized = {}
        for k, v in weights.items():
            if "num_batches_tracked" in k:
                continue

            # Remap top-level block indices (NOT res2net_block.blocks)
            k = k.replace("embedding_model.blocks.0.", "embedding_model.block0.")
            k = k.replace("embedding_model.blocks.1.", "embedding_model.block1.")
            k = k.replace("embedding_model.blocks.2.", "embedding_model.block2.")
            k = k.replace("embedding_model.blocks.3.", "embedding_model.block3.")

            # Flatten SpeechBrain double-nesting
            k = k.replace(".conv.conv.", ".conv.")
            k = k.replace(".norm.norm.", ".norm.")

            # SE block Conv1d wrappers
            k = k.replace(".se_block.conv1.conv.", ".se_block.conv1.")
            k = k.replace(".se_block.conv2.conv.", ".se_block.conv2.")

            # ASP BN
            k = k.replace(".asp_bn.norm.", ".asp_bn.")

            # FC conv
            k = k.replace(".fc.conv.", ".fc.")

            sanitized[k] = v

        return sanitized

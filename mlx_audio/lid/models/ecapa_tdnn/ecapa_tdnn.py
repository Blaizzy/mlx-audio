"""ECAPA-TDNN model for spoken language identification (107 languages).

Based on the SpeechBrain ``speechbrain/lang-id-voxlingua107-ecapa`` model,
trained on VoxLingua107 dataset. Input is raw 16 kHz mono audio; output is
a probability distribution over 107 languages.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .mel import compute_mel_spectrogram

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TDNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(nn.relu(self.conv(x)))


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()
        self.scale = scale
        hidden = channels // scale
        self.blocks = [
            TDNNBlock(hidden, hidden, kernel_size, dilation) for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        chunks = mx.split(x, self.scale, axis=-1)
        y = [chunks[0]]
        for i, block in enumerate(self.blocks):
            inp = chunks[i + 1] + y[-1] if i > 0 else chunks[i + 1]
            y.append(block(inp))
        return mx.concatenate(y, axis=-1)


class SEBlock(nn.Module):
    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, bottleneck, 1)
        self.conv2 = nn.Conv1d(bottleneck, in_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        s = mx.mean(x, axis=1, keepdims=True)
        s = nn.relu(self.conv1(s))
        s = mx.sigmoid(self.conv2(s))
        return x * s


class SERes2NetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        res2net_scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        self.tdnn1 = TDNNBlock(channels, channels, 1)
        self.res2net_block = Res2NetBlock(
            channels, kernel_size, dilation, res2net_scale
        )
        self.tdnn2 = TDNNBlock(channels, channels, 1)
        self.se_block = SEBlock(channels, se_channels)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.tdnn1(x)
        out = self.res2net_block(out)
        out = self.tdnn2(out)
        out = self.se_block(out)
        return out + x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.tdnn = TDNNBlock(channels * 3, attention_channels, 1)
        self.conv = nn.Conv1d(attention_channels, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        m = mx.mean(x, axis=1, keepdims=True)
        v = mx.var(x, axis=1, keepdims=True)
        s = mx.sqrt(v + 1e-9)
        m_exp = mx.broadcast_to(m, x.shape)
        s_exp = mx.broadcast_to(s, x.shape)

        attn = self.tdnn(mx.concatenate([x, m_exp, s_exp], axis=-1))
        attn = mx.tanh(attn)
        attn = self.conv(attn)
        attn = mx.softmax(attn, axis=1)

        weighted_mean = mx.sum(attn * x, axis=1)
        weighted_var = mx.sum(attn * (x * x), axis=1) - weighted_mean * weighted_mean
        weighted_std = mx.sqrt(mx.maximum(weighted_var, 1e-9))

        return mx.concatenate([weighted_mean, weighted_std], axis=-1)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------


class EcapaTdnnEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        ch = config.channels
        self.block0 = TDNNBlock(config.n_mels, ch, 5)
        self.block1 = SERes2NetBlock(ch, 3, 2, config.res2net_scale, config.se_channels)
        self.block2 = SERes2NetBlock(ch, 3, 3, config.res2net_scale, config.se_channels)
        self.block3 = SERes2NetBlock(ch, 3, 4, config.res2net_scale, config.se_channels)
        self.mfa = TDNNBlock(ch * 3, ch * 3, 1)
        self.asp = AttentiveStatisticsPooling(ch * 3, config.attention_channels)
        self.asp_bn = nn.BatchNorm(ch * 6)
        self.fc = nn.Conv1d(ch * 6, config.embedding_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.block0(x)
        xl = []
        out = self.block1(out)
        xl.append(out)
        out = self.block2(out)
        xl.append(out)
        out = self.block3(out)
        xl.append(out)

        out = mx.concatenate(xl, axis=-1)
        out = self.mfa(out)
        out = self.asp(out)
        out = self.asp_bn(out)
        out = mx.expand_dims(out, axis=1)
        out = self.fc(out)
        return out


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
        self.embedding_model = EcapaTdnnEmbedding(config)
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

        probs_list = probs[0].tolist()
        indexed = sorted(enumerate(probs_list), key=lambda x: x[1], reverse=True)

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

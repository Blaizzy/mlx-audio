"""Wav2Vec2ForSequenceClassification — Language Identification via Wav2Vec2.

Wraps the existing Wav2Vec2Model backbone (from mlx_audio.stt) with a
classification head (projector + classifier) for spoken language identification.

Supports facebook/mms-lid-256 and similar HuggingFace checkpoints that use
the Wav2Vec2ForSequenceClassification architecture.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.wav2vec.wav2vec import Wav2Vec2Model

from .config import ModelConfig


class Wav2Vec2ForSequenceClassification(nn.Module):
    """Wav2Vec2 encoder with a sequence classification head for language ID.

    Architecture:
        Wav2Vec2Model (backbone) → mean pooling → projector → classifier

    Args:
        config: ModelConfig with classifier_proj_size, num_labels, id2label.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

    def __call__(
        self,
        input_values: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_values: Raw 16kHz waveform, shape (B, T). Should be
                zero-mean unit-variance normalized.
            attention_mask: Not currently supported by the underlying
                Wav2Vec2Model backbone. Accepted for API compatibility
                but always passed as None.

        Returns:
            Logits of shape (B, num_labels).
        """
        outputs = self.wav2vec2(
            input_values,
            attention_mask=None,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # (B, T', hidden_size)
        hidden_states = self.projector(hidden_states)  # (B, T', proj_size)
        pooled = mx.mean(hidden_states, axis=1)  # (B, proj_size)
        logits = self.classifier(pooled)  # (B, num_labels)
        return logits

    def sanitize(self, weights):
        """Remap HuggingFace weight keys to MLX model structure.

        Key differences from STT Wav2Vec2Model.sanitize():
        - Does NOT strip 'wav2vec2.' prefix (our model has self.wav2vec2)
        - Keeps 'projector.*' and 'classifier.*' (classification head)
        """
        sanitized = {}
        for k, v in weights.items():
            # Conv1d axis swap: HF (out, in, kernel) → MLX (out, kernel, in)
            if k.endswith(".conv.weight"):
                v = v.swapaxes(1, 2)

            # WNConv1d weight norm tensors also need axis swap
            if k.endswith(".conv.weight_v") or k.endswith(".conv.weight_g"):
                v = v.swapaxes(1, 2)

            # PyTorch parametrize API → MLX WNConv1d naming
            if k.endswith(".parametrizations.weight.original0"):
                k = k.replace(".parametrizations.weight.original0", ".weight_g")
                v = v.swapaxes(1, 2)
            if k.endswith(".parametrizations.weight.original1"):
                k = k.replace(".parametrizations.weight.original1", ".weight_v")
                v = v.swapaxes(1, 2)

            # Drop training-only keys
            if (
                k.startswith("quantizer.")
                or k.startswith("project_")
                or k == "masked_spec_embed"
                or "lm_head." in k
            ):
                continue

            sanitized[k] = v
        return sanitized

    def predict(
        self,
        input_values: mx.array,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Predict language from pre-processed audio.

        Args:
            input_values: Normalized waveform, shape (T,) or (1, T).
                Only single-sample inference is supported.
            top_k: Number of top predictions to return.

        Returns:
            List of (language_code, probability) tuples, sorted by probability.

        Raises:
            ValueError: If input has batch size > 1.
        """
        if input_values.ndim == 1:
            input_values = input_values[None, :]

        if input_values.shape[0] != 1:
            raise ValueError(
                f"predict() supports single-sample input only, "
                f"got batch size {input_values.shape[0]}. "
                f"Use __call__() for batched inference."
            )

        logits = self(input_values)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        probs_list = probs[0].tolist()
        indexed = sorted(enumerate(probs_list), key=lambda x: x[1], reverse=True)

        id2label = self.config.id2label or {}
        return [
            (id2label.get(str(idx), f"LABEL_{idx}"), prob)
            for idx, prob in indexed[:top_k]
        ]

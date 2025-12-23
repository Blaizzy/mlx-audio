# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import List, Optional, Tuple

import mlx.core as mx

from .config import T5EncoderConfig


class T5TextEncoder:
    """
    T5 text encoder wrapper for SAM-Audio.

    This wraps the HuggingFace T5 encoder model and converts outputs to MLX arrays.
    The T5 model runs on CPU/GPU via PyTorch, and outputs are converted to MLX.
    """

    def __init__(self, config: T5EncoderConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def _lazy_load(self):
        """Lazily load the T5 model and tokenizer."""
        if self.model is None:
            import torch
            import transformers

            self.model = transformers.T5EncoderModel.from_pretrained(self.config.name)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.name
            )
            # Set to eval mode
            self.model.eval()

    def __call__(self, texts: List[str]) -> Tuple[mx.array, mx.array]:
        """
        Encode text descriptions.

        Args:
            texts: List of text descriptions

        Returns:
            Tuple of (features, attention_mask) as MLX arrays
            - features: (batch, seq_len, dim)
            - attention_mask: (batch, seq_len) boolean mask
        """
        self._lazy_load()
        import torch

        # Tokenize
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_length,
            padding=self.config.pad_mode,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Encode
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            features = outputs.last_hidden_state

        # Convert to MLX arrays
        features_np = features.cpu().numpy()
        # HuggingFace: attention_mask=1 means "attend", =0 means "padding"
        # Our attention: mask=True means "mask out" (set to -inf)
        # So we need to invert: padding (0) -> True, real tokens (1) -> False
        mask_np = ~attention_mask.cpu().numpy().astype(bool)

        return mx.array(features_np), mx.array(mask_np)


class T5TextEncoderMLX:
    """
    Pure MLX T5 text encoder (placeholder for future implementation).

    This would be a full MLX implementation of T5 for better performance.
    For now, we use the PyTorch wrapper above.
    """

    def __init__(self, config: T5EncoderConfig):
        raise NotImplementedError(
            "Pure MLX T5 encoder not yet implemented. Use T5TextEncoder instead."
        )

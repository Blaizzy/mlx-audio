# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3 import Qwen3Model

try:
    from .config import Qwen3Config
except ImportError:
    from mlx_audio.sts.models.fun_audio_chat.config import Qwen3Config


class LanguageModel(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Store last hidden state for speech decoder
        self._last_hidden_state: Optional[mx.array] = None

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> mx.array:
        hidden_states = self.model(
            inputs=input_ids,
            input_embeddings=inputs_embeds,
            cache=cache,
        )

        # Store for speech decoder access
        self._last_hidden_state = hidden_states

        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    def get_last_hidden_state(self) -> Optional[mx.array]:
        """Get the last hidden state from the most recent forward pass.

        This is used by the speech decoder to generate audio tokens
        from the language model's hidden representations.

        Returns:
            The last hidden state tensor, or None if no forward pass has been made.
        """
        return self._last_hidden_state

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

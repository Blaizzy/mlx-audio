# Copyright © 2025 Apple Inc.
# Vendored verbatim from mlx-lm v0.31.3 (ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd),
# mlx_lm/models/gemma3.py. MIT licensed.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import gemma3_text
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vocab_size: int = 262208

    def __post_init__(self):
        self.text_config["vocab_size"] = self.vocab_size
        self.text_config["num_attention_heads"] = self.text_config.get(
            "num_attention_heads", 8
        )
        self.text_config["num_key_value_heads"] = self.text_config.get(
            "num_key_value_heads", 4
        )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma3_text.Model(
            gemma3_text.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        weights.pop("vision_tower", None)
        weights.pop("multi_modal_projector", None)
        lm_weights = dict(tree_flatten(weights["language_model"]))
        lm_weights = self.language_model.sanitize(lm_weights)
        weights["language_model"] = tree_unflatten(list(lm_weights.items()))
        return dict(tree_flatten(weights))

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

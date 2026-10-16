"""Drama 3's text backbone: the Qwen3.5-MoE text model, driven by frame embeddings.

The backbone is the Qwen3.5-MoE architecture (three gated-delta layers per
full-attention layer, routed experts plus a gated shared expert), so the vendored
Qwen3.5 model and its Metal gated-delta kernel run it unchanged. Its norms use
zero-centered gammas; convert.py stores them shifted to 1 + w in float32, because
bfloat16 only resolves steps of 1/128 near 1.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models.qwen3_5 import TextModel, TextModelArgs

# Settings that Qwen3.5 hardcodes; a checkpoint that differs would load and run,
# but compute something else.
FIXED_TEXT_SETTINGS = {
    "attn_output_gate": True,
    "hidden_act": "silu",
    "layernorm_zero_centered_gamma": True,
    "rotary_interleaved": False,
    "shared_expert_gate": True,
    "use_qk_norm": True,
}


class Fp32RMSNorm(nn.Module):
    """RMSNorm with a float32 gamma, computed in float32 and cast back."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps)
        return y.astype(x.dtype)


def text_model_args(text: dict[str, Any]) -> TextModelArgs:
    layers = text["num_hidden_layers"]
    full_every_fourth = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(layers)
    ]
    if text["layer_types"] != full_every_fourth:
        raise ValueError("expected three linear-attention layers per full one")
    if text.get("mlp_layer_types", ["moe"] * layers) != ["moe"] * layers:
        raise ValueError("expected a mixture-of-experts MLP in every layer")
    for key, value in FIXED_TEXT_SETTINGS.items():
        if text.get(key) != value:
            raise ValueError(f"unsupported text_config {key}={text.get(key)!r}")
    return TextModelArgs.from_dict(
        {
            **text,
            "model_type": "qwen3_5_moe_text",
            "tie_word_embeddings": False,
            # TextModelArgs renames rope_type in place; keep the config intact.
            "rope_parameters": dict(text["rope_parameters"]),
        }
    )


def build_backbone(text: dict[str, Any]) -> TextModel:
    """Qwen3.5 without its vocabulary head, with float32-gamma norms."""
    model = TextModel(text_model_args(text))
    # Only the semantic band and EOS can be sampled; Model.semantic_head holds
    # those rows instead of the full vocabulary head.
    model.pop("lm_head")
    dim, head_dim, eps = text["hidden_size"], text["head_dim"], text["rms_norm_eps"]
    inner = model.model
    inner.norm = Fp32RMSNorm(dim, eps)
    for layer in inner.layers:
        layer.input_layernorm = Fp32RMSNorm(dim, eps)
        layer.post_attention_layernorm = Fp32RMSNorm(dim, eps)
        if not layer.is_linear:
            layer.self_attn.q_norm = Fp32RMSNorm(head_dim, eps)
            layer.self_attn.k_norm = Fp32RMSNorm(head_dim, eps)
    return model

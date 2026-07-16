"""Pure-MLX Qwen2.5 LLM trunk for dots.tts (``llm.*`` + ``eos_proj.*``).

Mirrors ``DotsTtsCore.step_llm``: the dots.tts decode is schedule-driven, so the
LLM is a *contextual encoder* (+ an EOS signal), NOT a logit sampler. Each step we
want the **last-layer hidden state** — i.e. HF Qwen2's ``outputs.hidden_states[-1]``,
which is the output of the final RMSNorm (``model.norm``). mlx-lm's ``Qwen2Model``
returns exactly that: ``self.norm(h)`` (the trunk, pre-``lm_head``). We never compute
logits, so the tied ``lm_head`` (= ``embed_tokens``) is irrelevant to ``step``.

``step`` accepts EITHER token ids (embedded via ``embed_tokens``) OR ``inputs_embeds``
directly (re-encoded audio embeddings), matching the two ``step_llm`` entry points.
mlx-lm's ``Qwen2Model.__call__(inputs, cache=None, input_embeddings=None)`` supports
both natively — when ``input_embeddings`` is given it skips the embedding lookup.

The KV cache is incremental: ``make_cache()`` returns mlx-lm's per-layer
``KVCache`` list (via ``make_prompt_cache``); feeding one token at a time with the
cache equals a single full-sequence forward (mlx-lm's RoPE offsets off ``cache.offset``).

``eos_proj`` is a tiny ``Linear(1536,1536) -> SiLU -> Linear(1536,2)`` head; the decode
uses ``softmax(eos_proj(hidden))[..., 1] > 0.8`` for EOS.

Imports mlx + mlx_lm and makes no torch calls. (Importing mlx_lm pulls in
torch + transformers transitively for its tokenizer utilities; this module never
calls into them.) The Qwen2.5 config (vocab 151672, hidden
1536, 28 layers, 12 heads, 2 kv-heads/GQA, rope_theta 1e6, rms_norm_eps 1e-6, tied
embeddings) is read verbatim from the saved ``llm_config.json``; mlx-lm handles the
GQA / RoPE / norm details from those args. Loaded fp32 for parity (MLX fast matmul
rounds to ~tf32, but cosine over the 28 layers stays >= 0.9999 vs the torch oracle).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.models.qwen2 import ModelArgs as Qwen2Args

_LLM_PREFIX = "llm."


class _EosProj(nn.Module):
    """``Linear(H, H) -> SiLU -> Linear(H, 2)`` (the ``eos_proj`` Sequential).

    Submodule names ``0`` / ``2`` match the checkpoint keys (``eos_proj.0.*`` is the
    first Linear, index 1 is the SiLU, ``eos_proj.2.*`` is the output Linear).
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.layers = [nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class DotsLLM:
    """Qwen2.5 trunk (hidden + KV cache) + the ``eos_proj`` head.

    Use ``DotsLLM.from_core(...)`` to build from the converted ``core.safetensors``.
    """

    def __init__(self, model: Qwen2Model, eos: _EosProj):
        self._model = model
        self._eos = eos

    # region construction
    @classmethod
    def from_core(
        cls,
        core_safetensors: str,
        llm_config_json: str,
        dtype: mx.Dtype = mx.float32,
        quantization: "object | None" = None,  # QuantizationConfig | None
    ) -> "DotsLLM":
        """Build the Qwen2.5 trunk + eos head and bind ``core.safetensors`` weights.

        Reads the Qwen2.5 config from ``llm_config_json`` (the same config the
        trained checkpoint was built with), instantiates mlx-lm's Qwen2 ``Model``,
        then loads the ``llm.model.*`` weights (stripped of the ``llm.`` prefix) and
        the ``eos_proj.*`` weights. Embeddings are tied, so there is no separate
        ``lm_head`` to load (we never compute logits anyway).
        """
        cfg = json.loads(Path(llm_config_json).read_text())
        args = Qwen2Args(
            model_type=cfg.get("model_type", "qwen2"),
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            vocab_size=cfg["vocab_size"],
            num_key_value_heads=cfg["num_key_value_heads"],
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            rope_theta=cfg.get("rope_theta", 1000000.0),
            rope_traditional=cfg.get("rope_traditional", False),
            rope_scaling=cfg.get("rope_scaling"),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        )
        model = Qwen2Model(args)

        raw = mx.load(str(core_safetensors))
        llm_weights = {
            k[len(_LLM_PREFIX) :]: (v if v.dtype == mx.uint32 else v.astype(dtype))
            for k, v in raw.items()
            if k.startswith(_LLM_PREFIX)
        }
        if quantization is not None and "llm" in quantization.components:
            has_scales = any(k.endswith(".scales") for k in llm_weights)
            if not has_scales:
                raise ValueError(
                    "config.json declares the llm is quantized, but core.safetensors "
                    "has no '.scales' tensors — the weights and the quantization block "
                    "disagree. Re-run `python -m dots_tts_mlx.quantize`."
                )
            nn.quantize(
                model, group_size=quantization.group_size, bits=quantization.bits
            )
        # ``sanitize`` drops a tied ``lm_head.weight`` if present and any unused
        # rotary buffers; our keys have neither, so it is a no-op safeguard.
        llm_weights = model.sanitize(llm_weights)
        model.load_weights(list(llm_weights.items()))
        model.set_dtype(dtype)
        model.eval()

        eos = _EosProj(args.hidden_size)
        eos.update(
            {
                "layers": [
                    {
                        "weight": raw["eos_proj.0.weight"].astype(dtype),
                        "bias": raw["eos_proj.0.bias"].astype(dtype),
                    },
                    {},  # SiLU — no parameters
                    {
                        "weight": raw["eos_proj.2.weight"].astype(dtype),
                        "bias": raw["eos_proj.2.bias"].astype(dtype),
                    },
                ]
            }
        )
        eos.eval()

        return cls(model, eos)

    # endregion construction

    def make_cache(self) -> list:
        """Fresh incremental per-layer KV cache for a decode (mlx-lm ``KVCache``)."""
        return make_prompt_cache(self._model)

    def step(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: list | None = None,
    ) -> tuple[mx.array, list | None]:
        """One LLM step -> ``(last_hidden, cache)``.

        Provide exactly one of ``input_ids`` ``[B, L]`` (token path) or
        ``inputs_embeds`` ``[B, L, H]`` (re-encoded-embedding path). ``last_hidden``
        is ``[B, L, H]`` = the final-RMSNorm output (HF ``hidden_states[-1]``). When
        ``cache`` is given it is updated in place and returned for the next step.
        """
        provided = int(input_ids is not None) + int(inputs_embeds is not None)
        if provided != 1:
            raise ValueError(
                "Exactly one of input_ids or inputs_embeds must be provided to step()."
            )

        if inputs_embeds is not None:
            hidden = self._model.model(
                inputs=None, cache=cache, input_embeddings=inputs_embeds
            )
        else:
            hidden = self._model.model(input_ids, cache=cache)
        return hidden, cache

    def eos_logits(self, hidden: mx.array) -> mx.array:
        """EOS head: ``eos_proj(hidden)`` -> ``[..., 2]`` logits."""
        return self._eos(hidden)

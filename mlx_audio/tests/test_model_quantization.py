"""Compatibility of checkpoint quantization for models without a load hook."""

import mlx.nn as nn
import pytest

from mlx_audio.utils import apply_quantization


@pytest.mark.parametrize(
    "policy", [False, True, {"group_size": 64, "bits": 4, "mode": "affine"}]
)
def test_existing_model_predicate_is_used_without_load_hook(policy):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(64, 64, bias=False)

        def model_quant_predicate(self, path, module):
            return policy

    model = Model()
    apply_quantization(
        model,
        {"quantization": {"group_size": 64, "bits": 8}},
        {"layer.scales": None},
        model.model_quant_predicate,
    )

    if policy is False:
        assert isinstance(model.layer, nn.Linear)
    else:
        assert isinstance(model.layer, nn.QuantizedLinear)
        assert model.layer.bits == (policy["bits"] if isinstance(policy, dict) else 8)

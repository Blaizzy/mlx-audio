from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import load_file, save_file


def _load_onnx_initializers(onnx_path: Path) -> Dict[str, np.ndarray]:
    model = onnx.load(onnx_path.as_posix(), load_external_data=True)
    return {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }


def convert_campplus_onnx_to_safetensors(
    onnx_path: Path | str,
    output_path: Path | str | None = None,
    verify_allclose: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Path:
    """Convert campplus ONNX parameters to safetensors and optionally verify with allclose."""
    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_suffix(".safetensors")
    output_path = Path(output_path)

    weights = _load_onnx_initializers(onnx_path)
    if not weights:
        raise ValueError(f"No initializer tensors found in {onnx_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, output_path.as_posix(), metadata={"format": "numpy"})

    if verify_allclose:
        reloaded = load_file(output_path.as_posix())
        missing = set(weights) - set(reloaded)
        if missing:
            raise ValueError(
                f"Missing tensors after conversion: {sorted(list(missing))[:5]}"
            )
        for name, original in weights.items():
            converted = reloaded[name]
            if not np.allclose(original, converted, rtol=rtol, atol=atol):
                diff = np.max(np.abs(original - converted))
                raise ValueError(
                    f"allclose failed for tensor '{name}' (max abs diff: {diff})"
                )

    return output_path

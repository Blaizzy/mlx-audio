from .bailingmm import Model, ModelConfig


def convert_campplus_onnx_to_safetensors(*args, **kwargs):
    # Conversion dependencies are only imported when this legacy utility is called.
    from .convert import convert_campplus_onnx_to_safetensors as convert

    return convert(*args, **kwargs)


__all__ = ["Model", "ModelConfig", "convert_campplus_onnx_to_safetensors"]

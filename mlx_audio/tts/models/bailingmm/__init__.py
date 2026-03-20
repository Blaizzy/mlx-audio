from .bailingmm import Model, ModelConfig

try:
    from .convert import convert_campplus_onnx_to_safetensors
except ModuleNotFoundError as exc:
    if exc.name != "onnx":
        raise

    _ONNX_IMPORT_ERROR = exc

    def convert_campplus_onnx_to_safetensors(*args, **kwargs):
        raise ModuleNotFoundError(
            "onnx is required for convert_campplus_onnx_to_safetensors function."
            "Please install onnx using `pip install onnx`."
        ) from _ONNX_IMPORT_ERROR


__all__ = ["Model", "ModelConfig", "convert_campplus_onnx_to_safetensors"]

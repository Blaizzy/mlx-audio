import mlx.core as mx


def load_voice_tensor(path: str) -> mx.array:
    """
    Load a voice pack .safetensors file into an MLX array.

    Args:
        path: Path to the .safetensors voice file

    Returns:
        mx.array: The voice tensor
    """
    weights = mx.load(path)
    return weights["voice"]

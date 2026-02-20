from .kitten_tts import Model, ModelConfig

# Hints for model detection in mlx_audio.convert
DETECTION_HINTS = {
    "config_keys": {
        "kitten_tts": {
            "model_file",
            "voices",
            "voice_aliases",
            "speed_priors",
            "type",
        }
    },
    "path_patterns": {"kitten_tts": {"kitten-tts", "kitten_tts"}},
}

__all__ = ["Model", "ModelConfig"]

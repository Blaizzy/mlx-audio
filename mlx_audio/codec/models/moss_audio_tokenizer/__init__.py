from .config import MossAudioTokenizerConfig
from .moss_audio_tokenizer import MossAudioTokenizer

ModelConfig = MossAudioTokenizerConfig
Model = MossAudioTokenizer

DETECTION_HINTS = {
    "config_keys": {
        "quantizer_kwargs",
        "decoder_kwargs",
        "downsample_rate",
        "code_dim",
    },
    "architectures": {"MossAudioTokenizerModel"},
    "path_patterns": {"moss_audio_tokenizer", "mossaudiotokenizer", "speech_tokenizer"},
}

__all__ = ["MossAudioTokenizer", "MossAudioTokenizerConfig", "Model", "ModelConfig"]

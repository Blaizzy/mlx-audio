from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class SpecialTokens:
    """
    Dataclass containing special tokens used for text and audio processing.
    """

    bos: str = "<|im_start|>"
    eos: str = "<|im_end|>"
    c1: str = "<|c1_{}|>"
    c2: str = "<|c2_{}|>"
    text_start: str = "<|text_start|>"
    text_end: str = "<|text_end|>"
    voice_characteristic_start: str = "<|voice_characteristic_start|>"
    voice_characteristic_end: str = "<|voice_characteristic_end|>"
    emotion_start: str = "<|emotion_start|>"
    emotion_end: str = "<|emotion_end|>"
    audio_start: str = "<|audio_start|>"
    audio_end: str = "<|audio_end|>"
    time: str = "<|t_{:.2f}|>"
    code: str = "<|code|>"
    energy: str = "<|energy_{}|>"
    spectral_centroid: str = "<|spectral_centroid_{}|>"
    pitch: str = "<|pitch_{}|>"
    word_start: str = "<|word_start|>"
    word_end: str = "<|word_end|>"
    features: str = "<|features|>"
    global_features_start: str = "<|global_features_start|>"
    global_features_end: str = "<|global_features_end|>"

    def to_dict(self) -> Dict[str, str]:
        """Convert the dataclass instance to a dictionary using asdict."""
        return asdict(self)

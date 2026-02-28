from .config import ModelConfig
from .wav2vec_lid import Wav2Vec2ForSequenceClassification as Model

DETECTION_HINTS = {
    "config_keys": {"classifier_proj_size", "id2label", "num_labels"},
    "architectures": {"Wav2Vec2ForSequenceClassification"},
}

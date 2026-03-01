from .config import ModelConfig
from .ecapa_tdnn import EcapaTdnn as Model

DETECTION_HINTS = {
    "config_keys": {"n_mels", "res2net_scale", "se_channels", "embedding_dim"},
    "architectures": {"EcapaTdnn"},
}

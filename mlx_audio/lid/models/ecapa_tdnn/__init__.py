from .config import ModelConfig
from .ecapa_tdnn import ECAPA_TDNN as Model

DETECTION_HINTS = {
    "config_keys": {"n_mels", "res2net_scale", "se_channels", "embedding_dim"},
    "architectures": {"ECAPA_TDNN"},
}

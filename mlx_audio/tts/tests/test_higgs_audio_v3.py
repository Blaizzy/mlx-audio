from mlx_audio import utils
from mlx_audio.tts.utils import MODEL_REMAPPING


def test_higgs_v3_model_type_remapping():
    arch, resolved = utils.get_model_class(
        model_type="higgs_multimodal_qwen3",
        model_name=["bosonai", "higgs-audio-v3-tts-4b"],
        category="tts",
        model_remapping=MODEL_REMAPPING,
    )

    assert resolved == "higgs_audio_v3"
    assert arch.__name__ == "mlx_audio.tts.models.higgs_audio_v3"

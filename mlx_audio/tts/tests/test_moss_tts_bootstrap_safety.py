import unittest

import mlx_audio.tts.models.moss_tts as moss_tts_module
from mlx_audio.convert import Domain, get_detection_hints
from mlx_audio.tts.models.moss_tts import MossNormalizedRequest
from mlx_audio.tts.models.moss_tts.audit import MossNormalizedRequest as AuditRequest


class TestMossTTSPhase2Setup(unittest.TestCase):
    def test_audit_request_alias_points_to_runtime_contract(self):
        self.assertIs(AuditRequest, MossNormalizedRequest)

    def test_detection_hints_are_gated_until_model_entry_points_exist(self):
        self.assertFalse(hasattr(moss_tts_module, "Model"))
        self.assertFalse(hasattr(moss_tts_module, "DETECTION_HINTS"))

    def test_convert_detection_skips_moss_tts_bootstrap_module(self):
        from mlx_audio import convert as convert_module

        convert_module._detection_hints_cache.pop(Domain.TTS.value, None)
        hints = get_detection_hints(Domain.TTS)
        self.assertNotIn("moss_tts", hints.get("path_patterns", {}))


if __name__ == "__main__":
    unittest.main()

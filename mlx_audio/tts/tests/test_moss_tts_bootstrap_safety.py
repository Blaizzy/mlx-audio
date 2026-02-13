import unittest
from unittest.mock import patch

import mlx_audio.tts.models.moss_tts as moss_tts_module
from mlx_audio.convert import Domain, get_detection_hints
from mlx_audio.tts.models.moss_tts import MossNormalizedRequest
from mlx_audio.tts.models.moss_tts.audit import MossNormalizedRequest as AuditRequest
from mlx_audio.tts.utils import get_model_and_args


class TestMossTTSPhase2Runtime(unittest.TestCase):
    def test_audit_request_alias_points_to_runtime_contract(self):
        self.assertIs(AuditRequest, MossNormalizedRequest)

    def test_runtime_entry_points_are_exported(self):
        self.assertTrue(hasattr(moss_tts_module, "Model"))
        self.assertTrue(hasattr(moss_tts_module, "ModelConfig"))
        self.assertTrue(hasattr(moss_tts_module, "DETECTION_HINTS"))

    def test_convert_detection_includes_moss_tts_runtime_hints(self):
        from mlx_audio import convert as convert_module

        convert_module._detection_hints_cache.pop(Domain.TTS.value, None)
        with patch.object(convert_module, "get_model_types", return_value={"moss_tts"}):
            hints = get_detection_hints(Domain.TTS)
        self.assertIn("moss_tts", hints.get("path_patterns", {}))
        self.assertIn("moss-tts", hints["path_patterns"]["moss_tts"])

    def test_model_remapping_resolves_moss_tts_delay_to_moss_tts(self):
        module, resolved = get_model_and_args(
            model_type="moss_tts_delay",
            model_name=["moss", "tts", "local", "transformer"],
        )
        self.assertEqual(resolved, "moss_tts")
        self.assertTrue(hasattr(module, "Model"))

    def test_stage1_remap_precedence_over_repo_name_hints(self):
        module, resolved = get_model_and_args(
            model_type="qwen3_tts",
            model_name=["qwen3", "tts", "voice", "clone"],
        )
        self.assertEqual(resolved, "qwen3_tts")
        self.assertTrue(hasattr(module, "Model"))


if __name__ == "__main__":
    unittest.main()

import re
import unittest
from pathlib import Path

from mlx_audio.tts.models.moss_tts.audit import (
    MossNormalizedRequest,
    load_moss_audio_tokenizer_audit,
    load_moss_variant_invariants,
)


def _repo_root() -> Path:
    # .../mlx_audio/tts/tests -> project root
    return Path(__file__).resolve().parents[3]


class TestMossPhase0Audit(unittest.TestCase):
    def test_load_variant_invariants_from_reference_configs(self):
        invariants = load_moss_variant_invariants(
            _repo_root() / "REFERENCE" / "MOSS-TTS-HF-Repos"
        )

        self.assertEqual(
            set(invariants.keys()),
            {
                "MOSS-TTS",
                "MOSS-TTS-Local",
                "MOSS-TTSD",
                "MOSS-VoiceGenerator",
                "MOSS-SoundEffect",
                "MOSS-TTS-Realtime",
            },
        )

        # High-signal table checks
        self.assertEqual(invariants["MOSS-TTS"].n_vq, 32)
        self.assertEqual(invariants["MOSS-TTS-Local"].n_vq, 32)
        self.assertEqual(invariants["MOSS-TTSD"].n_vq, 16)
        self.assertEqual(invariants["MOSS-VoiceGenerator"].n_vq, 16)
        self.assertEqual(invariants["MOSS-SoundEffect"].n_vq, 16)
        self.assertEqual(invariants["MOSS-TTS-Realtime"].n_vq, 16)

        self.assertEqual(invariants["MOSS-TTS-Realtime"].audio_vocab_size, 1027)
        self.assertEqual(invariants["MOSS-TTS"].audio_vocab_size, 1024)

        self.assertTrue(invariants["MOSS-TTS-Local"].has_local_transformer)
        self.assertFalse(invariants["MOSS-TTS"].has_local_transformer)

    def test_audio_tokenizer_source_is_pinned_and_has_expected_contract(self):
        audit = load_moss_audio_tokenizer_audit(
            _repo_root() / "REFERENCE" / "MOSS-Audio-Tokenizer"
        )
        self.assertTrue(re.match(r"^[0-9a-f]{40}$", audit.commit_hash or ""))
        self.assertEqual(audit.frame_rate_hz, 12.5)
        self.assertEqual(audit.num_quantizers, 32)
        self.assertEqual(audit.codebook_size, 1024)
        # Upstream config still uses a legacy model_type string.
        self.assertEqual(audit.config_model_type, "speech_tokenizer")


class TestMossNormalizedRequest(unittest.TestCase):
    def test_maps_generate_aliases_to_upstream_fields(self):
        request = MossNormalizedRequest.from_generate_kwargs(
            text="hello",
            ref_audio="ref.wav",
            instruct="warm narrator",
            tokens=40,
            quality="studio",
            sound_event="none",
            ambient_sound="office",
            language="en",
        )
        self.assertEqual(request.reference, ["ref.wav"])
        self.assertEqual(request.instruction, "warm narrator")

        payload = request.to_user_message_kwargs()
        self.assertEqual(
            set(payload.keys()),
            {
                "text",
                "reference",
                "instruction",
                "tokens",
                "quality",
                "sound_event",
                "ambient_sound",
                "language",
            },
        )


if __name__ == "__main__":
    unittest.main()

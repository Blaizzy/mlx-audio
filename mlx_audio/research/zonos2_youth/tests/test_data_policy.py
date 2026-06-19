from __future__ import annotations

import unittest

from mlx_audio.research.zonos2_youth.data import (
    assert_split_isolation,
    attach_reference_pairs,
    common_voice_items,
    duplicate_keys,
    map_common_voice_age,
    normalize_transcript_for_prompt,
    stable_hash,
)
from mlx_audio.research.zonos2_youth.rights import default_rights_records
from mlx_audio.research.zonos2_youth.schema import ValidationError


class TestYouthNaturalDataPolicy(unittest.TestCase):
    def test_common_voice_age_mapping_preserves_provided_metadata(self):
        self.assertEqual(map_common_voice_age("teens"), "teen")
        self.assertEqual(map_common_voice_age("twenties"), "adult")
        self.assertEqual(map_common_voice_age(""), "unknown")

    def test_common_voice_filters_validated_teens_without_rehost_assumption(self):
        rows = [
            {
                "client_id": "speaker-a",
                "session_id": "s1",
                "path": "a.mp3",
                "sentence": "  Yeah, I don't know.  ",
                "age": "teens",
                "status": "validated",
                "audio_sha256": "a" * 64,
            },
            {
                "client_id": "speaker-b",
                "session_id": "s1",
                "path": "b.mp3",
                "sentence": "adult row",
                "age": "twenties",
                "status": "validated",
                "audio_sha256": "b" * 64,
            },
            {
                "client_id": "speaker-c",
                "session_id": "s1",
                "path": "c.mp3",
                "sentence": "pending row",
                "age": "teens",
                "status": "pending",
                "audio_sha256": "c" * 64,
            },
        ]
        items = common_voice_items(
            rows,
            dataset="common_voice_scripted",
            release="26.0",
            terms_hash=stable_hash("terms"),
            required_age_band="teen",
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].age_band, "teen")
        self.assertEqual(items[0].original_transcript, "  Yeah, I don't know.  ")
        self.assertEqual(items[0].normalized_transcript, "Yeah, I don't know.")
        self.assertEqual(items[0].rights_lane, "permissive_release")

    def test_speaker_split_isolation_and_reference_separation(self):
        rows = [
            {
                "client_id": "speaker-a",
                "session_id": "s1",
                "path": "a1.mp3",
                "sentence": "one",
                "age": "teens",
                "status": "validated",
                "audio_sha256": "1",
            },
            {
                "client_id": "speaker-a",
                "session_id": "s2",
                "path": "a2.mp3",
                "sentence": "two",
                "age": "teens",
                "status": "validated",
                "audio_sha256": "2",
            },
            {
                "client_id": "speaker-b",
                "session_id": "s1",
                "path": "b1.mp3",
                "sentence": "three",
                "age": "teens",
                "status": "validated",
                "audio_sha256": "3",
            },
        ]
        items = common_voice_items(
            rows,
            dataset="common_voice_scripted",
            release="26.0",
            terms_hash=stable_hash("terms"),
            required_age_band="teen",
        )
        assert_split_isolation(items)
        paired = attach_reference_pairs(items)
        first = next(item for item in paired if item.original_transcript == "one")
        self.assertEqual(len(first.reference_audio_ids), 1)
        self.assertNotEqual(first.reference_audio_ids[0], first.recording_id)

    def test_duplicate_detection_uses_transcript_and_audio_hash(self):
        rows = [
            {
                "client_id": "speaker-a",
                "session_id": "s1",
                "path": "a1.mp3",
                "sentence": "same",
                "age": "teens",
                "status": "validated",
                "audio_sha256": "hash",
            },
            {
                "client_id": "speaker-b",
                "session_id": "s1",
                "path": "b1.mp3",
                "sentence": "same",
                "age": "teens",
                "status": "validated",
                "audio_sha256": "hash",
            },
        ]
        items = common_voice_items(
            rows,
            dataset="common_voice_scripted",
            release="26.0",
            terms_hash=stable_hash("terms"),
            required_age_band="teen",
        )
        self.assertEqual(len(duplicate_keys(items)), 1)

    def test_rights_lanes_are_explicit(self):
        records = default_rights_records()
        lanes = {record.source: record.rights_lane for record in records}
        self.assertEqual(
            lanes["Mozilla Common Voice Scripted Speech English"], "permissive_release"
        )
        self.assertEqual(lanes["MyST Children's Conversational Speech"], "research_noncommercial")
        self.assertEqual(lanes["CSLU Kids Speech"], "separately_licensed")
        self.assertEqual(lanes["CMU Kids Corpus"], "separately_licensed")

    def test_invalid_rights_lane_rejected(self):
        rows = [
            {
                "client_id": "speaker-a",
                "session_id": "s1",
                "path": "a.mp3",
                "sentence": "hello",
                "age": "teens",
                "status": "validated",
            }
        ]
        with self.assertRaises(ValidationError):
            common_voice_items(
                rows,
                dataset="x",
                release="x",
                terms_hash="x",
                rights_lane="mystery",
            )

    def test_transcript_preserves_disfluencies(self):
        text = " um, I-- I don't know "
        self.assertEqual(normalize_transcript_for_prompt(text), "um, I-- I don't know")


if __name__ == "__main__":
    unittest.main()

import types
import unittest

from mlx_audio.stt.generate import _get_cues, _split_cue


class TestSubtitleSegmentation(unittest.TestCase):
    def test_short_cue_is_not_split(self):
        cue = {"start": 0.0, "end": 2.0, "text": "Hello world."}
        self.assertEqual(_split_cue(cue, 40), [cue])

    def test_long_cue_split_preserves_span_and_order(self):
        cue = {"start": 10.0, "end": 30.0, "text": "一二三。四五六。七八九。"}
        out = _split_cue(cue, 2)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["start"], 10.0)
        self.assertEqual(out[-1]["end"], 30.0)
        for a, b in zip(out, out[1:]):
            self.assertEqual(a["end"], b["start"])
        self.assertEqual("".join(c["text"] for c in out), cue["text"])

    def test_get_cues_default_keeps_single_segment(self):
        seg = types.SimpleNamespace(
            segments=[{"start": 0.0, "end": 1.0, "text": "a。b。"}]
        )
        self.assertEqual(len(_get_cues(seg)), 1)
        self.assertEqual(len(_get_cues(seg, max_line_length=1)), 2)


if __name__ == "__main__":
    unittest.main()

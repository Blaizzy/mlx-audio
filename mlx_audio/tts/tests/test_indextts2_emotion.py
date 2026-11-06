import unittest


class TestIndexTTS2Emotion(unittest.TestCase):
    def test_parse_json_english(self):
        from mlx_audio.tts.indextts2.emotion import (
            EMOTION_KEYS,
            normalize_emo_vector,
            parse_emotion_response,
        )

        resp = '{"happy": 0.5, "angry": 0.1, "sad": 0.0, "afraid": 0.0, "disgusted": 0, "melancholic": 0.0, "surprised": 0.2, "calm": 0.0}'
        emo = parse_emotion_response(resp)
        vec_dict, vec = normalize_emo_vector(emo, apply_bias=False)
        self.assertEqual(list(vec_dict.keys()), EMOTION_KEYS)
        self.assertEqual(len(vec), 8)
        self.assertAlmostEqual(vec_dict["happy"], 0.5)
        self.assertAlmostEqual(vec_dict["angry"], 0.1)
        self.assertAlmostEqual(vec_dict["surprised"], 0.2)

    def test_parse_json_chinese_keys(self):
        from mlx_audio.tts.indextts2.emotion import (
            normalize_emo_vector,
            parse_emotion_response,
        )

        resp = '{"高兴": 1.0, "愤怒": 0.2, "悲伤": 0.3, "自然": 0.1}'
        emo = parse_emotion_response(resp)
        vec_dict, _ = normalize_emo_vector(emo, apply_bias=False)
        self.assertAlmostEqual(vec_dict["happy"], 1.0)
        self.assertAlmostEqual(vec_dict["angry"], 0.2)
        self.assertAlmostEqual(vec_dict["sad"], 0.3)
        self.assertAlmostEqual(vec_dict["calm"], 0.1)

    def test_regex_fallback(self):
        from mlx_audio.tts.indextts2.emotion import (
            normalize_emo_vector,
            parse_emotion_response,
        )

        resp = "happy: 0.7, angry=0.2 calm:0"
        emo = parse_emotion_response(resp)
        vec_dict, _ = normalize_emo_vector(emo, apply_bias=False)
        self.assertAlmostEqual(vec_dict["happy"], 0.7)
        self.assertAlmostEqual(vec_dict["angry"], 0.2)

    def test_default_calm_when_empty(self):
        from mlx_audio.tts.indextts2.emotion import normalize_emo_vector

        vec_dict, vec = normalize_emo_vector({}, apply_bias=False)
        self.assertAlmostEqual(vec_dict["calm"], 1.0)
        self.assertEqual(sum(vec), 1.0)

    def test_sum_clamp(self):
        from mlx_audio.tts.indextts2.emotion import normalize_emo_vector

        emo = {"happy": 1.2, "angry": 1.2, "sad": 1.2}
        vec_dict, vec = normalize_emo_vector(emo, apply_bias=False, max_sum=0.8)
        self.assertLessEqual(sum(vec), 0.8 + 1e-6)
        self.assertGreater(vec_dict["happy"], 0.0)


if __name__ == "__main__":
    unittest.main()

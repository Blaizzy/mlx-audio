import unittest
from dataclasses import fields

from mlx_audio.stt.models.whisper.decoding import DecodingOptions
from mlx_audio.stt.models.whisper.whisper import (
    _DECODING_OPTION_NAMES,
    _filter_decode_options,
)


class TestWhisperDecodeOptionFiltering(unittest.TestCase):
    def test_names_match_decoding_options(self):
        self.assertEqual(
            _DECODING_OPTION_NAMES, {field.name for field in fields(DecodingOptions)}
        )

    def test_known_options_are_preserved(self):
        options = {
            "language": "en",
            "task": "transcribe",
            "prefix": "Hello",
            "sample_len": 200,
            "suppress_blank": False,
            "fp16": True,
        }
        self.assertEqual(_filter_decode_options(options), options)

    def test_unknown_options_are_dropped(self):
        filtered = _filter_decode_options(
            {
                "language": "en",
                "frame_threshold": 25,
                "prefill_step_size": 2048,
                "max_tokens": 1024,
                "generation_stream": False,
                "not_a_real_option": object(),
            }
        )
        self.assertEqual(filtered, {"language": "en"})

    def test_filtered_options_construct_decoding_options(self):
        filtered = _filter_decode_options(
            {"language": "en", "frame_threshold": 25, "prefill_step_size": 2048}
        )
        self.assertEqual(DecodingOptions(**filtered).language, "en")

    def test_empty_and_all_unknown(self):
        self.assertEqual(_filter_decode_options({}), {})
        self.assertEqual(_filter_decode_options({"frame_threshold": 25}), {})

    def test_input_is_not_mutated(self):
        options = {"language": "en", "frame_threshold": 25}
        _filter_decode_options(options)
        self.assertEqual(options, {"language": "en", "frame_threshold": 25})


if __name__ == "__main__":
    unittest.main()

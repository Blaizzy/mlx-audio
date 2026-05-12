"""
ASR prompt construction for MiMo-V2.5-ASR.

Ported from ``src/mimo_audio/process_speechdata.py`` and ``mimo_audio.py``.

Builds the interleaved text+speech input format:
  Row 0: text token IDs (with group_size filler tokens between real tokens)
  Rows 1-8: speech token IDs (zero for text-only regions, codes for audio regions)
"""

import random
from typing import List, Optional

import mlx.core as mx


# ── ASR prompt templates ────────────────────────────────────────────

ASR_ZH_TEMPLATES = [
    "请将这段语音转换为文字",
    "帮我识别这个音频文件中的内容",
    "把这段录音转成文本",
    "请转录这段语音",
    "将音频内容转换成文字格式",
    "识别并转写这段语音",
    "把语音内容写成文字",
    "转录这个音频片段",
    "将这段对话转换为文本",
    "麻烦帮我把这段录音整理成详细的文字记录",
]

ASR_EN_TEMPLATES = [
    "Please transcribe this audio file",
    "Convert this speech recording to text",
    "Transcribe the following voice message",
    "Turn this audio into readable text",
    "Please convert the recording to written format",
    "Transcribe what you hear in this audio",
    "Convert this spoken content to text",
    "Please write down what is said in this recording",
    "Transcribe this voice recording",
    "Could you please help me transcribe this important recording?",
    "Would you mind converting this voice message into a readable text format?",
    "I'd really appreciate it if you could turn this audio file into a written document",
]


# ── Input segment builder ───────────────────────────────────────────

class InputSegment:
    """
    A segment of the ASR prompt — either text-only or audio+text.

    Produces a (audio_channels+1, seg_len) tensor where:
      - Row 0: text token IDs
      - Rows 1..audio_channels: speech token IDs
    """

    def __init__(
        self,
        text_token_ids: Optional[mx.array] = None,
        text_str: Optional[str] = None,
        audio_codes: Optional[mx.array] = None,  # (n_tokens,), int32
        speech_empty_ids: Optional[List[int]] = None,
        text_empty_id: int = 151667,  # <|empty|>
        sosp_id: int = 151665,
        eosp_id: int = 151666,
        group_size: int = 4,
        audio_channels: int = 8,
        add_sosp_eosp: bool = True,
    ):
        self.text_token_ids = text_token_ids
        self.text_str = text_str
        self.audio_codes = audio_codes
        self.speech_empty_ids = speech_empty_ids or [128] * audio_channels
        self.text_empty_id = text_empty_id
        self.sosp_id = sosp_id
        self.eosp_id = eosp_id
        self.group_size = group_size
        self.audio_channels = audio_channels
        self.add_sosp_eosp = add_sosp_eosp

    def expand_with_fillers(self, ids: mx.array, filler: int) -> mx.array:
        """
        Insert (group_size - 1) filler tokens between each real token.

        Example (group_size=4):
          [a, b, c] → [a, _, _, _, b, _, _, _, c, _, _, _]
                           ^filler^      ^filler^      ^filler^
        """
        gs = self.group_size
        n = ids.shape[0]
        # Result has n + (n-1)*(gs-1) = n*gs - gs + 1 tokens
        # But the reference code uses:
        # total = n + (n-1)*(gs-1) + gs-1 = n*gs
        # So we need n*gs positions, with the first token at position 0,
        # then tokens at positions gs, 2*gs, etc.
        out_len = n * gs
        result = mx.full((out_len,), filler, dtype=ids.dtype)
        # Place tokens at indices 0, gs, 2*gs, ...
        indices = mx.arange(0, out_len, gs, dtype=mx.int32)
        result[indices] = ids
        return result

    def build(self) -> mx.array:
        """
        Build the (audio_channels+1, seg_len) tensor.
        """
        ac = self.audio_channels
        gs = self.group_size

        if self.audio_codes is None:
            # ── Text-only segment ──
            text_ids = self.text_token_ids
            if text_ids is None and self.text_str is not None:
                raise ValueError(
                    "text_token_ids not provided; use tokenizer.encode() first"
                )

            # Expand with filler tokens
            text_ids = self.expand_with_fillers(text_ids, filler=-100)

            seg_len = text_ids.shape[0]

            # Speech rows: fill with empty/padding IDs
            speech_rows = mx.zeros((ac, seg_len), dtype=mx.int32)
            for i in range(ac):
                speech_rows[i, :] = self.speech_empty_ids[i]

            return mx.concatenate([text_ids[None, :], speech_rows], axis=0)

        else:
            # ── Audio segment ──
            audio_codes = self.audio_codes  # (total_tokens,)

            # Verify divisible by group_size * audio_channels.
            assert audio_codes.shape[0] % (gs * ac) == 0, (
                f"Audio codes length {audio_codes.shape[0]} is not divisible "
                f"by group_size * audio_channels ({gs} * {ac})"
            )

            # Reshape to [n_groups, group_size, audio_channels]
            n_groups = audio_codes.shape[0] // (gs * ac)
            speech_mat = audio_codes.reshape(n_groups, gs * ac)  # (n_groups, gs*ac)

            # Text tokens: all empty
            text_len = n_groups
            text_ids = mx.full((text_len,), self.text_empty_id, dtype=mx.int32)

            if self.add_sosp_eosp:
                text_ids = mx.concatenate([
                    mx.array([self.sosp_id], dtype=mx.int32),
                    text_ids,
                    mx.array([self.eosp_id], dtype=mx.int32),
                ])
                n_groups += 2

                # Add zero speech rows for sosp/eosp markers
                sosp_speech = mx.zeros((gs, ac), dtype=mx.int32)
                eosp_speech = mx.zeros((gs, ac), dtype=mx.int32)
                for i in range(ac):
                    sosp_speech[:, i] = self.speech_empty_ids[i]
                    eosp_speech[:, i] = self.speech_empty_ids[i]

                speech_mat = mx.concatenate([
                    sosp_speech.reshape(1, -1),
                    speech_mat,
                    eosp_speech.reshape(1, -1),
                ], axis=0)

            # Expand text with filler tokens
            text_ids = self.expand_with_fillers(text_ids, filler=-100)
            seg_len = text_ids.shape[0]

            # Build speech rows: [audio_channels, seq_len].
            # Reference: audio.reshape(-1, audio_channels).T.
            speech_rows = (
                speech_mat.reshape(n_groups, gs, ac)
                .transpose(2, 0, 1)
                .reshape(ac, n_groups * gs)
            )

            return mx.concatenate([text_ids[None, :], speech_rows], axis=0)


# ── ASR prompt builder ──────────────────────────────────────────────

def build_asr_prompt(
    audio_codes: mx.array,  # (total_tokens,) — flattened speech codes [T*8]
    text_token_ids: List[mx.array],  # list of tokenized text segments
    config,  # MiMoAudioConfig
    tokenizer=None,
    language: str = "auto",  # "zh", "en", or "auto"
) -> mx.array:
    """
    Build the full ASR prompt input_ids.

    Prompt structure (matching reference get_asr_sft_prompt):
      <|im_start|>user\\n
      [audio tokens: sosp, speech codes..., eosp]
      ASR instruction (zh/en template)
      <|im_end|>\\n
      <|im_start|>assistant\\n
      thinking\\n\\nresponse\\n

    Returns
    -------
    mx.array, shape (audio_channels+1, total_seq_len)
    """
    ac = config.audio_channels
    gs = config.group_size
    empty_ids = config.parsed_speech_empty_ids()

    # Use FIXED template matching reference model's training tokenization
    # Actual reference text: "将音频内容转换成文字格式"
    if language == "zh":
        template_tokens = [44063, 111268, 43815, 105359, 12857, 87335, 68805]
    elif language == "en":
        template_tokens = tokenizer.encode(random.choice(ASR_EN_TEMPLATES))
    else:
        template_tokens = [44063, 111268, 43815, 105359, 12857, 87335, 68805]

    segments = []

    # Segment 1: <|im_start|>user\n
    seg1_text = mx.array(tokenizer.encode("<|im_start|>user\n"), dtype=mx.int32)
    segments.append(
        InputSegment(
            text_token_ids=seg1_text,
            speech_empty_ids=empty_ids,
            text_empty_id=config.empty_idx,
            sosp_id=config.sosp_idx,
            eosp_id=config.eosp_idx,
            group_size=gs,
            audio_channels=ac,
            add_sosp_eosp=True,
        ).build()
    )

    # Segment 2: audio codes
    segments.append(
        InputSegment(
            audio_codes=audio_codes,
            speech_empty_ids=empty_ids,
            text_empty_id=config.empty_idx,
            sosp_id=config.sosp_idx,
            eosp_id=config.eosp_idx,
            group_size=gs,
            audio_channels=ac,
            add_sosp_eosp=True,
        ).build()
    )

    # Segment 3: ASR instruction
    seg3_text = mx.array(template_tokens, dtype=mx.int32)
    segments.append(
        InputSegment(
            text_token_ids=seg3_text,
            speech_empty_ids=empty_ids,
            text_empty_id=config.empty_idx,
            group_size=gs,
            audio_channels=ac,
            add_sosp_eosp=False,
        ).build()
    )

    # Segment 4: <|im_end|>\n
    seg4_text = mx.array(tokenizer.encode("<|im_end|>\n"), dtype=mx.int32)
    segments.append(
        InputSegment(
            text_token_ids=seg4_text,
            speech_empty_ids=empty_ids,
            text_empty_id=config.empty_idx,
            group_size=gs,
            audio_channels=ac,
            add_sosp_eosp=False,
        ).build()
    )

    # Segment 5: <|im_start|>assistant\n
    seg5_text = mx.array(tokenizer.encode("<|im_start|>assistant\n"), dtype=mx.int32)
    segments.append(
        InputSegment(
            text_token_ids=seg5_text,
            speech_empty_ids=empty_ids,
            text_empty_id=config.empty_idx,
            group_size=gs,
            audio_channels=ac,
            add_sosp_eosp=False,
        ).build()
    )

    # Segment 6: hardcoded reference tokenization (matches model training)
    if language == "zh":
        seg6_tokens = [13708, 766, 1339, 522, 26865, 397, 27, 331, 7346, 29]
    elif language == "en":
        seg6_tokens = [13708, 766, 1339, 522, 26865, 397, 27, 974, 975, 678, 29]
    else:
        seg6_tokens = tokenizer.encode(" thinking\n\n response\n<chinese>")
    seg6_text = mx.array(seg6_tokens, dtype=mx.int32)
    segments.append(
        InputSegment(
            text_token_ids=seg6_text,
            speech_empty_ids=empty_ids,
            text_empty_id=config.empty_idx,
            group_size=gs,
            audio_channels=ac,
            add_sosp_eosp=False,
        ).build()
    )

    return mx.concatenate(segments, axis=1)

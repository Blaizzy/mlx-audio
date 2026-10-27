"""Training prompt and control-caption contracts for Ming-3.0-tts-tiny."""

import json
import re

LANGUAGES = {
    "zh": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "french": "French",
    "ko": "Korean",
    "korean": "Korean",
    "ja": "Japanese",
    "japanese": "Japanese",
}
ALIASES = {
    "tts": "seed",
    "clone": "seed",
    "style": "voice_design",
    "voice_actor": "ip",
    "nvbench": "paralinguistic",
}
REFERENCE_TASKS = {"seed", "multilingual", "paralinguistic", "podcast"}
SPEAKER_TASKS = {"basic", "emotion", "dialect"}
MASKED_TASKS = {"voice_design", "ip", "freeform"}


def control_instruction(task, *, instruct=None, **controls):
    controls = {key: value for key, value in controls.items() if value is not None}
    if task in REFERENCE_TASKS:
        if instruct is not None or controls:
            raise ValueError(f"{task} does not accept style controls")
        return None
    if task == "freeform":
        if controls or not isinstance(instruct, str) or not instruct.strip():
            raise ValueError("freeform requires instruct and no structured controls")
        return (
            instruct
            if instruct.startswith("Speaking instruction:")
            else f"Speaking instruction: Speak in this style: {instruct}"
        )
    required = (
        {"speed", "pitch", "volume"}
        if task == "basic"
        else {"style" if task == "voice_design" else task}
    )
    if (
        instruct is not None
        or set(controls) != required
        or any(
            not isinstance(value, str) or not value.strip()
            for value in controls.values()
        )
    ):
        raise ValueError(f"{task} requires exactly these controls: {sorted(required)}")
    dialects = {
        "cantonese": "广粤话",
        "sichuanese": "川渝话",
        "chuanyu dialect": "川渝话",
    }
    emotions = {
        "happy": "高兴",
        "sad": "悲伤",
        "angry": "愤怒",
        "neutral": "中性",
        "surprised": "惊讶",
        "fearful": "恐惧",
        "disgusted": "厌恶",
    }
    if "dialect" in controls:
        controls["dialect"] = dialects.get(
            controls["dialect"].lower(), controls["dialect"]
        )
    if "emotion" in controls:
        controls["emotion"] = emotions.get(
            controls["emotion"].lower(), controls["emotion"]
        )
    allowed = {
        "dialect": {"广粤话", "川渝话"},
        "emotion": set(emotions.values()),
        "speed": {"慢速", "中速", "快速"},
        "pitch": {"低", "中", "高"},
        "volume": {"低", "中", "高"},
    }
    for key, value in controls.items():
        if key in allowed and value not in allowed[key]:
            raise ValueError(f"Unsupported {key}: {value}")
    fields = ("方言", "情感", "风格", "语速", "基频", "音量")
    names = ("dialect", "emotion", "style", "speed", "pitch", "volume")
    if task == "emotion":
        fields, names = ("方言", "风格", "语速", "基频", "音量", "情感"), (
            "dialect",
            "style",
            "speed",
            "pitch",
            "volume",
            "emotion",
        )
    item = {"序号": 1, "说话人": "speaker_1"}
    item.update({field: controls.get(name) for field, name in zip(fields, names)})
    item["BGM"] = {
        key: None for key in ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")
    }
    item["IP"] = controls.get("ip")
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


def build_prompt(
    tokenizer,
    config,
    text,
    *,
    instruction=None,
    language=None,
    speakers=1,
    prompt_text=None,
    prompt_steps=0,
):
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    prefix = (
        "<role>HUMAN</role>Please generate speech based on the following description.\n"
    )
    if language:
        prefix += f"Target language: {language}.\n"
    # Freeform belongs to the user turn; structured task captions to the assistant.
    assistant_instruction = instruction is not None and not instruction.startswith(
        "Speaking instruction:"
    )
    if instruction is not None and not assistant_instruction:
        prefix += instruction + "\n"
    if speakers == 1:
        prefix += "  speaker_1:"
    ids = encode(prefix)
    positions = []
    for index in range(speakers):
        if speakers > 1:
            ids += encode(("\n" if index else "") + f"  speaker_{index + 1}:")
        ids.append(config["speaker_start_token_id"])
        positions.append(len(ids))
        ids += [config["audio_patch_token_id"], config["speaker_end_token_id"]]
    ids += encode("\n Text input:\n")
    if prompt_text is not None:
        ids += encode(prompt_text)
    ids += encode(text) + encode("<role>ASSISTANT</role>")
    if assistant_instruction:
        ids += encode(instruction) + [tokenizer.eos_token_id]
    ids += [config["audio_start_token_id"]] + [
        config["audio_patch_token_id"]
    ] * prompt_steps
    return ids, positions


def podcast_transcript(texts):
    return "".join(f" speaker_{i + 1}:{text}\n" for i, text in enumerate(texts))


def validate_podcast(text, count):
    if count < 2:
        raise ValueError("Podcast requires at least two speakers")
    seen = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\s*speaker_(\d+)\s*:\s*\S", line)
        if match is None or not 1 <= int(match[1]) <= count:
            raise ValueError("Every podcast line must start with a valid speaker_N:")
        seen.add(int(match[1]))
    if seen != set(range(1, count + 1)):
        raise ValueError("Podcast dialogue must include every reference speaker")

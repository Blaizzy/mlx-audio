import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


EMOTION_KEYS = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
]


CN_TO_EN = {
    "高兴": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "afraid",
    "反感": "disgusted",
    "低落": "melancholic",
    "惊讶": "surprised",
    "自然": "calm",
}


EMO_BIAS = {
    # Bias factors from the official IndexTTS2 inference helper.
    # Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    "happy": 0.9375,
    "angry": 0.875,
    "sad": 1.0,
    "afraid": 1.0,
    "disgusted": 0.9375,
    "melancholic": 0.9375,
    "surprised": 0.6875,
    "calm": 0.5625,
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _coerce_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_emotion_response(text: str) -> Dict[str, float]:
    """Parse a model response into an emotion dict.

    Accepts either:
    - JSON with English keys
    - JSON with Chinese keys (mapped via CN_TO_EN)
    - Loose `key: number` pairs in free-form text
    """

    text = text.strip()

    # Try to extract a JSON object substring first (common for chatty outputs).
    m = re.search(r"\{[\s\S]*\}", text)
    json_blob = m.group(0) if m else None

    candidates = [json_blob, text] if json_blob else [text]
    for blob in candidates:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return _normalize_emotion_dict(obj)
        except Exception:
            pass

    # Fallback: regex parse key/value pairs.
    # Matches: happy: 0.5, "angry":0.2, 高兴: 1.0, etc.
    pairs: Dict[str, float] = {}
    for key, val in re.findall(
        r"([A-Za-z_]+|[\u4e00-\u9fff]+)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        text,
    ):
        f = _coerce_float(val)
        if f is None:
            continue
        pairs[key] = f

    return _normalize_emotion_dict(pairs)


def _normalize_emotion_dict(obj: Dict[str, Any]) -> Dict[str, float]:
    # Map keys to English and drop unknown keys.
    out: Dict[str, float] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        key = k.strip()
        if key in CN_TO_EN:
            key = CN_TO_EN[key]

        if key not in EMOTION_KEYS:
            continue

        f = _coerce_float(v)
        if f is None:
            continue

        out[key] = f

    return out


def normalize_emo_vector(
    emo: Dict[str, float],
    *,
    min_score: float = 0.0,
    max_score: float = 1.2,
    apply_bias: bool = True,
    max_sum: float = 0.8,
) -> Tuple[Dict[str, float], list[float]]:
    """Clamp + bias + sum-normalize emotion vectors.

    Returns both a dict (by key) and a list in EMOTION_KEYS order.
    """

    vec: Dict[str, float] = {k: 0.0 for k in EMOTION_KEYS}
    for k in EMOTION_KEYS:
        if k in emo:
            vec[k] = _clamp(float(emo[k]), min_score, max_score)

    # Default to neutral/calm if empty.
    if all(v <= 0.0 for v in vec.values()):
        vec["calm"] = 1.0

    if apply_bias:
        for k in EMOTION_KEYS:
            vec[k] *= EMO_BIAS[k]

    s = sum(vec.values())
    if s > max_sum and s > 0:
        scale = max_sum / s
        for k in EMOTION_KEYS:
            vec[k] *= scale

    return vec, [vec[k] for k in EMOTION_KEYS]


@dataclass
class QwenEmotionConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct-4bit"
    max_tokens: int = 256
    temperature: float = 0.0
    apply_bias: bool = True


_LLM_CACHE: Dict[str, Tuple[Any, Any]] = {}


class QwenEmotion:
    """Emotion-from-text using an MLX LLM (Qwen-family recommended).

    This is MLX-native (mlx_lm) and returns an IndexTTS2-style emotion vector.
    """

    def __init__(self, config: Optional[QwenEmotionConfig] = None):
        self.config = config or QwenEmotionConfig()

        # Words that should tilt sad->melancholic (mirrors official helper hack).
        self._melancholic_words = {
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }

    def _load_llm(self):
        if self.config.model in _LLM_CACHE:
            return _LLM_CACHE[self.config.model]

        from mlx_lm.utils import load as load_llm

        llm, tokenizer = load_llm(self.config.model)
        _LLM_CACHE[self.config.model] = (llm, tokenizer)
        return llm, tokenizer

    def _prompt(self, text: str) -> str:
        llm, tokenizer = self._load_llm()
        del llm

        system = (
            "You are a text emotion classifier. "
            "Return ONLY valid JSON with exactly these keys: "
            "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm. "
            "Values must be numbers in range [0.0, 1.2]."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def infer(self, text: str) -> Tuple[Dict[str, float], list[float]]:
        llm, tokenizer = self._load_llm()

        from mlx_lm.generate import generate

        prompt = self._prompt(text)
        resp = generate(
            llm,
            tokenizer,
            prompt,
            max_tokens=self.config.max_tokens,
            temp=self.config.temperature,
            verbose=False,
        )

        emo = parse_emotion_response(resp)

        # Sad vs melancholic swap workaround.
        text_lower = text.lower()
        if any(w in text_lower for w in self._melancholic_words):
            emo["sad"], emo["melancholic"] = emo.get("melancholic", 0.0), emo.get(
                "sad", 0.0
            )

        return normalize_emo_vector(emo, apply_bias=self.config.apply_bias)

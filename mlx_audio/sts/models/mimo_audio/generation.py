from dataclasses import dataclass

import mlx.core as mx

from mlx_audio.lm.sample_utils import apply_top_p


@dataclass
class Sampler:
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 0.95

    def __post_init__(self):
        if self.temperature < 0 or self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid sampling parameters")

    def __call__(self, logits, removed=()):
        scores = logits.astype(mx.float32)
        if self.temperature:
            scores = scores / self.temperature
            if 0 < self.top_k < scores.shape[-1]:
                threshold = mx.sort(scores, axis=-1)[..., -self.top_k, None]
                scores = mx.where(scores < threshold, -float("inf"), scores)
            if self.top_p < 1:
                scores = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
                scores = apply_top_p(scores, self.top_p)
        if removed:
            indices = mx.array(list(removed))
            scores = mx.put_along_axis(
                scores, indices[None], mx.array(-float("inf")), axis=-1
            )
            # A padding-only top-k set can otherwise produce NaN probabilities.
            fallback = mx.put_along_axis(
                logits.astype(mx.float32),
                indices[None],
                mx.array(-float("inf")),
                axis=-1,
            )
            scores = mx.where(
                mx.any(mx.isfinite(scores), axis=-1, keepdims=True), scores, fallback
            )
        return (
            mx.random.categorical(scores)
            if self.temperature
            else mx.argmax(scores, axis=-1)
        )


@dataclass
class GenerationResult:
    text: str
    audio: mx.array | None
    audio_codes: mx.array
    sample_rate: int
    tokens: mx.array
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str
    processing_time: float = 0.0
    reasoning: str | None = None
    raw_text: str = ""


def split_reasoning(text, *, thinking_prefix=False):
    """Separate a completed answer from optional, possibly prefilled thinking."""
    text = text.strip()
    if text.startswith("<think>"):
        thinking_prefix = True
        text = text[len("<think>") :].lstrip()
    if thinking_prefix:
        reasoning, separator, answer = text.partition("</think>")
        return answer.strip() if separator else "", reasoning.strip()
    return text, None


def generate_tokens(
    model,
    prompt,
    *,
    max_new_tokens=2048,
    min_new_tokens=0,
    stop_tokens=(),
    global_sampler=None,
    local_sampler=None,
    prefill_step_size=256,
):
    """Offline nested decoding. Token counts are global positions (audio patches)."""
    c = model.config
    if (
        prompt.ndim != 3
        or prompt.shape[0] != 1
        or prompt.shape[1] != c.audio_channels + 1
    ):
        raise ValueError("MiMo generation expects (1, audio_channels + 1, frames)")
    if prompt.shape[-1] == 0 or prompt.shape[-1] % c.group_size:
        raise ValueError("The prompt must contain complete, nonempty patches")
    prompt_length = prompt.shape[-1] // c.group_size
    if (
        max_new_tokens <= 0
        or not 0 <= min_new_tokens <= max_new_tokens
        or prefill_step_size <= 0
    ):
        raise ValueError("Invalid generation or prefill length")
    available = c.max_position_embeddings - prompt_length
    if available <= 0 or min_new_tokens > available:
        raise ValueError("The prompt exceeds the model context budget")
    limit = min(max_new_tokens, available)
    global_sampler = global_sampler or Sampler()
    local_sampler = local_sampler or Sampler(temperature=0.9)
    cache = model.make_cache()
    step = prefill_step_size * c.group_size
    for start in range(0, prompt.shape[-1], step):
        output = model(prompt[:, :, start : start + step], cache=cache)
        mx.eval(
            output.text_logits,
            output.local_hidden_states,
            [layer.state for layer in cache],
        )
    parts = []
    finish_reason = "length"
    stop_tokens = set(stop_tokens)
    for i in range(limit):
        text_token = global_sampler(
            output.text_logits[:, -1], stop_tokens if i < min_new_tokens else ()
        )
        token_id = text_token.item()
        if token_id == c.empty_token_id:
            speech = model.local_forward(output.local_hidden_states, local_sampler)
        else:
            speech = mx.broadcast_to(
                mx.array(c.speech_empty_ids)[None, None],
                (1, c.group_size, c.audio_channels),
            )
        text = mx.broadcast_to(text_token[:, None, None], (1, c.group_size, 1))
        part = mx.concatenate([text, speech], axis=-1).transpose(0, 2, 1)
        mx.eval(part)
        parts.append(part)
        if token_id in stop_tokens:
            finish_reason = "stop"
            break
        if i + 1 < limit:
            output = model(part, cache=cache)
    return mx.concatenate(parts, axis=-1)[0], finish_reason

"""CTC proposals with greedy LLM verification and KV-cache rollback.

The BPE head and posterior-weighted pooling follow IBM's reference notebook:
https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/speculative_decoding_bpe.ipynb
Unlike its confidence-threshold shortcut, every emitted token is an LLM argmax.
Batched low-precision kernels can still differ from single-token evaluation at
near ties; this is not a guarantee of bitwise-identical greedy output.
"""

import time

import mlx.core as mx


def posterior_weighted_pool(hidden, importance, window_size=4):
    """Pool encoder frames using their mid-layer nonblank probabilities."""
    batch, length, dim = hidden.shape
    padding = (-length) % window_size
    if padding:
        hidden = mx.pad(hidden, [(0, 0), (0, padding), (0, 0)])
        importance = mx.pad(importance, [(0, 0), (0, padding)])
    hidden = hidden.reshape(batch, -1, window_size, dim)
    importance = importance.reshape(batch, -1, window_size)
    weights = importance / (mx.sum(importance, axis=-1, keepdims=True) + 1e-8)
    return mx.sum(hidden * weights[..., None], axis=2)


def collapse_bpe_tokens(labels):
    """Collapse CTC repeats/blanks and map label i to LLM token i-1."""
    tokens = []
    previous = None
    for label in labels:
        if label != previous and label != 0:
            tokens.append(label - 1)
        previous = label
    return tokens


class DraftLookup:
    """Resynchronize a fixed CTC draft after an LLM correction or insertion."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.positions = {}
        for index, token in enumerate(tokens):
            self.positions.setdefault(token, []).append(index)
        self.cursor = 0

    def propose(self, history, count):
        if not history or count == 0:
            return []
        positions = self.positions.get(history[-1], ())
        for size in range(min(4, len(history)), 0, -1):
            suffix = history[-size:]
            for end in positions:
                if end + 1 < self.cursor or end + 1 - size < 0:
                    continue
                if self.tokens[end + 1 - size : end + 1] == suffix:
                    self.cursor = end + 1
                    return self.tokens[self.cursor : self.cursor + count]
        return []


def speculative_generate_step(
    prompt,
    model,
    *,
    input_embeddings,
    draft_tokens,
    max_tokens=4096,
    num_draft_tokens=16,
    prefill_step_size=2048,
    stats=None,
):
    """Yield greedy tokens, verifying CTC continuations in parallel.

    A rejected suffix is removed from every cache. The correction (or bonus
    token after a fully accepted block) remains uncached until the next call.
    Empty/exhausted drafts naturally reduce to ordinary greedy decoding.
    """
    from mlx_audio.lm.generate import generation_stream

    if num_draft_tokens < 1 or prefill_step_size < 1 or max_tokens < 0:
        raise ValueError("Invalid speculative decoding token or prefill limit.")
    if max_tokens == 0:
        return
    stats = stats if stats is not None else {}
    stats.update(
        draft_tokens=len(draft_tokens),
        proposed_tokens=0,
        accepted_tokens=0,
        verification_steps=0,
        autoregressive_steps=0,
    )
    cache = model.make_cache()
    lookup = DraftLookup(draft_tokens)
    start = time.perf_counter()
    with mx.stream(generation_stream):
        remaining = len(prompt)
        offset = 0
        while remaining > 1:
            count = min(prefill_step_size, remaining - 1)
            model(
                prompt[offset : offset + count][None],
                cache=cache,
                input_embeddings=input_embeddings[offset : offset + count][None],
            )
            mx.eval([layer.state for layer in cache])
            remaining -= count
            offset += count
        logits = model(
            prompt[offset:][None],
            cache=cache,
            input_embeddings=input_embeddings[offset:][None],
        )
        current = mx.argmax(logits[0, -1], axis=-1).item()
    stats["prefill_time"] = time.perf_counter() - start
    history = [current]
    eos = model._tokenizer.eos_token_id
    yield current, None
    while len(history) < max_tokens and current != eos:
        count = min(num_draft_tokens, max_tokens - len(history) - 1)
        proposals = lookup.propose(history, count)
        with mx.stream(generation_stream):
            logits = model(mx.array([[current, *proposals]]), cache=cache)
            predictions = mx.argmax(logits[0], axis=-1).tolist()
        accepted = 0
        for proposal, prediction in zip(proposals, predictions):
            if proposal != prediction:
                break
            accepted += 1
        for layer in cache:
            layer.trim(len(proposals) - accepted)
        lookup.cursor += accepted
        stats["proposed_tokens"] += len(proposals)
        stats["accepted_tokens"] += accepted
        stats["verification_steps" if proposals else "autoregressive_steps"] += 1
        for token in predictions[: accepted + 1]:
            history.append(token)
            current = token
            yield token, None
            if token == eos:
                return

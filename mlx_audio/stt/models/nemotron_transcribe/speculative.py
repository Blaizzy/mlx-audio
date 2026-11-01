"""Greedy MTP drafting with parallel target verification and hybrid rollback."""

import mlx.core as mx

from mlx_audio.lm.generate import generation_stream
from mlx_audio.lm.models.cache import KVCache

from .language_model import MambaCache


def speculative_generate_step(
    prompt,
    model,
    *,
    input_embeddings,
    max_tokens=1024,
    num_draft_tokens=4,
    prefill_step_size=512,
    logits_processors=None,
    stats=None,
):
    """Emit target argmax tokens; sampling is deliberately handled by the caller.

    Batched low-precision evaluation can differ at near ties from single-token
    evaluation. Every emitted token is verified by the target, including EOS.
    All caches belong to this generator, so closing a stream discards them.
    """
    if max_tokens < 0 or num_draft_tokens < 1 or prefill_step_size < 1:
        raise ValueError("Invalid speculative decoding token or prefill limit")
    stats = stats if stats is not None else {}
    stats.update(proposed_tokens=0, accepted_tokens=0, verification_steps=0)
    if max_tokens == 0:
        return
    lm = model.llm
    cache, draft_cache = lm.make_cache(), [KVCache()]
    history = prompt

    def predict(hidden, context):
        logits = lm.lm_head(hidden)
        for processor in logits_processors or ():
            logits = processor(context, logits)
        return mx.argmax(logits, axis=-1)

    with mx.stream(generation_stream):
        for start in range(0, len(prompt), prefill_step_size):
            end = min(start + prefill_step_size, len(prompt))
            hidden = lm.backbone(
                input_embeddings=input_embeddings[None, start:end], cache=cache
            )
            shifted = input_embeddings[None, start + 1 : end + 1]
            if end == len(prompt):
                current = predict(hidden[:, -1], history).item()
                shifted = mx.concatenate(
                    [shifted, lm.backbone.embeddings(mx.array([[current]]))], axis=1
                )
            draft_hidden = lm.mtp(shifted, hidden, draft_cache)
            mx.eval(draft_hidden, [c.state for c in cache], draft_cache[0].state)
    emitted = 1
    yield current, None
    while emitted < max_tokens and current not in model.config.eos_token_id:
        count = min(num_draft_tokens, max_tokens - emitted - 1)
        proposals = []
        with mx.stream(generation_stream):
            draft_offset = draft_cache[0].offset
            for index in range(count):
                context = mx.concatenate([history, mx.array([current, *proposals])])
                proposal = predict(draft_hidden[:, -1], context).item()
                proposals.append(proposal)
                if proposal in model.config.eos_token_id or index + 1 == count:
                    break
                draft_hidden = lm.mtp(
                    lm.backbone.embeddings(mx.array([[proposal]])),
                    draft_hidden[:, -1:],
                    draft_cache,
                )
            for layer_cache in cache:
                if isinstance(layer_cache, MambaCache):
                    layer_cache.recording = True
            hidden = lm.backbone(mx.array([[current, *proposals]]), cache=cache)
            # Process each row with exactly the history ordinary generation sees.
            # This also makes repetition penalties correct after partial acceptance.
            logits = lm.lm_head(hidden)[0]
            if logits_processors:
                rows = []
                for i in range(len(proposals) + 1):
                    row = logits[i : i + 1]
                    context = mx.concatenate(
                        [history, mx.array([current, *proposals[:i]])]
                    )
                    for processor in logits_processors:
                        row = processor(context, row)
                    rows.append(row)
                logits = mx.concatenate(rows)
            predictions = mx.argmax(logits, axis=-1).tolist()
            accepted = 0
            for proposal, prediction in zip(proposals, predictions):
                if proposal != prediction:
                    break
                accepted += 1
            rejected = len(proposals) - accepted
            lm.finish_verification(cache, accepted + 1, rejected)
            stats["proposed_tokens"] += len(proposals)
            stats["accepted_tokens"] += accepted
            stats["verification_steps"] += 1
            output = predictions[: accepted + 1]
            history = mx.concatenate(
                [history, mx.array([current, *proposals[:accepted]])]
            )
            # Replace recursively drafted states with verified target states.
            # The existing draft prefix already contains the current token.
            draft_cache[0].trim(draft_cache[0].offset - draft_offset)
            draft_hidden = lm.mtp(
                lm.backbone.embeddings(mx.array([output])),
                hidden[:, : accepted + 1],
                draft_cache,
            )[:, -1:]
            mx.eval(draft_hidden, [c.state for c in cache], draft_cache[0].state)
        for token in output:
            current = token
            emitted += 1
            yield token, None
            if token in model.config.eos_token_id:
                return

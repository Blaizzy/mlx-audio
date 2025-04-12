import torch


def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def apply_unified(probs: torch.Tensor, linear: float, conf: float, quad: float) -> torch.Tensor:
    """Sample next token using unified sampling approach that combines linear scaling, confidence, and quadratic terms.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        linear (float): Linear scaling factor applied to log probabilities.
        conf (float): Confidence factor that scales the entropy term.
        quad (float): Quadratic penalty factor applied to squared log probabilities.
    Returns:
        torch.Tensor: Modified probability distribution after applying unified sampling.
    """
    logprobs = torch.log(probs.clamp_min(1e-20))
    entropy = -torch.sum(probs * logprobs, dim=-1, keepdim=True)
    raw = logprobs * (linear + entropy * conf) - logprobs**2 * quad
    return raw.softmax(dim=-1)

def apply_top_k(
    probs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    v, _ = torch.topk(probs, min(k, probs.size(-1)))
    pivot = v.select(-1, -1).unsqueeze(-1)
    probs = torch.where(probs < pivot, 0.0, probs)
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs = probs.scatter(-1, probs_idx, probs_sort)
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Sample next token using min-p sampling.

    Args:
        scores (torch.FloatTensor): Input logits with token candidates on the last dimension.
        min_p (float): Minimum token probability, scaled by the probability of the most likely token.
                       Must be between 0 and 1. Typical values are in the 0.01-0.2 range.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    tokens_to_remove = probs < (min_p * top_probs)
    probs = probs.masked_fill(tokens_to_remove, 0.0)
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def modify_logit_for_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    repetition_penalty: float,
    repetition_penalty_window: int,
):
    """See https://arxiv.org/abs/1909.05858
    Apply repetition penalty over a sliding window of the last `repetition_penalty_window` tokens.
    logits: (batch_size, n_codebooks, vocab_size)
    generated_tokens: (batch_size, n_codebooks, seq_len)
    """
    generated_tokens = generated_tokens[..., -repetition_penalty_window:]
    generated_tokens = generated_tokens.clamp_max(logits.shape[-1] - 1).to(torch.int64)
    rp = torch.full_like(logits, repetition_penalty)
    factors = torch.ones_like(logits).scatter_reduce(2, generated_tokens, rp, reduce="prod")
    return torch.where(logits <= 0, logits * factors, logits / factors)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    linear: float = 0.0,
    conf: float = 0.0,
    quad: float = 0.0,
    generated_tokens: torch.Tensor | None = None,
    repetition_penalty: float = 3.0,
    repetition_penalty_window: int = 2,
) -> torch.Tensor:
    """Sample next token from logits using temperature, top-p, top-k, or min-p sampling.

    Args:
        logits (torch.Tensor): Input logits with token candidates on the last dimension.
        temperature (float): Sampling temperature. Lower temperature results in more deterministic samples.
        top_p (float): The p in “top-p”.
        top_k (int): The k in “top-k”.
        min_p (float): Minimum token probability, scaled by the probability of the most likely token.
                       Must be between 0 and 1. Typical values are in the 0.01-0.2 range.

    Returns:
        torch.Tensor: Sampled tokens.
    """
    if repetition_penalty != 1.0 and generated_tokens is not None:
        logits = modify_logit_for_repetition_penalty(logits, generated_tokens, repetition_penalty, repetition_penalty_window)

    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        if linear > 0.0:
            probs = apply_unified(probs, linear, conf, quad)
        if top_p > 0:
            probs = apply_top_p(probs, top_p)
        if top_k > 0:
            probs = apply_top_k(probs, top_k)
        if min_p > 0:
            probs = apply_min_p(probs, min_p)

        next_token = multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    return next_token  # [batch_size, num_codebooks, 1]
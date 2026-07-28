"""Cache-aware streaming for the Nemotron FastConformer encoder.

Each conformer layer keeps an attention cache (last ``left_context`` attention-input
frames) and a causal-conv cache (last ``conv_kernel-1`` GLU-output frames);
subsampling is incremental with a small mel cache. With the window sized to the
allowed left context, no attention mask is needed, so the streamed encoder output is
frame-identical to the offline ``chunked_limited`` encoder at the native chunk size
(``right_context + 1``). This yields the model's native O(n), no-recompute streaming.
"""

import mlx.core as mx
import mlx.nn as nn

_PRE_ENCODE_MEL_CACHE = 16  # >= causal receptive field of the 8x dw-striding stack


def _stream_block(block, x, pos_enc, attn_cache, conv_cache, left_cache, conv_left):
    # half-step FFN 1
    residual = x + 0.5 * block.feed_forward1(block.norm_feed_forward1(x))

    # cache-aware self-attention: Q = chunk, K/V = [cache ++ chunk]
    xn = block.norm_self_att(residual)
    kv = xn if attn_cache is None else mx.concatenate([attn_cache, xn], axis=1)
    pos_emb = pos_enc.pos_emb_for(kv.shape[1], x.dtype)
    residual = residual + block.self_attn.stream(xn, kv, pos_emb)
    attn_next = kv[:, -left_cache:] if left_cache > 0 else kv[:, :0]

    # cache-aware causal conv: prepend conv cache instead of zero-padding
    xc = block.norm_conv(residual)
    g = nn.glu(block.conv.pointwise_conv1(xc), axis=-1)  # (B, c, d)
    if conv_cache is None:
        conv_cache = mx.zeros((g.shape[0], conv_left, g.shape[2]), dtype=g.dtype)
    din = mx.concatenate([conv_cache, g], axis=1)
    dw = block.conv.depthwise_conv(din)  # valid conv -> (B, c, d)
    conv_next = din[:, -conv_left:]
    y = block.conv.batch_norm(dw)
    y = block.conv.activation(y)
    residual = residual + block.conv.pointwise_conv2(y)

    # half-step FFN 2 + final norm
    residual = residual + 0.5 * block.feed_forward2(block.norm_feed_forward2(residual))
    return block.norm_out(residual), attn_next, conv_next


def stream_encode_chunks(
    model, mel_chunks, language, chunk_frames=None, att_context_size=None
):
    """Yield post-prompt encoder frames from one or more mel chunks.

    The encoder/conv/subsampling caches persist across input mel chunks, so callers
    can keep STFT memory bounded without resetting model context at chunk boundaries.
    """
    enc = model.encoder
    acs = att_context_size or model.default_att_context_size
    left_cache = int(acs[0])
    right = int(acs[1])
    cf = chunk_frames or (right + 1)
    sf = enc.args.subsampling_factor
    chunk_mel = cf * sf
    conv_left = enc.args.conv_kernel_size - 1

    n = len(enc.layers)
    attn_cache = [None] * n
    conv_cache = [None] * n
    mel_cache = None
    emitted = 0
    consumed = 0
    pending = None

    def append_pending(chunk):
        nonlocal pending
        if chunk.ndim == 2:
            chunk = mx.expand_dims(chunk, 0)
        if chunk.shape[1] == 0:
            return
        pending = chunk if pending is None else mx.concatenate([pending, chunk], axis=1)

    def encode_mel_chunk(m, is_final):
        nonlocal mel_cache, emitted, consumed
        cache_len = 0 if mel_cache is None else mel_cache.shape[1]
        win = m if mel_cache is None else mx.concatenate([mel_cache, m], axis=1)
        win_len = win.shape[1]
        sub = enc.pre_encode(win, mx.array([win_len], dtype=mx.int32))[0]  # (1, k, d)

        end = consumed + m.shape[1]
        base = (consumed - cache_len) // sf
        lo = emitted - base
        hi = sub.shape[1] if is_final else (end // sf - base)
        consumed = end
        mel_cache = win[:, -_PRE_ENCODE_MEL_CACHE:]

        if hi <= lo:
            emitted = base + max(lo, hi)
            return
        emitted = base + hi
        h = sub[:, lo:hi]
        for li, block in enumerate(enc.layers):
            h, attn_cache[li], conv_cache[li] = _stream_block(
                block,
                h,
                enc.pos_enc,
                attn_cache[li],
                conv_cache[li],
                left_cache,
                conv_left,
            )
        yield model.apply_prompt(h, language)

    def encode_ready(is_final):
        nonlocal pending
        while pending is not None and pending.shape[1] > 0:
            if pending.shape[1] < chunk_mel and not is_final:
                break

            take = min(chunk_mel, pending.shape[1])
            if is_final and pending.shape[1] <= chunk_mel:
                take = pending.shape[1]

            m = pending[:, :take]
            pending = pending[:, take:]
            is_final_chunk = is_final and pending.shape[1] == 0
            yield from encode_mel_chunk(m, is_final_chunk)

    iterator = iter(mel_chunks)
    try:
        current = next(iterator)
    except StopIteration:
        return

    for next_chunk in iterator:
        append_pending(current)
        yield from encode_ready(is_final=False)
        current = next_chunk

    append_pending(current)
    yield from encode_ready(is_final=True)


def stream_encode(model, mel, language, chunk_frames=None, att_context_size=None):
    """Yield post-prompt encoder frames (1, c, d) per chunk, cache-aware.

    Frame-identical to ``encoder(...)`` + ``apply_prompt(...)`` at the native chunk
    size (right_context + 1).
    """
    yield from stream_encode_chunks(
        model,
        [mel],
        language,
        chunk_frames=chunk_frames,
        att_context_size=att_context_size,
    )

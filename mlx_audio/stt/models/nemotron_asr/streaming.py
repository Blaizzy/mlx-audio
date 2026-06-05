"""Cache-aware streaming for the Nemotron FastConformer encoder.

Each conformer layer keeps two caches (matching NeMo cache_last_channel /
cache_last_time):
  - attn cache: last `left_cache` frames of the attention input (norm_self_att out)
  - conv cache: last `conv_kernel-1` frames of the depthwise-conv input (GLU out)

Per chunk: Q = new frames, K/V = [attn_cache + new]; depthwise conv prepends the
conv_cache instead of zero-padding. With the window sized to the allowed left
context, no attention mask is needed — so streaming output is frame-identical to
the offline (chunked_limited) encoder. Subsampling is done up-front on the
provided mel; incremental subsampling can wrap this later.
"""

import mlx.core as mx
import mlx.nn as nn


def _stream_block(layer, x, pos_enc, attn_cache, conv_cache, left_cache, conv_left):
    # half-step FFN 1
    residual = x + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(x))

    # cache-aware self-attention
    xn = layer.norm_self_att(residual)
    kv = xn if attn_cache is None else mx.concatenate([attn_cache, xn], axis=1)
    pos_emb = pos_enc.pos_emb_for(kv.shape[1], x.dtype)
    residual = residual + layer.self_attn.stream(xn, kv, pos_emb)
    attn_cache_next = kv[:, -left_cache:]

    # cache-aware causal conv
    xc = layer.norm_conv(residual)
    g = nn.glu(layer.conv.pointwise_conv1(xc), axis=-1)  # (B, c, d)
    if conv_cache is None:
        conv_cache = mx.zeros((g.shape[0], conv_left, g.shape[2]), dtype=g.dtype)
    din = mx.concatenate([conv_cache, g], axis=1)
    dw = layer.conv.depthwise_conv(din)  # valid conv -> (B, c, d)
    conv_cache_next = din[:, -conv_left:]
    cc = layer.conv.batch_norm(dw)
    cc = cc * mx.sigmoid(cc)  # SiLU
    residual = residual + layer.conv.pointwise_conv2(cc)

    # half-step FFN 2 + final norm
    residual = residual + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(residual))
    return layer.norm_out(residual), attn_cache_next, conv_cache_next


def native_chunk_frames(conformer) -> int:
    """Block size that makes streaming frame-identical to offline:
    chunk = right_context + 1 (the chunked_limited block granularity)."""
    return int(conformer.args.att_context_size[1]) + 1


def _iter_chunks(conformer, mel, chunk_frames):
    """Yield (encoder_frames_for_chunk) streaming over the conformer."""
    left_cache = int(conformer.args.att_context_size[0])
    conv_left = conformer.args.conv_kernel_size - 1
    feats = conformer.pre_encode(mel)  # (1, T', d) — subsample up-front
    T = feats.shape[1]
    n = len(conformer.layers)
    attn_cache = [None] * n
    conv_cache = [None] * n
    for s in range(0, T, chunk_frames):
        x = feats[:, s : s + chunk_frames]
        for li, layer in enumerate(conformer.layers):
            x, attn_cache[li], conv_cache[li] = _stream_block(
                layer,
                x,
                conformer.pos_enc,
                attn_cache[li],
                conv_cache[li],
                left_cache,
                conv_left,
            )
        yield x


def stream_conformer(conformer, mel, chunk_frames=None):
    """Cache-aware streamed conformer over `mel`; returns (1, T', d).
    Frame-identical to offline when chunk_frames == native_chunk_frames."""
    cf = chunk_frames or native_chunk_frames(conformer)
    return mx.concatenate(list(_iter_chunks(conformer, mel, cf)), axis=1)


def stream_transcribe(model, mel, prompt_id, chunk_frames=None):
    """Cache-aware streaming RNN-T decode. Yields (new_token_ids, all_token_ids)
    per chunk; persists LSTM/last-token state across chunks. Token sequence
    equals the offline greedy result at the native chunk size."""
    cf = chunk_frames or native_chunk_frames(model.encoder)
    last_token = model.blank_id
    hidden = None
    all_ids: list[int] = []
    for enc in _iter_chunks(model.encoder, mel, cf):
        post = model._fuse(enc, prompt_id)  # (1, c, d) language-conditioned
        new: list[int] = []
        for t in range(post.shape[1]):
            feat = post[:, t : t + 1]
            sym = 0
            while True:
                cur = (
                    mx.array([[last_token]], dtype=mx.int32)
                    if last_token != model.blank_id
                    else None
                )
                dec_out, (h, c) = model.decoder(cur, hidden)
                logits = model.joint(feat, dec_out.astype(feat.dtype))
                pred = int(mx.argmax(logits))
                if pred == model.blank_id:
                    break
                last_token = pred
                hidden = (h.astype(feat.dtype), c.astype(feat.dtype))
                new.append(pred)
                sym += 1
                if model.max_symbols is not None and sym >= model.max_symbols:
                    break
        all_ids.extend(new)
        yield new, all_ids

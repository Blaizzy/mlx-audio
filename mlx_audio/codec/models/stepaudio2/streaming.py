# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu, Zhihao Du)
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified streaming algorithm from StepAudio2/CosyVoice2 and ESPnet.

"""StepAudio2 streaming primitives with explicit sequence caches.

Adapted from native StepAudio2/CosyVoice2 streaming and the MLX Audio 0.4.5
source baseline. Existing StepAudio2 weighted networks are reused unchanged.
No runtime/device package is imported here; public entry points load MLX lazily.

Cache updates are functional. ESPnet position encoders maintain a deterministic
position-table memo on the supplied network; network owners must serialize calls.
"""

import math
import time
from dataclasses import dataclass

ATTENTION_MODES = ("eager", "sdpa")


def _position(embed, length):
    # The MLX ESPnet method takes size first (unlike the native wrapper).
    embed.pos_enc._extend_pe(length)
    return embed.pos_enc.position_encoding(length)


def encoder_chunk(encoder, xs, last_chunk, cnn_cache, att_cache, mx, nn):
    """Native two-stage Conformer cache layout, with immutable MLX updates."""
    stride, lookahead = (
        encoder.up_layer.stride,
        encoder.pre_lookahead_layer.pre_lookahead_len,
    )
    if stride != 2 or lookahead != 3:
        raise ValueError(
            "StepAudio2 streaming encoder requires stride 2 and lookahead 3"
        )
    if att_cache is not None and att_cache.shape[3] % 2:
        raise ValueError("Conformer attention cache length must be even")
    if cnn_cache is not None and cnn_cache.shape[2] != 2 + stride * 2:
        raise ValueError("Conformer CNN cache requires six history frames")
    offset = att_cache.shape[3] // 2 if att_cache is not None else 0
    lower = len(encoder.encoders)
    xs, _, _ = encoder.embed(xs, None)
    if last_chunk:
        xs = mx.pad(xs, [(0, 0), (0, lookahead), (0, 0)])
    if xs.shape[1] <= lookahead:
        raise ValueError(
            "Non-final chunk must contain lookahead and at least one content token"
        )
    layer = encoder.pre_lookahead_layer
    y = nn.leaky_relu(layer.conv1(xs))
    new_cnn1 = mx.transpose(y[:, -2:, :], (0, 2, 1))
    old1 = (
        mx.zeros((xs.shape[0], 2, xs.shape[2]), dtype=xs.dtype)
        if cnn_cache is None
        else mx.transpose(cnn_cache[:, :, :2], (0, 2, 1))
    )
    xs = layer.conv2(mx.concatenate([old1, y], axis=1)) + xs[:, :-lookahead, :]
    pos = _position(encoder.embed, offset + xs.shape[1])
    cache1 = []
    for i, layer in enumerate(encoder.encoders):
        old = None if att_cache is None else att_cache[i, :, :, :offset, :]
        xs, _, new, _ = layer(xs, None, pos, att_cache=old)
        cache1.append(new)
    y = mx.repeat(xs, stride, axis=1)
    old2 = (
        mx.zeros((xs.shape[0], stride * 2, xs.shape[2]), dtype=xs.dtype)
        if cnn_cache is None
        else mx.transpose(cnn_cache[:, :, 2:], (0, 2, 1))
    )
    y = mx.concatenate([old2, y], axis=1)
    new_cnn2 = mx.transpose(y[:, -stride * 2 :, :], (0, 2, 1))
    xs = encoder.up_layer.conv(y)
    xs, _, _ = encoder.up_embed(xs, None)
    pos = _position(encoder.embed, offset * stride + xs.shape[1])
    cache2 = []
    for i, layer in enumerate(encoder.up_encoders):
        old = None if att_cache is None else att_cache[lower + i]
        xs, _, new, _ = layer(xs, None, pos, att_cache=old)
        cache2.append(new)
    if encoder.normalize_before:
        xs = encoder.after_norm(xs)
    packed_att = mx.concatenate(
        [mx.tile(mx.stack(cache1), (1, 1, 1, 2, 1)), mx.stack(cache2)], axis=0
    )
    return xs, mx.concatenate([new_cnn1, new_cnn2], axis=2), packed_att


def _attention_mode(mode):
    if mode not in ATTENTION_MODES:
        raise ValueError("StepAudio2 MLX waveform attention must be eager or sdpa")
    return mode


def _guard_sdpa_input(layer, x, cache, mx):
    """Fail before projections for the pinned FP32, eight-head DiT geometry."""
    if (
        getattr(layer, "num_heads", None) != 8
        or getattr(layer, "head_dim", None) != 64
        or layer.inner_dim != 512
        or layer.scale != 0.125
    ):
        raise ValueError(
            "StepAudio2 waveform SDPA requires eight 64-wide heads and scale 0.125"
        )
    if (
        x.dtype != mx.float32
        or x.ndim != 3
        or min(x.shape[:2]) <= 0
        or x.shape[-1] != 512
    ):
        raise ValueError(
            "StepAudio2 waveform SDPA requires nonempty FP32 [batch,time,512] input"
        )
    if cache is not None and (
        cache.dtype != mx.float32
        or cache.ndim != 4
        or cache.shape[:2] != (x.shape[0], 8)
        or cache.shape[-1] != 128
    ):
        raise ValueError(
            "StepAudio2 waveform SDPA requires FP32 packed [batch,8,history,128] cache"
        )
    if not callable(
        getattr(getattr(mx, "fast", None), "scaled_dot_product_attention", None)
    ):
        raise ValueError("StepAudio2 waveform SDPA requires the MLX GPU attention API")


def _dit_attention(layer, x, cache, mx, attention_mode="eager"):
    mode = _attention_mode(attention_mode)
    if mode == "sdpa":
        _guard_sdpa_input(layer, x, cache, mx)
    q, k, v = (layer.to_heads(proj(x)) for proj in (layer.to_q, layer.to_k, layer.to_v))
    if layer.q_norm is not None:
        q, k = layer.q_norm(q), layer.k_norm(k)
    if mode == "sdpa":
        expected = (x.shape[0], 8, x.shape[1], 64)
        if any(
            value.dtype != mx.float32 or value.shape != expected for value in (q, k, v)
        ):
            raise ValueError(
                "StepAudio2 waveform SDPA requires matching FP32 Q/K/V head tensors"
            )
    if cache is not None:
        old_k, old_v = mx.split(cache, 2, axis=-1)
        # Native DiT uses current-before-history, unlike Conformer.
        k, v = mx.concatenate([k, old_k], axis=2), mx.concatenate([v, old_v], axis=2)
    new = mx.concatenate([k, v], axis=-1)
    if mode == "sdpa":
        # Noncausal attention over current-before-history K/V, exactly as eager.
        # The fused FP32 reduction order can differ; this is an audition mode.
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=layer.scale, mask=None, stream=mx.gpu
        )
        if out.dtype != mx.float32 or out.shape != q.shape:
            raise ValueError(
                "StepAudio2 waveform SDPA returned an unsupported head tensor"
            )
    else:
        scores = (q @ mx.swapaxes(k, -1, -2)) * layer.scale
        out = mx.softmax(scores, axis=-1) @ v
    out = mx.transpose(out, (0, 2, 1, 3)).reshape(
        x.shape[0], x.shape[1], layer.inner_dim
    )
    return layer.proj(out), new


def _dit_conv(layer, x, cache, mx):
    if cache is None:
        cache1 = mx.zeros((x.shape[0], 2, layer.in_channels), dtype=x.dtype)
        cache2 = mx.zeros((x.shape[0], 2, layer.out_channels), dtype=x.dtype)
    else:
        cache1 = mx.transpose(cache[:, : layer.in_channels, :], (0, 2, 1))
        cache2 = mx.transpose(cache[:, layer.in_channels :, :], (0, 2, 1))
    y = mx.concatenate([cache1, x], axis=1)
    new1 = y[:, -2:, :]
    y = layer.block[1](y)
    y = layer.block[4](layer.block[3](y))
    y = mx.concatenate([cache2, y], axis=1)
    new2 = y[:, -2:, :]
    out = layer.block[6](y)
    return out, mx.transpose(mx.concatenate([new1, new2], axis=-1), (0, 2, 1))


def dit_chunk(
    estimator, x, mu, t, spks, cond, cnn_cache, att_cache, mx, attention_mode="eager"
):
    c = mx.expand_dims(estimator.t_embedder(t), 1)
    speaker = mx.broadcast_to(
        mx.expand_dims(spks, -1), (spks.shape[0], spks.shape[1], x.shape[-1])
    )
    x = mx.transpose(mx.concatenate([x, mu, speaker, cond], axis=1), (0, 2, 1))
    x = estimator.in_proj(x)
    new_cnn, new_att = [], []
    for index, block in enumerate(estimator.blocks):
        mod = c
        for layer in block.adaLN_modulation:
            mod = layer(mod)
        s_a, z_a, g_a, s_m, z_m, g_m, s_c, z_c, g_c = mx.split(mod, 9, axis=-1)
        att, ac = _dit_attention(
            block.attn,
            block.norm1(x) * (1 + z_a) + s_a,
            None if att_cache is None else att_cache[index],
            mx,
            attention_mode,
        )
        x = x + g_a * att
        conv, cc = _dit_conv(
            block.conv,
            block.norm3(x) * (1 + z_c) + s_c,
            None if cnn_cache is None else cnn_cache[index],
            mx,
        )
        x = x + g_c * conv
        x = x + g_m * block.mlp(block.norm2(x) * (1 + z_m) + s_m)
        new_cnn.append(cc)
        new_att.append(ac)
    x = estimator.final_layer(x, c)
    return mx.transpose(x, (0, 2, 1)), mx.stack(new_cnn), mx.stack(new_att)


def cfm_chunk(
    decoder,
    mu,
    spks,
    cond,
    n_timesteps,
    cnn_cache,
    att_cache,
    mx,
    attention_mode="eager",
):
    if n_timesteps != 10:
        raise ValueError("StepAudio2 MLX decoder requires ten flow steps")
    offset = 0 if att_cache is None else att_cache.shape[4]
    if offset + mu.shape[2] > decoder.rand_noise.shape[2]:
        raise ValueError("StepAudio2 flow noise trajectory exhausted")
    x = decoder.rand_noise[:, :, offset : offset + mu.shape[2]]
    t_span = mx.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
    t_span = 1 - mx.cos(t_span * (0.5 * math.pi))
    t = mx.expand_dims(t_span[0], 0)
    dt = t_span[1] - t_span[0]
    mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
    spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
    cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)
    new_cnn, new_att = [], []
    for step in range(n_timesteps):
        velocity, cc, ac = dit_chunk(
            decoder.estimator,
            mx.concatenate([x, x], axis=0),
            mu_in,
            mx.concatenate([t, t], axis=0),
            spks_in,
            cond_in,
            None if cnn_cache is None else cnn_cache[step],
            None if att_cache is None else att_cache[step],
            mx,
            attention_mode,
        )
        conditional, unconditional = mx.split(velocity, 2, axis=0)
        velocity = (
            1 + decoder.inference_cfg_rate
        ) * conditional - decoder.inference_cfg_rate * unconditional
        x = x + dt * velocity
        t = t + dt
        if step < n_timesteps - 1:
            dt = t_span[step + 2] - t
        new_cnn.append(cc)
        new_att.append(ac)
    return x, mx.stack(new_cnn), mx.stack(new_att)


def flow_chunk(
    flow,
    token,
    mel,
    speaker,
    cache,
    last_chunk,
    n_timesteps,
    mx,
    nn,
    attention_mode="eager",
):
    # torch.nn.functional.normalize uses max(norm, eps), not norm + eps.
    speaker = speaker / mx.maximum(
        mx.linalg.norm(speaker, axis=1, keepdims=True), 1e-12
    )
    speaker = flow.spk_embed_affine_layer(speaker)
    hidden, cc, ac = encoder_chunk(
        flow.encoder,
        flow.input_embedding(token),
        last_chunk,
        None if cache is None else cache["conformer_cnn_cache"],
        None if cache is None else cache["conformer_att_cache"],
        mx,
        nn,
    )
    hidden = flow.encoder_proj(hidden)
    condition = mx.zeros_like(hidden) if mel is None else mel
    if condition.shape != hidden.shape:
        raise ValueError("Prompt mel and streaming encoder lengths disagree")
    output, ec, ea = cfm_chunk(
        flow.decoder,
        mx.transpose(hidden, (0, 2, 1)),
        speaker,
        mx.transpose(condition, (0, 2, 1)),
        n_timesteps,
        None if cache is None else cache["estimator_cnn_cache"],
        None if cache is None else cache["estimator_att_cache"],
        mx,
        attention_mode,
    )
    return output, {
        "conformer_cnn_cache": cc,
        "conformer_att_cache": ac,
        "estimator_cnn_cache": ec,
        "estimator_att_cache": ea,
    }


def crop_flow_cache(cache, prompt_frames, mx):
    """Preserve native prompt + last-100 crop, including native DiT ordering."""
    result = dict(cache)
    for name, axis in (("estimator_att_cache", 4), ("conformer_att_cache", 3)):
        value = result[name]
        if value.shape[axis] > prompt_frames + 100:
            first, tail = [slice(None)] * value.ndim, [slice(None)] * value.ndim
            first[axis], tail[axis] = slice(0, prompt_frames), slice(-100, None)
            result[name] = mx.concatenate(
                [value[tuple(first)], value[tuple(tail)]], axis=axis
            )
    return result


def vocoder_chunk(hift, chunk_mel, cache, last_chunk, window, mx):
    mel = mx.concatenate([cache["mel"], chunk_mel], axis=2)
    speech, source = hift(mel, cache["source"])
    overlap = window.shape[0] // 2
    first = cache["speech"].shape[-1] == 0
    if not first:
        if speech.shape[-1] < overlap or cache["speech"].shape[-1] < overlap:
            raise ValueError("Vocoder overlap cache is shorter than 160ms")
        faded = (
            speech[:, :overlap] * window[:overlap]
            + cache["speech"][:, -overlap:] * window[overlap:]
        )
        speech = mx.concatenate([faded, speech[:, overlap:]], axis=1)
    new = {
        "mel": mel[:, :, -8:],
        "source": source[:, :, -overlap:],
        "speech": speech[:, -overlap:],
    }
    if not last_chunk:
        if speech.shape[-1] < overlap:
            raise ValueError("Non-final decoded chunk is shorter than 160ms")
        speech = speech[:, :-overlap]
        if first:
            speech = mx.concatenate(
                [mx.zeros((1, overlap), dtype=speech.dtype), speech], axis=1
            )
    return speech, new


def _runtime(mx, nn=None, *, need_nn=False):
    if mx is None:
        import mlx.core as mx
    if need_nn and nn is None:
        import mlx.nn as nn
    return mx, nn


def reference_from_prompt(prompt, *, mx=None):
    """Convert a prepared offline prompt without extracting audio or mutating it.

    Streaming uses the aligned feature tensor's length. Existing offline prompt
    preparation can retain a pre-alignment prompt_feat_len; that metadata is not
    a streaming cache length and is deliberately not used here.
    """
    mx, _ = _runtime(mx)
    keys = {
        "prompt_token",
        "prompt_token_len",
        "embedding",
        "prompt_feat",
        "prompt_feat_len",
    }
    if not isinstance(prompt, dict) or set(prompt) != keys:
        raise CodecError("five prepared prompt fields required", "reference")
    feature = prompt["prompt_feat"]
    if getattr(feature, "ndim", None) != 3:
        raise CodecError("prompt feature layout", "reference")
    original_length = prompt["prompt_feat_len"]
    if getattr(original_length, "dtype", None) != mx.int32 or original_length.shape != (
        1,
    ):
        raise CodecError("prompt feature length layout", "reference")
    mx.eval(original_length)
    if int(original_length[0].item()) <= 0:
        raise CodecError("prompt feature length must be positive", "reference")
    reference = (
        prompt["prompt_token"],
        prompt["prompt_token_len"],
        prompt["embedding"],
        feature,
        mx.array([feature.shape[1]], dtype=mx.int32),
    )
    return validate_reference(reference, mx)


FLOW_KEYS = frozenset(
    (
        "conformer_cnn_cache",
        "conformer_att_cache",
        "estimator_cnn_cache",
        "estimator_att_cache",
    )
)
VOCODER_KEYS = frozenset(("mel", "source", "speech"))
SAMPLE_RATE = 24000
SILENCE_CODE = 4218
VOCAB_SIZE = 6561
LOOKAHEAD_CODES = 3
MEL_CHANNELS = 80
FLOW_STEPS = 10


class CodecError(ValueError):
    """A bounded diagnostic with no audio, codes or reference content."""

    def __init__(self, reason, stage, **details):
        self.codec_diagnostics = {"reason": reason, "stage": stage, **details}
        super().__init__(f"StepAudio2 codec rejected {reason} at {stage}")


@dataclass(frozen=True)
class StreamState:
    """Sequence state. Treat cache mappings and their array leaves as read-only.

    ``reset_stream`` clones leaves for independent branches. Shared weighted
    networks still have deterministic position-table memos and must be used by
    one serialized owner; independent cache state is not a thread-safety claim.
    """

    flow_cache: dict
    vocoder_cache: dict
    prompt_frames: int
    consumed_codes: int = 0
    chunks: int = 0
    finalized: bool = False


@dataclass(frozen=True)
class ChunkResult:
    waveform: object
    state: StreamState
    consumed_codes: int
    metrics: dict


def _finite(value, mx, stage, shape=None):
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise CodecError(
            "shape", stage, shape=list(value.shape), expected_shape=list(shape)
        )
    if value.dtype != mx.float32:
        raise CodecError("dtype", stage, dtype=str(value.dtype))
    valid = mx.all(mx.isfinite(value))
    mx.eval(value, valid)
    if not bool(valid.item()):
        raise CodecError("non-finite values", stage, shape=list(value.shape))


def _cache(cache, keys, mx, stage):
    if not isinstance(cache, dict) or set(cache) != keys:
        raise CodecError("cache keys", stage)
    for key, value in cache.items():
        _finite(value, mx, f"{stage}.{key}")
    mx.eval(cache)


def validate_reference(reference, mx):
    """Validate an already prepared native tuple; never extract or read audio.

    Layout: int32 codes [1,N], int32 code lengths [1], FP32 speaker [1,192],
    FP32 mel [1,2*N,80], int32 mel lengths [1]. No Torch conversion is implicit.
    """
    if not isinstance(reference, (tuple, list)) or len(reference) != 5:
        raise CodecError("five prepared reference arrays required", "reference")
    tokens, token_lengths, speaker, mel, mel_lengths = reference
    if (
        tokens.dtype != mx.int32
        or tokens.ndim != 2
        or tokens.shape[0] != 1
        or tokens.shape[1] <= 0
    ):
        raise CodecError("reference token layout", "reference")
    for lengths, expected in (
        (token_lengths, tokens.shape[1]),
        (mel_lengths, tokens.shape[1] * 2),
    ):
        if lengths.dtype != mx.int32 or lengths.shape != (1,):
            raise CodecError("reference length layout", "reference")
        mx.eval(lengths)
        if int(lengths[0].item()) != expected:
            raise CodecError("reference length mismatch", "reference")
    valid = mx.all((tokens >= 0) & (tokens < VOCAB_SIZE))
    mx.eval(tokens, valid)
    if not bool(valid.item()):
        raise CodecError("reference code range", "reference")
    _finite(speaker, mx, "reference.speaker", (1, 192))
    _finite(mel, mx, "reference.mel", (1, tokens.shape[1] * 2, MEL_CHANNELS))
    return tuple(reference)


def _codes(codes, mx, last_chunk):
    if type(last_chunk) is not bool:
        raise CodecError("last_chunk must be a boolean", "codes")
    if (
        not isinstance(codes, (list, tuple))
        or not codes
        or any(type(code) is not int or not 0 <= code < VOCAB_SIZE for code in codes)
    ):
        raise CodecError("nonempty integer codes in [0,6561) required", "codes")
    if not last_chunk and len(codes) <= LOOKAHEAD_CODES:
        raise CodecError("non-final chunk requires three lookahead codes", "codes")
    return mx.array([codes], dtype=mx.int32)


def hamming_window(mx=None):
    """Native FP32 320ms window, computed without importing a device runtime."""
    mx, _ = _runtime(mx)
    import numpy as np

    window = mx.array(np.hamming(7680).astype(np.float32))
    mx.eval(window)
    return window


def prepare_stream(flow, reference, *, mx=None, nn=None, attention_mode="eager"):
    """Prepare prompt flow cache and empty HiFT cache from resident arrays.

    Original flow noise and RNG are untouched. The returned state is a reset
    base; use ``reset_stream`` before running independent utterances/branches.
    """
    mx, nn = _runtime(mx, nn, need_nn=True)
    attention_mode = _attention_mode(attention_mode)
    if isinstance(reference, dict):
        reference = reference_from_prompt(reference, mx=mx)
    tokens, _, speaker, mel, _ = validate_reference(reference, mx)
    if getattr(flow, "up_rate", None) != 2:
        raise CodecError("two mel frames per code required", "prepare")
    token = mx.concatenate(
        [tokens, mx.full((1, 3), SILENCE_CODE, dtype=mx.int32)], axis=1
    )
    output, cache = flow_chunk(
        flow, token, mel, speaker, None, False, FLOW_STEPS, mx, nn, attention_mode
    )
    _finite(output, mx, "prompt_flow_mel", (1, MEL_CHANNELS, mel.shape[1]))
    _cache(cache, FLOW_KEYS, mx, "prompt_flow_cache")
    vocoder = {
        "mel": mx.zeros((1, MEL_CHANNELS, 0), dtype=mx.float32),
        "source": mx.zeros((1, 1, 0), dtype=mx.float32),
        "speech": mx.zeros((1, 0), dtype=mx.float32),
    }
    mx.eval(vocoder)
    return StreamState(cache, vocoder, mel.shape[1])


def clone_cache(cache, *, mx=None):
    """Fresh containers and evaluated array copies, with no hidden RNG/reset."""
    mx, _ = _runtime(mx)
    if not isinstance(cache, dict):
        raise CodecError("cache mapping required", "clone")
    mx.eval(cache)
    result = {key: mx.array(value) for key, value in cache.items()}
    mx.eval(result)
    return result


def reset_stream(base, *, mx=None):
    """Clone an unconsumed prompt base; never silently reset a finished stream."""
    mx, _ = _runtime(mx)
    if (
        not isinstance(base, StreamState)
        or base.chunks
        or base.consumed_codes
        or base.finalized
    ):
        raise CodecError("unconsumed prompt base required", "reset")
    _cache(base.flow_cache, FLOW_KEYS, mx, "reset_flow")
    _cache(base.vocoder_cache, VOCODER_KEYS, mx, "reset_vocoder")
    return StreamState(
        clone_cache(base.flow_cache, mx=mx),
        clone_cache(base.vocoder_cache, mx=mx),
        base.prompt_frames,
    )


def decode_chunk(
    flow,
    hift,
    codes,
    state,
    speaker,
    window=None,
    *,
    last_chunk=False,
    mx=None,
    nn=None,
    attention_mode="eager",
):
    """Decode one already buffered chunk, leaving the supplied state unchanged.

    Non-final N codes consume N-3; final N consumes N. First normal chunk is
    three silence codes + 25 generated codes (28 total). Continuation is the
    carried last three codes + 25 new codes. Final 3/17/25 refer to remaining
    buffer counts, not new-code counts or the complete native flush wrapper.

    Arrays/caches are evaluated and validated before a new state is returned.
    There is no sanitization, CPU fallback, reseeding or automatic final flush.
    """
    mx, nn = _runtime(mx, nn, need_nn=True)
    if window is None:
        window = hamming_window(mx)
    attention_mode = _attention_mode(attention_mode)
    if not isinstance(state, StreamState) or state.finalized:
        raise CodecError("open stream state required", "decode")
    if type(state.prompt_frames) is not int or state.prompt_frames <= 0:
        raise CodecError("prompt frame count", "decode")
    tokens = _codes(codes, mx, last_chunk)
    _cache(state.flow_cache, FLOW_KEYS, mx, "input_flow_cache")
    _cache(state.vocoder_cache, VOCODER_KEYS, mx, "input_vocoder_cache")
    _finite(speaker, mx, "speaker", (1, 192))
    _finite(window, mx, "window", (7680,))
    consumed = len(codes) if last_chunk else len(codes) - LOOKAHEAD_CODES
    start = time.perf_counter()
    mel, cache = flow_chunk(
        flow,
        tokens,
        None,
        speaker,
        state.flow_cache,
        last_chunk,
        FLOW_STEPS,
        mx,
        nn,
        attention_mode,
    )
    _finite(mel, mx, "flow_mel", (1, MEL_CHANNELS, consumed * 2))
    cache = crop_flow_cache(cache, state.prompt_frames, mx)
    _cache(cache, FLOW_KEYS, mx, "output_flow_cache")
    flow_seconds = time.perf_counter() - start
    start = time.perf_counter()
    waveform, vocoder = vocoder_chunk(
        hift, mel, state.vocoder_cache, last_chunk, window, mx
    )
    _finite(waveform, mx, "waveform")
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] <= 0:
        raise CodecError("mono nonempty waveform required", "waveform")
    _cache(vocoder, VOCODER_KEYS, mx, "output_vocoder_cache")
    vocoder_seconds = time.perf_counter() - start
    new = StreamState(
        cache,
        vocoder,
        state.prompt_frames,
        state.consumed_codes + consumed,
        state.chunks + 1,
        last_chunk,
    )
    return ChunkResult(
        waveform,
        new,
        consumed,
        {
            "input_codes": len(codes),
            "consumed_codes": consumed,
            "mel_frames": mel.shape[2],
            "output_samples": waveform.shape[1],
            "flow_seconds": flow_seconds,
            "vocoder_seconds": vocoder_seconds,
            "attention_mode": attention_mode,
            "last_chunk": last_chunk,
            "timing_scope": "host wall time including evaluation and finite checks; no output bridge",
        },
    )


def pcm16_bytes(waveform):
    """Native mono PCM16 packaging; amplitude clipping is not numeric repair."""
    import numpy as np

    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != 1 or not np.isfinite(wave).all():
        raise CodecError("finite mono waveform required", "pcm16")
    return (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()

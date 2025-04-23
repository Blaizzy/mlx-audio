import typing as tp

import mlx.core as mx

from .config import DataConfig


def build_delay_indices(
    B: int, T: int, C: int, delay_pattern: tp.List[int]
) -> tp.Tuple[mx.array, mx.array]:
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    """
    delay_arr = mx.array(delay_pattern, dtype=mx.int32)

    t_idx_BxT = mx.broadcast_to(
        mx.arange(T, dtype=mx.int32)[None, :],
        [B, T],
    )
    t_idx_BxTx1 = mx.expand_dims(t_idx_BxT, -1)
    t_idx_BxTxC = t_idx_BxTx1 - mx.reshape(delay_arr, (1, 1, C))

    b_idx_BxTxC = mx.broadcast_to(
        mx.reshape(mx.arange(B, dtype=mx.int32), (B, 1, 1)),
        [B, T, C],
    )
    c_idx_BxTxC = mx.broadcast_to(
        mx.reshape(mx.arange(C, dtype=mx.int32), (1, 1, C)),
        [B, T, C],
    )

    # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
    t_clamped_BxTxC = mx.clip(t_idx_BxTxC, 0, T - 1)

    indices_BTCx3 = mx.stack(
        [
            mx.reshape(b_idx_BxTxC, (-1,)),
            mx.reshape(t_clamped_BxTxC, (-1,)),
            mx.reshape(c_idx_BxTxC, (-1,)),
        ],
        axis=1,
    ).astype(mx.int32)

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio_BxTxC: mx.array,
    pad_value: int,
    bos_value: int,
    precomp: tp.Tuple[mx.array, mx.array],
) -> mx.array:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.

    Args:
        audio_BxTxC: [B, T, C] int16 audio tokens (or int32/float)
        pad_value: the padding token
        bos_value: the BOS token
        precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

    Returns:
        result_BxTxC: [B, T, C] delayed audio tokens
    """
    t_idx_BxTxC, indices_BTCx3 = precomp

    def gather_nd(array, indices):
        gathered = []
        for idx in range(indices.shape[0]):
            b, t, c = indices[idx, 0], indices[idx, 1], indices[idx, 2]
            gathered.append(array[b, t, c])
        return mx.array(gathered)

    # Apply gather
    gathered_flat = gather_nd(audio_BxTxC, indices_BTCx3)
    gathered_BxTxC = mx.reshape(gathered_flat, audio_BxTxC.shape)

    # Create masks
    mask_bos = t_idx_BxTxC < 0  # => place bos_value
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  # => place pad_value

    # Create scalar values
    bos_tensor = mx.full(1, bos_value, dtype=audio_BxTxC.dtype)
    pad_tensor = mx.full(1, pad_value, dtype=audio_BxTxC.dtype)

    # Apply masks (if mask_bos, BOS; else if mask_pad, PAD; else original gather)
    result_BxTxC = mx.where(
        mask_bos, bos_tensor, mx.where(mask_pad, pad_tensor, gathered_BxTxC)
    )

    return result_BxTxC


def audio_to_codebook(
    model,
    input_values,
    data_config: DataConfig,
    padding_mask=None,
    sample_rate=44100,
):
    """
    Encodes the input audio waveform into discrete codes.

    Args:
        model: The model to use for encoding.
        input_values (`mx.array` of shape `(batch_size, channels, sequence_length)`):
            Float values of the input audio waveform.
        padding_mask (`mx.array` of shape `(batch_size, channels, sequence_length)`):
            Padding mask used to pad the `input_values`.
        sample_rate (`int`, *optional*) :
            Signal sampling_rate

    Returns:
        A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
        factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
        `codebook` of shape `[batch_size, num_codebooks, frames]`.
        Scale is not used here.
    """
    audio_data = model.preprocess(input_values, sample_rate)

    if padding_mask is None:
        padding_mask = mx.ones_like(input_values).astype(mx.bool_)

    _, encoded_frame, _, _, _ = model.encode(audio_data, n_quantizers=None)  # 1, C, T
    seq_length = encoded_frame.shape[2]

    t_idx_BxTxC, indices_BTCx3 = build_delay_indices(
        B=1,
        T=seq_length,
        C=data_config.channels,
        delay_pattern=data_config.delay_pattern,
    )

    encoded_frame = apply_audio_delay(
        audio_BxTxC=mx.transpose(encoded_frame, (0, 2, 1)),  # 1, T, C
        pad_value=data_config.audio_pad_value,
        bos_value=data_config.audio_bos_value,
        precomp=(t_idx_BxTxC, indices_BTCx3),
    )

    return encoded_frame


def build_revert_indices(
    B: int, T: int, C: int, delay_pattern: tp.List[int]
) -> tp.Tuple[mx.array, mx.array]:
    """
    Precompute indices for the revert operation using MLX.

    Returns:
        A tuple (t_idx_BxTxC, indices_BTCx3) where:
            - t_idx_BxTxC is a tensor of shape [B, T, C] computed as time indices plus the delay.
            - indices_BTCx3 is a tensor of shape [B*T*C, 3] used for gathering, computed from:
                batch indices, clamped time indices, and channel indices.
    """
    delay_arr = mx.array(delay_pattern, dtype=mx.int32)

    t_idx_BT1 = mx.broadcast_to(mx.expand_dims(mx.arange(T), 0), [B, T])
    t_idx_BT1 = mx.expand_dims(t_idx_BT1, -1)

    t_idx_BxTxC = mx.minimum(
        t_idx_BT1 + mx.reshape(delay_arr, (1, 1, C)),
        mx.array(T - 1, dtype=mx.int32),
    )
    b_idx_BxTxC = mx.broadcast_to(mx.reshape(mx.arange(B), (B, 1, 1)), [B, T, C])
    c_idx_BxTxC = mx.broadcast_to(mx.reshape(mx.arange(C), (1, 1, C)), [B, T, C])

    indices_BTCx3 = mx.stack(
        [
            mx.reshape(b_idx_BxTxC, (-1,)),
            mx.reshape(t_idx_BxTxC, (-1,)),
            mx.reshape(c_idx_BxTxC, (-1,)),
        ],
        axis=1,
    ).astype(mx.int32)

    return t_idx_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: mx.array,
    pad_value: int,
    precomp: tp.Tuple[mx.array, mx.array],
    T: int,
) -> mx.array:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices (MLX version).

    Args:
        audio_BxTxC: Input delayed audio tensor
        pad_value: Padding value for out-of-bounds indices
        precomp: Precomputed revert indices tuple containing:
            - t_idx_BxTxC: Time offset indices tensor
            - indices_BTCx3: Gather indices tensor for original audio
        T: Original sequence length before padding

    Returns:
        Reverted audio tensor with same shape as input
    """
    t_idx_BxTxC, indices_BTCx3 = precomp

    def gather_nd(array, indices):
        gathered = []
        for idx in range(indices.shape[0]):
            b, t, c = indices[idx, 0], indices[idx, 1], indices[idx, 2]
            gathered.append(array[b, t, c])
        return mx.array(gathered)

    gathered_flat = gather_nd(audio_BxTxC, indices_BTCx3)
    gathered_BxTxC = mx.reshape(gathered_flat, audio_BxTxC.shape)
    pad_tensor = mx.full(1, pad_value, dtype=audio_BxTxC.dtype)
    T_tensor = mx.array(T)

    result_BxTxC = mx.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)

    return result_BxTxC


def decode(
    model,
    audio_codes,
):
    """
    Decodes the given frames into an output audio waveform
    """

    if len(audio_codes) != 1:
        raise ValueError(f"Expected one frame, got {len(audio_codes)}")

    try:
        audio_values = model.quantizer.from_codes(audio_codes)
        audio_values = model.decode(audio_values[0])

        return audio_values
    except Exception as e:
        print(f"Error in decode method: {str(e)}")
        raise


def codebook_to_audio(
    generated_codes: mx.array, model, delay_pattern, B=1, T=2600, C=9
):
    """Process a single codebook file to generate audio"""
    # Remove BOS token
    generated_codes = generated_codes[:, 1:]

    if generated_codes.shape[1] > T:
        generated_codes = generated_codes[:, :T]

    seq_length = generated_codes.shape[1]

    # Build revert indices
    t_idx_BxTxC, indices_BTCx3 = build_revert_indices(
        B=B, T=seq_length, C=C, delay_pattern=delay_pattern
    )

    # Transpose and add batch dimension
    audio_BxTxC = mx.expand_dims(mx.transpose(generated_codes, (1, 0)), 0)
    reverted_codebook = revert_audio_delay(
        audio_BxTxC=audio_BxTxC,
        pad_value=0,
        precomp=(t_idx_BxTxC, indices_BTCx3),
        T=seq_length,
    )
    reverted_codebook = reverted_codebook[:, :-30, :]

    codebook = mx.transpose(reverted_codebook, (0, 2, 1))

    min_valid_index = 0
    max_valid_index = 1023
    invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)

    num_invalid = mx.sum(invalid_mask).item()
    if num_invalid > 0:
        print(
            f"Warning: Clamping {num_invalid} indices outside range [{min_valid_index}, {max_valid_index}] to 0."
        )

    # Set invalid values to 0
    zeros = mx.zeros_like(codebook)
    codebook = mx.where(invalid_mask, zeros, codebook)

    audio_array = decode(model, codebook)

    return audio_array

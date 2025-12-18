from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from .convolution import ConvolutionModule
from .embedding import EspnetRelPositionalEncoding, RelPositionalEncoding
from .encoder_layer import ConformerEncoderLayer
from .positionwise_feed_forward import PositionwiseFeedForward
from .subsampling import LinearNoSubsampling


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (int): number of channels in the inputs and outputs.
        out_channels (int): number of output channels.
        stride (int): upsampling factor. Defaults to 2.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # In this mode, first repeat interpolate, then conv with stride=1
        self.conv = nn.Conv1d(
            self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0
        )

    def __call__(
        self, inputs: mx.array, input_lengths: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            inputs: Input tensor (B, C, T) - PyTorch format
            input_lengths: Lengths tensor (B,)

        Returns:
            outputs: Upsampled tensor (B, C, T*stride) - PyTorch format
            output_lengths: Updated lengths (B,)
        """
        # Upsample using nearest neighbor interpolation
        B, C, T = inputs.shape
        new_T = T * self.stride

        # Repeat each timestep stride times
        outputs = mx.repeat(inputs, self.stride, axis=2)

        # Pad on the left (axis 2 in PyTorch format B,C,T)
        outputs = mx.pad(outputs, [(0, 0), (0, 0), (self.stride * 2, 0)])

        # Transpose to MLX format (B, T, C) for conv
        outputs = mx.transpose(outputs, [0, 2, 1])

        # Apply convolution (MLX Conv1d expects B, T, C)
        outputs = self.conv(outputs)

        # Transpose back to PyTorch format (B, C, T)
        outputs = mx.transpose(outputs, [0, 2, 1])

        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead layer for causal processing."""

    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def __call__(self, inputs: mx.array, context: mx.array = None) -> mx.array:
        """
        Args:
            inputs: Input tensor (B, T, C)
            context: Optional context tensor for lookahead (B, pre_lookahead_len, C)
                     Used during streaming inference to provide future context.

        Returns:
            outputs: Output tensor (B, T, C)
        """
        # Input is (B, T, C) - MLX Conv1d expects (B, T, C)
        outputs = inputs

        # Look ahead padding on time dimension (axis 1)
        if context is None or context.shape[1] == 0:
            # No context - pad with zeros
            outputs = mx.pad(outputs, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        else:
            # Use context for lookahead
            assert (
                context.shape[1] == self.pre_lookahead_len
            ), f"Context length {context.shape[1]} != pre_lookahead_len {self.pre_lookahead_len}"
            outputs = mx.concatenate([outputs, context], axis=1)
            # Pad remaining if needed
            remaining = self.pre_lookahead_len - context.shape[1]
            if remaining > 0:
                outputs = mx.pad(outputs, [(0, 0), (0, remaining), (0, 0)])

        outputs = nn.leaky_relu(self.conv1(outputs))

        # Output padding on time dimension (axis 1)
        outputs = mx.pad(outputs, [(0, 0), (2, 0), (0, 0)])
        outputs = self.conv2(outputs)

        # Residual connection
        outputs = outputs + inputs
        return outputs


def make_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths: Batch of lengths (B,).
        max_len: Maximum length. If 0, uses max of lengths.

    Returns:
        Mask tensor containing indices of padded part (B, T).

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else int(mx.max(lengths).item())

    seq_range = mx.arange(max_len)
    seq_range_expand = mx.expand_dims(seq_range, 0)  # (1, T)
    seq_range_expand = mx.broadcast_to(
        seq_range_expand, (batch_size, max_len)
    )  # (B, T)

    seq_length_expand = mx.expand_dims(lengths, -1)  # (B, 1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
) -> mx.array:
    """Create mask for subsequent steps (size, size) with chunk size.

    This is for streaming encoder.

    Args:
        size: size of mask
        chunk_size: size of chunk
        num_left_chunks: number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks

    Returns:
        Mask tensor (size, size)

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    pos_idx = mx.arange(size)
    block_value = ((pos_idx // chunk_size) + 1) * chunk_size
    ret = mx.expand_dims(pos_idx, 0) < mx.expand_dims(block_value, 1)
    return ret


def add_optional_chunk_mask(
    xs: mx.array,
    masks: mx.array,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
) -> mx.array:
    """Apply optional mask for encoder.

    Args:
        xs: padded input, (B, L, D), L for max length
        masks: mask for xs, (B, 1, L)
        use_dynamic_chunk: whether to use dynamic chunk or not
        use_dynamic_left_chunk: whether to use dynamic left chunk for training.
        decoding_chunk_size: decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size: chunk size for static chunk training/decoding
        num_decoding_left_chunks: number of left chunks
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context:
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # For training, use random chunk size
            import random

            chunk_size = random.randint(1, max_len - 1)
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = random.randint(0, max_left_chunks)

        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size, num_left_chunks)
        chunk_masks = mx.expand_dims(chunk_masks, 0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(
            xs.shape[1], static_chunk_size, num_left_chunks
        )
        chunk_masks = mx.expand_dims(chunk_masks, 0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks

    # Check for all-false masks and fix them
    mask_sums = mx.sum(chunk_masks, axis=-1)
    if mx.any(mask_sums == 0):
        # Force set to true where all false
        chunk_masks = mx.where(
            mx.expand_dims(mask_sums == 0, -1), mx.ones_like(chunk_masks), chunk_masks
        )

    return chunk_masks


class UpsampleConformerEncoder(nn.Module):
    """Conformer encoder with upsampling for speech synthesis."""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        pre_lookahead_len: int = 3,
        upsample_stride: int = 2,
    ):
        """
        Args:
            input_size: input dimension
            output_size: dimension of attention
            attention_heads: the number of heads of multi head attention
            linear_units: the hidden units number of position-wise feed forward
            num_blocks: the number of encoder blocks before upsampling
            num_up_blocks: the number of encoder blocks after upsampling
            dropout_rate: dropout rate
            attention_dropout_rate: dropout rate in attention
            positional_dropout_rate: dropout rate after adding positional encoding
            input_layer: input layer type (currently only "linear" supported)
            pos_enc_layer_type: positional encoding layer type
            normalize_before: use layer_norm before each sub-block
            static_chunk_size: chunk size for static chunk training/decoding
            use_dynamic_chunk: whether use dynamic chunk size for training
            use_dynamic_left_chunk: whether use dynamic left chunk
            macaron_style: whether to use macaron style FFN
            selfattention_layer_type: self-attention layer type
            activation_type: activation function type
            use_cnn_module: whether to use convolution module
            cnn_module_kernel: kernel size of convolution module
            causal: whether to use causal convolution
            cnn_module_norm: normalization type for convolution module
            key_bias: whether use bias in attention.linear_k
            pre_lookahead_len: length of pre-lookahead window for streaming
            upsample_stride: stride for temporal upsampling (2 = 2x upsample)
        """
        super().__init__()
        self._output_size = output_size

        # Select positional encoding class based on pos_enc_layer_type
        if pos_enc_layer_type == "rel_pos_espnet":
            pos_enc_class = EspnetRelPositionalEncoding(
                output_size, positional_dropout_rate
            )
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding(output_size, positional_dropout_rate)
        else:
            raise ValueError(f"Unsupported pos_enc_layer_type: {pos_enc_layer_type}")

        # Input embedding layer
        if input_layer == "linear":
            self.embed = LinearNoSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class,
            )
        else:
            raise ValueError(f"Unsupported input_layer: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        # Activation function
        if activation_type == "swish":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")

        # Self-attention layer
        if selfattention_layer_type == "rel_selfattn":
            self_attn_class = RelPositionMultiHeadedAttention
        elif selfattention_layer_type == "selfattn":
            self_attn_class = MultiHeadedAttention
        else:
            raise ValueError(
                f"Unsupported selfattention_layer_type: {selfattention_layer_type}"
            )

        # Pre-lookahead layer
        self.pre_lookahead_layer = PreLookaheadLayer(
            channels=output_size, pre_lookahead_len=pre_lookahead_len
        )

        # Main encoder layers - use indexed attributes for weight loading
        self._num_encoders = num_blocks
        for i in range(num_blocks):
            layer = ConformerEncoderLayer(
                output_size,
                self_attn_class(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    key_bias,
                ),
                PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation
                ),
                (
                    PositionwiseFeedForward(
                        output_size, linear_units, dropout_rate, activation
                    )
                    if macaron_style
                    else None
                ),
                (
                    ConvolutionModule(
                        output_size,
                        cnn_module_kernel,
                        activation,
                        cnn_module_norm,
                        causal,
                    )
                    if use_cnn_module
                    else None
                ),
                dropout_rate,
                normalize_before,
            )
            setattr(self, f"encoders_{i}", layer)

        # Upsampling layer
        self.upsample_stride = upsample_stride
        self.up_layer = Upsample1D(
            channels=output_size, out_channels=output_size, stride=upsample_stride
        )

        # Upsampling embedding layer (uses same pos_enc_layer_type)
        if pos_enc_layer_type == "rel_pos_espnet":
            up_pos_enc_class = EspnetRelPositionalEncoding(
                output_size, positional_dropout_rate
            )
        elif pos_enc_layer_type == "rel_pos":
            up_pos_enc_class = RelPositionalEncoding(
                output_size, positional_dropout_rate
            )
        else:
            raise ValueError(f"Unsupported pos_enc_layer_type: {pos_enc_layer_type}")

        if input_layer == "linear":
            self.up_embed = LinearNoSubsampling(
                input_size,
                output_size,
                dropout_rate,
                up_pos_enc_class,
            )
        else:
            raise ValueError(f"Unsupported input_layer: {input_layer}")

        # Upsampling encoder layers - use indexed attributes for weight loading
        self._num_up_encoders = num_up_blocks
        for i in range(num_up_blocks):
            layer = ConformerEncoderLayer(
                output_size,
                self_attn_class(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    key_bias,
                ),
                PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation
                ),
                (
                    PositionwiseFeedForward(
                        output_size, linear_units, dropout_rate, activation
                    )
                    if macaron_style
                    else None
                ),
                (
                    ConvolutionModule(
                        output_size,
                        cnn_module_kernel,
                        activation,
                        cnn_module_norm,
                        causal,
                    )
                    if use_cnn_module
                    else None
                ),
                dropout_rate,
                normalize_before,
            )
            setattr(self, f"up_encoders_{i}", layer)

    def output_size(self) -> int:
        return self._output_size

    def __call__(
        self,
        xs: mx.array,
        xs_lens: mx.array,
        context: mx.array = None,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        streaming: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            context: optional context tensor for lookahead (B, pre_lookahead_len, D)
                     Used during streaming inference to provide future context.
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
            streaming: whether to use streaming (chunk-based) attention
                When True, uses static_chunk_size for causal masking
                When False, uses full context attention (chunk_size=0)

        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T*2, D)
            masks: batch padding mask (B, 1, T' ~= T*2)
        """
        T = xs.shape[1]
        masks = mx.logical_not(make_pad_mask(xs_lens, T))
        masks = mx.expand_dims(masks, 1)  # (B, 1, T)

        xs, pos_emb, masks = self.embed(xs, masks)

        # Embed context if provided
        embedded_context = None
        if context is not None and context.shape[1] > 0:
            context_masks = mx.ones((1, 1, context.shape[1]), dtype=mx.bool_)
            embedded_context, _, _ = self.embed(
                context, context_masks, offset=xs.shape[1]
            )

        mask_pad = masks  # (B, 1, T)

        # Use static_chunk_size when streaming, otherwise full context (0)
        effective_chunk_size = self.static_chunk_size if streaming else 0

        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            effective_chunk_size,
            num_decoding_left_chunks,
        )

        # Lookahead + conformer encoder
        xs = self.pre_lookahead_layer(xs, context=embedded_context)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        # Upsample + conformer encoder
        xs = mx.transpose(xs, [0, 2, 1])  # (B, D, T)
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = mx.transpose(xs, [0, 2, 1])  # (B, T', D)

        T = xs.shape[1]
        masks = mx.logical_not(make_pad_mask(xs_lens, T))
        masks = mx.expand_dims(masks, 1)  # (B, 1, T')

        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks  # (B, 1, T')

        # Scale chunk size by upsample stride for upsampled encoder
        effective_up_chunk_size = effective_chunk_size * self.up_layer.stride

        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            effective_up_chunk_size,
            num_decoding_left_chunks,
        )

        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks

    @property
    def encoders(self):
        """Get encoder layers as a list."""
        return [getattr(self, f"encoders_{i}") for i in range(self._num_encoders)]

    @property
    def up_encoders(self):
        """Get up_encoder layers as a list."""
        return [getattr(self, f"up_encoders_{i}") for i in range(self._num_up_encoders)]

    def forward_layers(
        self, xs: mx.array, chunk_masks: mx.array, pos_emb: mx.array, mask_pad: mx.array
    ) -> mx.array:
        """Forward through main encoder layers."""
        for i in range(self._num_encoders):
            layer = getattr(self, f"encoders_{i}")
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(
        self, xs: mx.array, chunk_masks: mx.array, pos_emb: mx.array, mask_pad: mx.array
    ) -> mx.array:
        """Forward through upsampling encoder layers."""
        for i in range(self._num_up_encoders):
            layer = getattr(self, f"up_encoders_{i}")
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

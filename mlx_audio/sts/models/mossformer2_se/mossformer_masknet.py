import mlx.core as mx
import mlx.nn as nn

from .computation_block import Computation_Block
from .globallayernorm import GlobalLayerNorm
from .scaledsinuembedding import ScaledSinuEmbedding


class MossFormer_MaskNet(nn.Module):
    """
    MLX implementation of MossFormer_MaskNet for mask prediction.

    This class is designed for predicting masks used in source separation tasks.
    It processes input tensors through various layers including convolutional layers,
    normalization, and a computation block to produce the final output.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the MossFormer2 blocks.
    out_channels_final : int
        Number of channels that are finally outputted.
    num_blocks : int
        Number of layers in the Dual Computation Block.
    norm : str
        Normalization type ('ln' for LayerNorm, 'gln' for GlobalLayerNorm).
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        If True, applies skip connections around intra-block connections.
    use_global_pos_enc : bool
        If True, uses global positional encodings.
    max_length : int
        Maximum sequence length for input tensors.

    Example
    ---------
    >>> import mlx.core as mx
    >>> mossformer_masknet = MossFormer_MaskNet(180, 512, out_channels_final=961, num_spks=2)
    >>> x = mx.random.normal((10, 180, 2000))  # Example input
    >>> x = mossformer_masknet(x)  # Forward pass
    >>> x.shape  # Expected output shape
    (10, 961, 2000)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_channels_final: int,
        num_blocks: int = 24,
        norm: str = "gln",
        num_spks: int = 2,
        skip_around_intra: bool = True,
        use_global_pos_enc: bool = True,
        max_length: int = 20000,
    ):
        super().__init__()

        # Store parameters for weight loading
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_final = out_channels_final
        self.num_blocks = num_blocks
        self.norm_type = norm
        self.num_spks = num_spks
        self.skip_around_intra = skip_around_intra
        self.use_global_pos_enc = use_global_pos_enc
        self.max_length = max_length

        # Initialize normalization layer
        if norm == "gln":
            self.norm = GlobalLayerNorm(in_channels, 3)
        elif norm == "ln":
            # Group norm with 1 group is equivalent to layer norm
            self.norm = nn.GroupNorm(
                1, in_channels, eps=1e-8, affine=True, pytorch_compatible=True
            )
        else:
            raise ValueError(f"Unsupported norm type: {norm}")

        # Encoder convolutional layer (1x1 convolution)
        self.conv1d_encoder = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        # Positional encoding
        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        # Computation block
        self.mdl = Computation_Block(
            num_blocks=num_blocks,
            out_channels=out_channels,
            norm="ln",
            skip_around_intra=skip_around_intra,
            use_mossformer2=False,  # Use MossFormerM (default)
        )

        # Output layers
        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1, bias=True
        )
        self.conv1_decoder = nn.Conv1d(
            out_channels, out_channels_final, kernel_size=1, bias=False
        )

        # PReLU activation
        self.prelu = nn.PReLU()

        # ReLU activation for final output
        self.activation = nn.ReLU()

        # Gated output layers
        self.output = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)
        self.output_gate = nn.Conv1d(
            out_channels, out_channels, kernel_size=1, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the MossFormer_MaskNet.

        Args:
            x (mx.array): Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            mx.array: Output tensor of shape (batch_size, out_channels_final, sequence_length)
        """
        # Normalize the input
        # [B, N, L]
        if self.norm_type == "gln":
            x = self.norm(x)
            # Apply encoder convolution
            # MLX Conv1d expects NLC format, so transpose from NCL to NLC
            x = mx.transpose(x, (0, 2, 1))  # [B, L, N]
            x = self.conv1d_encoder(x)  # [B, L, out_channels]
            x = mx.transpose(x, (0, 2, 1))  # [B, out_channels, L]
        else:
            # For GroupNorm, transpose once to NLC and keep it
            x = mx.transpose(x, (0, 2, 1))  # [B, L, N]
            x = self.norm(x)
            # Apply encoder convolution (already in NLC format)
            x = self.conv1d_encoder(x)  # [B, L, out_channels]
            x = mx.transpose(x, (0, 2, 1))  # [B, out_channels, L]

        if self.use_global_pos_enc:
            base = x  # Store the base for adding positional embedding
            x = mx.transpose(
                x, (0, 2, 1)
            )  # Change shape to [B, L, out_channels] for positional encoding
            emb = self.pos_enc(x)  # Get positional embeddings
            # Note: pos_enc returns [seq_len, dim], we need to expand to [B, L, dim]
            if len(emb.shape) == 2:
                # If positional encoding returns [L, dim], expand to [B, L, dim]
                batch_size = x.shape[0]
                emb = mx.broadcast_to(
                    mx.expand_dims(emb, axis=0),
                    (batch_size, emb.shape[0], emb.shape[1]),
                )
            emb = mx.transpose(emb, (0, 2, 1))  # Change back to [B, out_channels, L]
            x = base + emb  # Add positional embeddings to the base

        # Process through the computation block
        # [B, out_channels, L]
        x = self.mdl(x)

        x = self.prelu(x)  # Apply activation

        # Expand to multiple speakers
        # [B, out_channels, L] -> [B, out_channels*num_spks, L]
        x = mx.transpose(x, (0, 2, 1))  # [B, L, out_channels]
        x = self.conv1d_out(x)  # [B, L, out_channels*num_spks]
        x = mx.transpose(x, (0, 2, 1))  # [B, out_channels*num_spks, L]
        B, _, S = x.shape  # Unpack the batch size and sequence length

        # Reshape to [B*num_spks, out_channels, L]
        # This prepares the output for gating
        x = mx.reshape(x, (B * self.num_spks, -1, S))

        # Apply gated output layers
        # [B*num_spks, out_channels, L]
        x = mx.transpose(x, (0, 2, 1))  # [B*num_spks, L, out_channels]
        output_val = mx.tanh(self.output(x))  # [B*num_spks, L, out_channels]
        gate_val = mx.sigmoid(self.output_gate(x))  # [B*num_spks, L, out_channels]
        x = output_val * gate_val  # Element-wise multiplication for gating

        # Decode to final output (already in NLC format)
        # [B*num_spks, L, out_channels] -> [B*num_spks, L, out_channels_final]
        x = self.conv1_decoder(x)  # [B*num_spks, L, out_channels_final]
        x = mx.transpose(x, (0, 2, 1))  # [B*num_spks, out_channels_final, L]

        # Reshape to [B, num_spks, out_channels_final, L] for output
        _, N, L = x.shape
        x = mx.reshape(x, (B, self.num_spks, N, L))
        x = self.activation(x)  # Apply final activation

        # Transpose to [num_spks, B, out_channels_final, L] for output
        # return the 1st spk signal as the target speech
        x = mx.transpose(x, (1, 0, 2, 3))  # [num_spks, B, out_channels_final, L]
        result = mx.transpose(
            x[0], (0, 2, 1)
        )  # Return [B, L, out_channels_final] (first speaker only)
        return result

    def eval(self):
        """Set the model to evaluation mode."""
        # Set all sub-modules to eval mode
        if hasattr(self.mdl, "eval"):
            self.mdl.eval()
        if hasattr(self.pos_enc, "eval"):
            self.pos_enc.eval()
        return self

    def train(self):
        """Set the model to training mode."""
        # Set all sub-modules to train mode
        if hasattr(self.mdl, "train"):
            self.mdl.train()
        if hasattr(self.pos_enc, "train"):
            self.pos_enc.train()
        return self

# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .mossformer_masknet import MossFormer_MaskNet


class TestNet(nn.Module):
    """
    The TestNet class for testing the MossFormer MaskNet implementation in MLX.

    This class builds a model that integrates the MossFormer_MaskNet
    for processing input audio and generating masks for source separation.

    Arguments
    ---------
    n_layers : int
        The number of layers in the model. It determines the depth
        of the model architecture, we leave this para unused at this moment.
    """

    def __init__(self, n_layers: int = 18):
        super().__init__()
        self.n_layers = n_layers  # Set the number of layers
        # Initialize the MossFormer MaskNet with specified input and output channels
        self.mossformer = MossFormer_MaskNet(
            in_channels=180, out_channels=512, out_channels_final=961
        )
        # Compile the forward pass for better performance
        self._forward_compiled = mx.compile(self._forward)

    def _forward(self, input: mx.array) -> List[mx.array]:
        """
        Internal forward pass implementation.
        """
        out_list = []  # Initialize output list to store outputs

        # Input is [B, time, channels] but MaskNet expects [B, channels, time]
        x = mx.transpose(input, (0, 2, 1))

        # Get the mask from the MossFormer MaskNet
        mask = self.mossformer(x)  # Forward pass through the MossFormer_MaskNet
        # Stop gradient for inference mode (no backprop needed)
        mask = mx.stop_gradient(mask)
        out_list.append(mask)  # Append the mask to the output list

        return out_list  # Return the list containing the mask

    def __call__(self, input: mx.array) -> List[mx.array]:
        """
        Forward pass through the TestNet model (compiled for performance).

        Arguments
        ---------
        input : mx.array
            Input tensor of dimension [B, N, S], where B is the batch size,
            N is the number of input channels (180), and S is the sequence length.

        Returns
        -------
        out_list : list
            List containing the mask tensor predicted by the MossFormer_MaskNet.
        """
        return self._forward_compiled(input)


class MossFormer2SE(nn.Module):
    """
    MossFormer2 SE model for speech enhancement in MLX.

    This class encapsulates the functionality of the MossFormer MaskNet
    within a higher-level model. It processes input audio data to produce
    enhanced outputs and corresponding masks.

    Arguments
    ---------
    args : Namespace
        Configuration arguments that may include hyperparameters
        and model settings (not utilized in this implementation but
        can be extended for flexibility).

    Example
    ---------
    >>> model = MossFormer2SE(args)
    >>> x = mx.random.normal((10, 180, 2000))  # Example input
    >>> outputs, mask = model(x)  # Forward pass
    >>> outputs.shape, mask.shape  # Check output shapes
    """

    def __init__(self, args=None):
        super().__init__()
        # Initialize the TestNet model, which contains the MossFormer MaskNet
        self.model = TestNet()  # Instance of TestNet

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Forward pass through the model.

        Arguments
        ---------
        x : mx.array
            Input tensor of dimension [B, N, S], where B is the batch size,
            N is the number of channels (180 in this case), and S is the
            sequence length (e.g., time frames).

        Returns
        -------
        out_list : List[mx.array]
            List containing the mask tensor predicted by the model for speech separation.
        """
        # Get outputs from TestNet (returns a list with one mask)
        out_list = self.model(x)

        # Return the list to match PyTorch behavior
        return out_list

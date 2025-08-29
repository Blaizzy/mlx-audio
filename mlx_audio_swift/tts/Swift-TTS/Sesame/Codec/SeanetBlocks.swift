//
// SeanetBlocks for Sesame TTS Mimi Codec
// Shared building blocks used by both encoder and decoder

import Foundation
import MLX
import MLXNN

/// Residual block for Seanet using MLXNN components
/// Shared between SeanetEncoder and SeanetDecoder
class SeanetResidualBlock: Module {
    @ModuleInfo var conv1: MLXNN.Conv1d
    @ModuleInfo var conv2: MLXNN.Conv1d
    @ModuleInfo var norm1: MLXNN.RMSNorm
    @ModuleInfo var norm2: MLXNN.RMSNorm

    init(channels: Int, kernelSize: Int, dilation: Int, causal: Bool) {
        let padding = causal ? (kernelSize - 1) * dilation : kernelSize

        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernelSize,
            padding: padding,
            dilation: dilation,
            bias: true
        )

        self._conv2.wrappedValue = MLXNN.Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1,
            bias: true
        )

        // RMSNorm layers for better stability (Sesame's preferred normalization)
        self._norm1.wrappedValue = MLXNN.RMSNorm(dimensions: channels, eps: 1e-5)
        self._norm2.wrappedValue = MLXNN.RMSNorm(dimensions: channels, eps: 1e-5)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm architecture (modern transformer style)
        var residual = x
        residual = norm1(residual)

        // First convolution with ELU activation
        let y = MLXNN.elu(conv1(residual))

        // Second convolution with residual
        let out = conv2(y)

        // Final normalization and residual connection
        return norm2(out + x)
    }
}

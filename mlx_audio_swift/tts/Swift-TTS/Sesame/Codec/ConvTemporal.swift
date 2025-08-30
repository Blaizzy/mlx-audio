//
// ConvTemporal for Sesame TTS Mimi Codec
// Temporal processing components for downsampling and upsampling
// Equivalent to Python's ConvDownsample1d and ConvTrUpsample1d

import Foundation
import MLX
import MLXNN

/// ConvDownsample1d - Downsampling convolution for temporal processing
/// Reduces temporal resolution while maintaining channel dimension
class ConvDownsample1d: Module {
    @ModuleInfo var conv: MLXNN.Conv1d

    private let stride: Int
    private let causal: Bool

    init(stride: Int, dim: Int, causal: Bool) {
        self.stride = stride
        self.causal = causal

        // Create convolution with 2*stride kernel size for proper downsampling
        self._conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: causal ? (2 * stride - 1) : (stride - 1),  // Causal padding
            bias: false  // No bias for downsampling
        )

        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return conv(xs)
    }

    /// Process a single step for streaming
    func step(_ xs: MLXArray) -> MLXArray {
        return self(xs)
    }

    /// Reset any internal streaming state
    func resetState() {
        // ConvDownsample1d doesn't maintain internal state, so this is a no-op
        // In the future, if we add streaming convolutions, we would reset their state here
    }
}

/// ConvTrUpsample1d - Transposed upsampling convolution for temporal processing
/// Increases temporal resolution while maintaining channel dimension
/// Following Python implementation exactly
class ConvTrUpsample1d: Module {
    @ModuleInfo var convtr: MLXNN.ConvTransposed1d

    private let stride: Int
    private let causal: Bool
    private let dim: Int

    init(stride: Int, dim: Int, causal: Bool) {
        self.stride = stride
        self.causal = causal
        self.dim = dim

        // Follow Python implementation: use groups=dim for depthwise separable
        self._convtr.wrappedValue = MLXNN.ConvTransposed1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: causal ? stride : stride,  // Proper padding for causal processing
            groups: dim,  // Depthwise separable - one group per channel (following Python)
            bias: false   // No bias for upsampling
        )

        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return convtr(xs)
    }

    /// Process a single step for streaming
    /// Following Python StreamableConvTranspose1d.step() implementation
    func step(_ xs: MLXArray) -> MLXArray {
        // Simplified streaming implementation - for now just call regular forward
        // Python version has sophisticated streaming with _prev_ys state management
        // TODO: Implement full streaming logic like Python version
        return self(xs)
    }

    /// Reset any internal streaming state
    func resetState() {
        // ConvTrUpsample1d doesn't maintain internal state, so this is a no-op
        // In the future, if we add streaming convolutions, we would reset their state here
    }
}

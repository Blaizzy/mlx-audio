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
        print("🔍 DEBUG ConvDownsample1d: input shape = \(xs.shape)")
        print("🔍 DEBUG ConvDownsample1d: conv weight shape = \(conv.weight.shape)")
        
        let result = conv(xs)
        
        print("🔍 DEBUG ConvDownsample1d: output shape = \(result.shape)")
        return result
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
    @ParameterInfo(key: "convtr.convtr.convtr.weight") var weight: MLXArray
    @ParameterInfo(key: "convtr.convtr.convtr.bias") var bias: MLXArray?

    private let stride: Int
    private let causal: Bool
    private let dim: Int
    private let kernelSize: Int
    private let padding: Int
    private let groups: Int
    private let inChannels: Int
    private let outChannels: Int

    init(stride: Int, dim: Int, causal: Bool) {
        self.stride = stride
        self.causal = causal
        self.dim = dim
        self.kernelSize = 2 * stride
        self.padding = causal ? stride : stride
        self.groups = dim  // Depthwise separable
        self.inChannels = dim
        self.outChannels = dim

        // Initialize weight for depthwise separable: [out_channels/groups, kernel_size, in_channels]
        // For depthwise: [1, kernel_size, dim] since out_channels/groups = dim/dim = 1
        let scale = 1.0 / Float(dim * kernelSize)
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, 
            high: scale, 
            [1, kernelSize, dim]
        )
        
        // No bias for upsampling
        self._bias.wrappedValue = nil

        super.init()
        
        print("🔍 DEBUG ConvTrUpsample1d init: initialized weight with shape \(weight.shape)")
        print("🔍 DEBUG ConvTrUpsample1d init: weight range: \(weight.min().item(Float.self)) to \(weight.max().item(Float.self))")
        print("🔍 DEBUG ConvTrUpsample1d init: using depthwise separable (groups=\(groups))")
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        print("🔍 DEBUG ConvTrUpsample1d: input shape = \(xs.shape)")
        print("🔍 DEBUG ConvTrUpsample1d: weight shape = \(weight.shape)")
        print("🔍 DEBUG ConvTrUpsample1d: weight range: \(weight.min().item(Float.self)) to \(weight.max().item(Float.self))")
        print("🔍 DEBUG ConvTrUpsample1d: config - stride: \(stride), padding: \(padding), groups: \(groups)")
        
        // Following Python MLX implementation:
        // 1. For depthwise separable (groups == in_channels && groups == out_channels)
        // 2. Expand weight and use groups=1
        
        var expandedWeight: MLXArray
        var expandedGroups: Int
        
        if groups == inChannels && groups == outChannels {
            // Create identity matrix for depthwise expansion
            let eye = MLXArray.eye(outChannels)
                .reshaped([outChannels, 1, outChannels])
            let eyeRepeated = MLXArray.repeated(eye, count: kernelSize, axis: 1)
            
            // Expand weight: repeat across groups and multiply by identity
            let weightRepeated = MLXArray.repeated(weight, count: groups, axis: 0)
            expandedWeight = weightRepeated * eyeRepeated
            expandedGroups = 1
            
            print("🔧 DEBUG ConvTrUpsample1d: Expanded weight from \(weight.shape) to \(expandedWeight.shape)")
        } else {
            expandedWeight = weight
            expandedGroups = groups
        }
        
        // MLX uses channel-last format, so swap axes before and after convolution
        let xsSwapped = xs.swappedAxes(-1, -2)  // [1, 512, 3] -> [1, 3, 512]
        
        print("🔍 DEBUG ConvTrUpsample1d: input after swapaxes: \(xsSwapped.shape)")
        print("🔍 DEBUG ConvTrUpsample1d: expanded weight shape: \(expandedWeight.shape)")
        print("🔍 DEBUG ConvTrUpsample1d: expanded groups: \(expandedGroups)")
        
        let result = MLX.convTransposed1d(
            xsSwapped,
            expandedWeight,
            stride: stride,
            padding: padding,
            dilation: 1,
            groups: expandedGroups
        )
        
        print("🔍 DEBUG ConvTrUpsample1d: conv result shape: \(result.shape)")
        
        if result.shape.isEmpty {
            print("🚨 ERROR ConvTrUpsample1d: Got empty output")
            // Return a properly shaped tensor for debugging
            let debugOutput = MLXArray.zeros([xs.shape[0], xs.shape[2], dim])
            return debugOutput.swappedAxes(-1, -2)  // Swap back to original format
        }
        
        // Add bias if present
        var finalResult = result
        if let bias = bias {
            finalResult = result + bias
        }
        
        // Swap axes back to original format
        let output = finalResult.swappedAxes(-1, -2)  // [1, 6, 512] -> [1, 512, 6]
        
        print("🔍 DEBUG ConvTrUpsample1d: final output shape = \(output.shape)")
        
        return output
    }

    /// Process a single step for streaming
    func step(_ xs: MLXArray) -> MLXArray {
        return self(xs)
    }

    /// Reset any internal streaming state
    func resetState() {
        // No internal state to reset
    }
}
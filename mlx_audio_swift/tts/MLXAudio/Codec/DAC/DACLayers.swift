//
//  DACLayers.swift
//  MLXAudio
//
//  DAC (Descript Audio Codec) neural network layers
//  Ported from: mlx_audio/codec/models/descript/nn/layers.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Weight Normalization Helper

/// Normalize weight tensor along all axes except the specified dimension
func normalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    guard x.ndim == 3 else {
        fatalError("Input tensor must have 3 dimensions")
    }
    let axes = Array(0..<x.ndim).filter { $0 != exceptDim }
    let xSquared = MLX.pow(x, 2)
    let sumSquared = MLX.sum(xSquared, axes: axes, keepDims: true)
    return MLX.sqrt(sumSquared)
}

// MARK: - Snake Activation

/// Snake activation function: x + (1/alpha) * sin^2(alpha * x)
func snake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let recip = MLX.reciprocal(alpha + 1e-9)
    return x + recip * MLX.pow(MLX.sin(alpha * x), 2)
}

/// Snake1d activation layer with learnable alpha parameter
class DACSnake1d: Module, UnaryLayer {
    let alpha: MLXArray

    init(channels: Int) {
        self.alpha = MLX.ones([1, 1, channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return snake(x, alpha: alpha)
    }
}

// MARK: - Weight-Normalized Convolution

/// Weight-normalized 1D convolution layer
class DACWNConv1d: Module, UnaryLayer {
    // Use @ModuleInfo to map Python snake_case keys to Swift camelCase
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    var bias: MLXArray?

    let kernelSize: Int
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        // Initialize bias
        self.bias = bias ? MLX.zeros([outChannels]) : nil

        // Initialize weights with weight normalization
        let scale = sqrt(1.0 / Double(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels]
        )
        self._weightG.wrappedValue = normalizeWeight(weightInit)
        self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue + 1e-12)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input x: [batch, time, channels] - channels-last format (like Python)
        // Output: [batch, time, out_channels] - channels-last format
        // NO internal transposition - callers handle format conversion

        // Compute normalized weight
        let normV = normalizeWeight(weightV)
        let weight = weightG * weightV / (normV + 1e-12)

        // Apply convolution
        var y = MLX.conv1d(
            x,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )

        // Add bias if present
        if let bias = bias {
            y = y + bias
        }

        return y
    }
}

// MARK: - Weight-Normalized Transposed Convolution

/// Weight-normalized transposed 1D convolution layer
class DACWNConvTranspose1d: Module, UnaryLayer {
    // Use @ModuleInfo to map Python snake_case keys to Swift camelCase
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    var bias: MLXArray?

    let kernelSize: Int
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        // Initialize bias
        self.bias = bias ? MLX.zeros([outChannels]) : nil

        // Initialize weights with weight normalization
        // For transposed conv, weight shape is [outChannels, kernelSize, inChannels/groups]
        let scale = sqrt(1.0 / Double(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels / groups]
        )
        self._weightG.wrappedValue = normalizeWeight(weightInit, exceptDim: 2)
        self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue + 1e-12)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input x: [batch, time, channels] - channels-last format (like Python)
        // Output: [batch, time, out_channels] - channels-last format
        // NO internal transposition - callers handle format conversion

        // Compute normalized weight
        let normV = normalizeWeight(weightV, exceptDim: 2)
        let weight = weightG * weightV / (normV + 1e-12)

        // Apply transposed convolution
        var y = MLX.convTransposed1d(
            x,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )

        // Add bias if present
        if let bias = bias {
            y = y + bias
        }

        return y
    }
}

// MARK: - Residual Unit

/// Residual unit with dilated convolutions and snake activations
/// Uses Sequential to match Python weight key structure
class DACResidualUnit: Module, UnaryLayer {
    let block: Sequential

    init(dim: Int, dilation: Int = 1) {
        let pad = ((7 - 1) * dilation) / 2

        self.block = Sequential(layers: [
            DACSnake1d(channels: dim),
            DACWNConv1d(
                inChannels: dim,
                outChannels: dim,
                kernelSize: 7,
                padding: pad,
                dilation: dilation
            ),
            DACSnake1d(channels: dim),
            DACWNConv1d(
                inChannels: dim,
                outChannels: dim,
                kernelSize: 1
            )
        ])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = block(x)

        // Handle padding difference for residual connection
        let pad = (x.shape[2] - y.shape[2]) / 2
        var residual = x
        if pad > 0 {
            residual = x[0..<x.shape[0], 0..<x.shape[1], pad..<(x.shape[2] - pad)]
        }

        return residual + y
    }
}

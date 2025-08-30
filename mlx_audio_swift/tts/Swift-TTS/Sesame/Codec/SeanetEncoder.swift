//
// SeanetEncoder for Sesame TTS Mimi Codec
// Using MLXNN directly (like Kokoro/Orpheus) for standard components

import Foundation
import MLX
import MLXNN

/// Configuration for SeanetEncoder (matches Python config)
struct SeanetConfig: Codable {
    let dimension: Int
    let channels: Int
    let causal: Bool
    let nfilters: Int
    let nresidualLayers: Int
    let ratios: [Int]
    let ksize: Int
    let residualKsize: Int
    let lastKsize: Int
    let dilationBase: Int
    let padMode: String
    let trueSkip: Bool
    let compress: Int

    enum CodingKeys: String, CodingKey {
        case dimension, channels, causal, nfilters, nresidualLayers = "nresidual_layers"
        case ratios, ksize, residualKsize = "residual_ksize", lastKsize = "last_ksize"
        case dilationBase = "dilation_base", padMode = "pad_mode"
        case trueSkip = "true_skip", compress
    }
}

// Note: SeanetResidualBlock is defined in SeanetBlocks.swift
// and imported here to avoid duplication

/// SeanetEncoder - convolutional encoder for audio
class SeanetEncoder: Module {
    @ModuleInfo var initialConv: MLXNN.Conv1d
    private let residualBlocks: [SeanetResidualBlock]
    private let downsampleLayers: [MLXNN.Conv1d]
    @ModuleInfo var finalConv: MLXNN.Conv1d
    @ModuleInfo var finalNorm: MLXNN.RMSNorm

    private let config: SeanetConfig

    init(_ config: SeanetConfig) {
        self.config = config

        // Initial convolution using MLXNN.Conv1d
        self._initialConv.wrappedValue = MLXNN.Conv1d(
            inputChannels: config.channels,
            outputChannels: config.nfilters,
            kernelSize: config.ksize,
            padding: config.ksize,
            groups: 1,  // Following Python implementation
            bias: true
        )

        // Build residual blocks and downsampling layers
        var residualBlocks: [SeanetResidualBlock] = []
        var downLayers: [MLXNN.Conv1d] = []
        var currentChannels = config.nfilters

        for (i, ratio) in config.ratios.enumerated() {
            // Add residual blocks for this level
            for layer in 0..<config.nresidualLayers {
                let dilation = Int(pow(Float(config.dilationBase), Float(layer)))
                let block = SeanetResidualBlock(
                    channels: currentChannels,
                    kernelSize: config.residualKsize,
                    dilation: dilation,
                    causal: config.causal
                )
                residualBlocks.append(block)
            }

            // Downsampling layer (if not the last level)
            if i < config.ratios.count - 1 {
                let nextChannels = min(currentChannels * 2, config.dimension)
                let downConv = MLXNN.Conv1d(
                    inputChannels: currentChannels,
                    outputChannels: nextChannels,
                    kernelSize: ratio * 2,
                    stride: ratio,
                    padding: ratio,
                    groups: 1,  // Following Python implementation
                    bias: true
                )
                downLayers.append(downConv)
                currentChannels = nextChannels
            }
        }

        self.residualBlocks = residualBlocks
        self.downsampleLayers = downLayers

        // Final convolution to target dimension
        self._finalConv.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: config.dimension,
            kernelSize: config.lastKsize,
            padding: config.lastKsize,
            groups: 1,  // Following Python implementation
            bias: true
        )

        // Final RMSNorm for output stability
        self._finalNorm.wrappedValue = MLXNN.RMSNorm(dimensions: config.dimension, eps: 1e-5)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = initialConv(x)

        // Apply residual blocks and downsampling
        var blockIndex = 0
        for downLayer in downsampleLayers {
            // Apply residual blocks for this level
            for _ in 0..<config.nresidualLayers {
                y = residualBlocks[blockIndex](y)
                blockIndex += 1
            }
            // Downsample
            y = downLayer(y)
        }

        // Apply remaining residual blocks
        for i in blockIndex..<residualBlocks.count {
            y = residualBlocks[i](y)
        }

        // Final projection and normalization
        let encoded = finalConv(y)
        return finalNorm(encoded)
    }

    // MARK: - Streaming Methods

    /// Process a single step for streaming (same as forward for now)
    func step(_ x: MLXArray) -> MLXArray {
        return self(x)
    }

    /// Reset any internal streaming state
    func resetState() {
        // SeanetEncoder doesn't maintain internal state, so this is a no-op
        // In the future, if we add streaming convolutions, we would reset their state here
    }
}

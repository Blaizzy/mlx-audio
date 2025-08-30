//
// SeanetDecoder for Sesame TTS Mimi Codec
// Using MLXNN directly (like Kokoro/Orpheus) for standard components

import Foundation
import MLX
import MLXNN

// Note: SeanetResidualBlock is defined in SeanetBlocks.swift
// and imported here to avoid duplication

/// SeanetDecoder - convolutional decoder for audio using MLXNN components
class SeanetDecoder: Module {
    @ModuleInfo var initialConv: MLXNN.Conv1d
    private let residualBlocks: [SeanetResidualBlock]
    private let upsampleLayers: [MLXNN.ConvTransposed1d]
    @ModuleInfo var finalConv: MLXNN.Conv1d
    @ModuleInfo var finalNorm: MLXNN.RMSNorm

    private let config: SeanetConfig

    init(_ config: SeanetConfig) {
        self.config = config

        // Initial convolution to expand from latent dimension
        self._initialConv.wrappedValue = MLXNN.Conv1d(
            inputChannels: config.dimension,
            outputChannels: config.nfilters,
            kernelSize: config.ksize,
            padding: config.ksize,
            groups: 1,  // Following Python implementation
            bias: true
        )

        // Build residual blocks and upsampling layers (reverse order)
        var residualBlocks: [SeanetResidualBlock] = []
        var upLayers: [MLXNN.ConvTransposed1d] = []

        // Start from the final dimension and work backwards
        var currentChannels = config.nfilters

        for (i, ratio) in config.ratios.reversed().enumerated() {
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

            // Upsampling layer (if not the first level)
            if i < config.ratios.count - 1 {
                let nextChannels = max(currentChannels / 2, config.channels)
                let upConv = MLXNN.ConvTransposed1d(
                    inputChannels: currentChannels,
                    outputChannels: nextChannels,
                    kernelSize: ratio * 2,
                    stride: ratio,
                    padding: ratio,
                    groups: 1,  // Following Python implementation
                    bias: true
                )
                upLayers.append(upConv)
                currentChannels = nextChannels
            }
        }

        self.residualBlocks = residualBlocks.reversed()  // Reverse to match encoder order
        self.upsampleLayers = upLayers.reversed()

        // Final convolution to output channels (mono audio)
        self._finalConv.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: config.channels,
            kernelSize: config.lastKsize,
            padding: config.lastKsize,
            groups: 1,  // Following Python implementation
            bias: true
        )

        // Final RMSNorm for output stability
        self._finalNorm.wrappedValue = MLXNN.RMSNorm(dimensions: config.channels, eps: 1e-5)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = initialConv(x)

        // Apply residual blocks and upsampling (reverse order of encoder)
        var blockIndex = 0
        for upLayer in upsampleLayers {
            // Apply residual blocks for this level
            for _ in 0..<config.nresidualLayers {
                y = residualBlocks[blockIndex](y)
                blockIndex += 1
            }
            // Upsample
            y = upLayer(y)
        }

        // Apply remaining residual blocks
        for i in blockIndex..<residualBlocks.count {
            y = residualBlocks[i](y)
        }

        // Final projection and normalization
        let decoded = finalConv(y)
        return finalNorm(decoded)
    }

    // MARK: - Streaming Methods

    /// Process a single step for streaming (same as forward for now)
    func step(_ x: MLXArray) -> MLXArray {
        return self(x)
    }

    /// Reset any internal streaming state
    func resetState() {
        // SeanetDecoder doesn't maintain internal state, so this is a no-op
        // In the future, if we add streaming convolutions, we would reset their state here
    }
}

// Note: SeanetResidualBlock is defined in SeanetBlocks.swift
// and imported here to avoid duplication

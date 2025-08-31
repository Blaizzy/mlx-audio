//
// ProjectedTransformer for Sesame TTS Mimi Codec
// Main transformer component with input/output projections

import Foundation
import MLX
import MLXNN

/// ProjectedTransformer - Main transformer with input/output projections
/// Equivalent to MLX Python's ProjectedTransformer class
class ProjectedTransformer: Module {
    @ModuleInfo var transformer: Transformer
    @ModuleInfo var inputProj: MLXNN.Linear?
    var outputProjs: [MLXNN.Linear?]

    private let config: TransformerConfig
    private let inputDim: Int
    private let outputDims: [Int]
    private let convLayout: Bool

    init(config: TransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.config = config
        self.inputDim = inputDim
        self.outputDims = outputDims
        self.convLayout = config.convLayout

        // Initialize the core transformer
        self._transformer.wrappedValue = Transformer(config)

        // Initialize input projection (only if input_dim != d_model)
        if inputDim != config.dModel {
            self._inputProj.wrappedValue = MLXNN.Linear(inputDim, config.dModel, bias: false)
        } else {
            self._inputProj.wrappedValue = nil
        }

        // Initialize output projections
        var outputProjections: [MLXNN.Linear?] = []
        for outputDim in outputDims {
            if outputDim == config.dModel {
                outputProjections.append(nil)
            } else {
                outputProjections.append(MLXNN.Linear(config.dModel, outputDim, bias: false))
            }
        }
        self.outputProjs = outputProjections

        super.init()
    }

    /// Forward pass through the projected transformer (matches Python exactly)
    func callAsFunction(_ xs: MLXArray, cache: [LayerCache], crossAttentionSrc: MLXArray? = nil) -> [MLXArray] {
        var x = xs

        // Handle conv layout (transpose for convolution input)
        if convLayout {
            x = x.swappedAxes(1, 2)
        }

        // Apply input projection if needed
        if let inputProj = inputProj {
            x = inputProj(x)
        }

        // Apply transformer
        x = transformer(x, cache: cache, crossAttentionSrc: crossAttentionSrc)

        // Apply output projections
        var outputs: [MLXArray] = []
        for outputProj in outputProjs {
            var output = x
            if let proj = outputProj {
                output = proj(x)
            }

            // Handle conv layout (transpose back for convolution output)
            if convLayout {
                output = output.swappedAxes(1, 2)
            }

            outputs.append(output)
        }

        return outputs
    }

    /// Create standard cache (matches Python make_cache)
    func makeCache() -> [LayerCache] {
        return transformer.makeCache()
    }

    /// Create rotating cache (matches Python make_rot_cache)
    func makeRotCache() -> [LayerCache] {
        return transformer.makeRotCache()
    }

    /// Get transformer configuration
    var transformerConfig: TransformerConfig {
        return config
    }

    /// Get input dimension
    var inputDimension: Int {
        return inputDim
    }

    /// Get output dimensions
    var outputDimensions: [Int] {
        return outputDims
    }
}

/// Convenience initializers for common configurations
extension ProjectedTransformer {

    /// Encoder transformer (matches Python/official config)
    static func encoder(dModel: Int = 512, inputDim: Int = 512, outputDim: Int = 512) -> ProjectedTransformer {
        let config = TransformerConfig.encoderConfig(dModel: dModel)
        return ProjectedTransformer(config: config, inputDim: inputDim, outputDims: [outputDim])
    }

    /// Decoder transformer (matches Python/official config)
    static func decoder(dModel: Int = 512, inputDim: Int = 512, outputDim: Int = 512) -> ProjectedTransformer {
        let config = TransformerConfig.decoderConfig(dModel: dModel)
        return ProjectedTransformer(config: config, inputDim: inputDim, outputDims: [outputDim])
    }

    /// Multi-output transformer for complex architectures
    static func multiOutput(dModel: Int = 512, inputDim: Int = 512, outputDims: [Int]) -> ProjectedTransformer {
        let config = TransformerConfig.defaultConfig(dModel: dModel)
        return ProjectedTransformer(config: config, inputDim: inputDim, outputDims: outputDims)
    }
}
//
// TransformerConfig for Sesame TTS ProjectedTransformer
// Configuration for transformer architecture matching Python implementation

import Foundation

/// Configuration for Sesame TTS Transformer components (matches MLX Python TransformerConfig)
struct TransformerConfig {
    let dModel: Int
    let numHeads: Int
    let numLayers: Int
    let causal: Bool
    let normFirst: Bool
    let biasFF: Bool
    let biasAttn: Bool
    let layerScale: Float?
    let positionalEmbedding: String  // "rope", "none", "sin", "sin_rope"
    let useConvBlock: Bool
    let crossAttention: Bool
    let convKernelSize: Int
    let useConvBias: Bool
    var gating: Bool
    let norm: String
    let context: Int
    let maxPeriod: Int
    let maxSeqLen: Int
    var kvRepeat: Int
    let dimFeedforward: Int
    let convLayout: Bool

    /// Computed property for head dimension
    var headDim: Int {
        return dModel / numHeads
    }

    /// Default configuration for Sesame TTS
    static func defaultConfig(dModel: Int = 512, numHeads: Int = 8, numLayers: Int = 6) -> TransformerConfig {
        return TransformerConfig(
            dModel: dModel,
            numHeads: numHeads,
            numLayers: numLayers,
            causal: true,
            normFirst: true,
            biasFF: true,
            biasAttn: true,
            layerScale: 0.1,
            positionalEmbedding: "rope",
            useConvBlock: false,
            crossAttention: false,
            convKernelSize: 3,
            useConvBias: true,
            gating: true,
            norm: "rms_norm",
            context: 250,
            maxPeriod: 10000,
            maxSeqLen: 2048,
            kvRepeat: 1,
            dimFeedforward: 4 * dModel,
            convLayout: true
        )
    }

    /// Encoder-specific configuration (matches MLX Python mimi_202407)
    static func encoderConfig(dModel: Int = 512) -> TransformerConfig {
        return TransformerConfig(
            dModel: dModel,
            numHeads: 8,
            numLayers: 8,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: 0.01,
            positionalEmbedding: "rope",  // Python uses rope for Mimi
            useConvBlock: false,
            crossAttention: false,
            convKernelSize: 3,
            useConvBias: true,
            gating: false,
            norm: "layer_norm",  // Python uses layer_norm for Mimi
            context: 250,
            maxPeriod: 10000,
            maxSeqLen: 8192,
            kvRepeat: 1,
            dimFeedforward: 2048,
            convLayout: true
        )
    }

    /// Decoder-specific configuration (matches MLX Python mimi_202407)
    static func decoderConfig(dModel: Int = 512) -> TransformerConfig {
        return TransformerConfig(
            dModel: dModel,
            numHeads: 8,
            numLayers: 8,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: 0.01,
            positionalEmbedding: "rope",  // Python uses rope for Mimi
            useConvBlock: false,
            crossAttention: false,
            convKernelSize: 3,
            useConvBias: true,
            gating: false,
            norm: "layer_norm",  // Python uses layer_norm for Mimi
            context: 250,
            maxPeriod: 10000,
            maxSeqLen: 8192,
            kvRepeat: 1,
            dimFeedforward: 2048,
            convLayout: true
        )
    }
}
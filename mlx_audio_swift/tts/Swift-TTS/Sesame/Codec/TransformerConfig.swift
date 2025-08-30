//
// TransformerConfig for Sesame TTS ProjectedTransformer
// Configuration for transformer architecture matching Python implementation

import Foundation

/// Configuration for Sesame TTS Transformer components
struct TransformerConfig {
    let dModel: Int
    let numHeads: Int
    let numLayers: Int
    let causal: Bool
    let normFirst: Bool
    let biasFF: Bool
    let biasAttn: Bool
    let layerScale: Float?
    let positionalEmbedding: String
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

    /// Encoder-specific configuration
    static func encoderConfig(dModel: Int = 512) -> TransformerConfig {
        var config = defaultConfig(dModel: dModel, numHeads: 8, numLayers: 6)
        config.gating = false
        return config
    }

    /// Decoder-specific configuration
    static func decoderConfig(dModel: Int = 512) -> TransformerConfig {
        var config = defaultConfig(dModel: dModel, numHeads: 16, numLayers: 6)
        config.kvRepeat = 4  // 16 query heads, 4 key/value heads for GQA
        return config
    }
}

//
// SesameModelArgs for Sesame TTS
// Configuration for Llama model components
// Based on Python mlx_lm.models.llama.ModelArgs
//

import Foundation

/// Model arguments for Llama-based models in Sesame TTS
/// Equivalent to Python's ModelArgs from mlx_lm
public struct LlamaModelArgs {
    public let modelType: String
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int?
    public let headDim: Int?
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool?
    public let mlpBias: Bool?
    public let ropeTheta: Float?
    public let ropeScaling: [String: Any]?

    // Text-specific parameters for Sesame
    public let textVocabSize: Int
    public let audioVocabSize: Int
    public let audioNumCodebooks: Int
    public let audioNumCodebooksTotal: Int
    public let backboneFlavor: String

    /// Default initializer with common defaults
    init(
        modelType: String = "llama",
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int? = nil,
        headDim: Int? = nil,
        hiddenSize: Int,
        intermediateSize: Int,
        rmsNormEps: Float = 1e-5,
        vocabSize: Int,
        maxPositionEmbeddings: Int,
        attentionBias: Bool? = false,
        mlpBias: Bool? = false,
        ropeTheta: Float? = 500000.0,
        ropeScaling: [String: Any]? = nil,
        textVocabSize: Int,
        audioVocabSize: Int,
        audioNumCodebooks: Int,
        audioNumCodebooksTotal: Int,
        backboneFlavor: String
    ) {
        self.modelType = modelType
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.textVocabSize = textVocabSize
        self.audioVocabSize = audioVocabSize
        self.audioNumCodebooks = audioNumCodebooks
        self.audioNumCodebooksTotal = audioNumCodebooksTotal
        self.backboneFlavor = backboneFlavor
    }

    /// Create Llama-1B configuration
    static func llama1B(
        textVocabSize: Int = 128256,
        audioVocabSize: Int = 2048,
        audioNumCodebooks: Int = 32,
        audioNumCodebooksTotal: Int = 32
    ) -> LlamaModelArgs {
        return LlamaModelArgs(
            numHiddenLayers: 16,
            numAttentionHeads: 32,
            numKeyValueHeads: 8,
            headDim: 64,
            hiddenSize: 2048,
            intermediateSize: 8192,
            vocabSize: textVocabSize,
            maxPositionEmbeddings: 2048,
            attentionBias: false,
            mlpBias: false,
            ropeTheta: 500000.0,
            ropeScaling: [
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            ],
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooksTotal,
            backboneFlavor: "llama-1B"
        )
    }

    /// Create Llama-100M configuration
    static func llama100M(
        textVocabSize: Int = 128256,
        audioVocabSize: Int = 2048,
        audioNumCodebooks: Int = 32,
        audioNumCodebooksTotal: Int = 32
    ) -> LlamaModelArgs {
        return LlamaModelArgs(
            numHiddenLayers: 12,
            numAttentionHeads: 12,
            numKeyValueHeads: 3,
            headDim: 64,
            hiddenSize: 768,
            intermediateSize: 2048,
            vocabSize: textVocabSize,
            maxPositionEmbeddings: 2048,
            attentionBias: false,
            mlpBias: false,
            ropeTheta: 500000.0,
            ropeScaling: [
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            ],
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooksTotal,
            backboneFlavor: "llama-100M"
        )
    }

    /// Create backbone model args for encoder
    func createBackboneArgs() -> LlamaModelArgs {
        return LlamaModelArgs(
            modelType: modelType,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            rmsNormEps: rmsNormEps,
            vocabSize: textVocabSize,
            maxPositionEmbeddings: maxPositionEmbeddings,
            attentionBias: attentionBias,
            mlpBias: mlpBias,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooksTotal,
            backboneFlavor: backboneFlavor
        )
    }

    /// Create decoder model args for audio generation
    func createDecoderArgs() -> LlamaModelArgs {
        return LlamaModelArgs(
            modelType: modelType,
            numHiddenLayers: 8,  // Decoder typically has fewer layers
            numAttentionHeads: 16,
            numKeyValueHeads: 4,
            headDim: headDim,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            rmsNormEps: rmsNormEps,
            vocabSize: audioVocabSize,
            maxPositionEmbeddings: audioNumCodebooks,
            attentionBias: attentionBias,
            mlpBias: mlpBias,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooksTotal,
            backboneFlavor: backboneFlavor
        )
    }
}

//
// SesameModelArgs for Sesame TTS
// Configuration for Llama model components
// Based on Python mlx_lm.models.llama.ModelArgs
//

import Foundation
import MLX

/// Rope scaling configuration
public struct RopeScalingConfig: Codable {
    public let factor: Float?
    public let highFreqFactor: Float?
    public let lowFreqFactor: Float?
    public let originalMaxPositionEmbeddings: Int?
    public let ropeType: String?

    private enum CodingKeys: String, CodingKey {
        case factor
        case highFreqFactor = "high_freq_factor"
        case lowFreqFactor = "low_freq_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case ropeType = "rope_type"
    }
}

/// Depth decoder configuration for Sesame TTS
/// Equivalent to Python's DepthDecoderConfig
public struct DepthDecoderConfig: Codable {
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let backboneHiddenSize: Int
    public let headDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let initializerRange: Float
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let mlpBias: Bool
    public let modelType: String
    public let numAttentionHeads: Int
    public let numCodebooks: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Float
    public let ropeScaling: RopeScalingConfig?
    public let ropeTheta: Float
    public let useCache: Bool
    public let vocabSize: Int

    private enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case backboneHiddenSize = "backbone_hidden_size"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case initializerRange = "initializer_range"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case mlpBias = "mlp_bias"
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case numCodebooks = "num_codebooks"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
        case useCache = "use_cache"
        case vocabSize = "vocab_size"
    }
}

/// Segment struct (shared type to avoid circular imports)
/// Represents a piece of audio with text and speaker information
public struct SesameSegment {
    public let speaker: Int
    public let text: String
    public let audio: MLXArray
    
    public init(speaker: Int, text: String, audio: MLXArray) {
        self.speaker = speaker
        self.text = text
        self.audio = audio
    }
}

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
    public let ropeScaling: RopeScalingConfig?

    // Text-specific parameters for Sesame
    public let textVocabSize: Int
    public let audioVocabSize: Int
    public let audioNumCodebooks: Int
    public let audioNumCodebooksTotal: Int
    public let backboneFlavor: String
    public let depthDecoderConfig: DepthDecoderConfig?
    
    // Audio token IDs (following Marvis TTS pattern)
    public let audioTokenId: Int
    public let audioEosTokenId: Int
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int

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
        ropeScaling: RopeScalingConfig? = nil,
        textVocabSize: Int,
        audioVocabSize: Int,
        audioNumCodebooks: Int,
        audioNumCodebooksTotal: Int,
        backboneFlavor: String,
        depthDecoderConfig: DepthDecoderConfig? = nil,
        audioTokenId: Int = 128002,
        audioEosTokenId: Int = 128003,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        padTokenId: Int = 0
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
        self.depthDecoderConfig = depthDecoderConfig
        self.audioTokenId = audioTokenId
        self.audioEosTokenId = audioEosTokenId
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
    }

    /// Create Llama-1B configuration
    static func llama1B(
        textVocabSize: Int = 49152,
        audioVocabSize: Int = 2051,
        audioNumCodebooks: Int = 32,
        audioNumCodebooksTotal: Int = 32,
        depthDecoderConfig: DepthDecoderConfig? = nil
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
            ropeScaling: RopeScalingConfig(
                factor: 32.0,
                highFreqFactor: 4.0,
                lowFreqFactor: 1.0,
                originalMaxPositionEmbeddings: 8192,
                ropeType: "llama3"
            ),
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooksTotal,
            backboneFlavor: "llama-1B",
            depthDecoderConfig: depthDecoderConfig,
            audioTokenId: textVocabSize - 2,
            audioEosTokenId: textVocabSize - 1,
            bosTokenId: 1,
            eosTokenId: 2,
            padTokenId: 0
        )
    }

    /// Create Llama-100M configuration
    static func llama100M(
        textVocabSize: Int = 49152,
        audioVocabSize: Int = 2051,
        audioNumCodebooks: Int = 32,
        audioNumCodebooksTotal: Int = 32,
        depthDecoderConfig: DepthDecoderConfig? = nil
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
            ropeScaling: RopeScalingConfig(
                factor: 32.0,
                highFreqFactor: 4.0,
                lowFreqFactor: 1.0,
                originalMaxPositionEmbeddings: 8192,
                ropeType: "llama3"
            ),
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooksTotal,
            backboneFlavor: "llama-100M",
            depthDecoderConfig: depthDecoderConfig,
            audioTokenId: textVocabSize - 2,
            audioEosTokenId: textVocabSize - 1,
            bosTokenId: 1,
            eosTokenId: 2,
            padTokenId: 0
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
            backboneFlavor: backboneFlavor,
            depthDecoderConfig: depthDecoderConfig,
            audioTokenId: audioTokenId,
            audioEosTokenId: audioEosTokenId,
            bosTokenId: bosTokenId,
            eosTokenId: eosTokenId,
            padTokenId: padTokenId
        )
    }

    /// Create decoder model args for audio generation using depth_decoder_config
    func createDecoderArgs() -> LlamaModelArgs {
        guard let decoderConfig = depthDecoderConfig else {
            // Fallback to hardcoded values if no config provided (for backward compatibility)
            return LlamaModelArgs(
                modelType: modelType,
                numHiddenLayers: 8,
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
                backboneFlavor: backboneFlavor,
                depthDecoderConfig: depthDecoderConfig,
                audioTokenId: audioTokenId,
                audioEosTokenId: audioEosTokenId,
                bosTokenId: bosTokenId,
                eosTokenId: eosTokenId,
                padTokenId: padTokenId
            )
        }

        // Use actual config from depth_decoder_config (like Python implementation)
        return LlamaModelArgs(
            modelType: decoderConfig.modelType,
            numHiddenLayers: decoderConfig.numHiddenLayers,
            numAttentionHeads: decoderConfig.numAttentionHeads,
            numKeyValueHeads: decoderConfig.numKeyValueHeads,
            headDim: decoderConfig.headDim,
            hiddenSize: decoderConfig.hiddenSize,  // Use decoder's hiddenSize, not backbone's
            intermediateSize: decoderConfig.intermediateSize,
            rmsNormEps: decoderConfig.rmsNormEps,
            vocabSize: decoderConfig.vocabSize,  // Use decoder's vocabSize
            maxPositionEmbeddings: decoderConfig.maxPositionEmbeddings,
            attentionBias: decoderConfig.attentionBias,
            mlpBias: decoderConfig.mlpBias,
            ropeTheta: decoderConfig.ropeTheta,
            ropeScaling: decoderConfig.ropeScaling,
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooksTotal,
            backboneFlavor: backboneFlavor,
            depthDecoderConfig: depthDecoderConfig,
            audioTokenId: audioTokenId,
            audioEosTokenId: audioEosTokenId,
            bosTokenId: bosTokenId,
            eosTokenId: eosTokenId,
            padTokenId: padTokenId
        )
    }

    /// Test function to validate configuration loading
    /// - Parameter configPath: Path to the sesame_config.json file
    /// - Returns: Validation results
    static func validateSesameConfig(configPath: String) throws -> (backboneHiddenSize: Int, decoderHiddenSize: Int, projectionShape: (Int, Int)) {
        let config = try fromSesameConfig(configPath: configPath)

        let backboneHiddenSize = config.hiddenSize
        let decoderHiddenSize = config.depthDecoderConfig?.hiddenSize ?? config.hiddenSize
        let projectionShape = (backboneHiddenSize, decoderHiddenSize)

        print("✅ Configuration Validation:")
        print("  - Backbone hidden size: \(backboneHiddenSize)")
        print("  - Decoder hidden size: \(decoderHiddenSize)")
        print("  - Projection shape: \(projectionShape.0) -> \(projectionShape.1)")

        return (backboneHiddenSize, decoderHiddenSize, projectionShape)
    }

    /// Load Sesame configuration from JSON file and create LlamaModelArgs
    /// - Parameter configPath: Path to the sesame_config.json file
    /// - Returns: LlamaModelArgs configured for Sesame TTS
    static func fromSesameConfig(configPath: String) throws -> LlamaModelArgs {
        let configURL = URL(fileURLWithPath: configPath)
        let configData = try Data(contentsOf: configURL)

        // Parse the full config to extract depth_decoder_config
        let jsonObject = try JSONSerialization.jsonObject(with: configData, options: []) as! [String: Any]

        // Extract depth_decoder_config
        guard let depthDecoderJson = jsonObject["depth_decoder_config"] as? [String: Any] else {
            throw NSError(domain: "SesameConfigError", code: 1, userInfo: [NSLocalizedDescriptionKey: "depth_decoder_config not found in config file"])
        }

        // Convert depth_decoder_config to JSON data and decode
        let depthDecoderData = try JSONSerialization.data(withJSONObject: depthDecoderJson, options: [])
        let depthDecoderConfig = try JSONDecoder().decode(DepthDecoderConfig.self, from: depthDecoderData)

        // Parse rope scaling configuration
        var ropeScalingConfig: RopeScalingConfig? = nil
        if let ropeScalingJson = jsonObject["rope_scaling"] as? [String: Any] {
            let ropeScalingData = try JSONSerialization.data(withJSONObject: ropeScalingJson, options: [])
            ropeScalingConfig = try JSONDecoder().decode(RopeScalingConfig.self, from: ropeScalingData)
        }

        // Extract other required parameters
        let textVocabSize = jsonObject["text_vocab_size"] as? Int ?? 49152
        let audioVocabSize = jsonObject["audio_vocab_size"] as? Int ?? 2051
        let audioNumCodebooks = jsonObject["audio_num_codebooks"] as? Int ?? 32
        let hiddenSize = jsonObject["hidden_size"] as? Int ?? 1536
        let numAttentionHeads = jsonObject["num_attention_heads"] as? Int ?? 12
        let numKeyValueHeads = jsonObject["num_key_value_heads"] as? Int ?? 3
        let numHiddenLayers = jsonObject["num_hidden_layers"] as? Int ?? 6
        let intermediateSize = jsonObject["intermediate_size"] as? Int ?? 8192
        let maxPositionEmbeddings = jsonObject["max_position_embeddings"] as? Int ?? 2048
        let backboneFlavor = jsonObject["backbone_flavor"] as? String ?? "llama-250M"
        
        // Extract token IDs (following Marvis TTS pattern)
        let audioTokenId = jsonObject["audio_token_id"] as? Int ?? (textVocabSize - 2)
        let audioEosTokenId = jsonObject["audio_eos_token_id"] as? Int ?? (textVocabSize - 1)
        let bosTokenId = jsonObject["bos_token_id"] as? Int ?? 1
        let eosTokenId = jsonObject["eos_token_id"] as? Int ?? 2
        let padTokenId = jsonObject["pad_token_id"] as? Int ?? 0

        return LlamaModelArgs(
            modelType: "sesame",
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: 128,  // Based on config analysis
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            rmsNormEps: 1e-5,
            vocabSize: textVocabSize,
            maxPositionEmbeddings: maxPositionEmbeddings,
            attentionBias: false,
            mlpBias: false,
            ropeTheta: 500000.0,
            ropeScaling: ropeScalingConfig,
            textVocabSize: textVocabSize,
            audioVocabSize: audioVocabSize,
            audioNumCodebooks: audioNumCodebooks,
            audioNumCodebooksTotal: audioNumCodebooks,
            backboneFlavor: backboneFlavor,
            depthDecoderConfig: depthDecoderConfig,
            audioTokenId: audioTokenId,
            audioEosTokenId: audioEosTokenId,
            bosTokenId: bosTokenId,
            eosTokenId: eosTokenId,
            padTokenId: padTokenId
        )
    }
}
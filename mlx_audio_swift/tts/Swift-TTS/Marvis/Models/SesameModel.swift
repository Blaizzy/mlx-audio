
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configs

public enum JSONValue: Codable, Equatable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let i = try? c.decode(Int.self) { self = .number(Double(i)); return }
        if let d = try? c.decode(Double.self) { self = .number(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let a = try? c.decode([JSONValue].self) { self = .array(a); return }
        if let o = try? c.decode([String: JSONValue].self) { self = .object(o); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "Unsupported JSON value")
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
}

public struct DepthDecoderConfig: Codable, Sendable {
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let backboneHiddenSize: Int
    public let headDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let initializerRange: Double
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let mlpBias: Bool
    public let modelType: String
    public let numAttentionHeads: Int
    public let numCodebooks: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Double
    public let ropeScaling: [String: JSONValue]?
    public let ropeTheta: Int
    public let useCache: Bool
    public let vocabSize: Int

    public init(
        attentionBias: Bool,
        attentionDropout: Double,
        backboneHiddenSize: Int,
        headDim: Int,
        hiddenAct: String,
        hiddenSize: Int,
        initializerRange: Double,
        intermediateSize: Int,
        maxPositionEmbeddings: Int,
        mlpBias: Bool,
        modelType: String,
        numAttentionHeads: Int,
        numCodebooks: Int,
        numHiddenLayers: Int,
        numKeyValueHeads: Int,
        rmsNormEps: Double,
        ropeScaling: [String: JSONValue]?,
        ropeTheta: Int,
        useCache: Bool,
        vocabSize: Int
    ) {
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.backboneHiddenSize = backboneHiddenSize
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.initializerRange = initializerRange
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.mlpBias = mlpBias
        self.modelType = modelType
        self.numAttentionHeads = numAttentionHeads
        self.numCodebooks = numCodebooks
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.rmsNormEps = rmsNormEps
        self.ropeScaling = ropeScaling
        self.ropeTheta = ropeTheta
        self.useCache = useCache
        self.vocabSize = vocabSize
    }

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

public struct SesameModelArgs: Codable, Sendable {
    public let modelType: String
    public let backboneFlavor: String
    public let decoderFlavor: String
    public let textVocabSize: Int
    public let audioVocabSize: Int
    public let audioNumCodebooks: Int
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let audioEosTokenId: Int
    public let audioTokenId: Int
    public let bosTokenId: Int
    public let codebookEosTokenId: Int
    public let codebookPadTokenId: Int
    public let depthDecoderConfig: DepthDecoderConfig?
    public let headDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let initializerRange: Double
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let mlpBias: Bool
    public let numAttentionHeads: Int
    public let numCodebooks: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let padTokenId: Int
    public let rmsNormEps: Double
    public let ropeScaling: [String: JSONValue]?
    public let ropeTheta: Int
    public let tieCodebooksEmbeddings: Bool
    public let tieWordEmbeddings: Bool
    public let useCache: Bool
    public let vocabSize: Int
    public let quantization: [String: Int]?

    public init(
        modelType: String,
        backboneFlavor: String,
        decoderFlavor: String,
        textVocabSize: Int,
        audioVocabSize: Int,
        audioNumCodebooks: Int,
        attentionBias: Bool,
        attentionDropout: Double,
        audioEosTokenId: Int,
        audioTokenId: Int,
        bosTokenId: Int,
        codebookEosTokenId: Int,
        codebookPadTokenId: Int,
        depthDecoderConfig: DepthDecoderConfig?,
        headDim: Int,
        hiddenAct: String,
        hiddenSize: Int,
        initializerRange: Double,
        intermediateSize: Int,
        maxPositionEmbeddings: Int,
        mlpBias: Bool,
        numAttentionHeads: Int,
        numCodebooks: Int,
        numHiddenLayers: Int,
        numKeyValueHeads: Int,
        padTokenId: Int,
        rmsNormEps: Double,
        ropeScaling: [String: JSONValue]?,
        ropeTheta: Int,
        tieCodebooksEmbeddings: Bool,
        tieWordEmbeddings: Bool,
        useCache: Bool,
        vocabSize: Int,
        quantization: [String: Int]?,
    ) {
        self.modelType = modelType
        self.backboneFlavor = backboneFlavor
        self.decoderFlavor = decoderFlavor
        self.textVocabSize = textVocabSize
        self.audioVocabSize = audioVocabSize
        self.audioNumCodebooks = audioNumCodebooks
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.audioEosTokenId = audioEosTokenId
        self.audioTokenId = audioTokenId
        self.bosTokenId = bosTokenId
        self.codebookEosTokenId = codebookEosTokenId
        self.codebookPadTokenId = codebookPadTokenId
        self.depthDecoderConfig = depthDecoderConfig
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.initializerRange = initializerRange
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.mlpBias = mlpBias
        self.numAttentionHeads = numAttentionHeads
        self.numCodebooks = numCodebooks
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.padTokenId = padTokenId
        self.rmsNormEps = rmsNormEps
        self.ropeScaling = ropeScaling
        self.ropeTheta = ropeTheta
        self.tieCodebooksEmbeddings = tieCodebooksEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
        self.useCache = useCache
        self.vocabSize = vocabSize
        self.quantization = quantization
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneFlavor = "backbone_flavor"
        case decoderFlavor = "decoder_flavor"
        case textVocabSize = "text_vocab_size"
        case audioVocabSize = "audio_vocab_size"
        case audioNumCodebooks = "audio_num_codebooks"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case audioEosTokenId = "audio_eos_token_id"
        case audioTokenId = "audio_token_id"
        case bosTokenId = "bos_token_id"
        case codebookEosTokenId = "codebook_eos_token_id"
        case codebookPadTokenId = "codebook_pad_token_id"
        case depthDecoderConfig = "depth_decoder_config"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case initializerRange = "initializer_range"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case mlpBias = "mlp_bias"
        case numAttentionHeads = "num_attention_heads"
        case numCodebooks = "num_codebooks"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case padTokenId = "pad_token_id"
        case rmsNormEps = "rms_norm_eps"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
        case tieCodebooksEmbeddings = "tie_codebooks_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useCache = "use_cache"
        case vocabSize = "vocab_size"
        case quantization
    }
}

public extension SesameModelArgs {
    static func load(from data: Data) throws -> SesameModelArgs {
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .useDefaultKeys
        return try dec.decode(SesameModelArgs.self, from: data)
    }

    static func load(from url: URL) throws -> SesameModelArgs {
        try load(from: Data(contentsOf: url))
    }
}

private func toStringOrNumber(_ dict: [String: JSONValue]?) -> [String: StringOrNumber]? {
    guard let dict else { return nil }
    var out: [String: StringOrNumber] = [:]
    for (k, v) in dict {
        switch v {
        case .string(let s): out[k] = .string(s)
        case .number(let d): out[k] = .float(Float(d))
        case .bool(let b): out[k] = .float(b ? 1.0 : 0.0)
        case .null, .array, .object: continue
        }
    }
    return out.isEmpty ? nil : out
}

public func createLlamaConfigurationForBackbone(_ cfg: SesameModelArgs) -> LlamaConfiguration {
    LlamaConfiguration(
        hiddenSize: cfg.hiddenSize,
        hiddenLayers: cfg.numHiddenLayers,
        intermediateSize: cfg.intermediateSize,
        attentionHeads: cfg.numAttentionHeads,
        headDimensions: cfg.headDim,
        rmsNormEps: Float(cfg.rmsNormEps),
        vocabularySize: cfg.textVocabSize,
        kvHeads: cfg.numKeyValueHeads,
        maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        ropeTheta: Float(cfg.ropeTheta),
        ropeTraditional: false,
        ropeScaling: toStringOrNumber(cfg.ropeScaling),
        tieWordEmbeddings: cfg.tieWordEmbeddings,
        attentionBias: cfg.attentionBias,
        mlpBias: cfg.mlpBias
    )
}

public func createLlamaConfigurationForDecoder(_ d: DepthDecoderConfig) -> LlamaConfiguration {
    LlamaConfiguration(
        hiddenSize: d.hiddenSize,
        hiddenLayers: d.numHiddenLayers,
        intermediateSize: d.intermediateSize,
        attentionHeads: d.numAttentionHeads,
        headDimensions: d.headDim,
        rmsNormEps: Float(d.rmsNormEps),
        vocabularySize: d.vocabSize,
        kvHeads: d.numKeyValueHeads,
        maxPositionEmbeddings: d.maxPositionEmbeddings,
        ropeTheta: Float(d.ropeTheta),
        ropeTraditional: false,
        ropeScaling: toStringOrNumber(d.ropeScaling),
        tieWordEmbeddings: true,
        attentionBias: d.attentionBias,
        mlpBias: d.mlpBias
    )
}

public func createLlamaConfiguration(flavor: String) throws -> LlamaConfiguration {
    switch flavor {
    case "llama-1B":
        return LlamaConfiguration(
            hiddenSize: 2048,
            hiddenLayers: 16,
            intermediateSize: 8192,
            attentionHeads: 32,
            headDimensions: 64,
            rmsNormEps: 1e-5,
            vocabularySize: 128256,
            kvHeads: 8,
            maxPositionEmbeddings: 2048,
            ropeTheta: 500000,
            ropeTraditional: false,
            ropeScaling: [
                "factor": .float(32.0),
                "low_freq_factor": .float(1.0),
                "high_freq_factor": .float(4.0),
                "original_max_position_embeddings": .float(8192.0),
                "rope_type": .string("llama3"),
            ],
            tieWordEmbeddings: true,
            attentionBias: false,
            mlpBias: false
        )

    case "llama-100M":
        return LlamaConfiguration(
            hiddenSize: 1024,
            hiddenLayers: 4,
            intermediateSize: 8192,
            attentionHeads: 8,
            headDimensions: 128,
            rmsNormEps: 1e-5,
            vocabularySize: 128256,
            kvHeads: 2,
            maxPositionEmbeddings: 2048,
            ropeTheta: 500000,
            ropeTraditional: false,
            ropeScaling: [
                "factor": .float(32.0),
                "low_freq_factor": .float(1.0),
                "high_freq_factor": .float(4.0),
                "original_max_position_embeddings": .float(8192.0),
                "rope_type": .string("llama3"),
            ],
            tieWordEmbeddings: true,
            attentionBias: false,
            mlpBias: false
        )

    default:
        struct UnknownFlavor: Error {}
        throw UnknownFlavor()
    }
}

// MARK: - Model

public final class SesameModel: Module {
    public let args: SesameModelArgs

    @ModuleInfo public var backbone: LlamaModel
    @ModuleInfo public var decoder: LlamaModel

    @ModuleInfo public var text_embeddings: Embedding
    @ModuleInfo public var audio_embeddings: Embedding
    @ModuleInfo public var projection: Linear // backbone_dim -> decoder_dim
    @ModuleInfo public var codebook0_head: Linear // logits for codebook 0
    public var audio_head: MLXArray // [nq-1, decoder_dim, audio_vocab]

    private var backboneCausalMask: MLXArray? = nil
    private var decoderCausalMask: MLXArray? = nil

    public var backboneCache: [KVCache]? = nil
    public var decoderCache: [KVCache]? = nil
    public var cachesEnabled: Bool = false

    public init(config: SesameModelArgs) {
        self.args = config

        let backCfg: LlamaConfiguration
        let decCfg: LlamaConfiguration
        if let depth = config.depthDecoderConfig {
            backCfg = createLlamaConfigurationForBackbone(config)
            decCfg = createLlamaConfigurationForDecoder(depth)
        } else {
            backCfg = try! createLlamaConfiguration(flavor: config.backboneFlavor)
            decCfg = try! createLlamaConfiguration(flavor: config.decoderFlavor)
        }

        self._backbone = ModuleInfo(wrappedValue: LlamaModel(backCfg))
        self._decoder = ModuleInfo(wrappedValue: LlamaModel(decCfg))

        let backboneDim = backCfg.hiddenSize
        let decoderDim = decCfg.hiddenSize

        self._text_embeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: args.textVocabSize, dimensions: backboneDim))
        let audioVocabCombined = args.audioVocabSize * args.audioNumCodebooks
        self._audio_embeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: audioVocabCombined, dimensions: backboneDim))

        self._projection = ModuleInfo(wrappedValue: Linear(backboneDim, decoderDim, bias: false))
        self._codebook0_head = ModuleInfo(wrappedValue: Linear(backboneDim, args.audioVocabSize, bias: false))

        let restCodebooks = max(args.audioNumCodebooks - 1, 0)
        self.audio_head = MLXArray.zeros([restCodebooks, decoderDim, args.audioVocabSize])

        self.backboneCache = nil
        self.decoderCache = nil
        self.cachesEnabled = false
    }

    public func cachesAreEnabled() -> Bool { cachesEnabled }

    public func resetCaches() {
        let backCfg: LlamaConfiguration
        let decCfg: LlamaConfiguration
        
        if let depth = args.depthDecoderConfig {
            backCfg = createLlamaConfigurationForBackbone(args)
            decCfg = createLlamaConfigurationForDecoder(depth)
        } else {
            backCfg = try! createLlamaConfiguration(flavor: args.backboneFlavor)
            decCfg = try! createLlamaConfiguration(flavor: args.decoderFlavor)
        }
        
        backboneCache = (0..<backCfg.hiddenLayers).map { _ in KVCache(headDim: backCfg.resolvedHeadDimensions, nKVHeads: backCfg.kvHeads) }
        decoderCache = (0..<decCfg.hiddenLayers).map { _ in KVCache(headDim: decCfg.resolvedHeadDimensions, nKVHeads: decCfg.kvHeads) }
        cachesEnabled = true
    }

    public func generateFrame(
        tokens: MLXArray,
        tokensMask: MLXArray,
        inputPos: MLXArray,
        sampler: (MLXArray) -> MLXArray
    ) -> MLXArray {
        precondition(cachesEnabled, "backbone caches are not enabled")

        let embeds = _embedTokens(tokens) // [B, T, Cb+1, D]
        let masked = embeds * tokensMask.expandedDimensions(axis: -1) // [B, T, Cb+1, D]
        var h = sum(masked, axis: 2) // [B, T, D]

        h = backbone(h, cache: backboneCache) // [B, T, D]

        let B = h.shape[0]
        let D_backbone = h.shape[2]
        let lastT = h.shape[1] - 1
        let split1 = split(h, indices: [lastT], axis: 1)
        let lastSlice = split(split1[1], indices: [1], axis: 1)[0] // [B, 1, D]
        let lastH = lastSlice.reshaped([B, D_backbone]) // [B, D]

        let c0Logits = codebook0_head(lastH) // [B, vocab_audio]
        let c0SampleVec = sampler(c0Logits) // [B]
        let c0Sample = c0SampleVec.expandedDimensions(axis: -1) // [B, 1]
        let c0Embed = _embedAudio(codebook: 0, tokens: c0Sample) // [B, 1, D_backbone]

        let lastH3 = expandedDimensions(lastH, axis: 1) // [B, 1, D_backbone]
        var currH = concatenated([lastH3, c0Embed], axis: 1) // [B, 2, D_backbone]
        var currSample = c0Sample // [B, 1]

        let basePos = MLXArray.arange(2).reshaped([1, 2])
        var currPos = repeated(basePos, count: B, axis: 0) // [B, 2]

        let decCfg: LlamaConfiguration
        if let depth = args.depthDecoderConfig {
            decCfg = createLlamaConfigurationForDecoder(depth)
        } else {
            decCfg = try! createLlamaConfiguration(flavor: args.decoderFlavor)
        }
        decoderCache = (0..<decCfg.hiddenLayers).map { _ in KVCache(headDim: decCfg.resolvedHeadDimensions, nKVHeads: decCfg.kvHeads) }

        let Cb = args.audioNumCodebooks
        if Cb > 1 {
            for i in 1 ..< Cb {
                let decH = decoder(projection(currH), cache: decoderCache) // [B, Tcur, D_dec]

                let D_dec = decH.shape[2]
                let lastSplit1 = split(decH, indices: [decH.shape[1] - 1], axis: 1)
                let lastDec = split(lastSplit1[1], indices: [1], axis: 1)[0].reshaped([B, D_dec]) // [B, D_dec]

                let Wi = take2DHead(audio_head, index: i - 1)
                let ciLogits = matmul(lastDec, Wi) // [B, vocab_audio]

                let ciSampleVec = sampler(ciLogits) // [B]
                let ciSample = expandedDimensions(ciSampleVec, axis: -1) // [B, 1]
                let ciEmbed = _embedAudio(codebook: i, tokens: ciSample) // [B, 1, D_backbone]

                currH = ciEmbed // [B, 1, D_backbone]
                currSample = concatenated([currSample, ciSample], axis: 1)
                currPos = split(currPos, indices: [1], axis: 1)[1] + MLXArray(1)
            }
        }

        return currSample // [B, Cb]
    }

    private func _embedAudio(codebook: Int, tokens: MLXArray) -> MLXArray {
        let offset = codebook * args.audioVocabSize
        let shifted = tokens + MLXArray(offset)
        return audio_embeddings(shifted)
    }

    private func _embedTokens(_ tokens: MLXArray) -> MLXArray {
        let B = tokens.shape[0]
        let T = tokens.shape[1]
        let CbPlus = tokens.shape[2]
        let Cb = CbPlus - 1

        let split1 = split(tokens, indices: [Cb], axis: 2)
        let audioIds = split1[0] // [B, T, Cb]
        let textIds = split(split1[1], indices: [1], axis: 2)[0].reshaped([B, T]) // [B, T]

        var textEmb = text_embeddings(textIds) // [B, T, D]
        textEmb = expandedDimensions(textEmb, axis: -2) // [B, T, 1, D]

        let cbIdx = MLXArray.arange(Cb) // [Cb]
        let cbOffsets = (cbIdx * MLXArray(Int32(args.audioVocabSize))).reshaped([1, 1, Cb])
        let shiftedAudioIds = audioIds + cbOffsets // [B, T, Cb]

        let flat = shiftedAudioIds.flattened() // [B*T*Cb]
        let audioFlatEmb = audio_embeddings(flat) // [B*T*Cb, D]
        let D = audioFlatEmb.shape[1]
        let audioEmb = audioFlatEmb.reshaped([B, T, Cb, D]) // [B, T, Cb, D]

        return concatenated([audioEmb, textEmb], axis: 2) // [B, T, Cb+1, D]
    }

    private func take2DHead(_ W: MLXArray, index i: Int) -> MLXArray {
        if W.ndim == 3 {
            let left = split(W, indices: [i], axis: 0)
            let tail = split(left[1], indices: [1], axis: 0)
            return tail[0].reshaped([W.shape[1], W.shape[2]])
        }
        return W
    }
}

public extension MLXArray {
    static func arange(_ size: Int) -> MLXArray {
        MLXArray(Array(0 ..< size))
    }
}

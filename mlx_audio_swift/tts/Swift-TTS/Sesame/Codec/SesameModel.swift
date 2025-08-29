//
// SesameModel for Sesame TTS
// Main dual-transformer model for text-to-audio conversion
// Based on Python mlx_audio/tts/models/sesame/sesame.py
//

import Foundation
import MLX
import MLXNN

/// SesameModel - Main dual-transformer model
/// Equivalent to Python's SesameModel class
class SesameModel: Module {
    @ModuleInfo var backbone: LlamaModel
    @ModuleInfo var decoder: LlamaModel
    @ModuleInfo var textEmbeddings: MLXNN.Embedding
    @ModuleInfo var audioEmbeddings: MLXNN.Embedding
    @ModuleInfo var projection: MLXNN.Linear
    @ModuleInfo var codebook0Head: MLXNN.Linear
    @ModuleInfo var audioHead: MLXArray

    private let args: LlamaModelArgs
    private var backboneCausalMask: MLXArray?
    private var decoderCausalMask: MLXArray?
    var backboneCache: [KVCacheProtocol]?
    var decoderCache: [KVCacheProtocol]?
    var cachesEnabled: Bool = false

    /// Initialize SesameModel
    /// - Parameter args: Model configuration arguments
    init(_ args: LlamaModelArgs) {
        self.args = args

        super.init()

        // Create backbone model (for text understanding)
        self._backbone.wrappedValue = createLlamaModel(args.createBackboneArgs())

        // Create decoder model (for audio generation)
        self._decoder.wrappedValue = createLlamaModel(args.createDecoderArgs())

        // Replace attention layers with SesameAttention
        replaceAttentionLayers(model: self._backbone.wrappedValue, args: args.createBackboneArgs())
        replaceAttentionLayers(model: self._decoder.wrappedValue, args: args.createDecoderArgs())

        // Initialize embeddings
        self._textEmbeddings.wrappedValue = MLXNN.Embedding(
            embeddingCount: args.textVocabSize,
            dimensions: args.hiddenSize
        )

        self._audioEmbeddings.wrappedValue = MLXNN.Embedding(
            embeddingCount: args.audioVocabSize * args.audioNumCodebooks,
            dimensions: args.hiddenSize
        )

        // Initialize projection layer
        self._projection.wrappedValue = MLXNN.Linear(
            args.hiddenSize,
            args.hiddenSize,
            bias: false
        )

        // Initialize codebook heads
        self._codebook0Head.wrappedValue = MLXNN.Linear(
            args.hiddenSize,
            args.audioVocabSize,
            bias: false
        )

        // Initialize audio head for remaining codebooks
        self._audioHead.wrappedValue = MLXArray.zeros([
            args.audioNumCodebooks - 1,
            args.hiddenSize,
            args.audioVocabSize
        ])
    }

    /// Setup KV caches for efficient generation
    /// - Parameter maxBatchSize: Maximum batch size for caching
    func setupCaches(maxBatchSize: Int = 1) {
        let backboneArgs = args.createBackboneArgs()
        _ = args.createDecoderArgs() // Decoder args not used in this function

        // Create causal masks
        self.backboneCausalMask = createCausalMask(seqLen: backboneArgs.maxPositionEmbeddings)
        self.decoderCausalMask = createCausalMask(seqLen: args.audioNumCodebooks)

        // Initialize caches
        self.backboneCache = makePromptCache(backbone)
        self.decoderCache = makePromptCache(decoder)
        self.cachesEnabled = true
    }

    /// Check if caches are enabled
    func cachesAreEnabled() -> Bool {
        return cachesEnabled
    }

    /// Reset all caches
    func resetCaches() {
        if backboneCache != nil {
            self.backboneCache = makePromptCache(backbone)
        }

        if decoderCache != nil {
            self.decoderCache = makePromptCache(decoder)
        }
    }

    /// Generate audio tokens from text tokens
    /// - Parameters:
    ///   - tokens: Text token sequence [batch, seq_len]
    ///   - tokensMask: Attention mask [batch, seq_len]
    ///   - inputPos: Position indices for incremental generation
    ///   - sampler: Sampling function for token selection
    /// - Returns: Generated audio tokens [batch, num_codebooks]
    func generateFrame(
        tokens: MLXArray,
        tokensMask: MLXArray,
        inputPos: MLXArray,
        sampler: (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard cachesAreEnabled() else {
            fatalError("Backbone caches are not enabled")
        }

        // Create backbone causal mask
        let currBackboneMask = indexCausalMask(
            mask: backboneCausalMask!,
            inputPos: inputPos
        )

        // Embed tokens
        let embeds = embedTokens(tokens)
        let maskedEmbeds = embeds * tokensMask.expandedDimensions(axis: -1)

        // Process through backbone
        var h = maskedEmbeds.sum(axis: 2)
        h = backbone(h, mask: currBackboneMask, cache: backboneCache).0

        // Get last hidden state
        let lastH = h[0..., -1, 0...]

        // Generate first codebook token
        let c0Logits = codebook0Head(lastH)
        let c0Sample = sampler(c0Logits).expandedDimensions(axis: -1)

        // Embed first codebook token
        let c0Embed = embedAudio(codebook: 0, tokens: c0Sample)
        var currH = MLX.concatenated([lastH.expandedDimensions(axis: 1), c0Embed], axis: 1)
        var currSample = c0Sample

        // Generate position indices for decoder
        let seqLen = currH.shape[1]
        var currPos = MLXArray(Array(0..<seqLen))
        currPos = currPos.expandedDimensions(axis: 0)
        currPos = MLX.broadcast(currPos, to: [currH.shape[0], currH.shape[1]])

        // Reset decoder cache for new frame
        self.decoderCache = makePromptCache(decoder)

        // Generate remaining codebook tokens
        for i in 1..<args.audioNumCodebooks {
            let currDecoderMask = indexCausalMask(
                mask: decoderCausalMask!,
                inputPos: currPos
            )

            // Process through decoder
            let decoderH = decoder(projection(currH), mask: currDecoderMask, cache: decoderCache).0

            // Generate next codebook token
            let ciLogits = MLX.matmul(decoderH[0..., -1, 0...], audioHead[i - 1])
            let ciSample = sampler(ciLogits).expandedDimensions(axis: -1)

            // Embed token and update state
            let ciEmbed = embedAudio(codebook: i, tokens: ciSample)
            currH = ciEmbed
            currSample = MLX.concatenated([currSample, ciSample], axis: 1)

            // Update position for next iteration
            currPos = currPos[0..., -1, 0...].expandedDimensions(axis: -1) + 1
        }

        return currSample
    }

    /// Embed text tokens
    /// - Parameter tokens: Text tokens [batch, seq_len]
    /// - Returns: Embedded tokens [batch, seq_len, num_codebooks + 1, hidden_size]
    private func embedTokens(_ tokens: MLXArray) -> MLXArray {
        let textEmbeds = textEmbeddings(tokens[0..., 0..., -1])
        let textEmbedsExpanded = textEmbeds.expandedDimensions(axis: -2)

        // Create audio token embeddings
        let codebookIndices = MLXArray(Array(0..<args.audioNumCodebooks))
        let codebookOffsets = codebookIndices * args.audioVocabSize

        let audioTokens = tokens[0..., 0..., 0..<(args.audioNumCodebooks)] +
                         codebookOffsets.expandedDimensions(axis: 1)

        let audioEmbedsFlat = audioEmbeddings(audioTokens.flattened())
        let audioEmbeds = audioEmbedsFlat.reshaped([
            tokens.shape[0],
            tokens.shape[1],
            args.audioNumCodebooks,
            -1
        ])

        return MLX.concatenated([audioEmbeds, textEmbedsExpanded], axis: -2)
    }

    /// Embed audio tokens for specific codebook
    /// - Parameters:
    ///   - codebook: Codebook index
    ///   - tokens: Audio tokens [batch, seq_len]
    /// - Returns: Embedded tokens [batch, seq_len, hidden_size]
    private func embedAudio(codebook: Int, tokens: MLXArray) -> MLXArray {
        let tokenIndices = tokens + codebook * args.audioVocabSize
        return audioEmbeddings(tokenIndices)
    }

    /// Create Llama model with custom attention layers
    private func createLlamaModel(_ args: LlamaModelArgs) -> LlamaModel {
        let model = LlamaModel(args)
        replaceAttentionLayers(model: model, args: args)
        return model
    }

    /// Replace standard attention with SesameAttention
    private func replaceAttentionLayers(model: LlamaModel, args: LlamaModelArgs) {
        // Replace attention layers in all transformer layers
        for layer in model.layers {
            // Create new attention layer and assign it
            let newAttention = SesameAttention(args: args)
            layer.selfAttention = newAttention
        }
    }

    /// Create causal mask for attention
    private func createCausalMask(seqLen: Int) -> MLXArray {
        let mask = MLX.tril(MLX.ones([seqLen, seqLen]))
        return mask
    }

    /// Index causal mask for specific positions
    private func indexCausalMask(mask: MLXArray, inputPos: MLXArray) -> MLXArray {
        let maskIndexed = mask[inputPos[0..., 0..., 0...], 0..., 0...]
        let seqLen = inputPos.shape[1]
        let maskIndexedResized = maskIndexed[0..., 0..., 0..<seqLen]

        // Reshape to (batch_size, 1, seq_len, seq_len) for broadcasting
        return maskIndexedResized.expandedDimensions(axis: 1)
    }

    /// Create prompt cache for model
    private func makePromptCache(_ model: LlamaModel) -> [KVCacheProtocol] {
        // Create KV caches for all transformer layers
        var caches: [KVCacheProtocol] = []

        for _ in model.layers {
            let headDim = model.args.hiddenSize / model.args.numAttentionHeads
            let nKvHeads = model.args.numKeyValueHeads ?? model.args.numAttentionHeads
            let cache = KVCache(headDim: headDim, nKvHeads: nKvHeads)
            caches.append(cache)
        }

        return caches
    }
}

/// LlamaModel for Sesame TTS
/// Full Llama implementation with proper attention layers
class LlamaModel: Module {
    let args: LlamaModelArgs
    var layers: [LlamaTransformerLayer]

    init(_ args: LlamaModelArgs) {
        self.args = args
        self.layers = []

        // Initialize transformer layers
        for _ in 0..<args.numHiddenLayers {
            layers.append(LlamaTransformerLayer(args))
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: [KVCacheProtocol]? = nil
    ) -> (MLXArray, [KVCacheProtocol]?) {
        var hidden = x
        var updatedCache = cache

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (newHidden, newCache) = layer(hidden, mask: mask, cache: layerCache)
            hidden = newHidden
            if var cacheArray = updatedCache, let cache = newCache {
                cacheArray[i] = cache
                updatedCache = cacheArray
            }
        }

        return (hidden, updatedCache)
    }
}

/// LlamaTransformerLayer for Sesame TTS
class LlamaTransformerLayer: Module {
    @ModuleInfo var selfAttention: SesameAttention!
    @ModuleInfo var mlp: MLP
    @ModuleInfo var inputNorm: MLXNN.LayerNorm
    @ModuleInfo var postNorm: MLXNN.LayerNorm

    init(_ args: LlamaModelArgs) {
        super.init()

        // Initialize components after calling super.init()
        self.selfAttention = SesameAttention(args: args)
        self._mlp.wrappedValue = MLP(args)
        self._inputNorm.wrappedValue = MLXNN.LayerNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        self._postNorm.wrappedValue = MLXNN.LayerNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCacheProtocol? = nil
    ) -> (MLXArray, KVCacheProtocol?) {
        // Self-attention with residual connection
        let normedX = inputNorm(x)
        let attnOut = selfAttention(normedX, mask: mask, cache: cache)
        let updatedCache = cache
        var residual = x + attnOut

        // MLP with residual connection
        let normedResidual = postNorm(residual)
        let mlpOut = mlp(normedResidual)
        residual = residual + mlpOut

        return (residual, updatedCache)
    }
}

/// MLP for transformer layers
class MLP: Module {
    @ModuleInfo var gateProj: MLXNN.Linear
    @ModuleInfo var upProj: MLXNN.Linear
    @ModuleInfo var downProj: MLXNN.Linear

    init(_ args: LlamaModelArgs) {
        let hiddenSize = args.hiddenSize
        let intermediateSize = args.intermediateSize

        self._gateProj.wrappedValue = MLXNN.Linear(hiddenSize, intermediateSize, bias: args.mlpBias ?? false)
        self._upProj.wrappedValue = MLXNN.Linear(hiddenSize, intermediateSize, bias: args.mlpBias ?? false)
        self._downProj.wrappedValue = MLXNN.Linear(intermediateSize, hiddenSize, bias: args.mlpBias ?? false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = MLXNN.silu(gateProj(x))
        let up = upProj(x)
        let fused = gate * up
        return downProj(fused)
    }
}

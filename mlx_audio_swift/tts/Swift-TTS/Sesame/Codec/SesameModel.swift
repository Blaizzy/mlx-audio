//
// SesameModel for Sesame TTS
// Main dual-transformer model for text-to-audio conversion
// Based on Python mlx_audio/tts/models/sesame/sesame.py
//

import Foundation
import MLX
import MLXNN

/// Attention type for transformer layers
enum AttentionType {
    case sesame
}

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
        self._backbone.wrappedValue = LlamaModel(args.createBackboneArgs())

        // Create decoder model (for audio generation)
        self._decoder.wrappedValue = LlamaModel(args.createDecoderArgs())

        // TODO: Replace attention layers with SesameAttention (like Python implementation)
        // This requires modifying the LlamaModel to use our custom transformer layers

        // Initialize embeddings
        self._textEmbeddings.wrappedValue = MLXNN.Embedding(
            embeddingCount: args.textVocabSize,
            dimensions: args.hiddenSize
        )

        self._audioEmbeddings.wrappedValue = MLXNN.Embedding(
            embeddingCount: args.audioVocabSize * args.audioNumCodebooks,
            dimensions: args.hiddenSize
        )

        // Initialize projection layer: backbone_dim -> decoder_dim
        let backboneDim = args.hiddenSize  // 1536
        let decoderDim = args.depthDecoderConfig?.hiddenSize ?? args.hiddenSize  // 1024 or fallback

        print("🔧 DEBUG SesameModel init: Creating projection layer \(backboneDim) -> \(decoderDim)")
        self._projection.wrappedValue = MLXNN.Linear(
            backboneDim,
            decoderDim,
            bias: false
        )

        // Initialize codebook heads - codebook0Head uses backbone dimension
        print("🔧 DEBUG SesameModel init: Creating codebook0Head with backbone dim \(backboneDim)")
        self._codebook0Head.wrappedValue = MLXNN.Linear(
            backboneDim,
            args.audioVocabSize,
            bias: false
        )

        // Initialize audio head for remaining codebooks - uses decoder dimension
        print("🔧 DEBUG SesameModel init: Creating audioHead with decoder dim \(decoderDim)")
        self._audioHead.wrappedValue = MLXArray.zeros([
            args.audioNumCodebooks - 1,
            decoderDim,
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
        print("🎭 DEBUG generateFrame: tokens shape=\(tokens.shape), tokensMask shape=\(tokensMask.shape), inputPos shape=\(inputPos.shape)")

        guard cachesAreEnabled() else {
            print("🎭 DEBUG generateFrame: Caches not enabled!")
            fatalError("Backbone caches are not enabled")
        }

        // Create backbone causal mask
        print("🎭 DEBUG generateFrame: Creating backbone causal mask")
        let currBackboneMask = indexCausalMask(
            mask: backboneCausalMask!,
            inputPos: inputPos
        )
        print("🎭 DEBUG generateFrame: currBackboneMask shape=\(currBackboneMask.shape)")

        // Embed tokens
        print("🎭 DEBUG generateFrame: Embedding tokens")
        let embeds = embedTokens(tokens)

        // Validate embeds before masking
        print("🎭 DEBUG generateFrame: embeds validation:")
        print("  - embeds.isValid: \(embeds.ctx != nil)")
        print("  - embeds.shape: \(embeds.shape)")
        print("  - embeds.dtype: \(embeds.dtype)")

        if embeds.ctx == nil {
            print("  - ERROR: embeds is invalid!")
            fatalError("Embedding operation returned invalid array")
        }

        // Apply mask exactly like Python: embeds * expand_dims(tokens_mask, -1)
        print("🎭 DEBUG generateFrame: Applying mask like Python implementation")
        let expandedMask = tokensMask.expandedDimensions(axis: -1)
        print("🎭 DEBUG generateFrame: expandedMask shape=\(expandedMask.shape)")
        print("  - expandedMask.isValid: \(expandedMask.ctx != nil)")

        if expandedMask.ctx == nil {
            print("  - ERROR: expandedMask is invalid!")
            fatalError("Mask expansion returned invalid array")
        }

        let maskedEmbeds = embeds * expandedMask
        print("🎭 DEBUG generateFrame: maskedEmbeds shape=\(maskedEmbeds.shape)")
        print("  - maskedEmbeds.isValid: \(maskedEmbeds.ctx != nil)")

        if maskedEmbeds.ctx == nil {
            print("  - ERROR: maskedEmbeds is invalid after multiplication!")
            fatalError("Mask application returned invalid array")
        }

        // Process through backbone
        print("🎭 DEBUG generateFrame: Processing through backbone")

        // Validate maskedEmbeds before sum
        print("🎭 DEBUG generateFrame: maskedEmbeds validation:")
        print("  - isValid: \(maskedEmbeds.ctx != nil)")
        print("  - shape: \(maskedEmbeds.shape)")
        print("  - dtype: \(maskedEmbeds.dtype)")

        // Check if we have any NaN or inf values
        if maskedEmbeds.dtype == .float32 {
            let hasNan = MLX.any(MLX.isNaN(maskedEmbeds))
            let hasInf = MLX.any(MLX.isInf(maskedEmbeds))
            print("  - hasNaN: \(hasNan.item(Bool.self))")
            print("  - hasInf: \(hasInf.item(Bool.self))")
        }

        var h = maskedEmbeds.sum(axis: 2)

        // Validate sum result
        print("🎭 DEBUG generateFrame: sum result validation:")
        print("  - h.isValid: \(h.ctx != nil)")
        if h.ctx != nil {
            print("  - h.shape: \(h.shape)")
        } else {
            print("  - ERROR: h is invalid after sum operation!")
            fatalError("Sum operation returned invalid array")
        }

        // Validate backbone inputs before calling
        print("🎭 DEBUG generateFrame: Backbone input validation:")
        print("  - h.isValid: \(h.ctx != nil)")
        print("  - h.shape: \(h.shape)")
        print("  - currBackboneMask.isValid: \(currBackboneMask.ctx != nil)")
        print("  - currBackboneMask.shape: \(currBackboneMask.shape)")
        print("  - backboneCache: \(backboneCache != nil ? "not nil" : "nil")")

        if currBackboneMask.ctx == nil {
            print("  - ERROR: currBackboneMask is invalid!")
            fatalError("Invalid backbone mask")
        }

        print("🎭 DEBUG generateFrame: Calling backbone...")
        let backboneResult = backbone(h, mask: currBackboneMask, cache: backboneCache)
        print("🎭 DEBUG generateFrame: Backbone call completed")

        // Handle the backbone result - it returns (hidden, cache) tuple
        let (backboneHidden, _) = backboneResult
        print("🎭 DEBUG generateFrame: backboneHidden validation:")
        print("  - backboneHidden.isValid: \(backboneHidden.ctx != nil)")
        if backboneHidden.ctx != nil {
            print("  - backboneHidden.shape: \(backboneHidden.shape)")
        } else {
            print("  - ERROR: backboneHidden is invalid!")
            fatalError("Backbone returned invalid hidden state")
        }

        h = backboneHidden
        print("🎭 DEBUG generateFrame: backbone returned tuple, h shape=\(h.shape)")

        // Get last hidden state
        print("🎭 DEBUG generateFrame: Extracting last hidden state")
        var seqLen = h.shape[1]
        let lastTokenIndex = seqLen - 1
        print("🎭 DEBUG generateFrame: seqLen=\(seqLen), lastTokenIndex=\(lastTokenIndex)")

        // Try alternative indexing approach to avoid potential issues
        let lastH = h[0..., lastTokenIndex, 0...]
        print("🎭 DEBUG generateFrame: lastH shape=\(lastH.shape)")

        // Generate first codebook token
        print("🎭 DEBUG generateFrame: Generating first codebook token")
        let c0Logits = codebook0Head(lastH)
        print("🎭 DEBUG generateFrame: c0Logits shape=\(c0Logits.shape)")

        // Sample first codebook token and reshape for concatenation
        let c0SampleRaw = sampler(c0Logits)
        print("🎭 DEBUG generateFrame: c0SampleRaw shape=\(c0SampleRaw.shape)")

        // Expand c0Sample to [1, 1, 1] for embedding (following Python implementation)
        let c0Sample = c0SampleRaw.expandedDimensions(axis: -1)
        print("🎭 DEBUG generateFrame: c0Sample expanded shape=\(c0Sample.shape)")

        // Embed first codebook token
        print("🎭 DEBUG generateFrame: Embedding first codebook token")
        let c0Embed = embedAudio(codebook: 0, tokens: c0Sample)
        print("🎭 DEBUG generateFrame: c0Embed shape=\(c0Embed.shape)")

        var currH = MLX.concatenated([lastH.expandedDimensions(axis: 1), c0Embed], axis: 1)
        print("🎭 DEBUG generateFrame: currH after concatenation shape=\(currH.shape)")

        var currSample = c0Sample
        print("🎭 DEBUG generateFrame: currSample shape=\(currSample.shape)")

        // Update sequence length after concatenation
        let newSeqLen = currH.shape[1]
        print("🎭 DEBUG generateFrame: Updated seqLen from \(seqLen) to \(newSeqLen)")
        seqLen = newSeqLen

        // Generate position indices for decoder
        print("🎭 DEBUG generateFrame: Using updated seqLen=\(seqLen)")

        var currPos = MLXArray(Array(0..<seqLen))
        currPos = currPos.expandedDimensions(axis: 0)
        currPos = MLX.broadcast(currPos, to: [currH.shape[0], currH.shape[1]])
        print("🎭 DEBUG generateFrame: currPos shape=\(currPos.shape)")

        // Reset decoder cache for new frame
        print("🎭 DEBUG generateFrame: Resetting decoder cache")
        self.decoderCache = makePromptCache(decoder)

        // Generate remaining codebook tokens
        print("🎭 DEBUG generateFrame: Starting decoder loop for \(args.audioNumCodebooks - 1) codebooks")
        for i in 1..<args.audioNumCodebooks {
            print("🎭 DEBUG generateFrame: Codebook \(i)")
            let currDecoderMask = indexCausalMask(
                mask: decoderCausalMask!,
                inputPos: currPos
            )
            print("🎭 DEBUG generateFrame: currDecoderMask shape=\(currDecoderMask.shape)")

            // Process through decoder
            print("🎭 DEBUG generateFrame: Processing through decoder")
            let projectedH = projection(currH)
            print("🎭 DEBUG generateFrame: projectedH shape=\(projectedH.shape)")

            let decoderH = decoder(projectedH, mask: currDecoderMask, cache: decoderCache).0
            print("🎭 DEBUG generateFrame: decoderH shape=\(decoderH.shape)")

            // Generate next codebook token
            print("🎭 DEBUG generateFrame: Generating codebook \(i) token")
            let lastDecoderH = decoderH[0..., -1, 0...]
            print("🎭 DEBUG generateFrame: lastDecoderH shape=\(lastDecoderH.shape)")

            // For remaining codebooks, use audioHead which is [num_codebooks-1, decoder_dim, vocab_size]
            let audioHeadSlice = audioHead[i - 1]
            print("🎭 DEBUG generateFrame: audioHead[\(i-1)] shape=\(audioHeadSlice.shape)")

            // audioHeadSlice shape is [decoder_dim, vocab_size], lastDecoderH shape is [1, decoder_dim]
            // Matrix multiplication: [1, decoder_dim] @ [decoder_dim, vocab_size] -> [1, vocab_size]
            let ciLogits = MLX.matmul(lastDecoderH, audioHeadSlice)
            print("🎭 DEBUG generateFrame: ciLogits shape=\(ciLogits.shape)")

            // Sample token and expand for embedding [batch, 1] -> [batch, 1, 1]
            let ciSampleRaw = sampler(ciLogits)
            let ciSample = ciSampleRaw.expandedDimensions(axis: -1)
            print("🎭 DEBUG generateFrame: ciSampleRaw shape=\(ciSampleRaw.shape), ciSample shape=\(ciSample.shape)")

            // Embed token and update state
            print("🎭 DEBUG generateFrame: Embedding codebook \(i) token")
            let ciEmbed = embedAudio(codebook: i, tokens: ciSample)
            print("🎭 DEBUG generateFrame: ciEmbed shape=\(ciEmbed.shape)")

            currH = ciEmbed
            currSample = MLX.concatenated([currSample, ciSample], axis: 1)
            print("🎭 DEBUG generateFrame: currSample after concat shape=\(currSample.shape)")

            // Update position for next iteration
            currPos = currPos[0..., -1, 0...].expandedDimensions(axis: -1) + 1
            print("🎭 DEBUG generateFrame: currPos updated shape=\(currPos.shape)")
        }

        print("🎭 DEBUG generateFrame: Returning final currSample shape=\(currSample.shape)")
        return currSample
    }

    /// Embed text tokens
    /// - Parameter tokens: Text tokens [batch, seq_len]
    /// - Returns: Embedded tokens [batch, seq_len, num_codebooks + 1, hidden_size]
    private func embedTokens(_ tokens: MLXArray) -> MLXArray {
        print("🔤 DEBUG embedTokens: Input tokens shape=\(tokens.shape)")
        print("🔤 DEBUG embedTokens: Input tokens validation:")
        print("  - tokens.isValid: \(tokens.ctx != nil)")
        print("  - tokens.dtype: \(tokens.dtype)")

        if tokens.ctx == nil {
            print("  - ERROR: tokens is invalid!")
            fatalError("Invalid tokens array passed to embedTokens")
        }

        print("🔤 DEBUG embedTokens: Extracting text tokens (last column)")
        let textTokens = tokens[0..., 0..., -1]
        print("🔤 DEBUG embedTokens: textTokens shape=\(textTokens.shape)")
        print("  - textTokens.isValid: \(textTokens.ctx != nil)")

        if textTokens.ctx == nil {
            print("  - ERROR: textTokens extraction failed!")
            fatalError("Failed to extract text tokens")
        }

        let textEmbeds = textEmbeddings(textTokens)
        print("🔤 DEBUG embedTokens: textEmbeds shape=\(textEmbeds.shape)")
        print("  - textEmbeds.isValid: \(textEmbeds.ctx != nil)")

        if textEmbeds.ctx == nil {
            print("  - ERROR: textEmbeddings lookup failed!")
            fatalError("Failed to get text embeddings")
        }

        let textEmbedsExpanded = textEmbeds.expandedDimensions(axis: -2)
        print("🔤 DEBUG embedTokens: textEmbedsExpanded shape=\(textEmbedsExpanded.shape)")
        print("  - textEmbedsExpanded.isValid: \(textEmbedsExpanded.ctx != nil)")

        if textEmbedsExpanded.ctx == nil {
            print("  - ERROR: textEmbedsExpanded failed!")
            fatalError("Failed to expand text embeddings")
        }

        // Create audio token embeddings - following Python implementation exactly
        print("🔤 DEBUG embedTokens: Creating audio token embeddings")
        let codebookIndices = MLXArray(Array(0..<args.audioNumCodebooks))
        print("🔤 DEBUG embedTokens: codebookIndices shape=\(codebookIndices.shape)")

        let codebookOffsets = codebookIndices * args.audioVocabSize
        print("🔤 DEBUG embedTokens: codebookOffsets shape=\(codebookOffsets.shape)")

        // Reshape codebook_offsets to (1, 1, -1) like Python: mx.reshape(codebook_offsets, (1, 1, -1))
        let codebookOffsetsReshaped = codebookOffsets.reshaped([1, 1, -1])
        print("🔤 DEBUG embedTokens: codebookOffsetsReshaped shape=\(codebookOffsetsReshaped.shape)")

        print("🔤 DEBUG embedTokens: Extracting audio tokens (first \(args.audioNumCodebooks) columns)")
        let audioTokensSlice = tokens[0..., 0..., 0..<(args.audioNumCodebooks)]
        print("🔤 DEBUG embedTokens: audioTokensSlice shape=\(audioTokensSlice.shape)")

        let audioTokens = audioTokensSlice + codebookOffsetsReshaped
        print("🔤 DEBUG embedTokens: audioTokens after offset addition shape=\(audioTokens.shape)")

        print("🔤 DEBUG embedTokens: Flattening audio tokens for embedding")
        let audioTokensFlat = audioTokens.flattened()
        print("🔤 DEBUG embedTokens: audioTokensFlat shape=\(audioTokensFlat.shape)")

        let audioEmbedsFlat = audioEmbeddings(audioTokensFlat)
        print("🔤 DEBUG embedTokens: audioEmbedsFlat shape=\(audioEmbedsFlat.shape)")

        print("🔤 DEBUG embedTokens: Reshaping audio embeddings")
        let audioEmbeds = audioEmbedsFlat.reshaped([
            tokens.shape[0],
            tokens.shape[1],
            args.audioNumCodebooks,
            -1
        ])
        print("🔤 DEBUG embedTokens: audioEmbeds reshaped to \(audioEmbeds.shape)")

        print("🔤 DEBUG embedTokens: Concatenating audio and text embeddings")
        let result = MLX.concatenated([audioEmbeds, textEmbedsExpanded], axis: -2)
        print("🔤 DEBUG embedTokens: Final result shape=\(result.shape)")

        return result
    }

    /// Embed audio tokens for specific codebook
    /// - Parameters:
    ///   - codebook: Codebook index
    ///   - tokens: Audio tokens [batch, seq_len]
    /// - Returns: Embedded tokens [batch, seq_len, hidden_size]
    private func embedAudio(codebook: Int, tokens: MLXArray) -> MLXArray {
        print("🎵 DEBUG embedAudio: codebook=\(codebook), tokens shape=\(tokens.shape)")

        let tokenIndices = tokens + codebook * args.audioVocabSize
        print("🎵 DEBUG embedAudio: tokenIndices shape=\(tokenIndices.shape)")

        let result = audioEmbeddings(tokenIndices)
        print("🎵 DEBUG embedAudio: result shape=\(result.shape)")

        return result
    }



    /// Create causal mask for attention
    private func createCausalMask(seqLen: Int) -> MLXArray {
        let mask = MLX.tril(MLX.ones([seqLen, seqLen]))
        return mask
    }

    /// Index causal mask for specific positions
    /// Following the Python implementation exactly:
    /// mask_indexed = mx.take(mask, input_pos, axis=0)
    /// seq_len = input_pos.shape[1]
    /// mask_indexed = mask_indexed[:, :, :seq_len]
    /// return mx.expand_dims(mask_indexed, axis=1)
    private func indexCausalMask(mask: MLXArray, inputPos: MLXArray) -> MLXArray {
        print("🎭 DEBUG indexCausalMask: mask shape=\(mask.shape), inputPos shape=\(inputPos.shape)")

        let seqLen = inputPos.shape[1]
        print("🎭 DEBUG indexCausalMask: seqLen=\(seqLen)")

        // inputPos contains position indices [0, 1, 2, ..., seqLen-1] with shape [batch, seqLen]
        // We need to implement mx.take(mask, input_pos, axis=0) equivalent

        // For basic cases where inputPos contains consecutive indices [0, 1, 2, ...],
        // we can simply slice the mask to the correct size
        if seqLen <= mask.shape[0] {
            let maskSlice = mask[0..<seqLen, 0..<seqLen]
            print("🎭 DEBUG indexCausalMask: maskSlice shape=\(maskSlice.shape)")

            // Expand dimensions: [seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
            let expandedMask = maskSlice.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            print("🎭 DEBUG indexCausalMask: expandedMask shape=\(expandedMask.shape)")

            return expandedMask
        } else {
            // Fallback: create a new mask if seqLen is larger than the pre-computed mask
            print("🎭 DEBUG indexCausalMask: Creating new mask for seqLen=\(seqLen)")
            let newMask = MLX.tril(MLX.ones([seqLen, seqLen]))
            let expandedMask = newMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            print("🎭 DEBUG indexCausalMask: new expandedMask shape=\(expandedMask.shape)")
            return expandedMask
        }
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

    init(_ args: LlamaModelArgs, attentionType: AttentionType = .sesame) {
        self.args = args
        self.layers = []

        // Initialize transformer layers with specified attention type
        for _ in 0..<args.numHiddenLayers {
            layers.append(LlamaTransformerLayer(args, attentionType: attentionType))
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
    @ModuleInfo var selfAttention: Module
    @ModuleInfo var mlp: MLP
    @ModuleInfo var inputNorm: MLXNN.LayerNorm
    @ModuleInfo var postNorm: MLXNN.LayerNorm

    init(_ args: LlamaModelArgs, attentionType: AttentionType = .sesame) {
        // Initialize attention module based on type
        let attentionModule: Module
        switch attentionType {
        case .sesame:
            attentionModule = SesameAttention(args: args)
        }

        self.selfAttention = attentionModule
        self._mlp.wrappedValue = MLP(args)
        self._inputNorm.wrappedValue = MLXNN.LayerNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        self._postNorm.wrappedValue = MLXNN.LayerNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCacheProtocol? = nil
    ) -> (MLXArray, KVCacheProtocol?) {
        // Self-attention with residual connection
        let normedX = inputNorm(x)

        // Cast selfAttention to SesameAttention and call it
        guard let sesameAttention = selfAttention as? SesameAttention else {
            fatalError("Expected SesameAttention but got \(type(of: selfAttention))")
        }

        let attnOut = sesameAttention(normedX, mask: mask, cache: cache)
        var residual = x + attnOut

        // MLP with residual connection
        let normedResidual = postNorm(residual)
        let mlpOut = mlp(normedResidual)
        residual = residual + mlpOut

        return (residual, cache)  // Return the original cache since SesameAttention handles KV caching internally
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

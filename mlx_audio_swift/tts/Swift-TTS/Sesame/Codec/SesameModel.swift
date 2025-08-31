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

        self._projection.wrappedValue = MLXNN.Linear(
            backboneDim,
            decoderDim,
            bias: false
        )

        // Initialize codebook heads - codebook0Head uses backbone dimension
        self._codebook0Head.wrappedValue = MLXNN.Linear(
            backboneDim,
            args.audioVocabSize,
            bias: false
        )

        // Initialize audio head for remaining codebooks - uses decoder dimension
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
        let decoderArgs = args.createDecoderArgs()

        // Create causal masks
        self.backboneCausalMask = createCausalMask(seqLen: backboneArgs.maxPositionEmbeddings)
        // Decoder mask should accommodate the maximum possible sequence length in decoder
        // which is 1 (last_h) + 1 (c0_embed) + remaining codebooks = audioNumCodebooks + 1
        self.decoderCausalMask = createCausalMask(seqLen: args.audioNumCodebooks + 1)

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
    ///   - tokens: Text token sequence [batch, seq_len, num_codebooks+1]
    ///   - tokensMask: Attention mask [batch, seq_len, num_codebooks+1]
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

        print("🎵 DEBUG generateFrame: Input shapes - tokens: \(tokens.shape), tokensMask: \(tokensMask.shape), inputPos: \(inputPos.shape)")

        // Create backbone causal mask - follow Python exactly
        // Python: curr_backbone_mask = index_causal_mask(self._backbone_causal_mask, input_pos)
        let currBackboneMask = indexCausalMask(
            mask: backboneCausalMask!,
            inputPos: inputPos
        )
        print("🎵 DEBUG generateFrame: currBackboneMask shape: \(currBackboneMask.shape)")

        // Embed tokens
        print("🎵 DEBUG generateFrame: About to embed tokens...")
        let embeds = embedTokens(tokens)
        print("🎵 DEBUG generateFrame: embeds shape: \(embeds.shape)")

        // Apply mask exactly like Python: embeds * expand_dims(tokens_mask, -1)
        let expandedMask = tokensMask.expandedDimensions(axis: -1)
        print("🎵 DEBUG generateFrame: expandedMask shape: \(expandedMask.shape)")
        let maskedEmbeds = embeds * expandedMask
        print("🎵 DEBUG generateFrame: maskedEmbeds shape: \(maskedEmbeds.shape)")

        // Process through backbone
        var h = maskedEmbeds.sum(axis: 2)
        print("🎵 DEBUG generateFrame: h after sum shape: \(h.shape)")

        let backboneResult = backbone(h, mask: currBackboneMask, cache: backboneCache)

        // Handle the backbone result - it returns (hidden, cache) tuple
        let (backboneHidden, _) = backboneResult
        h = backboneHidden
        print("🎵 DEBUG generateFrame: backbone hidden shape: \(h.shape)")

        // Get last hidden state - Python: last_h = h[:, -1, :]
        let lastH = h[0..., -1, 0...]
        print("🎵 DEBUG generateFrame: lastH shape: \(lastH.shape)")

        // Generate first codebook token - Python: c0_logits = self.codebook0_head(last_h)
        let c0Logits = codebook0Head(lastH)
        print("🎵 DEBUG generateFrame: c0Logits shape: \(c0Logits.shape)")

        // Sample first codebook token - Python: c0_sample = mx.expand_dims(sampler(c0_logits), axis=-1)
        let c0SampleFlat = sampler(c0Logits)  // [batch]
        let c0Sample = c0SampleFlat.expandedDimensions(axis: -1)  // [batch, 1]
        print("🎵 DEBUG generateFrame: c0Sample shape: \(c0Sample.shape)")

        // Embed first codebook token - Python: c0_embed = self._embed_audio(0, c0_sample)
        let c0Embed = embedAudio(codebook: 0, tokens: c0Sample)
        print("🎵 DEBUG generateFrame: c0Embed shape: \(c0Embed.shape)")

        // Python: curr_h = mx.concat([mx.expand_dims(last_h, 1), c0_embed], axis=1)
        var currH = MLX.concatenated([lastH.expandedDimensions(axis: 1), c0Embed], axis: 1)
        print("🎵 DEBUG generateFrame: currH initial shape: \(currH.shape)")

        // Initialize current sample with first codebook token (keep it as [batch] not [batch, 1])
        var currSample = c0SampleFlat  // [batch]

        // Python: curr_pos = mx.arange(curr_h.shape[1], dtype=mx.int32)
        //         curr_pos = mx.expand_dims(curr_pos, 0)  
        //         curr_pos = mx.broadcast_to(curr_pos, (curr_h.shape[0], curr_h.shape[1]))
        var currPos = MLXArray.arange(start: 0, stop: currH.shape[1], dtype: .int32)
            .expandedDimensions(axis: 0)  // [1, seq_len]
        currPos = MLX.broadcast(currPos, to: [currH.shape[0], currH.shape[1]])
        print("🎵 DEBUG generateFrame: currPos shape: \(currPos.shape)")

        // Reset decoder cache for new frame - Python: self.decoder_cache = make_prompt_cache(self.decoder)
        self.decoderCache = makePromptCache(decoder)

        // Generate remaining codebook tokens
        for i in 1..<args.audioNumCodebooks {
            print("🎵 DEBUG generateFrame: Processing codebook \(i)")
            
            // Python: curr_decoder_mask = index_causal_mask(self._decoder_causal_mask, curr_pos)
            let currDecoderMask = indexCausalMask(
                mask: decoderCausalMask!,
                inputPos: currPos
            )

            // Python: decoder_h = self.decoder(self.projection(curr_h), mask=curr_decoder_mask, cache=self.decoder_cache)
            let projectedH = projection(currH)
            print("🎵 DEBUG generateFrame: projectedH shape: \(projectedH.shape)")
            let decoderH = decoder(projectedH, mask: currDecoderMask, cache: decoderCache).0
            print("🎵 DEBUG generateFrame: decoderH shape: \(decoderH.shape)")

            // Python: ci_logits = mx.matmul(decoder_h[:, -1, :], self.audio_head[i - 1])
            let lastDecoderH = decoderH[0..., -1, 0...]
            print("🎵 DEBUG generateFrame: lastDecoderH shape: \(lastDecoderH.shape)")
            let audioHeadSlice = audioHead[i - 1]
            print("🎵 DEBUG generateFrame: audioHeadSlice shape: \(audioHeadSlice.shape)")
            let ciLogits = MLX.matmul(lastDecoderH, audioHeadSlice)

            // Python: ci_sample = mx.expand_dims(sampler(ci_logits), axis=-1)
            let ciSampleFlat = sampler(ciLogits)  // [batch]
            let ciSample = ciSampleFlat.expandedDimensions(axis: -1)  // [batch, 1]

            // Python: ci_embed = self._embed_audio(i, ci_sample)
            let ciEmbed = embedAudio(codebook: i, tokens: ciSample)

            // Python: curr_h = ci_embed (NOT concatenated!)
            currH = ciEmbed
            
            // Accumulate tokens - Python: curr_sample = mx.concat([curr_sample, ci_sample], axis=1)
            // Convert currSample to [batch, 1] if it's [batch], then concatenate ciSample [batch, 1]
            if currSample.ndim == 1 {
                currSample = currSample.expandedDimensions(axis: -1)  // [batch] -> [batch, 1]
            }
            currSample = MLX.concatenated([currSample, ciSample], axis: 1)  // [batch, i+1]
            
            // Python: curr_pos = curr_pos[:, -1:] + 1
            let lastIndex = currPos.shape[1] - 1
            let lastPos = currPos[0..., lastIndex..<currPos.shape[1]]  // Get [:, -1:]
            currPos = lastPos + 1
        }
        
        print("🎵 DEBUG generateFrame: Final currSample shape: \(currSample.shape)")
        return currSample  // Should be [batch, num_codebooks]
    }

    /// Embed text tokens
    /// - Parameter tokens: Text tokens [batch, seq_len]
    /// - Returns: Embedded tokens [batch, seq_len, num_codebooks + 1, hidden_size]
    private func embedTokens(_ tokens: MLXArray) -> MLXArray {
        print("🎵 DEBUG embedTokens: Input tokens shape: \(tokens.shape)")
        
        // Extract text tokens from the last column (column 32 for 33 total columns)
        let textTokens = tokens[0..., 0..., -1]
        print("🎵 DEBUG embedTokens: textTokens shape: \(textTokens.shape)")
        
        let textEmbeds = textEmbeddings(textTokens)
        print("🎵 DEBUG embedTokens: textEmbeds shape: \(textEmbeds.shape)")
        
        let textEmbedsExpanded = textEmbeds.expandedDimensions(axis: -2)
        print("🎵 DEBUG embedTokens: textEmbedsExpanded shape: \(textEmbedsExpanded.shape)")

        // Create audio token embeddings - following Python implementation exactly
        let codebookIndices = MLXArray(Array(0..<args.audioNumCodebooks))
        let codebookOffsets = codebookIndices * args.audioVocabSize

        // Reshape codebook_offsets to (1, 1, -1) like Python: mx.reshape(codebook_offsets, (1, 1, -1))
        let codebookOffsetsReshaped = codebookOffsets.reshaped([1, 1, -1])
        print("🎵 DEBUG embedTokens: codebookOffsetsReshaped shape: \(codebookOffsetsReshaped.shape)")

        let audioTokensSlice = tokens[0..., 0..., 0..<(args.audioNumCodebooks)]
        print("🎵 DEBUG embedTokens: audioTokensSlice shape: \(audioTokensSlice.shape)")
        
        let audioTokens = audioTokensSlice + codebookOffsetsReshaped
        print("🎵 DEBUG embedTokens: audioTokens shape: \(audioTokens.shape)")

        let audioTokensFlat = audioTokens.flattened()
        let audioEmbedsFlat = audioEmbeddings(audioTokensFlat)
        print("🎵 DEBUG embedTokens: audioEmbedsFlat shape: \(audioEmbedsFlat.shape)")

        let audioEmbeds = audioEmbedsFlat.reshaped([
            tokens.shape[0],
            tokens.shape[1],
            args.audioNumCodebooks,
            -1
        ])
        print("🎵 DEBUG embedTokens: audioEmbeds shape: \(audioEmbeds.shape)")

        let result = MLX.concatenated([audioEmbeds, textEmbedsExpanded], axis: -2)
        print("🎵 DEBUG embedTokens: Final result shape: \(result.shape)")

        return result
    }

    /// Embed audio tokens for specific codebook
    /// - Parameters:
    ///   - codebook: Codebook index
    ///   - tokens: Audio tokens
    /// - Returns: Embedded tokens
    private func embedAudio(codebook: Int, tokens: MLXArray) -> MLXArray {
        // Match Python implementation exactly: self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)
        let tokenIndices = tokens + codebook * args.audioVocabSize

        // For embedding lookup, we need to flatten the token indices, then reshape the result
        let originalShape = tokenIndices.shape
        let tokenIndicesFlat = tokenIndices.flattened()

        let embeddingsFlat = audioEmbeddings(tokenIndicesFlat)

        // Reshape back to original shape + embedding dimension
        let resultShape = originalShape + [args.hiddenSize]
        let result = embeddingsFlat.reshaped(resultShape)

        return result
    }



    /// Create causal mask for attention
    private func createCausalMask(seqLen: Int) -> MLXArray {
        // Python: return mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        let mask = MLX.tril(MLX.ones([seqLen, seqLen], dtype: .bool))
        return mask
    }

    /// Index causal mask for specific positions
    /// Following Python implementation exactly:
    /// def index_causal_mask(mask: mx.array, input_pos: mx.array) -> mx.array:
    ///     mask_indexed = mx.take(mask, input_pos, axis=0)
    ///     seq_len = input_pos.shape[1]
    ///     mask_indexed = mask_indexed[:, :, :seq_len]
    ///     return mx.expand_dims(mask_indexed, axis=1)
    private func indexCausalMask(mask: MLXArray, inputPos: MLXArray, attentionSeqLen: Int? = nil) -> MLXArray {
        // Python exact implementation:
        // mask_indexed = mx.take(mask, input_pos, axis=0)
        // seq_len = input_pos.shape[1]
        // mask_indexed = mask_indexed[:, :, :seq_len]
        // return mx.expand_dims(mask_indexed, axis=1)

        // Use mx.take equivalent to index the mask based on input positions
        let maskIndexed = mask.take(inputPos, axis: 0)

        // Get sequence length from input_pos
        let seqLen = inputPos.shape[1]

        // Slice to the correct sequence length: mask_indexed[:, :, :seq_len]
        // After mx.take, maskIndexed has shape (batch, seq_len, seq_len)
        let maskSliced = maskIndexed[0..., 0..., 0..<seqLen]

        // Add head dimension for broadcasting: mx.expand_dims(mask_indexed, axis=1)
        // This creates shape (batch, 1, seq_len, seq_len) which broadcasts to (batch, n_heads, seq_len, seq_len)
        let maskWithHeadDim = maskSliced.expandedDimensions(axis: 1)
        return maskWithHeadDim
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
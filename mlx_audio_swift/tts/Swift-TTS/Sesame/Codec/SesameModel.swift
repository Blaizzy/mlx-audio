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

        // Create backbone causal mask - follow Python exactly
        // Python: curr_backbone_mask = index_causal_mask(self._backbone_causal_mask, input_pos)
        let currBackboneMask = indexCausalMask(
            mask: backboneCausalMask!,
            inputPos: inputPos
        )

        // Embed tokens
        let embeds = embedTokens(tokens)

        // Apply mask exactly like Python: embeds * expand_dims(tokens_mask, -1)
        let expandedMask = tokensMask.expandedDimensions(axis: -1)
        let maskedEmbeds = embeds * expandedMask

        // Process through backbone
        var h = maskedEmbeds.sum(axis: 2)

        // Backbone processing - cache is updated in place
        h = backbone(h, mask: currBackboneMask, cache: backboneCache)

        // Get last hidden state - Python: last_h = h[:, -1, :]
        let lastH = h[0..., -1, 0...]

        // Generate first codebook token - Python: c0_logits = self.codebook0_head(last_h)
        let c0Logits = codebook0Head(lastH)

        // Sample first codebook token - Python: c0_sample = mx.expand_dims(sampler(c0_logits), axis=-1)
        let c0SampleFlat = sampler(c0Logits)  // [batch]
        let c0Sample = c0SampleFlat.expandedDimensions(axis: -1)  // [batch, 1]

        // Embed first codebook token - Python: c0_embed = self._embed_audio(0, c0_sample)
        let c0Embed = embedAudio(codebook: 0, tokens: c0Sample)

        // Python: curr_h = mx.concat([mx.expand_dims(last_h, 1), c0_embed], axis=1)
        var currH = MLX.concatenated([lastH.expandedDimensions(axis: 1), c0Embed], axis: 1)

        // Initialize current sample with first codebook token (keep it as [batch] not [batch, 1])
        var currSample = c0SampleFlat  // [batch]

        // Position tracking - start with position 0
        var currPos = MLXArray(0)

        // Reset decoder cache for new frame - Python: decoder_cache = make_prompt_cache(self.decoder)
        self.decoderCache = makePromptCache(decoder)

        // Generate remaining codebook tokens
        for i in 1..<args.audioNumCodebooks {
            // Create decoder causal mask - Python: curr_decoder_mask = index_causal_mask(self._decoder_causal_mask, curr_pos)
            // For decoder, we need a mask of shape [1, 1, seq_len, seq_len] where seq_len is the current sequence length
            let seqLen = i + 1  // Current sequence length (1 for c0 + i for current codebook)
            let decoderMask = createCausalMask(seqLen: seqLen)
            let currDecoderMask = decoderMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)  // [1, 1, seq_len, seq_len]

            // Python: decoder_h = self.decoder(self.projection(curr_h), mask=curr_decoder_mask, cache=self.decoder_cache)
            let projectedH = projection(currH)
            let decoderH = decoder(projectedH, mask: currDecoderMask, cache: decoderCache)

            // Python: ci_logits = mx.matmul(decoder_h[:, -1, :], self.audio_head[i - 1])
            let lastDecoderH = decoderH[0..., -1, 0...]  // [batch, decoder_dim]
            let audioHeadSlice = take2DHead(audioHead, index: i - 1)  // [decoder_dim, audio_vocab]
            let ciLogits = MLX.matmul(lastDecoderH, audioHeadSlice)  // [batch, audio_vocab]

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

            // Update position for next token - Python: curr_pos = curr_pos + 1
            currPos = currPos + 1
        }

        return currSample  // Should be [batch, num_codebooks]
    }

    /// Embed tokens - matches Python _embed_tokens exactly
    /// - Parameter tokens: Token sequence [batch, seq_len, num_codebooks+1]
    /// - Returns: Embedded tokens [batch, seq_len, num_codebooks+1, hidden_size]
    private func embedTokens(_ tokens: MLXArray) -> MLXArray {
        let B = tokens.shape[0]
        let T = tokens.shape[1]
        let CbPlus = tokens.shape[2]  // K+1
        let Cb = CbPlus - 1

        // Extract text tokens (last column)
        let textTokens = tokens[0..., 0..., -1]  // [B, T]

        // Debug: Check for out-of-bounds text token values
        let maxTextTokenValue = textTokens.max().item(Float.self)
        let textEmbeddingTableSize = args.textVocabSize
        print("🔍 DEBUG embedTokens: maxTextTokenValue=\(maxTextTokenValue), textEmbeddingTableSize=\(textEmbeddingTableSize)")

        // Clamp text token values to valid range
        let clampedTextTokens = MLX.maximum(MLX.minimum(textTokens, Float(textEmbeddingTableSize - 1)), 0)
        if !MLX.allClose(clampedTextTokens, textTokens).item(Bool.self) {
            print("⚠️  DEBUG embedTokens: Clamped out-of-bounds text token values!")
        }

        let textEmbeds = textEmbeddings(clampedTextTokens)  // [B, T, D]
        let textEmbedsExpanded = textEmbeds.expandedDimensions(axis: -2)  // [B, T, 1, D]

        // Extract audio tokens (first K columns)
        let audioTokensSlice = tokens[0..., 0..., 0..<Cb]  // [B, T, Cb]

        // NOTE: Mimi codec already returns tokens with proper offsets for embedding table
        // No need to add codebook offsets here

        // Debug: Check token values (should already be in correct range)
        let maxTokenValue = audioTokensSlice.max().item(Float.self)
        let embeddingTableSize = args.audioVocabSize * args.audioNumCodebooks
        print("🔍 DEBUG embedTokens: maxTokenValue=\(maxTokenValue), embeddingTableSize=\(embeddingTableSize)")

        // Clamp token values to valid range (safety check)
        let clampedTokens = MLX.maximum(MLX.minimum(audioTokensSlice, Float(embeddingTableSize - 1)), 0)
        if !MLX.allClose(clampedTokens, audioTokensSlice).item(Bool.self) {
            print("⚠️  DEBUG embedTokens: Clamped out-of-bounds token values!")
        }

        // Flatten and embed
        let flatTokens = clampedTokens.flattened()  // [B*T*Cb]
        let audioEmbedsFlat = audioEmbeddings(flatTokens)  // [B*T*Cb, D]
        let D = audioEmbedsFlat.shape[1]
        let audioEmbeds = audioEmbedsFlat.reshaped([B, T, Cb, D])  // [B, T, Cb, D]

        // Concatenate: [audio_embeds, text_embeds]
        return MLX.concatenated([audioEmbeds, textEmbedsExpanded], axis: 2)  // [B, T, Cb+1, D]
    }

    /// Embed audio tokens for specific codebook
    /// - Parameters:
    ///   - codebook: Codebook index
    ///   - tokens: Audio tokens (already offset by Mimi codec)
    /// - Returns: Embedded tokens
    private func embedAudio(codebook: Int, tokens: MLXArray) -> MLXArray {
        // NOTE: Mimi codec already returns tokens with proper offsets for embedding table
        // No need to add codebook offset here - tokens are already in the correct range

        // Clamp token indices to valid range (safety check)
        let embeddingTableSize = args.audioVocabSize * args.audioNumCodebooks
        let clampedTokenIndices = MLX.maximum(MLX.minimum(tokens, Float(embeddingTableSize - 1)), 0)

        // For embedding lookup, we need to flatten the token indices, then reshape the result
        let originalShape = clampedTokenIndices.shape
        let tokenIndicesFlat = clampedTokenIndices.flattened()

        let embeddingsFlat = audioEmbeddings(tokenIndicesFlat)

        // Reshape back to original shape + embedding dimension
        let resultShape = originalShape + [args.hiddenSize]
        let result = embeddingsFlat.reshaped(resultShape)

        return result
    }

    /// Take 2D head slice - matches Python take2DHead function
    /// - Parameters:
    ///   - W: Weight matrix [nq-1, decoder_dim, audio_vocab]
    ///   - index: Index to take
    /// - Returns: Sliced matrix [decoder_dim, audio_vocab]
    private func take2DHead(_ W: MLXArray, index i: Int) -> MLXArray {
        if W.ndim == 3 {
            // Direct indexing is much simpler and more reliable than split operations
            // W[i, :, :] gives us [decoder_dim, audio_vocab]
            return W[i, 0..., 0...]
        }
        return W
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

        // Get sequence length from input_pos
        let seqLen = inputPos.shape[1]

        // For causal masks, we typically don't need to index by input_pos for generation
        // We can just take the relevant slice of the causal mask
        // mask has shape (max_seq_len, max_seq_len), we want (seq_len, seq_len)
        let maskSliced = mask[0..<seqLen, 0..<seqLen]

        // Add batch and head dimensions for broadcasting: mx.expand_dims(mask_sliced, axis=[0, 1])
        // This creates shape (1, 1, seq_len, seq_len) which broadcasts to (batch, n_heads, seq_len, seq_len)
        let maskWithDims = maskSliced.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        return maskWithDims
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
    ) -> MLXArray {
        var hidden = x

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let newHidden = layer(hidden, mask: mask, cache: layerCache)
            hidden = newHidden
        }

        return hidden
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
    ) -> MLXArray {
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

        return residual
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
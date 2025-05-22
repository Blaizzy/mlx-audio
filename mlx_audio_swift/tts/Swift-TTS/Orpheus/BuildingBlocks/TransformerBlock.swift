//
//  TransformerBlock.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 21/05/2025.
//
import Foundation
import MLX
import MLXNN

class TransformerBlock {
    private let weights: [String: MLXArray]
    private let layerIndex: Int
    private let hiddenSize: Int
    private let intermediateSize: Int
    private let numAttentionHeads: Int
    private let numKeyValueHeads: Int
    private let numRepeats: Int
    private let headDim: Int
    private let rope: RoPE

    // Pre-transposed Weights
    private let q_proj_w_T: MLXArray
    private let k_proj_w_T: MLXArray
    private let v_proj_w_T: MLXArray
    private let o_proj_w_T: MLXArray
    private let gate_proj_w_T: MLXArray
    private let up_proj_w_T: MLXArray
    private let down_proj_w_T: MLXArray
    private let inputNormWeight: MLXArray
    private let postNormWeight: MLXArray
    
    init(weights: [String: MLXArray], layerIndex: Int = 0) {        
        self.weights = weights
        self.layerIndex = layerIndex
        self.hiddenSize = 3072
        self.intermediateSize = 8192 // Llama 3B config
        self.numAttentionHeads = 24 // Llama 3B config
        self.headDim = hiddenSize / numAttentionHeads // 128

        // Set numKeyValueHeads to 8 as specified in config.json
        self.numKeyValueHeads = 8
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            fatalError("numAttentionHeads (\(numAttentionHeads)) must be divisible by numKeyValueHeads (\(numKeyValueHeads))")
        }
        self.numRepeats = self.numAttentionHeads / self.numKeyValueHeads // 3
        
        // Initialize RoPE (dims = headDim)
        self.rope = RoPE(dims: self.headDim)
        
        // Initialize and pre-transpose weights for a small speed bump
        self.inputNormWeight = weights["model.layers.\(layerIndex).input_layernorm.weight"]!
        self.postNormWeight = weights["model.layers.\(layerIndex).post_attention_layernorm.weight"]!

        // Weights are assumed to be loaded in [in_features, out_features] format if "column-major"
        self.q_proj_w_T = weights["model.layers.\(layerIndex).self_attn.q_proj.weight"]!
        self.k_proj_w_T = weights["model.layers.\(layerIndex).self_attn.k_proj.weight"]!
        self.v_proj_w_T = weights["model.layers.\(layerIndex).self_attn.v_proj.weight"]!
        self.o_proj_w_T = weights["model.layers.\(layerIndex).self_attn.o_proj.weight"]!
        
        self.gate_proj_w_T = weights["model.layers.\(layerIndex).mlp.gate_proj.weight"]!
        self.up_proj_w_T = weights["model.layers.\(layerIndex).mlp.up_proj.weight"]!
        self.down_proj_w_T = weights["model.layers.\(layerIndex).mlp.down_proj.weight"]!
    }
    
    func call(_ x: MLXArray, mask: MLXArray? = nil, cache: Cache? = nil) -> (output: MLXArray, updatedCache: Cache?) {
        // Fast path: avoid repeated dictionary look-ups and debug IO.
        let B = x.shape[0] // Batch size
        let L = x.shape[1] // Current sequence length of input x

        // Input RMSNorm
        let normedX = MLX.rmsNorm(x, weight: self.inputNormWeight, eps: 1e-5)

        // --- Self attention ---
        let q_proj = TransformerBlock.linear(x: normedX, weight: q_proj_w_T)
        let k_proj = TransformerBlock.linear(x: normedX, weight: k_proj_w_T)
        let v_proj = TransformerBlock.linear(x: normedX, weight: v_proj_w_T)

        // Reshape and transpose for multi-head attention
        // queries: [B, numAttentionHeads, L, headDim]
        // keys:    [B, numKeyValueHeads, L, headDim]
        // values:  [B, numKeyValueHeads, L, headDim]
        var queries = q_proj.reshaped([B, L, numAttentionHeads, headDim]).transposed(0, 2, 1, 3)
        var keys = k_proj.reshaped([B, L, numKeyValueHeads, headDim]).transposed(0, 2, 1, 3)
        var values = v_proj.reshaped([B, L, numKeyValueHeads, headDim]).transposed(0, 2, 1, 3)
        
        var updatedLayerCache: Cache? = cache

        if let currentLayerCache = updatedLayerCache {
            // Use existing cache – incremental decoding path.
            // Cache exists: Apply RoPE to new Q, K using cache's current offset.
            queries = rope.call(queries, offset: currentLayerCache.offset)
            keys = rope.call(keys, offset: currentLayerCache.offset)

            // Update the cache: updateAndFetch appends newKeys, newValues and updates its own offset.
            // The K,V returned are the full concatenated history.
            let (fetchedKeys, fetchedValues) = currentLayerCache.updateAndFetch(newKeys: keys, newValues: values)
            keys = fetchedKeys     // Use combined history for attention
            values = fetchedValues // Use combined history for attention
            // updatedLayerCache (which is currentLayerCache) is now internally updated by updateAndFetch.
        } else {
            // No cache (first pass / no caching desired for this layer yet):
            // Apply RoPE without offset (implicitly offset 0).
            // `queries`, `keys`, `values` are for the initial prompt (L = L_prompt)
            queries = rope.call(queries)
            keys = rope.call(keys)
            // Create a new cache to store these initial keys and values.
            // The offset of this new cache is the sequence length of these initial keys.
            updatedLayerCache = Cache(keys: keys, values: values, offset: keys.shape[2])
        }
        
        let scale = 1.0 / sqrt(Float(headDim))
        
        // Scaled Dot-Product Attention
        // queries are [B, numAttentionHeads, L_new, headDim]
        // keys/values are [B, numKeyValueHeads, L_total_kv, headDim]
        // Mask should be [B, numAttentionHeads, L_new, L_total_kv] or broadcastable.
        let attnOutput = MLX.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)

        // Reshape back to [B, L, hiddenSize]
        let attnOutputReshaped = attnOutput.transposed(0, 2, 1, 3).reshaped([B, L, hiddenSize])

        // Output projection
        let attnProj = TransformerBlock.linear(x: attnOutputReshaped, weight: o_proj_w_T)
        
        // First residual connection
        // x is the original input to the block, with shape [B, L, hiddenSize]. L here is L_new.
        let h = x + attnProj

        // Post attention RMSNorm
        let normedH = MLX.rmsNorm(h, weight: self.postNormWeight, eps: 1e-5)
        
        // MLP
        let gate = TransformerBlock.linear(x: normedH, weight: gate_proj_w_T)
        let up = TransformerBlock.linear(x: normedH, weight: up_proj_w_T)
        let gateUp = MLXNN.silu(gate) * up
        let down = TransformerBlock.linear(x: gateUp, weight: down_proj_w_T)

        // Second residual connection
        let output = h + down // output has shape [B, L_new, hiddenSize]
        
        return (output, updatedLayerCache)
    }
    
    public static func linear(x: MLXArray, weight: MLXArray, bias: MLXArray? = nil) -> MLXArray {
        // Assume incoming `weight` is [outFeatures, inFeatures]. We need W^T once.
        // Cache transposed version so we pay the cost only on first use.
        struct StaticCache {
            static var map: [UInt: MLXArray] = [:]   // key: pointer hash of underlying storage
        }
        let key = UInt(bitPattern: ObjectIdentifier(weight))
        let transposed: MLXArray
        if let cached = StaticCache.map[key] {
            transposed = cached
        } else {
            transposed = weight.transposed(1, 0)
            StaticCache.map[key] = transposed
        }

        var output = MLX.matmul(x, transposed)

        // Add bias if present (broadcast along all but last dim).
        if let b = bias {
            output = output + b
        }
        return output
    }
}

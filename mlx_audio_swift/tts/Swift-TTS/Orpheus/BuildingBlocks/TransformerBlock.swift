//
//  TransformerBlock.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 21/05/2025.
//
import Foundation
import MLX
import MLXNN

// MARK: - Profiling Helper (if not already defined)
struct BlockProfiler {
    static var enabled: Bool = false // Set to true to enable block-level profiling
    
    static func time<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
        guard enabled else { return try block() }
        
        let start = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let end = CFAbsoluteTimeGetCurrent()
        let duration = (end - start) * 1000 // Convert to milliseconds
        print("    ⚡ [BLOCK] \(label): \(String(format: "%.2f", duration))ms")
        return result
    }
}

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
        let normedX = BlockProfiler.time("RMSNorm (input)") {
            MLX.rmsNorm(x, weight: self.inputNormWeight, eps: 1e-5)
        }

        // Self attention projections
        let (q_proj, k_proj, v_proj) = BlockProfiler.time("Attention projections (Q,K,V)") {
            let q = TransformerBlock.linear(x: normedX, weight: q_proj_w_T)
            let k = TransformerBlock.linear(x: normedX, weight: k_proj_w_T)
            let v = TransformerBlock.linear(x: normedX, weight: v_proj_w_T)
            return (q, k, v)
        }

        // Reshape and transpose for multi-head attention
        var (queries, keys, values) = BlockProfiler.time("Attention reshape/transpose") {
            let q = q_proj.reshaped([B, L, numAttentionHeads, headDim]).transposed(0, 2, 1, 3)
            let k = k_proj.reshaped([B, L, numKeyValueHeads, headDim]).transposed(0, 2, 1, 3)
            let v = v_proj.reshaped([B, L, numKeyValueHeads, headDim]).transposed(0, 2, 1, 3)
            return (q, k, v)
        }
        
        var updatedLayerCache: Cache? = cache

        if let currentLayerCache = updatedLayerCache {
            // Use existing cache – incremental decoding path.
            (queries, keys, values) = BlockProfiler.time("Cache update & RoPE") {
                // Apply RoPE to new Q, K using cache's current offset.
                let q_rope = rope.call(queries, offset: currentLayerCache.offset)
                let k_rope = rope.call(keys, offset: currentLayerCache.offset)

                // Update the cache: updateAndFetch appends newKeys, newValues and updates its own offset.
                let (fetchedKeys, fetchedValues) = currentLayerCache.updateAndFetch(newKeys: k_rope, newValues: values)
                return (q_rope, fetchedKeys, fetchedValues)
            }
        } else {
            // No cache (first pass / no caching desired for this layer yet):
            (queries, keys) = BlockProfiler.time("RoPE (no cache)") {
                let q_rope = rope.call(queries)
                let k_rope = rope.call(keys)
                return (q_rope, k_rope)
            }
            // Create a new cache to store these initial keys and values.
            updatedLayerCache = Cache(keys: keys, values: values, offset: keys.shape[2])
        }
        
        let scale = 1.0 / sqrt(Float(headDim))
        
        // Scaled Dot-Product Attention
        let attnOutput = BlockProfiler.time("Scaled dot-product attention") {
            MLX.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }

        // Reshape back to [B, L, hiddenSize]
        let attnOutputReshaped = BlockProfiler.time("Attention output reshape") {
            attnOutput.transposed(0, 2, 1, 3).reshaped([B, L, hiddenSize])
        }

        // Output projection
        let attnProj = BlockProfiler.time("Attention output projection") {
            TransformerBlock.linear(x: attnOutputReshaped, weight: o_proj_w_T)
        }
        
        // First residual connection
        let h = BlockProfiler.time("First residual connection") {
            x + attnProj
        }

        // Post attention RMSNorm
        let normedH = BlockProfiler.time("RMSNorm (post-attention)") {
            MLX.rmsNorm(h, weight: self.postNormWeight, eps: 1e-5)
        }
        
        // MLP
        let (gate, up) = BlockProfiler.time("MLP projections (gate, up)") {
            let g = TransformerBlock.linear(x: normedH, weight: gate_proj_w_T)
            let u = TransformerBlock.linear(x: normedH, weight: up_proj_w_T)
            return (g, u)
        }
        
        let gateUp = BlockProfiler.time("MLP activation (SiLU)") {
            MLXNN.silu(gate) * up
        }
        
        let down = BlockProfiler.time("MLP down projection") {
            TransformerBlock.linear(x: gateUp, weight: down_proj_w_T)
        }

        // Second residual connection
        let output = BlockProfiler.time("Second residual connection") {
            h + down
        }
        
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

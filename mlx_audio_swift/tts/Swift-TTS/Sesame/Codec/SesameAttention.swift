//
// SesameAttention for Sesame TTS
// Custom attention implementation with Llama3ScaledRoPE and GQA support
// Based on Python mlx_audio/tts/models/sesame/attention.py
//

import Foundation
import MLX
import MLXNN
import MLXFast

/// Llama3ScaledRoPE for advanced positional embeddings
/// Equivalent to Python's Llama3ScaledRoPE class
class Llama3ScaledRoPE: Module {
    @ParameterInfo var cache: MLXArray?
    private var theta: MLXArray
    private let dim: Int
    private let base: Float
    private let maxSeqLen: Int
    private let scaleFactor: Float
    private let lowFreqFactor: Int
    private let highFreqFactor: Int
    private let oldContextLen: Int
    private var isCacheBuilt: Bool = false

    init(
        dim: Int,
        maxSeqLen: Int = 2048,
        base: Float = 500000.0,
        scaleFactor: Float = 32.0,
        lowFreqFactor: Int = 1,
        highFreqFactor: Int = 4,
        oldContextLen: Int = 8192
    ) {
        self.dim = dim
        self.base = base
        self.maxSeqLen = maxSeqLen
        self.scaleFactor = scaleFactor
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
        self.oldContextLen = oldContextLen

        // Initialize theta (will be computed in ropeInit)
        self.theta = MLXArray.zeros([dim / 2])

        super.init()
        ropeInit()
    }

    private func ropeInit() {
        // Create frequency indices: 0, 2, 4, ... up to dim/2
        let indices = Array(stride(from: 0, to: dim, by: 2))[0..<(dim/2)]
        let indicesFloat = MLXArray(indices.map { Float($0) })
        let freqs = 1.0 / pow(base, indicesFloat / Float(dim))
        let theta = applyScaling(
            freqs: freqs,
            scaleFactor: scaleFactor,
            lowFreqFactor: lowFreqFactor,
            highFreqFactor: highFreqFactor,
            oldContextLen: oldContextLen
        )
        self.theta = theta
        buildRopeCache(maxSeqLen: maxSeqLen)
        isCacheBuilt = true
    }

    private func buildRopeCache(maxSeqLen: Int) {
        let seqIdx = MLXArray(Array(0..<maxSeqLen).map { Float($0) })
        let idxTheta = MLX.matmul(seqIdx.expandedDimensions(axis: 1), theta.expandedDimensions(axis: 0))
        let cosValues = MLX.cos(idxTheta)
        let sinValues = MLX.sin(idxTheta)
        // Stack to get shape [maxSeqLen, dim/2, 2] - matches Python implementation
        self._cache.wrappedValue = MLX.stacked([cosValues, sinValues], axis: -1)
    }

    private func applyScaling(
        freqs: MLXArray,
        scaleFactor: Float,
        lowFreqFactor: Int,
        highFreqFactor: Int,
        oldContextLen: Int
    ) -> MLXArray {
        let lowFreqWavelen = Float(oldContextLen) / Float(lowFreqFactor)
        let highFreqWavelen = Float(oldContextLen) / Float(highFreqFactor)

        var newFreqs: [Float] = []

        for i in 0..<freqs.count {
            let freq = freqs[i].item(Float.self)
            let wavelen = 2 * Float.pi / freq

            if wavelen < highFreqWavelen {
                newFreqs.append(freq)
            } else if wavelen > lowFreqWavelen {
                newFreqs.append(freq / scaleFactor)
            } else {
                let smooth = (Float(oldContextLen) / wavelen - Float(lowFreqFactor)) /
                           (Float(highFreqFactor) - Float(lowFreqFactor))
                newFreqs.append((1 - smooth) * freq / scaleFactor + smooth * freq)
            }
        }

        return MLXArray(newFreqs)
    }

    func callAsFunction(_ x: MLXArray, offset: Int?) -> MLXArray {
        guard isCacheBuilt else {
            fatalError("RoPE cache is not built. Please call ropeInit() first.")
        }

        let seqLen = x.shape[1]

        // Follow Python implementation exactly
        let ropeCache: MLXArray
        if let offset = offset {
            // Python: self._cache[None, offset : offset + seq_len]
            ropeCache = cache![offset..<(offset + seqLen), 0..., 0...].expandedDimensions(axis: 0)
        } else {
            // Python: self._cache[:seq_len]
            ropeCache = cache![0..<seqLen, 0..., 0...]
        }

        // Python: xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        let xShaped = x.asType(.float32).reshaped(x.shape[0], x.shape[1], x.shape[2], -1, 2)

        // Python: rope_cache = rope_cache.reshape(-1, xshaped.shape[1], 1, xshaped.shape[3], 2)
        // Dynamic reshape to match Python exactly
        let ropeCacheReshaped = ropeCache.reshaped(
            xShaped.shape[0],  // batch size (dynamic)
            xShaped.shape[1],  // seq_len
            1,                 // 1 for head broadcasting
            xShaped.shape[3],  // head_dim/2
            2                  // 2 for cos/sin
        )

        // Python RoPE computation - match exactly with proper broadcasting
        let xOut0 = xShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] -
                    xShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]
        let xOut1 = xShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] +
                    xShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]

        // Stack and reshape back to original shape
        let xOut = MLX.stacked([xOut0, xOut1], axis: -1)
        return xOut.reshaped(x.shape)
    }
}

/// Custom Attention with Llama3ScaledRoPE and GQA support
/// Equivalent to Python's Attention class
class SesameAttention: Module {
    @ModuleInfo var qProj: MLXNN.Linear
    @ModuleInfo var kProj: MLXNN.Linear
    @ModuleInfo var vProj: MLXNN.Linear
    @ModuleInfo var oProj: MLXNN.Linear
    @ModuleInfo var rope: Llama3ScaledRoPE?
    
    private let nHeads: Int
    private let nKvHeads: Int
    private let headDim: Int
    private let scale: Float
    
    init(args: LlamaModelArgs) {
        let dim = args.hiddenSize
        self.nHeads = args.numAttentionHeads
        self.nKvHeads = args.numKeyValueHeads ?? nHeads
        self.headDim = args.headDim ?? dim / nHeads
        self.scale = pow(Float(headDim), -0.5)
        
        let attentionBias = args.attentionBias ?? false
        
        self._qProj.wrappedValue = MLXNN.Linear(dim, nHeads * headDim, bias: attentionBias)
        self._kProj.wrappedValue = MLXNN.Linear(dim, nKvHeads * headDim, bias: attentionBias)
        self._vProj.wrappedValue = MLXNN.Linear(dim, nKvHeads * headDim, bias: attentionBias)
        self._oProj.wrappedValue = MLXNN.Linear(nHeads * headDim, dim, bias: attentionBias)
        
        if let ropeTheta = args.ropeTheta,
           let ropeScaling = args.ropeScaling {
            self._rope.wrappedValue = Llama3ScaledRoPE(
                dim: headDim,
                base: ropeTheta,
                scaleFactor: ropeScaling.factor ?? 1.0
            )
        }
        
        super.init()
    }
    
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCacheProtocol? = nil
    ) -> MLXArray {
        // Python: b, s_x, _ = x.shape
        let b = x.shape[0]
        let sX = x.shape[1]
        
        // Python: y = x
        let y = x
        let sY = y.shape[1]
        
        // Python: q = self.q_proj(x)
        var q = qProj(x)
        
        // Python: q_per_kv = self.n_heads // self.n_kv_heads
        let qPerKv = nHeads / nKvHeads
        
        // Python: q = q.reshape(b, s_x, self.n_kv_heads * q_per_kv, self.head_dim)
        q = q.reshaped([b, sX, nKvHeads * qPerKv, headDim])
        
        // Python: if self.rope is not None: q = self.rope(q, offset=cache.offset if cache else 0)
        if let rope = rope {
            q = rope(q, offset: cache?.offset ?? 0)
        }
        
        // Python: q = q.swapaxes(1, 2)
        q = q.swappedAxes(1, 2)
        
        // Python: k = self.k_proj(y), v = self.v_proj(y)
        var k = kProj(y)
        var v = vProj(y)
        
        // Python: k = k.reshape(b, s_y, -1, self.head_dim), v = v.reshape(b, s_y, -1, self.head_dim)
        k = k.reshaped([b, sY, -1, headDim])
        v = v.reshaped([b, sY, -1, headDim])
        
        // Python: if self.rope is not None: k = self.rope(k, offset=cache.offset if cache else 0)
        if let rope = rope {
            k = rope(k, offset: cache?.offset ?? 0)
        }
        
        // Python: k = k.swapaxes(1, 2), v = v.swapaxes(1, 2)
        k = k.swappedAxes(1, 2)
        v = v.swappedAxes(1, 2)
        
        // Python: if cache: k, v = cache.update_and_fetch(k, v)
        if let cache = cache {
            (k, v) = cache.updateAndFetch(keys: k, values: v)
        }
        
        // Handle GQA (Grouped Query Attention)
        // Calculate actual head counts from tensor shapes, not config
        let actualQHeads = q.shape[1]  // Number of query heads
        let actualKvHeads = k.shape[1]  // Number of KV heads
        
        var finalK = k
        var finalV = v
        
        if actualQHeads != actualKvHeads {
            let qPerKv = actualQHeads / actualKvHeads
            
            print("🔍 DEBUG SesameAttention: GQA expansion needed")
            print("  - actualQHeads: \(actualQHeads), actualKvHeads: \(actualKvHeads), qPerKv: \(qPerKv)")
            print("  - k.shape before expansion: \(k.shape)")
            print("  - v.shape before expansion: \(v.shape)")
            
            // Expand each KV head to match number of Q heads
            // k shape: [b, nKvHeads, seqLen, headDim]
            // Target: [b, nKvHeads, qPerKv, seqLen, headDim]
            let kExpandShape = [b, actualKvHeads, qPerKv, k.shape[2], k.shape[3]]
            let vExpandShape = [b, actualKvHeads, qPerKv, v.shape[2], v.shape[3]]
            
            // First expand dimensions to prepare for broadcasting
            finalK = k.expandedDimensions(axis: 2)  // [b, nKvHeads, 1, seqLen, headDim]
            finalV = v.expandedDimensions(axis: 2)  // [b, nKvHeads, 1, seqLen, headDim]
            
            // Broadcast to repeat each head qPerKv times
            finalK = MLX.broadcast(finalK, to: kExpandShape)  // [b, nKvHeads, qPerKv, seqLen, headDim]
            finalV = MLX.broadcast(finalV, to: vExpandShape)  // [b, nKvHeads, qPerKv, seqLen, headDim]
            
            // Reshape to final shape: [b, nHeads, seqLen, headDim]
            finalK = finalK.reshaped([b, actualKvHeads * qPerKv, k.shape[2], k.shape[3]])
            finalV = finalV.reshaped([b, actualKvHeads * qPerKv, v.shape[2], v.shape[3]])
            
            print("  - finalK.shape after expansion: \(finalK.shape)")
            print("  - finalV.shape after expansion: \(finalV.shape)")
        }
        
        // Scaled dot product attention
        let output = scaledDotProductAttention(
            queries: q,
            keys: finalK,
            values: finalV,
            scale: scale,
            mask: mask
        )
        
        let outputReshaped = output.swappedAxes(1, 2).reshaped([b, sX, -1])
        return oProj(outputReshaped)
    }
    
    private func scaledDotProductAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXArray?
    ) -> MLXArray {
        var scores = MLX.matmul(queries, keys.swappedAxes(-2, -1)) * scale

        if let mask = mask {
            // Handle different mask shapes for attention masking
            var attentionMask = mask

            // If mask has fewer heads than queries, broadcast it
            if mask.shape[1] == 1 && scores.shape[1] > 1 {
                // Broadcast mask from [batch, 1, query_seq, key_seq] to [batch, n_heads, query_seq, key_seq]
                let broadcastShape = [mask.shape[0], scores.shape[1], mask.shape[2], mask.shape[3]]
                attentionMask = MLX.broadcast(mask, to: broadcastShape)
            }

            // If the mask and scores have different sequence lengths, we need to slice the mask
            // This handles cases where mask might be [batch, n_heads, full_seq, full_seq]
            // but scores might be [batch, n_heads, 1, current_seq]
            if attentionMask.shape[2] > scores.shape[2] || attentionMask.shape[3] > scores.shape[3] {
                // Slice the mask to match scores dimensions
                let slicedMask = attentionMask[0..., 0..., 0..<scores.shape[2], 0..<scores.shape[3]]
                scores = scores + slicedMask
            } else {
                scores = scores + attentionMask
            }
        }

        let attentionWeights = MLX.softmax(scores, axis: -1)

        let output = MLX.matmul(attentionWeights, values)
        
        return output
    }
}

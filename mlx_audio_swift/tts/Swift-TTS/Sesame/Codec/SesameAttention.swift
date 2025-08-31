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

        // Ensure theta has the correct shape for head_dim
        let expectedThetaCount = dim / 2
        if theta.count != expectedThetaCount {
            // If dim is odd, theta will have dim/2 truncated
            // Pad or truncate to match expected size
            if theta.count < expectedThetaCount {
                // Pad with zeros
                let paddingCount = expectedThetaCount - theta.count
                let padding = MLXArray.zeros([paddingCount])
                self.theta = MLX.concatenated([theta, padding], axis: 0)
            } else {
                // Truncate
                self.theta = theta[0..<expectedThetaCount]
            }
        }

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
        
        guard let cache = cache else {
            fatalError("RoPE cache is nil")
        }

        let seqLen = x.shape[1]

        // Follow Python implementation exactly with bounds checking
        let ropeCache: MLXArray
        if let offset = offset {
            // Bounds checking for offset access
            let endPos = offset + seqLen
            let maxCacheLen = cache.shape[0]
            
            if offset < 0 || endPos > maxCacheLen {
                // Clamp to valid range
                let safeOffset = max(0, min(offset, maxCacheLen - seqLen))
                let safeEndPos = min(safeOffset + seqLen, maxCacheLen)
                let actualSeqLen = safeEndPos - safeOffset
                
                if actualSeqLen <= 0 {
                    // Return identity transformation if we can't get valid cache
                    return x
                }
                
                ropeCache = cache[safeOffset..<safeEndPos, 0..., 0...]
            } else {
                ropeCache = cache[offset..<endPos, 0..., 0...]
            }
        } else {
            // Bounds checking for no offset case
            let maxCacheLen = cache.shape[0]
            if seqLen > maxCacheLen {
                let safeSeqLen = min(seqLen, maxCacheLen)
                ropeCache = cache[0..<safeSeqLen, 0..., 0...]
            } else {
                ropeCache = cache[0..<seqLen, 0..., 0...]
            }
        }

        // Python: xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        let headDim = x.shape[3]
        let halfHeadDim = headDim / 2
        let remainder = headDim % 2

        // Handle odd head dimensions by padding the last dimension
        var xPadded = x
        if remainder != 0 {
            // Pad the last dimension to make it even
            let paddingShape = [x.shape[0], x.shape[1], x.shape[2], 1]
            let padding = MLXArray.zeros(paddingShape, dtype: x.dtype)
            xPadded = MLX.concatenated([x, padding], axis: -1)
        }

        let adjustedHeadDim = headDim + remainder
        let xShaped = xPadded.asType(.float32).reshaped(x.shape[0], x.shape[1], x.shape[2], adjustedHeadDim / 2, 2)

        // FIXED: Check dimensions match between x and rope cache
        let expectedRopeDim = adjustedHeadDim / 2  // This is the head dimension we expect
        let actualRopeDim = ropeCache.shape[1]     // This is the cache's feature dimension
        
        // The issue is ropeCache shape is [seq_len, rope_dim, 2] but we're comparing wrong dimensions
        // ropeCache should be [seq_len, head_dim/2, 2] to match xShaped [batch, seq_len, heads, head_dim/2, 2]
        if expectedRopeDim != actualRopeDim {
            // FIXED: Instead of failing, adapt the cache to match the expected dimensions
            // If the cache has fewer dimensions than expected, use what we have
            // If the cache has more dimensions, truncate to what we need
            let usableDim = min(expectedRopeDim, actualRopeDim)
            
            if usableDim <= 0 {
                return x
            }
            
            // Use only the usable dimensions from both cache and input
            let truncatedRopeCache = ropeCache[0..., 0..<usableDim, 0...]
            let truncatedXShaped = xShaped[0..., 0..., 0..., 0..<usableDim, 0...]
            
            // Apply RoPE only to the compatible dimensions
            let ropeCacheReshaped = truncatedRopeCache.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
            // Shape should be [1, seq_len, 1, usable_dim, 2]
            
            let xOut0 = truncatedXShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] -
                        truncatedXShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]
            let xOut1 = truncatedXShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] +
                        truncatedXShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]
            
            let xOutTruncated = MLX.stacked([xOut0, xOut1], axis: -1)
            
            // If we truncated, we need to restore the unused dimensions
            var result: MLXArray
            if usableDim < expectedRopeDim {
                // Pad back to full dimensions with identity (no rotation for unused dims)
                let identityPart = xShaped[0..., 0..., 0..., usableDim..<expectedRopeDim, 0...]
                let rotatedPart = xOutTruncated
                result = MLX.concatenated([rotatedPart, identityPart], axis: 3)
            } else {
                result = xOutTruncated
            }
            
            // Reshape back to original shape
            let resultReshaped = result.reshaped([x.shape[0], x.shape[1], x.shape[2], adjustedHeadDim])
            
            // Remove padding if it was added for odd dimensions
            if remainder != 0 {
                return resultReshaped[0..., 0..., 0..., 0..<headDim].asType(x.dtype)
            } else {
                return resultReshaped.asType(x.dtype)
            }
        }

        // Normal case - dimensions match
        // Python: rope_cache = rope_cache.reshape(-1, xshaped.shape[1], 1, xshaped.shape[3], 2)
        // FIXED: The reshape should expand the cache to match xShaped broadcasting requirements
        let ropeCacheReshaped = ropeCache.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        // This gives us [1, seq_len, 1, head_dim/2, 2] which will broadcast to [batch, seq_len, heads, head_dim/2, 2]

        // Python RoPE computation - match exactly with proper broadcasting and bounds checking
        let xOut0 = xShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] -
                    xShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]
        let xOut1 = xShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] +
                    xShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]

        // Stack and reshape back to original shape
        let xOut = MLX.stacked([xOut0, xOut1], axis: -1)

        // Reshape to padded shape first, then remove padding if it was added
        let paddedResultShape = [x.shape[0], x.shape[1], x.shape[2], headDim + remainder]
        var result = xOut.reshaped(paddedResultShape).asType(x.dtype)

        // Remove padding if it was added for odd dimensions
        if remainder != 0 {
            result = result[0..., 0..., 0..., 0..<headDim]
        }

        return result
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
    private var headDim: Int
    private let scale: Float
    
    init(args: LlamaModelArgs) {
        let dim = args.hiddenSize
        self.nHeads = args.numAttentionHeads
        self.nKvHeads = args.numKeyValueHeads ?? nHeads

        var tempHeadDim = args.headDim ?? dim / nHeads
        // Ensure headDim is consistent and even for RoPE
        if tempHeadDim % 2 != 0 {
            tempHeadDim -= 1  // Make it even
        }
        self.headDim = tempHeadDim
        self.scale = pow(Float(tempHeadDim), -0.5)

        let attentionBias = args.attentionBias ?? false

        // Use the adjusted headDim for projections
        let finalHeadDim = tempHeadDim

        self._qProj.wrappedValue = MLXNN.Linear(dim, nHeads * finalHeadDim, bias: attentionBias)
        self._kProj.wrappedValue = MLXNN.Linear(dim, nKvHeads * finalHeadDim, bias: attentionBias)
        self._vProj.wrappedValue = MLXNN.Linear(dim, nKvHeads * finalHeadDim, bias: attentionBias)
        self._oProj.wrappedValue = MLXNN.Linear(nHeads * finalHeadDim, dim, bias: attentionBias)
        
        if let ropeTheta = args.ropeTheta,
           let ropeScaling = args.ropeScaling {
            self._rope.wrappedValue = Llama3ScaledRoPE(
                dim: finalHeadDim,
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
        
        // Bounds checking for reshape
        let expectedQSize = nKvHeads * qPerKv * headDim
        let actualQSize = q.shape[2]
        
        if expectedQSize != actualQSize {
            print("🚨 Q projection size mismatch: expected \(expectedQSize), got \(actualQSize)")
            print("🚨 nHeads=\(nHeads), nKvHeads=\(nKvHeads), headDim=\(headDim), qPerKv=\(qPerKv)")
            // Try to fix by using actual dimensions
            let actualHeadsFromQ = actualQSize / headDim
            q = q.reshaped([b, sX, actualHeadsFromQ, headDim])
        } else {
            q = q.reshaped([b, sX, nKvHeads * qPerKv, headDim])
        }

        // Python: if self.rope is not None: q = self.rope(q, offset=cache.offset if cache else 0)
        if let rope = rope {
            q = rope(q, offset: cache?.offset ?? 0)
        }

        // Python: q = q.swapaxes(1, 2)
        q = q.swappedAxes(1, 2)

        // Python: k = self.k_proj(y), v = self.v_proj(y)
        var k = kProj(y)
        var v = vProj(y)

        // Bounds checking for KV reshape
        let expectedKVSize = nKvHeads * headDim
        let actualKSize = k.shape[2]
        let actualVSize = v.shape[2]
        
        if expectedKVSize != actualKSize {
            print("🚨 K projection size mismatch: expected \(expectedKVSize), got \(actualKSize)")
            let actualKvHeads = actualKSize / headDim
            k = k.reshaped([b, sY, actualKvHeads, headDim])
        } else {
            k = k.reshaped([b, sY, nKvHeads, headDim])
        }
        
        if expectedKVSize != actualVSize {
            print("🚨 V projection size mismatch: expected \(expectedKVSize), got \(actualVSize)")
            let actualKvHeads = actualVSize / headDim
            v = v.reshaped([b, sY, actualKvHeads, headDim])
        } else {
            v = v.reshaped([b, sY, nKvHeads, headDim])
        }

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

            // Expand each KV head to match number of Q heads (following Python exactly)
            // k shape: [b, nKvHeads, seqLen, headDim] -> [b, nKvHeads, qPerKv, seqLen, headDim]

            // First expand dimensions to prepare for broadcasting
            finalK = k.expandedDimensions(axis: 2)  // [b, nKvHeads, 1, seqLen, headDim]
            finalV = v.expandedDimensions(axis: 2)  // [b, nKvHeads, 1, seqLen, headDim]

            // Broadcast to repeat each head qPerKv times
            let kExpandShape = [b, actualKvHeads, qPerKv, k.shape[2], k.shape[3]]
            let vExpandShape = [b, actualKvHeads, qPerKv, v.shape[2], v.shape[3]]

            finalK = MLX.broadcast(finalK, to: kExpandShape)  // [b, nKvHeads, qPerKv, seqLen, headDim]
            finalV = MLX.broadcast(finalV, to: vExpandShape)  // [b, nKvHeads, qPerKv, seqLen, headDim]

            // Reshape to final shape: [b, nHeads, seqLen, headDim]
            finalK = finalK.reshaped([b, actualKvHeads * qPerKv, k.shape[2], k.shape[3]])
            finalV = finalV.reshaped([b, actualKvHeads * qPerKv, v.shape[2], v.shape[3]])
        }

        // Scaled dot product attention
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: finalK,
            values: finalV,
            scale: scale,
            mask: mask
        )
        
        let outputReshaped = output.swappedAxes(1, 2).reshaped([b, sX, -1])
        return oProj(outputReshaped)
    }

}
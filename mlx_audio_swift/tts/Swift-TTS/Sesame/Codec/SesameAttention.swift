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
        let ropeCache: MLXArray
        if let offset = offset {
            ropeCache = cache![0..., offset..<(offset + seqLen), 0..., 0..., 0...]
        } else {
            ropeCache = cache![0..<seqLen, 0..., 0..., 0..., 0...]
        }

        let xShaped = x.reshaped(x.shape[0], x.shape[1], -1, 2)
        let ropeCacheReshaped = ropeCache.reshaped(-1, xShaped.shape[1], 1, xShaped.shape[3], 2)

        let xOut0 = xShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] -
                   xShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]
        let xOut1 = xShaped[0..., 0..., 0..., 0..., 1] * ropeCacheReshaped[0..., 0..., 0..., 0..., 0] +
                   xShaped[0..., 0..., 0..., 0..., 0] * ropeCacheReshaped[0..., 0..., 0..., 0..., 1]

        let xOut = MLX.stacked([xOut0, xOut1], axis: -1)
        return xOut.flattened(end: -1)
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
                scaleFactor: ropeScaling["factor"] as? Float ?? 1.0
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCacheProtocol? = nil
    ) -> MLXArray {
        let (b, sX, _) = (x.shape[0], x.shape[1], x.shape[2])
        let y = x
        let sY = y.shape[1]

        // Q projection
        var q = qProj(x)
        let qPerKv = nHeads / nKvHeads
        q = q.reshaped([b, sX, nKvHeads * qPerKv, headDim])

        // Apply RoPE to queries
        if let rope = rope {
            q = rope(q, offset: cache?.offset)
        }

        q = q.swappedAxes(1, 2)

        // K and V projections
        var k = kProj(y)
        var v = vProj(y)

        k = k.reshaped([b, sY, -1, headDim])
        v = v.reshaped([b, sY, -1, headDim])

        // Apply RoPE to keys
        if let rope = rope {
            k = rope(k, offset: cache?.offset)
        }

        k = k.swappedAxes(1, 2)
        v = v.swappedAxes(1, 2)

        // Update cache if provided
        if let cache = cache {
            (k, v) = cache.updateAndFetch(keys: k, values: v)
        }

        // Handle GQA (Grouped Query Attention)
        var finalK = k
        var finalV = v

        if nHeads != nKvHeads {
            let qPerKv = nHeads / nKvHeads

            finalK = k.expandedDimensions(axis: 2)
            finalV = v.expandedDimensions(axis: 2)

            let kExpandShape = [b, nKvHeads, qPerKv] + Array(k.shape[3...])
            let vExpandShape = [b, nKvHeads, qPerKv] + Array(v.shape[3...])

            finalK = MLX.broadcast(finalK, to: kExpandShape)
            finalV = MLX.broadcast(finalV, to: vExpandShape)

            finalK = finalK.reshaped([b, nKvHeads * qPerKv] + Array(finalK.shape[3...]))
            finalV = finalV.reshaped([b, nKvHeads * qPerKv] + Array(finalV.shape[3...]))
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
            scores = scores + mask
        }

        let attentionWeights = MLX.softmax(scores, axis: -1)
        return MLX.matmul(attentionWeights, values)
    }
}

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - Config

public struct TransformerConfig {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let causal: Bool
    public let normFirst: Bool
    public let biasFF: Bool
    public let biasAttn: Bool
    public let layerScale: Float?
    public let positionalEmbedding: String
    public let useConvBlock: Bool
    public let crossAttention: Bool
    public let convKernelSize: Int
    public let useConvBias: Bool
    public let gating: Bool
    public let norm: String
    public let context: Int
    public let maxPeriod: Int
    public let maxSeqLen: Int
    public let kvRepeat: Int
    public let dimFeedforward: Int
    public let convLayout: Bool

    public init(
        dModel: Int,
        numHeads: Int,
        numLayers: Int,
        causal: Bool,
        normFirst: Bool,
        biasFF: Bool,
        biasAttn: Bool,
        layerScale: Float?,
        positionalEmbedding: String,
        useConvBlock: Bool,
        crossAttention: Bool,
        convKernelSize: Int,
        useConvBias: Bool,
        gating: Bool,
        norm: String,
        context: Int,
        maxPeriod: Int,
        maxSeqLen: Int,
        kvRepeat: Int,
        dimFeedforward: Int,
        convLayout: Bool
    ) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.causal = causal
        self.normFirst = normFirst
        self.biasFF = biasFF
        self.biasAttn = biasAttn
        self.layerScale = layerScale
        self.positionalEmbedding = positionalEmbedding
        self.useConvBlock = useConvBlock
        self.crossAttention = crossAttention
        self.convKernelSize = convKernelSize
        self.useConvBias = useConvBias
        self.gating = gating
        self.norm = norm
        self.context = context
        self.maxPeriod = maxPeriod
        self.maxSeqLen = maxSeqLen
        self.kvRepeat = kvRepeat
        self.dimFeedforward = dimFeedforward
        self.convLayout = convLayout
    }

    public var headDim: Int { dModel / numHeads }
}

// MARK: - Utilities

@inline(__always)
func geluApprox(_ x: MLXArray) -> MLXArray {
    // 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
    let c0 = MLXArray(0.7978845608028654) // sqrt(2/pi)
    let c1 = MLXArray(0.044715)
    let x3 = x * x * x
    return 0.5 * x * (1 + tanh(c0 * (x + c1 * x3)))
}

public final class Id: Module {
    override public init() {}
    public func callAsFunction(_ xs: MLXArray) -> MLXArray { xs }
}

public final class LayerScale: Module {
    public var scale: MLXArray
    public init(dim: Int) {
        self.scale = MLXArray.ones([dim])
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        xs * scale
    }
}

// MARK: - Attention

public protocol AttentionCache: KVCache {
    func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray)
}

public final class Attention: Module {
    private let cfg: TransformerConfig
    @ModuleInfo public var in_proj: Linear
    @ModuleInfo public var out_proj: Linear
    @ModuleInfo public var rope: RoPE?

    private let scale: Float

    public init(cfg: TransformerConfig) {
        self.cfg = cfg
        // Only kv_repeat == 1 supported (parity with your python)
        precondition(cfg.kvRepeat == 1, "only kv_repeat == 1 is supported")

        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * (cfg.dModel / cfg.numHeads) // => 3*dModel for kv_repeat=1
        self._in_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, outDim, bias: cfg.biasAttn))
        self._out_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn))
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())

        if cfg.positionalEmbedding == "rope" {
            self._rope = ModuleInfo(wrappedValue: RoPE(dimensions: cfg.headDim, traditional: true, base: Float(cfg.maxPeriod)))
        } else {
            self._rope = ModuleInfo(wrappedValue: nil)
        }
    }

    public func callAsFunction(
        _ xs: MLXArray, // [B, T, D]
        cache: any AttentionCache,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]
        let hd = xs.shape[2] // d_model

        let qkv = in_proj(xs).reshaped([b, t, 3, cfg.numHeads, cfg.headDim])

        var q = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 0, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)
        var k = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 1, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)
        var v = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 2, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)

        if let rope {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        }

        (k, v) = cache.updateAndFetch(k, v)

        let kLen = k.shape[2]
        let kTargetLen = t + min(cfg.context, kLen - t)
        if kTargetLen < kLen {
            let start = kLen - kTargetLen
            k = split(k, indices: [start], axis: 2)[1]
            v = split(v, indices: [start], axis: 2)[1]
        }

        var out = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = swappedAxes(out, 1, 2).reshaped([b, t, hd])
        return out_proj(out)
    }
}

// MARK: - MLP

public final class MlpGating: Module {
    @ModuleInfo public var linear_in: Linear
    @ModuleInfo public var linear_out: Linear

    public init(cfg: TransformerConfig) {
        var hidden = 2 * cfg.dimFeedforward / 3
        if cfg.dimFeedforward == 4 * cfg.dModel {
            hidden = 11 * cfg.dModel / 4
        }
        self._linear_in = ModuleInfo(wrappedValue: Linear(cfg.dModel, 2 * hidden, bias: cfg.biasFF))
        self._linear_out = ModuleInfo(wrappedValue: Linear(hidden, cfg.dModel, bias: cfg.biasFF))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]
        let doubled = linear_in(xs) // [B, T, 2*H]
        let hidden = doubled.shape[2] / 2
        let split2 = doubled.reshaped([b, t, 2, hidden])

        // split along axis=2 at 1 -> [B,T,1,H], [B,T,1,H]
        let parts = split(split2, indices: [1], axis: 2)
        let a = parts[0] // gate input
        let bpart = parts[1]

        // SiLU(a) * b -> [B,T,1,H] then reshape to [B,T,H]
        let gated = silu(a) * bpart
        let flat = gated.reshaped([b, t, hidden])

        return linear_out(flat)
    }
}

public final class MlpNoGating: Module {
    @ModuleInfo public var linear1: Linear
    @ModuleInfo public var linear2: Linear

    public init(cfg: TransformerConfig) {
        self._linear1 = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dimFeedforward, bias: cfg.biasFF))
        self._linear2 = ModuleInfo(wrappedValue: Linear(cfg.dimFeedforward, cfg.dModel, bias: cfg.biasFF))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        linear2(geluApprox(linear1(xs)))
    }
}

// MARK: - Transformer layer

public final class TransformerLayer: Module {
    @ModuleInfo public var gating: Module
    @ModuleInfo public var norm1: Module
    @ModuleInfo public var norm2: Module
    @ModuleInfo public var layer_scale_1: Module
    @ModuleInfo public var layer_scale_2: Module
    @ModuleInfo public var self_attn: Attention

    public init(cfg: TransformerConfig) {
        precondition(!cfg.useConvBlock, "conv-block is not supported")
        precondition(!cfg.crossAttention, "cross-attn is not supported")

        if cfg.gating {
            self._gating = ModuleInfo(wrappedValue: MlpGating(cfg: cfg))
        } else {
            self._gating = ModuleInfo(wrappedValue: MlpNoGating(cfg: cfg))
        }

        switch cfg.norm {
        case "layer_norm":
            self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))
            self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))
        case "rms_norm":
            self._norm1 = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.dModel, eps: 1e-8))
            self._norm2 = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.dModel, eps: 1e-8))
        default:
            fatalError("unsupported norm type \(cfg.norm)")
        }

        if let _ = cfg.layerScale {
            self._layer_scale_1 = ModuleInfo(wrappedValue: LayerScale(dim: cfg.dModel))
            self._layer_scale_2 = ModuleInfo(wrappedValue: LayerScale(dim: cfg.dModel))
        } else {
            self._layer_scale_1 = ModuleInfo(wrappedValue: Id())
            self._layer_scale_2 = ModuleInfo(wrappedValue: Id())
        }

        self._self_attn = ModuleInfo(wrappedValue: Attention(cfg: cfg))
    }

    public func callAsFunction(
        _ xs: MLXArray,
        cache: any AttentionCache
    ) -> MLXArray {
        var x = xs
        var n1 = (norm1 as! UnaryLayer)(x)
        n1 = self_attn(n1, cache: cache)
        x = x + (layer_scale_1 as! LayerScale)(n1)
        x = x + (layer_scale_2 as! LayerScale)((gating as! MlpNoGating)((norm2 as! LayerNorm)(x)))
        return x
    }
}

// MARK: - Transformer

public final class Transformer: Module {
    private let cfg: TransformerConfig
    @ModuleInfo public var layers: [TransformerLayer]

    public init(cfg: TransformerConfig) {
        self.cfg = cfg
        self._layers = ModuleInfo(wrappedValue: (0..<cfg.numLayers).map { _ in TransformerLayer(cfg: cfg) })
    }

    public func callAsFunction(
        _ xs: MLXArray,
        cache: [AttentionCache]
    ) -> MLXArray {
        var x = xs
        for (layer, c) in zip(layers, cache) {
            x = layer(x, cache: c)
        }
        return x
    }

    public func makeCache() -> [KVCache] {
        // Assume your KVCache init matches the python: (head_dim, n_kv_heads)
        let numKVHeads = cfg.numHeads / cfg.kvRepeat
        return (0..<cfg.numLayers).map { _ in KVCache(headDim: cfg.headDim, nKVHeads: numKVHeads) }
    }
}

// MARK: - ProjectedTransformer

public final class ProjectedTransformer: Module {
    private let convLayout: Bool
    @ModuleInfo public var transformer: Transformer
    @ModuleInfo public var input_proj: Linear?
    @ModuleInfo public var output_projs: [Linear?]

    public init(cfg: TransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout
        self._transformer = ModuleInfo(wrappedValue: Transformer(cfg: cfg))

        if inputDim == cfg.dModel {
            self._input_proj = ModuleInfo(wrappedValue: nil)
        } else {
            self._input_proj = ModuleInfo(wrappedValue: Linear(inputDim, cfg.dModel, bias: false))
        }

        var outs: [Linear?] = []
        for od in outputDims {
            if od == cfg.dModel {
                outs.append(nil)
            } else {
                outs.append(Linear(cfg.dModel, od, bias: false))
            }
        }
        self._output_projs = ModuleInfo(wrappedValue: outs)
    }

    public func callAsFunction(
        _ xsIn: MLXArray,
        cache: [AttentionCache]
    ) -> [MLXArray] {
        var xs = xsIn
        if convLayout { xs = swappedAxes(xs, 1, 2) } // [B,C,T] -> [B,T,C]

        if let ip = input_proj { xs = ip(xs) }

        xs = transformer(xs, cache: cache)

        if output_projs.compactMap({ $0 }).count == 0 {
            return [swappedAxes(xs, 1, 2)]
        } else {
            var outs: [MLXArray] = []
            for op in output_projs {
                guard let op else { continue }
                var out = op(xs)
                if convLayout { out = swappedAxes(out, 1, 2) } // back to [B,C,T] if needed
                outs.append(out)
            }
            return outs
        }
    }

    public func makeCache() -> [KVCache] { transformer.makeCache() }
}

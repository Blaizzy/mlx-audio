//
// TransformerBlocks for Sesame TTS ProjectedTransformer
// Core transformer components: Attention, MLP, and Layer blocks
// Based on the MLX Python implementation from Kyutai Labs

import Foundation
import MLX
import MLXNN
import MLXFast

/// Identity operation (no-op layer)
class Identity: Module {
    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return xs
    }
}

/// LayerScale for scaling transformer outputs
class LayerScale: Module {
    @ParameterInfo var scale: MLXArray

    init(dim: Int) {
        self._scale.wrappedValue = MLXArray.ones([dim])
        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return xs * scale
    }
}

/// Layer cache that wraps KV cache (matches Python LayerCache)
class LayerCache {
    var selfAttn: KVCacheProtocol
    var crossAttn: (MLXArray, MLXArray)?

    init(selfAttn: KVCacheProtocol) {
        self.selfAttn = selfAttn
        self.crossAttn = nil
    }

    func reset() {
        selfAttn.reset()
        crossAttn = nil
    }
}

/// Attention layer matching MLX Python implementation
class Attention: Module {
    @ModuleInfo var inProj: MLXNN.Linear
    @ModuleInfo var outProj: MLXNN.Linear
    @ModuleInfo var rope: MLXNN.RoPE?

    private let config: TransformerConfig
    private let scale: Float

    init(_ config: TransformerConfig) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)

        let numKvHeads = config.numHeads / config.kvRepeat
        let outDim = config.dModel + 2 * numKvHeads * config.dModel / config.numHeads

        self._inProj.wrappedValue = MLXNN.Linear(config.dModel, outDim, bias: config.biasAttn)
        self._outProj.wrappedValue = MLXNN.Linear(config.dModel, config.dModel, bias: config.biasAttn)

        if config.positionalEmbedding == "rope" {
            self._rope.wrappedValue = MLXNN.RoPE(
                dimensions: config.headDim,
                traditional: true,
                base: Float(config.maxPeriod)
            )
        } else {
            self._rope.wrappedValue = nil
        }

        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: LayerCache, mask: MLXArray? = nil) -> MLXArray {
        assert(config.kvRepeat == 1, "only kv_repeat==1 is supported")

        let b = xs.shape[0]
        let t = xs.shape[1] 
        let hd = xs.shape[2]

        // Project to Q, K, V - following Python exactly
        let qkv = inProj(xs).reshaped([b, t, 3, config.numHeads, config.headDim])
        var q = qkv[0..., 0..., 0, 0..., 0...].swappedAxes(1, 2)  // [b, h, t, d]
        var k = qkv[0..., 0..., 1, 0..., 0...].swappedAxes(1, 2)  // [b, h, t, d] 
        let v = qkv[0..., 0..., 2, 0..., 0...].swappedAxes(1, 2)  // [b, h, t, d]

        // Apply RoPE if configured
        if let rope = rope {
            q = rope(q, offset: cache.selfAttn.offset)
            k = rope(k, offset: cache.selfAttn.offset)
        }

        // Update cache and get complete K, V
        let (fullK, fullV) = cache.selfAttn.updateAndFetch(keys: k, values: v)
        
        // Apply context trimming if needed (following Python logic)
        var finalK = fullK
        var finalV = fullV
        let kLen = fullK.shape[2]
        let kTargetLen = t + min(config.context, kLen - t)
        
        if kTargetLen < kLen {
            let startIdx = kLen - kTargetLen
            finalK = fullK[0..., 0..., startIdx..., 0...]
            finalV = fullV[0..., 0..., startIdx..., 0...]
        }

        // Scaled dot product attention
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: finalK,
            values: finalV,
            scale: scale,
            mask: mask
        )

        // Reshape back and project output
        let reshapedOutput = output.swappedAxes(1, 2).reshaped([b, t, hd])
        return outProj(reshapedOutput)
    }
}

/// Gated MLP (matches MLX Python MlpGating)
class MlpGating: Module {
    @ModuleInfo var linearIn: MLXNN.Linear
    @ModuleInfo var linearOut: MLXNN.Linear

    init(_ config: TransformerConfig) {
        let hidden = 2 * config.dimFeedforward / 3
        let actualHidden = config.dimFeedforward == 4 * config.dModel ? 11 * config.dModel / 4 : hidden

        self._linearIn.wrappedValue = MLXNN.Linear(
            config.dModel,
            2 * actualHidden,
            bias: config.biasFF
        )

        self._linearOut.wrappedValue = MLXNN.Linear(
            actualHidden,
            config.dModel,
            bias: config.biasFF
        )

        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let B = xs.shape[0]
        let T = xs.shape[1]

        // Apply gating: linear_in -> split -> silu(gate) * up -> linear_out
        var hidden = linearIn(xs)
        hidden = hidden.reshaped([B, T, 2, -1])

        // Apply gating: silu(gate) * up_projection
        // Split the hidden tensor into gate and up components
        let gate = hidden[0..., 0..., 0, 0...]
        let up = hidden[0..., 0..., 1, 0...]

        // Apply SiLU activation to gate and multiply with up projection
        let activatedGate = MLXNN.silu(gate)

        return linearOut(activatedGate * up)
    }
}

/// Standard MLP (matches MLX Python MlpNoGating)
class MlpNoGating: Module {
    @ModuleInfo var linear1: MLXNN.Linear
    @ModuleInfo var linear2: MLXNN.Linear

    init(_ config: TransformerConfig) {
        self._linear1.wrappedValue = MLXNN.Linear(
            config.dModel,
            config.dimFeedforward,
            bias: config.biasFF
        )

        self._linear2.wrappedValue = MLXNN.Linear(
            config.dimFeedforward,
            config.dModel,
            bias: config.biasFF
        )

        super.init()
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        // Standard transformer MLP: linear -> gelu -> linear
        return linear2(MLXNN.geluApproximate(linear1(xs)))
    }
}

/// Individual transformer layer (matches MLX Python TransformerLayer)
class TransformerLayer: Module {
    @ModuleInfo var norm1: Module
    @ModuleInfo var norm2: Module
    @ModuleInfo var layerScale1: Module
    @ModuleInfo var layerScale2: Module
    @ModuleInfo var selfAttn: Attention
    @ModuleInfo var gating: Module

    private let config: TransformerConfig

    init(_ config: TransformerConfig) {
        self.config = config

        // Initialize normalization layers (matching Python exactly)
        if config.norm == "layer_norm" {
            self._norm1.wrappedValue = MLXNN.LayerNorm(dimensions: config.dModel, eps: 1e-5)
            self._norm2.wrappedValue = MLXNN.LayerNorm(dimensions: config.dModel, eps: 1e-5)
        } else if config.norm == "rms_norm" {
            self._norm1.wrappedValue = MLXNN.RMSNorm(dimensions: config.dModel, eps: 1e-8)
            self._norm2.wrappedValue = MLXNN.RMSNorm(dimensions: config.dModel, eps: 1e-8)
        } else {
            fatalError("unsupported norm type \(config.norm)")
        }

        // Initialize layer scaling (matching Python exactly)
        if config.layerScale != nil {
            self._layerScale1.wrappedValue = LayerScale(dim: config.dModel)
            self._layerScale2.wrappedValue = LayerScale(dim: config.dModel)
        } else {
            self._layerScale1.wrappedValue = Identity()
            self._layerScale2.wrappedValue = Identity()
        }

        // Initialize attention
        self._selfAttn.wrappedValue = Attention(config)

        // Initialize MLP/gating (matching Python exactly)
        if config.gating {
            self._gating.wrappedValue = MlpGating(config)
        } else {
            self._gating.wrappedValue = MlpNoGating(config)
        }

        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: LayerCache, crossAttentionSrc: MLXArray? = nil) -> MLXArray {
        // Self-attention block (matching Python exactly)
        let n1: MLXArray
        if let layerNorm = norm1 as? MLXNN.LayerNorm {
            n1 = layerNorm(xs)
        } else if let rmsNorm = norm1 as? MLXNN.RMSNorm {
            n1 = rmsNorm(xs)
        } else {
            n1 = xs
        }

        let attnOut = selfAttn(n1, cache: cache)
        
        let layerScale1Out: MLXArray
        if let layerScale = layerScale1 as? LayerScale {
            layerScale1Out = layerScale(attnOut)
        } else if let identity = layerScale1 as? Identity {
            layerScale1Out = identity(attnOut)
        } else {
            layerScale1Out = attnOut
        }
        
        var residual = xs + layerScale1Out

        // TODO: Cross attention support if needed

        // MLP block (matching Python exactly)  
        let n2: MLXArray
        if let layerNorm = norm2 as? MLXNN.LayerNorm {
            n2 = layerNorm(residual)
        } else if let rmsNorm = norm2 as? MLXNN.RMSNorm {
            n2 = rmsNorm(residual)
        } else {
            n2 = residual
        }

        let mlpOut: MLXArray
        if let mlpGating = gating as? MlpGating {
            mlpOut = mlpGating(n2)
        } else if let mlpNoGating = gating as? MlpNoGating {
            mlpOut = mlpNoGating(n2)
        } else {
            mlpOut = n2  // Fallback
        }

        let layerScale2Out: MLXArray
        if let layerScale = layerScale2 as? LayerScale {
            layerScale2Out = layerScale(mlpOut)
        } else if let identity = layerScale2 as? Identity {
            layerScale2Out = identity(mlpOut)
        } else {
            layerScale2Out = mlpOut
        }

        residual = residual + layerScale2Out
        return residual
    }
}

/// Stack of transformer layers (matches MLX Python Transformer)
class Transformer: Module {
    @ModuleInfo var layers: [TransformerLayer]
    private let config: TransformerConfig

    init(_ config: TransformerConfig) {
        self.config = config
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in TransformerLayer(config) }
        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: [LayerCache], crossAttentionSrc: MLXArray? = nil) -> MLXArray {
        var output = xs
        
        for (layer, layerCache) in zip(layers, cache) {
            output = layer(output, cache: layerCache, crossAttentionSrc: crossAttentionSrc)
        }
        
        return output
    }

    func makeCache() -> [LayerCache] {
        let numKvHeads = config.numHeads / config.kvRepeat
        return layers.map { _ in
            LayerCache(selfAttn: KVCache(headDim: config.headDim, nKvHeads: numKvHeads))
        }
    }

    func makeRotCache() -> [LayerCache] {
        let numKvHeads = config.numHeads / config.kvRepeat
        return layers.map { _ in
            LayerCache(selfAttn: RotatingKVCache(
                headDim: config.headDim,
                nKvHeads: numKvHeads,
                maxSize: config.maxSeqLen
            ))
        }
    }
}
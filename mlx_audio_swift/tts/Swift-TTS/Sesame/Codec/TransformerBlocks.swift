//
// TransformerBlocks for Sesame TTS ProjectedTransformer
// Core transformer components: Attention, MLP, and Layer blocks

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

/// Multi-head attention with RoPE embeddings
class Attention: Module {
    @ModuleInfo var multiHeadAttention: MLXNN.MultiHeadAttention

    private let config: TransformerConfig
    private let scale: Float

    init(_ config: TransformerConfig) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)

        // Calculate dimensions for GQA
        let numKvHeads = config.numHeads / config.kvRepeat
        let queryDim = config.dModel
        let kvDim = queryDim / config.kvRepeat  // KV heads have smaller dimension in GQA

        // Initialize MultiHeadAttention with GQA support
        self._multiHeadAttention.wrappedValue = MLXNN.MultiHeadAttention(
            dimensions: config.dModel,
            numHeads: config.numHeads,
            queryInputDimensions: queryDim,
            keyInputDimensions: kvDim,
            valueInputDimensions: kvDim,
            valueDimensions: kvDim,
            bias: config.biasAttn
        )

        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: KVCacheProtocol, mask: MLXArray? = nil) -> MLXArray {
        // Apply multi-head attention with self-attention (queries = keys = values)
        // For self-attention, we use the same tensor for queries, keys, and values
        let attentionOutput = multiHeadAttention(xs, keys: xs, values: xs, mask: mask)

        // Note: RoPE is now handled by SesameAttention in the main model
        // This class provides standard multi-head attention for transformer blocks

        return attentionOutput
    }
}

/// Gated MLP (Gating + Feedforward)
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

/// Standard MLP (Feedforward only)
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

/// Individual transformer layer
class TransformerLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: MLXNN.RMSNorm
    @ModuleInfo(key: "norm2") var norm2: MLXNN.RMSNorm
    @ModuleInfo var layerScale1: Module
    @ModuleInfo var layerScale2: Module
    @ModuleInfo(key: "self_attn") var selfAttention: Attention
    @ModuleInfo var mlp: Module

    private let config: TransformerConfig

    init(_ config: TransformerConfig) {
        self.config = config

        // Initialize normalization layers
        if config.norm == "rms_norm" {
            self._norm1.wrappedValue = MLXNN.RMSNorm(dimensions: config.dModel, eps: 1e-8)
            self._norm2.wrappedValue = MLXNN.RMSNorm(dimensions: config.dModel, eps: 1e-8)
        } else {
            // Fallback to RMSNorm even if layer_norm is specified
            self._norm1.wrappedValue = MLXNN.RMSNorm(dimensions: config.dModel, eps: 1e-5)
            self._norm2.wrappedValue = MLXNN.RMSNorm(dimensions: config.dModel, eps: 1e-5)
        }

        // Initialize layer scaling
        if config.layerScale != nil {
            self._layerScale1.wrappedValue = LayerScale(dim: config.dModel)
            self._layerScale2.wrappedValue = LayerScale(dim: config.dModel)
        } else {
            self._layerScale1.wrappedValue = Identity()
            self._layerScale2.wrappedValue = Identity()
        }

        // Initialize attention
        self._selfAttention.wrappedValue = Attention(config)

        // Initialize MLP (gated or standard)
        if config.gating {
            self._mlp.wrappedValue = MlpGating(config)
        } else {
            self._mlp.wrappedValue = MlpNoGating(config)
        }

        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: KVCacheProtocol) -> MLXArray {
        // Pre-norm architecture: norm -> attention -> residual
        var residual = xs
        var attnInput = config.normFirst ? norm1(xs) : xs
        var attnOutput = selfAttention(attnInput, cache: cache)

        // Apply layer scaling
        if let layerScale = layerScale1 as? LayerScale {
            attnOutput = layerScale(attnOutput)
        } else if let identity = layerScale1 as? Identity {
            attnOutput = identity(attnOutput)
        }

        residual = residual + attnOutput

        // Pre-norm architecture: norm -> MLP -> residual
        var mlpInput = config.normFirst ? norm2(residual) : residual

        // Call the MLP based on its type
        var mlpOutput: MLXArray
        if let gatingMLP = mlp as? MlpGating {
            mlpOutput = gatingMLP(mlpInput)
        } else if let noGatingMLP = mlp as? MlpNoGating {
            mlpOutput = noGatingMLP(mlpInput)
        } else {
            // Fallback - this shouldn't happen in our implementation
            mlpOutput = mlpInput
        }

        // Apply layer scaling to MLP output
        if let layerScale = layerScale2 as? LayerScale {
            mlpOutput = layerScale(mlpOutput)
        } else if let identity = layerScale2 as? Identity {
            mlpOutput = identity(mlpOutput)
        }
        
        residual = residual + mlpOutput
        return residual
    }
}

/// Stack of transformer layers
class Transformer: Module {
    @ModuleInfo(key: "layers") var layers: [TransformerLayer]
    private let config: TransformerConfig

    init(_ config: TransformerConfig) {
        self.config = config
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in TransformerLayer(config) }
        super.init()
    }

    func callAsFunction(_ xs: MLXArray, cache: [KVCacheProtocol]) -> MLXArray {
        var output = xs
        for (layerIndex, layer) in layers.enumerated() {
            let layerCache = cache[layerIndex]  // Get the cache for this specific layer
            output = layer(output, cache: layerCache)
        }
        return output
    }

    func makeCache() -> [KVCache] {
        let numKvHeads = config.numHeads / config.kvRepeat
        return layers.map { _ in
            KVCache(headDim: config.headDim, nKvHeads: numKvHeads)
        }
    }

    func makeRotCache() -> [RotatingKVCache] {
        let numKvHeads = config.numHeads / config.kvRepeat
        return layers.map { _ in
            RotatingKVCache(
                headDim: config.headDim,
                nKvHeads: numKvHeads,
                maxSize: config.maxSeqLen
            )
        }
    }
}
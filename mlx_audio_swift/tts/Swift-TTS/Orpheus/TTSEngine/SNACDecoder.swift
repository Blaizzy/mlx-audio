import Foundation
import MLX
import MLXNN

class SNACDecoder {
    private let weights: [String: MLXArray]
    private let config: SNACConfig
    
    init(weights: [String: MLXArray], config: SNACConfig) {
        self.weights = weights
        self.config = config
    }
    
    func decode(codes: [[Int]]) -> MLXArray {
        // Convert codes to embeddings
        let embeddings = embedCodes(codes: codes)
        
        // Process through decoder layers
        var x = embeddings
        for i in 0..<config.numLayers {
            x = decoderLayer(x: x, layerIndex: i)
        }
        
        // Final projection to audio samples
        let audio = finalProjection(x: x)
        return audio
    }
    
    private func embedCodes(codes: [[Int]]) -> MLXArray {
        // Convert codes to embeddings using learned embedding matrix
        guard let codeEmbeddings = weights["code_embeddings"] else {
            print("Error: code_embeddings weight not found in model weights")
            return MLXArray([])
        }
        var embeddings: [MLXArray] = []
        
        for codeList in codes {
            let codeIndices = MLXArray(codeList)
            let embedded = codeEmbeddings[codeIndices]
            embeddings.append(embedded)
        }
        
        return MLX.concatenated(embeddings, axis: 0)
    }
    
    private func decoderLayer(x: MLXArray, layerIndex: Int) -> MLXArray {
        // Self-attention
        let attnOutput = selfAttention(x: x, layerIndex: layerIndex)
        let x1 = x + attnOutput
        
        // Feed-forward
        let ffOutput = feedForward(x: x1, layerIndex: layerIndex)
        return x1 + ffOutput
    }
    
    private func selfAttention(x: MLXArray, layerIndex: Int) -> MLXArray {
        // Project to query, key, value
        let q = linear(x: x, weight: weights["decoder.layers.\(layerIndex).self_attn.q_proj.weight"]!)
        let k = linear(x: x, weight: weights["decoder.layers.\(layerIndex).self_attn.k_proj.weight"]!)
        let v = linear(x: x, weight: weights["decoder.layers.\(layerIndex).self_attn.v_proj.weight"]!)
        
        // Compute attention scores
        let scores = MLX.matmul(q, k.transposed(0, 1)) / sqrt(Float(config.headDim))
        let probs = MLX.softmax(scores, axis: -1)
        
        // Apply attention
        let attnOutput = MLX.matmul(probs, v)
        
        // Project back to model dimension
        return linear(x: attnOutput, weight: weights["decoder.layers.\(layerIndex).self_attn.o_proj.weight"]!)
    }
    
    private func feedForward(x: MLXArray, layerIndex: Int) -> MLXArray {
        // First linear layer
        let h = linear(x: x, weight: weights["decoder.layers.\(layerIndex).mlp.gate_proj.weight"]!)
        let h1 = silu(h)
        
        // Second linear layer
        let h2 = linear(x: x, weight: weights["decoder.layers.\(layerIndex).mlp.up_proj.weight"]!)
        
        // Element-wise multiplication
        let h3 = h1 * h2
        
        // Final projection
        return linear(x: h3, weight: weights["decoder.layers.\(layerIndex).mlp.down_proj.weight"]!)
    }
    
    private func finalProjection(x: MLXArray) -> MLXArray {
        // Project to audio samples
        return linear(x: x, weight: weights["audio_proj.weight"]!)
    }
    
    private func linear(x: MLXArray, weight: MLXArray) -> MLXArray {
        return MLX.matmul(x, weight)
    }
    
    // SiLU (Sigmoid Linear Unit) activation function
    private func silu(_ x: MLXArray) -> MLXArray {
        return x * MLX.sigmoid(x)
    }
}

struct SNACConfig {
    let numLayers: Int = 32
    let headDim: Int = 128
    let modelDim: Int = 4096
    let numHeads: Int = 32
    let codebookSize: Int = 1024
} 
//
// Vector Quantization for Sesame TTS Mimi Codec
// Using MLXNN directly for standard components, custom logic only where needed

import Foundation
import MLX
import MLXNN

/// Euclidean Codebook for vector quantization
class EuclideanCodebook: Module {
    @ParameterInfo var embeddingSum: MLXArray
    @ParameterInfo var clusterUsage: MLXArray
    @ParameterInfo var initialized: MLXArray

    private let dim: Int
    private let codebookSize: Int
    private let epsilon: Float = 1e-5

    init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.codebookSize = codebookSize

        self._initialized.wrappedValue = MLXArray.zeros([1])
        self._embeddingSum.wrappedValue = MLXArray.zeros([codebookSize, dim])
        self._clusterUsage.wrappedValue = MLXArray.zeros([codebookSize])

        super.init()
    }

    /// Encode vectors to codebook indices
    func encode(_ x: MLXArray) -> MLXArray {
        // Flatten spatial dimensions for batch processing
        let targetShape = Array(x.shape[0..<x.ndim-1])
        let xFlat = x.flattened(end: -2)

        // Compute distances to all codebook vectors
        let embedding = getEmbedding()
        let dotProd = MLX.matmul(xFlat, embedding.swappedAxes(-1, -2))
        let distances = (embedding ** 2).sum(axis: -1).expandedDimensions(axis: 0) - 2 * dotProd

        // Find closest codebook vectors
        let indices = distances.argMin(axis: -1)
        return indices.reshaped(targetShape)
    }

    /// Decode codebook indices to vectors
    func decode(_ x: MLXArray) -> MLXArray {
        let embedding = getEmbedding()
        let targetShape = Array(x.shape) + [dim]
        return MLX.take(embedding, x.flattened(), axis: 0).reshaped(targetShape)
    }

    /// Get current embedding vectors (computed from running statistics)
    private func getEmbedding() -> MLXArray {
        let clusterUsage = MLX.maximum(self.clusterUsage, MLXArray(epsilon))
        return embeddingSum / clusterUsage.expandedDimensions(axis: -1)
    }

    /// Update codebook statistics during training
    func updateStatistics(_ x: MLXArray, _ indices: MLXArray) {
        // Note: In practice, you'd implement proper VQ statistics updating here
        // For now, this is a placeholder implementation

        let _ = getEmbedding()  // Keep for future implementation
        let _ = x.flattened(end: -2)  // Keep for future implementation
        let _ = indices.flattened()   // Keep for future implementation

        // Simplified placeholder - in a real implementation you'd update
        // embeddingSum and clusterUsage based on the current batch
        let zeroSum = MLXArray.zeros(like: embeddingSum)
        let zeroUsage = MLXArray.zeros(like: clusterUsage)

        embeddingSum._updateInternal(embeddingSum + zeroSum)
        clusterUsage._updateInternal(clusterUsage + zeroUsage)
    }
}

/// Single vector quantization layer
class VectorQuantization: Module {
    @ModuleInfo var codebook: EuclideanCodebook
    @ModuleInfo var projectIn: MLXNN.Linear?
    @ModuleInfo var projectOut: MLXNN.Linear?

    init(dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        let actualCodebookDim = codebookDim ?? dim

        self._codebook.wrappedValue = EuclideanCodebook(
            dim: actualCodebookDim,
            codebookSize: codebookSize
        )

        // Projection layers if dimensions don't match
        if dim != actualCodebookDim {
            self._projectIn.wrappedValue = MLXNN.Linear(dim, actualCodebookDim, bias: false)
            self._projectOut.wrappedValue = MLXNN.Linear(actualCodebookDim, dim, bias: false)
        }

        super.init()
    }

    func encode(_ x: MLXArray) -> MLXArray {
        // Python: xs = xs.swapaxes(-1, -2)  
        // Convert [batch, channels, time] -> [batch, time, channels] for processing
        var processed = x.swappedAxes(-1, -2)
        
        if let projectIn = projectIn {
            processed = projectIn(processed)
        }
        return codebook.encode(processed)
    }

    func decode(_ x: MLXArray) -> MLXArray {
        var decoded = codebook.decode(x)
        if let projectOut = projectOut {
            decoded = projectOut(decoded)
        }
        
        // Python: return xs.swapaxes(-1, -2)
        // Convert [batch, time, channels] -> [batch, channels, time] for output
        return decoded.swappedAxes(-1, -2)
    }
}

/// Residual Vector Quantization (RVQ) with multiple layers
class ResidualVectorQuantization: Module {
    private let layers: [VectorQuantization]
    private let numQuantizers: Int

    init(
        numQuantizers: Int,
        dim: Int,
        codebookSize: Int,
        codebookDim: Int? = nil
    ) {
        self.numQuantizers = numQuantizers

        // Create multiple VQ layers
        var vqLayers: [VectorQuantization] = []
        for _ in 0..<numQuantizers {
            let vq = VectorQuantization(
                dim: dim,
                codebookSize: codebookSize,
                codebookDim: codebookDim
            )
            vqLayers.append(vq)
        }
        self.layers = vqLayers

        super.init()
    }

    /// Encode with residual quantization
    func encode(_ x: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = x

        for layer in layers {
            let indices = layer.encode(residual)
            let quantized = layer.decode(indices)
            residual = residual - quantized
            codes.append(indices)
        }

        return MLX.stacked(codes)
    }

    /// Decode by summing all layers
    func decode(_ codes: MLXArray) -> MLXArray {
        // Python: seq_len = xs.shape[0] (this is the codebooks dimension after swapaxes)
        // Input should be [codebooks, batch, time] after swapaxes from ResidualVectorQuantizer
        
        print("🔍 DEBUG RVQ decode: input codes shape = \(codes.shape), numQuantizers = \(numQuantizers)")
        
        let seqLen = codes.shape[0]  // Python: seq_len = xs.shape[0]
        
        guard seqLen >= 1 else {
            fatalError("Expected at least 1 codebook, got \(seqLen)")
        }
        
        guard seqLen <= numQuantizers else {
            fatalError("Too many codebooks: expected \(numQuantizers), got \(seqLen) in shape \(codes.shape)")
        }
        
        // Python: quantized = self.layers[0].decode(xs[0])
        print("🔍 DEBUG RVQ decode: decoding layer 0 with codes[0] shape \(codes[0].shape)")
        var quantized = layers[0].decode(codes[0])
        print("🔍 DEBUG RVQ decode: layer 0 quantized shape = \(quantized.shape)")

        // Python: for i in range(1, seq_len):
        //             quantized = quantized + self.layers[i].decode(xs[i])
        for i in 1..<seqLen {
            print("🔍 DEBUG RVQ decode: decoding layer \(i) with codes[\(i)] shape \(codes[i].shape)")
            let layerQuantized = layers[i].decode(codes[i])
            print("🔍 DEBUG RVQ decode: layer \(i) quantized shape = \(layerQuantized.shape)")
            quantized = quantized + layerQuantized
            print("🔍 DEBUG RVQ decode: accumulated quantized shape = \(quantized.shape)")
        }

        return quantized
    }

    /// Get individual layer codes
    func getLayerCodes(_ codes: MLXArray, layerIndex: Int) -> MLXArray {
        return codes[layerIndex]
    }
}
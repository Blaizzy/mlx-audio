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
        var processed = x
        if let projectIn = projectIn {
            processed = projectIn(x)
        }
        return codebook.encode(processed)
    }

    func decode(_ x: MLXArray) -> MLXArray {
        var decoded = codebook.decode(x)
        if let projectOut = projectOut {
            decoded = projectOut(decoded)
        }
        return decoded
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
        var quantized = layers[0].decode(codes[0])

        for i in 1..<numQuantizers {
            quantized = quantized + layers[i].decode(codes[i])
        }

        return quantized
    }

    /// Get individual layer codes
    func getLayerCodes(_ codes: MLXArray, layerIndex: Int) -> MLXArray {
        return codes[layerIndex]
    }
}

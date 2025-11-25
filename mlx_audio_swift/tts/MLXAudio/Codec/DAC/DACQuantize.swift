//
//  DACQuantize.swift
//  MLXAudio
//
//  DAC Vector Quantization layers
//  Ported from: mlx_audio/codec/models/descript/nn/quantize.py
//

import Foundation
import MLX
import MLXNN

// MARK: - L2 Normalization

/// Normalize input tensor along specified dimension using L2 norm
func l2Normalize(_ input: MLXArray, dim: Int = 1, eps: Float = 1e-12) -> MLXArray {
    let norm = MLX.pow(
        MLX.sum(MLX.pow(MLX.abs(input), 2), axis: dim, keepDims: true),
        0.5
    )
    return input / MLX.maximum(norm, MLXArray(eps))
}

// MARK: - Vector Quantize

/// Single codebook vector quantization
class DACVectorQuantize: Module {
    let codebookSize: Int
    let codebookDim: Int

    // Use @ModuleInfo to map Python snake_case keys to Swift camelCase
    @ModuleInfo(key: "in_proj") var inProj: DACWNConv1d
    @ModuleInfo(key: "out_proj") var outProj: DACWNConv1d
    let codebook: Embedding

    init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim

        self._inProj.wrappedValue = DACWNConv1d(
            inChannels: inputDim,
            outChannels: codebookDim,
            kernelSize: 1
        )
        self._outProj.wrappedValue = DACWNConv1d(
            inChannels: codebookDim,
            outChannels: inputDim,
            kernelSize: 1
        )
        self.codebook = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)

        super.init()
    }

    /// Forward pass: quantize latent representation
    func callAsFunction(_ z: MLXArray) -> (zQ: MLXArray, commitmentLoss: MLXArray, codebookLoss: MLXArray, indices: MLXArray, zE: MLXArray) {
        // z: [batch, time, dim]
        // Project to codebook dimension
        let zE = inProj(z.transposed(axes: [0, 2, 1])).transposed(axes: [0, 2, 1])  // [batch, time, codebook_dim]

        // Find nearest codebook entries
        let (zQRaw, indices) = decodeLatents(zE)

        // Compute losses
        let commitmentLoss = MLX.mean(MLX.pow(zE - zQRaw, 2), axes: [1, 2])
        let codebookLoss = MLX.mean(MLX.pow(zQRaw - zE, 2), axes: [1, 2])

        // Straight-through estimator: forward pass uses quantized, backward uses continuous
        let zQ = zE + (zQRaw - zE)  // This is a no-op in forward, but allows gradients to flow

        // Project back to input dimension
        let zQOut = outProj(zQ.transposed(axes: [0, 2, 1])).transposed(axes: [0, 2, 1])

        return (zQOut, commitmentLoss, codebookLoss, indices, zE)
    }

    /// Embed code indices to vectors
    func embedCode(_ embedId: MLXArray) -> MLXArray {
        return codebook(embedId)
    }

    /// Decode code indices to vectors (transposed)
    func decodeCode(_ embedId: MLXArray) -> MLXArray {
        return embedCode(embedId).transposed(axes: [0, 2, 1])
    }

    /// Find nearest codebook entries for latent vectors
    func decodeLatents(_ latents: MLXArray) -> (zQ: MLXArray, indices: MLXArray) {
        let batch = latents.shape[0]
        let time = latents.shape[1]

        // Reshape: [batch, time, dim] -> [batch*time, dim]
        let encodings = latents.reshaped([batch * time, -1])
        let codebookWeight = codebook.weight  // [codebook_size, codebook_dim]

        // L2 normalize
        let encodingsNorm = l2Normalize(encodings)
        let codebookNorm = l2Normalize(codebookWeight)

        // Compute distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        let dist = MLX.sum(MLX.pow(encodingsNorm, 2), axis: 1, keepDims: true)
            - 2 * MLX.matmul(encodingsNorm, codebookNorm.transposed())
            + MLX.sum(MLX.pow(codebookNorm, 2), axis: 1, keepDims: true).transposed()

        // Find minimum distance (argmax of negative distance)
        let minDist = MLX.argMax(MLX.negative(dist), axis: 1)

        // Reshape indices back: [batch*time] -> [batch, time]
        let indices = minDist.reshaped([batch, time])

        // Get quantized vectors
        let zQ = decodeCode(indices)

        return (zQ, indices)
    }
}

// MARK: - Residual Vector Quantize

/// Residual vector quantization with multiple codebooks
class DACResidualVectorQuantize: Module {
    let nCodebooks: Int
    let codebookDim: [Int]
    let codebookSize: Int

    let quantizers: [DACVectorQuantize]

    init(inputDim: Int = 512, nCodebooks: Int = 9, codebookSize: Int = 1024, codebookDim: Int = 8) {
        self.nCodebooks = nCodebooks
        self.codebookDim = Array(repeating: codebookDim, count: nCodebooks)
        self.codebookSize = codebookSize

        var quantizers: [DACVectorQuantize] = []
        for i in 0..<nCodebooks {
            quantizers.append(DACVectorQuantize(
                inputDim: inputDim,
                codebookSize: codebookSize,
                codebookDim: self.codebookDim[i]
            ))
        }
        self.quantizers = quantizers

        super.init()
    }

    /// Forward pass with residual quantization
    func callAsFunction(_ z: MLXArray, nQuantizers: Int? = nil) -> (zQ: MLXArray, codes: MLXArray, latents: MLXArray, commitmentLoss: MLXArray, codebookLoss: MLXArray) {
        var zQ = MLXArray.zeros(z.shape)
        var residual = z
        var commitmentLoss = MLXArray(0.0)
        var codebookLoss = MLXArray(0.0)

        var codebookIndices: [MLXArray] = []
        var latents: [MLXArray] = []

        let numQuantizers = nQuantizers ?? nCodebooks

        for (i, quantizer) in quantizers.enumerated() {
            if i >= numQuantizers {
                break
            }

            let (zQi, commitmentLossI, codebookLossI, indicesI, zEi) = quantizer(residual)

            // Create mask for active quantizers
            let mask = MLX.full([z.shape[0]], values: Float(i < numQuantizers ? 1.0 : 0.0))

            // Accumulate quantized output
            zQ = zQ + zQi * mask.reshaped([z.shape[0], 1, 1])

            // Update residual
            residual = residual - zQi

            // Accumulate losses
            commitmentLoss = commitmentLoss + MLX.mean(commitmentLossI * mask)
            codebookLoss = codebookLoss + MLX.mean(codebookLossI * mask)

            codebookIndices.append(indicesI)
            latents.append(zEi)
        }

        // Stack codes: [batch, n_codebooks, time]
        let codes = MLX.stacked(codebookIndices, axis: 1)

        // Concatenate latents: [batch, sum(codebook_dims), time]
        let allLatents = MLX.concatenated(latents, axis: 1)

        return (zQ, codes, allLatents, commitmentLoss, codebookLoss)
    }

    /// Reconstruct from codes
    func fromCodes(_ codes: MLXArray) -> (zQ: MLXArray, zP: MLXArray, codes: MLXArray) {
        var zQ: MLXArray? = nil
        var zPList: [MLXArray] = []
        let numCodebooks = codes.shape[1]

        for i in 0..<numCodebooks {
            // Get codes for this codebook: [batch, time]
            // codes shape: [batch, n_codebooks, time] -> take codes[:, i, :]
            // Indexing with single integer removes that dimension, no squeeze needed
            let codesI = codes[0..., i, 0...]

            // Decode to vectors
            let zPi = quantizers[i].decodeCode(codesI)
            zPList.append(zPi)

            // Project through output projection
            let zQi = quantizers[i].outProj(zPi.transposed(axes: [0, 2, 1])).transposed(axes: [0, 2, 1])

            // Accumulate
            if zQ == nil {
                zQ = zQi
            } else {
                zQ = zQ! + zQi
            }
        }

        let zP = MLX.concatenated(zPList, axis: 1)
        return (zQ!, zP, codes)
    }
}

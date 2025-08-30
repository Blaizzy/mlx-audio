//
// SplitResidualVectorQuantization for Sesame TTS Mimi Codec
// Advanced quantizer architecture matching Python implementation
// Equivalent to Python's SplitResidualVectorQuantizer

import Foundation
import MLX
import MLXNN

/// Advanced Residual Vector Quantizer with input/output projections
class ResidualVectorQuantizer: Module {
    @ModuleInfo var inputProj: MLXNN.Conv1d?
    @ModuleInfo var outputProj: MLXNN.Conv1d?
    @ModuleInfo var vq: ResidualVectorQuantization

    private let dim: Int
    private let inputDim: Int
    private let outputDim: Int

    init(dim: Int, inputDim: Int? = nil, outputDim: Int? = nil, nq: Int, bins: Int, forceProjection: Bool = false) {
        self.dim = dim
        self.inputDim = inputDim ?? dim
        self.outputDim = outputDim ?? dim

        print("🔍 DEBUG ResidualVectorQuantizer init: dim=\(dim), inputDim=\(self.inputDim), outputDim=\(self.outputDim), nq=\(nq), forceProjection=\(forceProjection)")

        // Initialize projections if needed
        if inputDim == dim && !forceProjection {
            self._inputProj.wrappedValue = nil
            print("🔍 DEBUG ResidualVectorQuantizer init: No input projection needed")
        } else {
            let projInputDim = inputDim ?? dim
            // Use Conv1d with kernel size 1 to match Python implementation
            self._inputProj.wrappedValue = MLXNN.Conv1d(
                inputChannels: projInputDim,
                outputChannels: dim,
                kernelSize: 1,
                bias: false
            )
            print("🔍 DEBUG ResidualVectorQuantizer init: Created input projection \(projInputDim) -> \(dim)")
        }

        if outputDim == dim && !forceProjection {
            self._outputProj.wrappedValue = nil
            print("🔍 DEBUG ResidualVectorQuantizer init: No output projection needed")
        } else {
            let projOutputDim = outputDim ?? dim
            self._outputProj.wrappedValue = MLXNN.Conv1d(
                inputChannels: dim,
                outputChannels: projOutputDim,
                kernelSize: 1,
                bias: false
            )
            print("🔍 DEBUG ResidualVectorQuantizer init: Created output projection \(dim) -> \(projOutputDim)")
        }

        // Initialize the core RVQ
        self._vq.wrappedValue = ResidualVectorQuantization(
            numQuantizers: nq,
            dim: dim,
            codebookSize: bins,
            codebookDim: nil  // Use default (same as dim)
        )

        super.init()
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var x = xs
        if let inputProj = inputProj {
            x = inputProj(x)
        }
        return vq.encode(x).swappedAxes(0, 1)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        print("🔍 DEBUG RVQ decode: input shape = \(xs.shape)")
        
        // Python: xs = xs.swapaxes(0, 1)
        // Input: [batch, codebooks, time] -> [codebooks, batch, time]
        var x = xs.swappedAxes(0, 1)
        print("🔍 DEBUG RVQ decode: after swapaxes shape = \(x.shape)")
        
        // Python: quantized = self.vq.decode(xs)
        x = vq.decode(x)
        print("🔍 DEBUG RVQ decode: after vq.decode shape = \(x.shape)")
        
        if let outputProj = outputProj {
            print("🔍 DEBUG RVQ decode: applying output projection from \(x.shape)")
            print("🔍 DEBUG RVQ decode: Conv1d weight shape = \(outputProj.weight.shape)")
            
            // MLX Conv1d expects channel-last format: [batch, length, in_channels]
            // Our tensor is currently [batch, in_channels, time] = [1, 256, 3] 
            // We need to convert to [batch, time, in_channels] = [1, 3, 256]
            
            print("🔧 DEBUG RVQ decode: Converting to MLX channel-last format...")
            x = x.swappedAxes(1, 2)  // [1, 256, 3] -> [1, 3, 256]
            print("🔧 DEBUG RVQ decode: After conversion: \(x.shape)")
            
            // Apply Conv1d projection: [1, 3, 256] -> [1, 3, 512]
            x = outputProj(x)
            print("🔍 DEBUG RVQ decode: after output projection shape = \(x.shape)")
            
            // Convert back to channel-first format for rest of pipeline: [1, 3, 512] -> [1, 512, 3]
            print("🔧 DEBUG RVQ decode: Converting back to channel-first format...")
            x = x.swappedAxes(1, 2)  // [1, 3, 512] -> [1, 512, 3]
            print("🔧 DEBUG RVQ decode: Final output shape: \(x.shape)")
        }
        
        return x
    }
}

/// Split Residual Vector Quantizer - Main quantizer used in Mimi
/// Splits quantization across multiple RVQ layers for better performance
class SplitResidualVectorQuantizer: Module {
    @ModuleInfo var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo var rvqRest: ResidualVectorQuantizer

    private let nq: Int

    init(dim: Int, inputDim: Int? = nil, outputDim: Int? = nil, nq: Int, bins: Int) {
        self.nq = nq

        // First RVQ handles the first codebook
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dim: dim,
            inputDim: inputDim,
            outputDim: outputDim,
            nq: 1,
            bins: bins,
            forceProjection: true
        )

        // Rest RVQ handles remaining codebooks
        if nq > 1 {
            self._rvqRest.wrappedValue = ResidualVectorQuantizer(
                dim: dim,
                inputDim: inputDim,
                outputDim: outputDim,
                nq: nq - 1,
                bins: bins,
                forceProjection: true
            )
        } else {
            // If nq == 1, create a dummy RVQ that does nothing
            self._rvqRest.wrappedValue = ResidualVectorQuantizer(
                dim: dim,
                inputDim: dim,
                outputDim: dim,
                nq: 0,
                bins: bins,
                forceProjection: false
            )
        }

        super.init()
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var codes = rvqFirst.encode(xs)

        if nq > 1 {
            let restCodes = rvqRest.encode(xs)
            codes = MLX.concatenated([codes, restCodes], axis: 1)
        }

        return codes
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        // xs should have shape [batch, num_codebooks, time]
        // Split along the codebook dimension (axis 1)
        
        print("🔍 DEBUG SplitRVQ decode: input shape = \(xs.shape), nq = \(nq)")
        
        // Validate input shape
        guard xs.ndim == 3 else {
            fatalError("SplitRVQ decode expects 3D input [batch, codebooks, time], got \(xs.ndim)D: \(xs.shape)")
        }
        
        guard xs.shape[1] == nq else {
            fatalError("SplitRVQ decode expects \(nq) codebooks, got \(xs.shape[1]) in shape \(xs.shape)")
        }
        
        // Python: quantized = self.rvq_first.decode(xs[:, :1])
        let firstSlice = xs[0..., 0..<1, 0...]  // [batch, 1, time]
        print("🔍 DEBUG SplitRVQ decode: firstSlice shape = \(firstSlice.shape)")
        
        var quantized = rvqFirst.decode(firstSlice)
        print("🔍 DEBUG SplitRVQ decode: first quantized shape = \(quantized.shape)")

        if nq > 1 {
            // Python: quantized = quantized + self.rvq_rest.decode(xs[:, 1:])
            let restSlice = xs[0..., 1..., 0...]  // [batch, nq-1, time] - Python uses xs[:, 1:]
            print("🔍 DEBUG SplitRVQ decode: restSlice shape = \(restSlice.shape)")
            
            let restQuantized = rvqRest.decode(restSlice)
            print("🔍 DEBUG SplitRVQ decode: rest quantized shape = \(restQuantized.shape)")
            
            quantized = quantized + restQuantized
            print("🔍 DEBUG SplitRVQ decode: final quantized shape = \(quantized.shape)")
        }

        return quantized
    }

    /// Get the number of quantizers
    var numQuantizers: Int {
        return nq
    }

    /// Get the number of codebooks (same as numQuantizers for Split RVQ)
    var numCodebooks: Int {
        return nq
    }
}
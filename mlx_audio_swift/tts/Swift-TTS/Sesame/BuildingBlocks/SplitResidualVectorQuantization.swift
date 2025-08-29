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

        // Initialize projections if needed
        if inputDim == dim && !forceProjection {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = MLXNN.Conv1d(
                inputChannels: inputDim!,
                outputChannels: dim,
                kernelSize: 1,
                bias: false
            )
        }

        if outputDim == dim && !forceProjection {
            self._outputProj.wrappedValue = nil
        } else {
            self._outputProj.wrappedValue = MLXNN.Conv1d(
                inputChannels: dim,
                outputChannels: outputDim!,
                kernelSize: 1,
                bias: false
            )
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
        var x = xs.swappedAxes(0, 1)
        x = vq.decode(x)
        if let outputProj = outputProj {
            x = outputProj(x)
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
        // Split along the codebook dimension (axis 1)
        let firstSlice = xs[0..., 0..<(nq > 0 ? 1 : 0), 0...]
        var quantized = rvqFirst.decode(firstSlice)

        if nq > 1 {
            let restSlice = xs[0..., 1..<nq, 0...]
            let restQuantized = rvqRest.decode(restSlice)
            quantized = quantized + restQuantized
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

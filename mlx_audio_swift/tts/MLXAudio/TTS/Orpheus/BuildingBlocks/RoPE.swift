//
//  RoPE.swift
//  MLXAudio
//
//  Created by Ben Harraway on 21/05/2025.
//
import Foundation
import MLX
import MLXFast

/// Llama3-style RoPE implementation using MLXFast.RoPE fused kernel
///
/// This implementation pre-computes scaled frequencies for Llama3-style RoPE
/// and uses the optimized MLXFast.RoPE kernel for the actual computation.
class OrpheusRoPE {
    let dims: Int
    let traditional: Bool
    let base: Float
    let scale: Float
    let scaleFactor: Float
    let lowFreqFactor: Float
    let highFreqFactor: Float
    let oldContextLen: Float

    /// Pre-computed frequencies for Llama3-style RoPE (passed to MLXFast.RoPE)
    private var freqs: MLXArray?

    init(
        dims: Int,
        traditional: Bool = false,
        base: Float = 500000.0,
        maxSeqLen: Int = 2048,
        scaleFactor: Float = 32.0,
        lowFreqFactor: Float = 1.0,
        highFreqFactor: Float = 4.0,
        oldContextLen: Int = 8192
    ) {
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = 1.0
        self.scaleFactor = scaleFactor
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
        self.oldContextLen = Float(oldContextLen)

        computeFreqs()
    }

    /// Compute Llama3-style scaled frequencies
    private func computeFreqs() {
        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        // Compute base frequencies: base^(indices/dims)
        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(MLXArray(base), indices / Float(dims))

        // Wavelengths for determining frequency bands
        let wavelens = 2 * Float.pi * frequencies

        // High frequency band: wavelens > lowFreqWavelen -> scale by factor
        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen),
            frequencies * scaleFactor,
            frequencies
        )

        // Medium frequency band: interpolate between scaled and unscaled
        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )
        let smoothFactors = (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / scaleFactor + smoothFactors)

        freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
    }

    /// Apply RoPE using the optimized MLXFast kernel
    func call(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,  // Using pre-computed freqs instead
            scale: scale,
            offset: offset,
            freqs: freqs
        )
    }
}

//
//  RoPE.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 21/05/2025.
//
import Foundation
import MLX

// Implementation based on mlx_lm.models.rope_utils
class RoPE {
    let dims: Int
    let traditional: Bool
    let base: Float
    let maxSeqLen: Int
    let scaleFactor: Float
    let lowFreqFactor: Float
    let highFreqFactor: Float
    let oldContextLen: Int
    private var theta: MLXArray?
    private var cache: MLXArray?
    
    init(dims: Int, traditional: Bool = false, base: Float = 500000.0, maxSeqLen: Int = 2048, scaleFactor: Float = 32.0, lowFreqFactor: Float = 1.0, highFreqFactor: Float = 4.0, oldContextLen: Int = 8192) {
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.maxSeqLen = maxSeqLen
        self.scaleFactor = scaleFactor
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
        self.oldContextLen = oldContextLen
        self.ropeInit()
    }
    
    private func ropeInit() {
        // Calculate frequencies
        let baseArray = MLXArray(base)
        let exponents = MLXArray.arange(start: 0, stop: dims, step: 2)[..<(dims/2)].asType(.float32) / Float(dims)
        let freqs = 1.0 / MLX.pow(baseArray, exponents)
        
        // Apply scaling
        theta = applyScaling(freqs: freqs)
        
        // Build cache
        buildRopeCache(maxSeqLen: maxSeqLen)
    }
    
    private func applyScaling(freqs: MLXArray) -> MLXArray {
        // Match Python's mlx_lm.models.rope_utils.apply_scaling logic used for Llama-3 style RoPE
        // If the legacy ("traditional") flag is set, return freqs unchanged.
        if traditional {
            return freqs
        }

        // Constants converted to MLXArray / Float32 for broadcasting
        let sf: Float = scaleFactor                      // scaling factor (e.g. 32)
        let lowF: Float = lowFreqFactor                  // 1.0
        let highF: Float = highFreqFactor                // 4.0
        let oldLen: Float = Float(oldContextLen)         // 8192

        // Pre-compute wavelength cut-offs as in Python
        let lowFreqWavelen: Float = oldLen / lowF
        let highFreqWavelen: Float = oldLen / highF

        // Vectorised computation
        // wavelength = 2π / freq
        let twoPi = Float.pi * 2
        let wavelens = MLXArray(twoPi) / freqs

        // Case masks
        let maskLow = MLX.less(wavelens, MLXArray(highFreqWavelen))
        let maskHigh = MLX.greater(wavelens, MLXArray(lowFreqWavelen))

        // Pre-compute scaled frequency (freq / scale_factor)
        let freqDivScale = freqs / MLXArray(sf)

        // Smooth interpolation for mid-band frequencies
        let smoothNumer = MLXArray(oldLen) / wavelens - MLXArray(lowF)
        let smoothDenom = MLXArray(highF - lowF)
        let smooth = smoothNumer / smoothDenom
        let mid = (MLXArray(1.0) - smooth) * freqDivScale + smooth * freqs

        // Combine using nested MLX.where (equivalent to Python's if/elif/else)
        let step1 = MLX.where(maskHigh, freqDivScale, mid)
        let newFreqs = MLX.where(maskLow, freqs, step1)        

        return newFreqs.asType(freqs.dtype)
    }
    
    private func buildRopeCache(maxSeqLen: Int) {
        guard let theta = theta else { return }
        
        // Create sequence indices
        let seqIdx = MLXArray.arange(start: 0, stop: maxSeqLen, dtype: theta.dtype)
        
        // Calculate idx_theta = seq_idx * theta
        let idxTheta = MLX.einsum("i,j->ij", seqIdx, theta).asType(.float32)
        
        // Compute cosine & sine and stack them directly – avoids two expand-dims and one concat.
        let cosCache = MLX.cos(idxTheta)
        let sinCache = MLX.sin(idxTheta)

        // Shape after stack: [maxSeqLen, D_half, 2]
        cache = MLX.stacked([cosCache, sinCache], axis: 2)
    }
    
    func call(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard let cache = cache else { return x }
        
        let L = x.shape[2]
        let D_head = x.shape[3] // Head dimension
        let D_half = D_head / 2

        // Split input into real and imaginary parts
        let x1 = x[.ellipsis, ..<D_half] // Shape: [B, H, L, D_half]
        let x2 = x[.ellipsis, D_half...] // Shape: [B, H, L, D_half]

        // cache has shape [maxSeqLen, D_half, 2]
        // Slice the cache for the current sequence length L
        let cos_slice = cache[offset ..< (offset + L), .ellipsis, 0] // Shape: [L, D_half]
        let sin_slice = cache[offset ..< (offset + L), .ellipsis, 1] // Shape: [L, D_half]

        // Rely on implicit broadcasting; no reshaping necessary.
        let y1 = x1 * cos_slice - x2 * sin_slice
        let y2 = x2 * cos_slice + x1 * sin_slice
        
        let rpe = MLX.concatenated([y1, y2], axis: -1)
        
        return rpe
    }
}

//
//  Dequantizer.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 21/05/2025.
//
import Foundation
import MLX

public class Dequantizer {
    
    static public func dequantize(_ w_q: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int) -> MLXArray {
        guard bits == 4 else {
            fatalError("Only 4-bit dequantization is currently implemented.")
        }
        
        // Ensure we have UInt32 data
        let w_q_uint32: MLXArray
        if w_q.dtype == .uint32 {
            w_q_uint32 = w_q
        } else {
            print("WARNING: Converting quantized weights from \(w_q.dtype) to uint32")
            w_q_uint32 = w_q.asType(.uint32)
        }
        
        // Extract tensor dimensions (V = rows, D = packed columns)
        let V = w_q.shape[0]
        let D = w_q.shape[1]
        
        // Vectorised nibble unpack
        // Each UInt32 contains 8 packed 4-bit values (nibbles).
        // We broadcast 8 different right-shift amounts and mask with 0xF.
        let shiftAmounts: [UInt32] = [0, 4, 8, 12, 16, 20, 24, 28]
        let shifts = MLXArray(shiftAmounts).expandDims(at: 0).expandDims(at: 0) // [1,1,8]
        let expanded = w_q_uint32.expandDims(at: 2)                            // [V, D, 1]
        let shifted = MLX.rightShift(expanded, shifts)                          // [V, D, 8]
        let nibblesUInt32 = MLX.bitwiseAnd(shifted, MLXArray(UInt32(0xF)))      // [V, D, 8]
        let w_unpacked = nibblesUInt32.asType(.float32).reshaped([V, D * 8])    // [V, D*8]

        // Convert scales and biases to float32 for precise maths
        let scales_float32 = scales.asType(.float32)
        let biases_float32 = biases.asType(.float32)
        
        // Expand scales / biases to match unpacked shape
        let numGroups = (D * 8) / groupSize
        let scales_expanded = MLX.repeated(scales_float32.reshaped([V, numGroups, 1]), count: groupSize, axis: 2).reshaped([V, D * 8])
        
        let biases_expanded = MLX.repeated(biases_float32.reshaped([V, numGroups, 1]), count: groupSize, axis: 2).reshaped([V, D * 8])
        
        let dequantized = w_unpacked * scales_expanded + biases_expanded
        
        return dequantized.asType(scales.dtype) // Match original dtype
    }
}

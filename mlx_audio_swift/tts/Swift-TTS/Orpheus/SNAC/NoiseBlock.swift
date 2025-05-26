//
//  NoiseBlock.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 14/05/2025.
//
import Foundation
import MLX
import MLXNN

struct NoiseBlock {
    let linear: ConvWeighted
    
    init(dim: Int, weights: [String: MLXArray], basePath: String) {
        // Load weights for the linear layer within NoiseBlock
        // Example basePath: "decoder.model.layers.1.block.layers.1.linear"
        let weightG = weights["\(basePath).weight_g"]!
        let weightV = weights["\(basePath).weight_v"]!
        // Bias is false in Python implementation for NoiseBlock's WNConv1d
        self.linear = ConvWeighted(
            weightG: weightG,
            weightV: weightV,
            bias: nil, // No bias in Python's NoiseBlock WNConv1d
            padding: 0 // Padding should be 0 for kernel size 1
            // stride, dilation, groups default to 1
        )
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input shape is likely [N, C, T]
        let B = x.shape[0]
        let T = x.shape[2]
        
        // Generate noise [B, 1, T]
        let noise = MLXRandom.normal([B, 1, T])
        
        // Apply the linear transformation
        let h = linear(x, conv: MLX.conv1d)
        
        // Modulate noise by the linear output and add to input
        let n = noise * h
        return x + n
    }
}

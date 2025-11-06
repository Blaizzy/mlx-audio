//
//  ResidualUnit.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 14/05/2025.
//
import Foundation
import MLX
import MLXNN

struct ResidualUnit {
    let snake1Alpha: MLXArray
    let conv1: ConvWeighted
    let snake2Alpha: MLXArray
    let conv2: ConvWeighted
    
    // We need to pass the specific weights required by this unit
    init(dim: Int, dilation: Int, kernelSize: Int, groups: Int, weights: [String: MLXArray], basePath: String) {
        // Load weights for this specific ResidualUnit based on the basePath
        // Example basePath: "decoder.model.layers.1.block.layers.3.block.layers"
        self.snake1Alpha = weights["\(basePath).0.alpha"]!
        let conv1WeightG = weights["\(basePath).1.weight_g"]!
        let conv1WeightV = weights["\(basePath).1.weight_v"]!
        let conv1Bias = weights["\(basePath).1.bias"]
        let pad1 = ((kernelSize - 1) * dilation) / 2
        self.conv1 = ConvWeighted(
            weightG: conv1WeightG,
            weightV: conv1WeightV,
            bias: conv1Bias,
            padding: pad1,
            dilation: dilation,
            groups: groups
        )
        
        self.snake2Alpha = weights["\(basePath).2.alpha"]!
        let conv2WeightG = weights["\(basePath).3.weight_g"]!
        let conv2WeightV = weights["\(basePath).3.weight_v"]!
        let conv2Bias = weights["\(basePath).3.bias"]
        // Second conv has kernel_size=1, padding=0, dilation=1, groups=1 (implicitly)
        self.conv2 = ConvWeighted(
            weightG: conv2WeightG,
            weightV: conv2WeightV,
            bias: conv2Bias,
            padding: 0, // Kernel size 1 needs 0 padding
            dilation: 1,
            groups: 1 // Typically 1 for the final 1x1 conv
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var y = x
        // Ensure [B, C, T] for both
        if y.shape[1] != residual.shape[1] && y.shape[2] == residual.shape[1] {
            y = y.transposed(axes: [0, 2, 1])
        }

        if residual.shape[1] != y.shape[1] && residual.shape[2] == y.shape[1] {
            residual = residual.transposed(axes: [0, 2, 1])
        }

        // Apply the sequence: Snake -> Conv1 -> Snake -> Conv2
        if y.shape[1] != snake1Alpha.shape[1] {
            y = y.transposed(axes: [0, 2, 1])
        }
        y = SNACDecoder.snake(y, alpha: snake1Alpha) // Use static call

        y = conv1(y, conv: MLX.conv1d)

        if y.shape[1] != snake2Alpha.shape[1] {
            y = y.transposed(axes: [0, 2, 1])
        }
        y = SNACDecoder.snake(y, alpha: snake2Alpha) // Use static call

        y = conv2(y, conv: MLX.conv1d)

        // Crop residual if needed to match y's time dim
        let tRes = residual.shape[2]
        let tY = y.shape[2]

        if tRes != tY {
            let diff = tRes - tY
            if diff > 0 {
                let pad = diff / 2
                let end = tRes - (diff - pad)
                let b = residual.shape[0]
                let c = residual.shape[1]
                residual = residual[0..<b, 0..<c, pad..<end]
            }
        }
        return residual + y
    }
}

//
//  DecoderBlock.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 14/05/2025.
//
import Foundation
import MLX
import MLXNN

struct DecoderBlock {
    let snake1Alpha: MLXArray
    let convT: ConvWeightedTranspose1d
    let noiseBlock: NoiseBlock? // Optional noise block
    let residualUnit1: ResidualUnit
    let residualUnit2: ResidualUnit
    let residualUnit3: ResidualUnit

    init(inputDim: Int, outputDim: Int, stride: Int, noise: Bool, groups: Int, weights: [String: MLXArray], basePath: String) {
        self.snake1Alpha = weights["\(basePath).0.alpha"]!

        // Layer 1: WNConvTranspose1d
        let convTWeightG = weights["\(basePath).1.weight_g"]!
        let convTWeightV = weights["\(basePath).1.weight_v"]!
        let convTBias = weights["\(basePath).1.bias"]        
        let paddingT = Int(ceil(Double(stride) / 2.0))
        let outputPaddingT = stride % 2

        self.convT = ConvWeightedTranspose1d(
            weightG: convTWeightG, // Pass original weightG [input_dim, 1, 1]
            weightV: convTWeightV, // Pass original weightV [input_dim, kernel_size, output_dim/groups]
            bias: convTBias,
            stride: stride,
            padding: paddingT,
            outputPadding: outputPaddingT
        )

        // Layer 2 (Conditional): NoiseBlock
        var residualUnitBasePathStartIndex = 2 // Start index for ResidualUnits
        if noise {
            self.noiseBlock = NoiseBlock(dim: outputDim, weights: weights, basePath: "\(basePath).2.linear")
            residualUnitBasePathStartIndex = 3 // Shift index if NoiseBlock exists
        } else {
            self.noiseBlock = nil
        }

        // Layers 3, 4, 5 (or 2, 3, 4 if no noise): ResidualUnits
        let resBasePath1 = "\(basePath).\(residualUnitBasePathStartIndex).block.layers"
        self.residualUnit1 = ResidualUnit(dim: outputDim, dilation: 1, kernelSize: 7, groups: groups, weights: weights, basePath: resBasePath1)

        let resBasePath2 = "\(basePath).\(residualUnitBasePathStartIndex + 1).block.layers"
        self.residualUnit2 = ResidualUnit(dim: outputDim, dilation: 3, kernelSize: 7, groups: groups, weights: weights, basePath: resBasePath2)

        let resBasePath3 = "\(basePath).\(residualUnitBasePathStartIndex + 2).block.layers"
        self.residualUnit3 = ResidualUnit(dim: outputDim, dilation: 9, kernelSize: 7, groups: groups, weights: weights, basePath: resBasePath3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        // Always ensure [B, C, T] before any block logic
        if y.shape[1] != snake1Alpha.shape[1] && y.shape[2] == snake1Alpha.shape[1] {
            y = y.transposed(axes: [0, 2, 1])
        }
        
        y = SNACDecoder.snake(y, alpha: snake1Alpha) // Use static call
        
        y = convT(y)

        if let noiseBlock = noiseBlock {
            y = noiseBlock(y)
        }
        
        y = residualUnit1(y)
        y = residualUnit2(y)
        y = residualUnit3(y)
        
        return y
    }
}

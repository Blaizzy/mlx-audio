//
//  WNConv1d.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 14/05/2025.
//
import Foundation
import MLX
import MLXNN

class WNConv1d: Module {
    var weightG: MLXArray
    var weightV: MLXArray
    var bias: MLXArray?
    
    let kernelSize: Int
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int
    
    private static func normalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
        guard x.ndim == 3 else {
            fatalError("Input tensor must have 3 dimensions")
        }
        let axes = Array(0..<x.ndim).filter { $0 != exceptDim }
        let xSquared = MLX.pow(x, 2)
        let sumSquared = MLX.sum(xSquared, axes: axes, keepDims: true)
        return MLX.sqrt(sumSquared)
    }
    
    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weightG = MLXArray([])
        self.weightV = MLXArray([])
        self.bias = bias ? MLX.zeros([outChannels]) : nil
        
        super.init()
        
        let scale = sqrt(1.0 / Double(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels / groups]
        )
        self.weightG = Self.normalizeWeight(weightInit)
        self.weightV = weightInit / (self.weightG + 1e-12)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Ensure input is 3D: [batch, in_channels, time]
        let x3d: MLXArray
        if x.ndim == 2 {
            // [in_channels, time] -> [1, in_channels, time]
            x3d = x.reshaped([1, x.shape[0], x.shape[1]])
        } else {
            x3d = x
        }
        let xT = x3d.transposed(axes: [0, 2, 1]) // [batch, time, in_channels]
        let normV = Self.normalizeWeight(weightV)
        let weight = weightG * weightV / (normV + 1e-12)
        var y = MLX.conv1d(
            xT,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias = bias {
            y = y + bias
        }
        // Output shape is [batch, time, outChannels], transpose to [batch, outChannels, time]
        return y.transposed(axes: [0, 2, 1])
    }
}

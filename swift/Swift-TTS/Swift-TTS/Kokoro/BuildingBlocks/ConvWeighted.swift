//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

func computeNorm(
  x: MLXArray,
  p: Int,
  dim: [Int]? = nil,
  keepdim: Bool = false
) -> MLXArray {
  guard p == 1 || p == 2 else {
    fatalError("Only p-norms with p of 1 or 2 are supported")
  }

  let dimensions: [Int]
  if let dim = dim {
    dimensions = dim
  } else {
    dimensions = Array(0 ..< x.ndim)
  }

  if p == 1 {
    // L1 norm
    return MLX.sum(MLX.abs(x), axes: dimensions, keepDims: keepdim)
  } else {
    // L2 norm
    return MLX.sqrt(MLX.sum(x * x, axes: dimensions, keepDims: keepdim))
  }
}

func weightNorm(
  weightV: MLXArray,
  weightG: MLXArray,
  dim: Int? = nil
) -> MLXArray {
  let rank = weightV.shape.count

  var axes: [Int]

  if let dim = dim {
    var adjustedDim = dim
    if dim < 0 {
      adjustedDim += rank
    }

    axes = Array(0 ..< rank)
    if adjustedDim != -1 {
      axes.removeAll(where: { $0 == adjustedDim })
    }
  } else {
    axes = Array(0 ..< rank)
  }

  let normV = computeNorm(x: weightV, p: 2, dim: axes, keepdim: true)

  let normalizedWeight = weightV / (normV + 1e-7) // Add epsilon for numerical stability
  return normalizedWeight * weightG
}

/// Conv1d with weight normalization
class ConvWeighted: Module {
  var weightG: MLXArray
  var weightV: MLXArray
  var bias: MLXArray?

  let stride: Int
  let padding: Int
  let dilation: Int
  let groups: Int

  init(
    weightG: MLXArray,
    weightV: MLXArray,
    bias: MLXArray?,
    stride: Int = 1,
    padding: Int = 1,
    dilation: Int = 1,
    groups: Int = 1
  ) {
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups

    // Store parameters
    self.weightG = weightG
    self.weightV = weightV
    self.bias = bias

    super.init()
  }

  public func callAsFunction(_ x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
    let weight = weightNorm(weightV: weightV, weightG: weightG, dim: 0)
    bias = bias?.reshaped([1, 1, -1])

    func applyConv(x: MLXArray, weightToUse: MLXArray) -> MLXArray {
      let result = conv(
        x,
        weightToUse,
        self.stride,
        padding,
        dilation,
        groups,
        .default
      )

      if let bias = bias {
        return result + bias
      }
      return result
    }

    if x.shape.last == weight.shape.last || groups > 1 {
      return applyConv(x: x, weightToUse: weight)
    } else {
      return applyConv(x: x, weightToUse: weight.transposed())
    }
  }
}

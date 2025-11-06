//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdaLayerNorm: Module {
  let eps: Float
  let fc: Linear

  init(eps: Float = 1e-5, weight: MLXArray, bias: MLXArray?) {
    self.eps = eps
    fc = Linear(weight: weight, bias: bias)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    let h = fc(s)
    let reshaped = h.reshaped([h.shape[0], h.shape[1], 1])
    let split = reshaped.split(parts: 2, axis: 1)
    let gamma = split[0].transposed(2, 0, 1)
    let beta = split[1].transposed(2, 0, 1)

    let mean = MLX.mean(x, axes: [-1], keepDims: true)
    let variance = MLX.variance(x, axes: [-1], keepDims: true)
    let normalized = (x - mean) / MLX.sqrt(variance + eps)

    return (1 + gamma) * normalized + beta
  }
}

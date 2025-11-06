//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXFast
import MLXNN

class LayerNormInference: Module {
  public let eps: Float
  public let weight: MLXArray?
  public let bias: MLXArray?

  public init(weight: MLXArray, bias: MLXArray?, eps: Float = 1e-5) {
    self.weight = weight
    self.bias = bias
    self.eps = eps
  }

  open func callAsFunction(_ x: MLXArray) -> MLXArray {
    MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
  }
}

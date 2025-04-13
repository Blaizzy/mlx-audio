//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertOutput {
  let dense: Linear
  let layerNorm: LayerNorm

  init(config: AlbertModelArgs) {
    dense = Linear(config.intermediateSize, config.hiddenSize)
    layerNorm = LayerNorm(
      dimensions: config.hiddenSize,
      eps: config.layerNormEps
    )
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    inputTensor: MLXArray
  ) -> MLXArray {
    var output = dense(hiddenStates)
    output = layerNorm(output + inputTensor)
    return output
  }
}

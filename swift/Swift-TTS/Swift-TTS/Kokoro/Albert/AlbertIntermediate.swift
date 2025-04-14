//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertIntermediate {
  let dense: Linear

  init(config: AlbertModelArgs) {
    dense = Linear(config.hiddenSize, config.intermediateSize)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    var output = dense(hiddenStates)
    output = MLXNN.gelu(output)
    return output
  }
}

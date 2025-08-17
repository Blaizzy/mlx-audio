//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertLayerGroup {
  let albertLayers: [AlbertLayer]

  init(config: AlbertModelArgs, layerNum: Int, weights: [String: MLXArray]) {
    var layers: [AlbertLayer] = []
    for innerGroupNum in 0 ..< config.innerGroupNum {
      layers.append(AlbertLayer(weights: weights, config: config, layerNum: layerNum, innerGroupNum: innerGroupNum))
    }
    albertLayers = layers
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    var output = hiddenStates
    for layer in albertLayers {
      output = layer(output, attentionMask: attentionMask)
    }
    return output
  }
}

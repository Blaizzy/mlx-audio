//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class UpSample1d {
  private let layerType: String
  private let interpolate: Upsample

  init(layerType: String) {
    self.layerType = layerType
    interpolate = Upsample(
      scaleFactor: 2.0,
      mode: .nearest
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    if layerType == "none" {
      return x
    } else {
      return interpolate(x)
    }
  }
}

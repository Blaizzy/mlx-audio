//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Conv1dInference {
  public let weight: MLXArray
  public let bias: MLXArray?
  public let padding: Int
  public let dilation: Int
  public let stride: Int
  public let groups: Int

  public init(
    inputChannels _: Int,
    outputChannels _: Int,
    kernelSize _: Int,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    groups: Int = 1,
    weight: MLXArray,
    bias: MLXArray? = nil
  ) {
    self.weight = weight
    self.bias = bias
    self.padding = padding
    self.dilation = dilation
    self.stride = stride
    self.groups = groups
  }

  open func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = conv1d(
      x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups
    )

    if let bias {
      y = y + bias
    }
    return y
  }
}

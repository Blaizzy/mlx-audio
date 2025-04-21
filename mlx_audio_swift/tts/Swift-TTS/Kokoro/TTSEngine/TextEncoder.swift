//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Text encoder
class TextEncoder {
  let embedding: Embedding
  let cnn: [[Module]]
  let lstm: LSTM

  init(weights: [String: MLXArray], channels: Int, kernelSize: Int, depth: Int, nSymbols _: Int, actv: Module = LeakyReLU(negativeSlope: 0.2)) {
    embedding = Embedding(weight: weights["text_encoder.embedding.weight"]!)
    let padding = (kernelSize - 1) / 2

    var cnnLayers: [[Module]] = []
    for i in 0 ..< depth {
      cnnLayers.append([
        ConvWeighted(
          weightG: weights["text_encoder.cnn.\(i).0.weight_g"]!,
          weightV: weights["text_encoder.cnn.\(i).0.weight_v"]!,
          bias: weights["text_encoder.cnn.\(i).0.bias"]!,
          padding: padding
        ),
        LayerNormInference(
          weight: weights["text_encoder.cnn.\(i).1.gamma"]!,
          bias: weights["text_encoder.cnn.\(i).1.beta"]!
        ),
        actv,
      ])
    }
    cnn = cnnLayers

    lstm = LSTM(
      inputSize: channels,
      hiddenSize: channels / 2,
      wxForward: weights["text_encoder.lstm.weight_ih_l0"]!,
      whForward: weights["text_encoder.lstm.weight_hh_l0"]!,
      biasIhForward: weights["text_encoder.lstm.bias_ih_l0"]!,
      biasHhForward: weights["text_encoder.lstm.bias_hh_l0"]!,
      wxBackward: weights["text_encoder.lstm.weight_ih_l0_reverse"]!,
      whBackward: weights["text_encoder.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: weights["text_encoder.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: weights["text_encoder.lstm.bias_hh_l0_reverse"]!
    )
  }

  public func callAsFunction(_ x: MLXArray, inputLengths _: MLXArray, m: MLXArray) -> MLXArray {
    var x = embedding(x)
    x = x.transposed(0, 2, 1)
    let mask = m.expandedDimensions(axis: 1)
    x = MLX.where(mask, 0.0, x)

    for convBlock in cnn {
      for layer in convBlock {
        if layer is ConvWeighted || layer is LayerNormInference {
          x = MLX.swappedAxes(x, 2, 1)
          if let convWeighted = layer as? ConvWeighted {
            x = convWeighted(x, conv: MLX.conv1d)
          } else if let layer = layer as? LayerNormInference {
            x = layer(x)
          }
          x = MLX.swappedAxes(x, 2, 1)
        } else if let layer = layer as? LeakyReLU {
          x = layer(x)
        } else {
          fatalError("Unsupported layer type")
        }
        x = MLX.where(mask, 0.0, x)
      }
    }

    x = MLX.swappedAxes(x, 2, 1)
    let (lstmOutput, _) = lstm(x)
    x = MLX.swappedAxes(lstmOutput, 2, 1)

    let xPad = MLX.zeros([x.shape[0], x.shape[1], mask.shape[mask.shape.count - 1]])
    xPad.update(x)

    return MLX.where(mask, 0.0, xPad)
  }
}

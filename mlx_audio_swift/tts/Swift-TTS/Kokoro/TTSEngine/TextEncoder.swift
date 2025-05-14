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
        var features = embedding(x)
        features = features.transposed(0, 2, 1)
        let mask = m.expandedDimensions(axis: 1)
        features = MLX.where(mask, 0.0, features)

        for convBlock in cnn {
            for layer in convBlock {
                if layer is ConvWeighted || layer is LayerNormInference {
                    features = MLX.swappedAxes(features, 2, 1)
                    if let conv = layer as? ConvWeighted {
                        features = conv(features, conv: MLX.conv1d)
                    } else if let norm = layer as? LayerNormInference {
                        features = norm(features)
                    }
                    features = MLX.swappedAxes(features, 2, 1)
                } else if let activation = layer as? LeakyReLU {
                    features = activation(features)
                } else {
                    fatalError("Unsupported layer type")
                }
                features = MLX.where(mask, 0.0, features)
            }
        }

        features = MLX.swappedAxes(features, 2, 1)
        let (lstmOutput, _) = lstm(features)
        features = MLX.swappedAxes(lstmOutput, 2, 1)

        return MLX.where(mask, 0.0, features)
    }
}

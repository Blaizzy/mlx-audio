//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Duration Encoder creates nlayers pairs of [LSTM, AdaLayerNorm]
class DurationEncoder {
  var lstms: [Module] = []

  init(weights: [String: MLXArray], dModel: Int, styDim: Int, nlayers: Int) {
    // Create nlayers pairs of [LSTM, AdaLayerNorm] = nlayers * 2 total layers
    for i in 0 ..< nlayers {
      let lstmIdx = i * 2
      let adaLnIdx = i * 2 + 1

      // LSTM layer
      lstms.append(
        LSTM(inputSize: dModel + styDim,
             hiddenSize: dModel / 2,
             wxForward: weights["predictor.text_encoder.lstms.\(lstmIdx).weight_ih_l0"]!, // Crash: Task 37: Fatal error: Unexpectedly found nil while unwrapping an Optional value
             whForward: weights["predictor.text_encoder.lstms.\(lstmIdx).weight_hh_l0"]!,
             biasIhForward: weights["predictor.text_encoder.lstms.\(lstmIdx).bias_ih_l0"]!,
             biasHhForward: weights["predictor.text_encoder.lstms.\(lstmIdx).bias_hh_l0"]!,
             wxBackward: weights["predictor.text_encoder.lstms.\(lstmIdx).weight_ih_l0_reverse"]!,
             whBackward: weights["predictor.text_encoder.lstms.\(lstmIdx).weight_hh_l0_reverse"]!,
             biasIhBackward: weights["predictor.text_encoder.lstms.\(lstmIdx).bias_ih_l0_reverse"]!,
             biasHhBackward: weights["predictor.text_encoder.lstms.\(lstmIdx).bias_hh_l0_reverse"]!)
      )

      // AdaLayerNorm layer
      lstms.append(AdaLayerNorm(weight: weights["predictor.text_encoder.lstms.\(adaLnIdx).fc.weight"]!,
                                bias: weights["predictor.text_encoder.lstms.\(adaLnIdx).fc.bias"]!))
    }
  }

  func callAsFunction(_ x: MLXArray, style: MLXArray, textLengths _: MLXArray, m: MLXArray) -> MLXArray {
    var x = x.transposed(2, 0, 1)
    let s = MLX.broadcast(style, to: [x.shape[0], x.shape[1], style.shape[style.shape.count - 1]])
    x = MLX.concatenated([x, s], axis: -1)
    x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(1, 0, 2), MLXArray.zeros(like: x), x)
    x = x.transposed(1, 2, 0)

    for block in lstms {
      if let adaLayerNorm = block as? AdaLayerNorm {
        x = adaLayerNorm(x.transposed(0, 2, 1), style).transposed(0, 2, 1)
        x = MLX.concatenated([x, s.transposed(1, 2, 0)], axis: 1)
        x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(0, 2, 1), MLXArray.zeros(like: x), x)
      } else if let lstm = block as? LSTM {
        x = x.transposed(0, 2, 1)[0]
        let (lstmOutput, _) = lstm(x)
        x = lstmOutput.transposed(0, 2, 1)
        let xPad = MLXArray.zeros([x.shape[0], x.shape[1], m.shape[m.shape.count - 1]])
        xPad[0 ..< x.shape[0], 0 ..< x.shape[1], 0 ..< x.shape[2]] = x
        x = xPad
      }
    }

    return x.transposed(0, 2, 1)
  }
}

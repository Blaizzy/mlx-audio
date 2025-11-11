//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Prosody Predictor from StyleTTS2
class ProsodyPredictor {
  let shared: LSTM
  let F0: [AdainResBlk1d]
  let N: [AdainResBlk1d]
  let F0Proj: Conv1dInference
  let NProj: Conv1dInference

  public init(weights: [String: MLXArray], styleDim: Int, dHid: Int) {
    shared = LSTM(
      inputSize: dHid + styleDim,
      hiddenSize: dHid / 2,
      wxForward: weights["predictor.shared.weight_ih_l0"]!,
      whForward: weights["predictor.shared.weight_hh_l0"]!,
      biasIhForward: weights["predictor.shared.bias_ih_l0"]!,
      biasHhForward: weights["predictor.shared.bias_hh_l0"]!,
      wxBackward: weights["predictor.shared.weight_ih_l0_reverse"]!,
      whBackward: weights["predictor.shared.weight_hh_l0_reverse"]!,
      biasIhBackward: weights["predictor.shared.bias_ih_l0_reverse"]!,
      biasHhBackward: weights["predictor.shared.bias_hh_l0_reverse"]!
    )

    F0 = [
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
    ]

    N = [
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
    ]

    F0Proj = Conv1dInference(
      inputChannels: dHid / 2,
      outputChannels: 1,
      kernelSize: 1,
      padding: 0,
      weight: weights["predictor.F0_proj.weight"]!,
      bias: weights["predictor.F0_proj.bias"]!
    )

    NProj = Conv1dInference(
      inputChannels: dHid / 2,
      outputChannels: 1,
      kernelSize: 1,
      padding: 0,
      weight: weights["predictor.N_proj.weight"]!,
      bias: weights["predictor.N_proj.bias"]!
    )
  }

  func F0NTrain(x: MLXArray, s: MLXArray) -> (MLXArray, MLXArray) {
    let (x1, _) = shared(x.transposed(0, 2, 1))

    // F0 prediction
    var F0Val = x1.transposed(0, 2, 1)
    for block in F0 {
      F0Val = block(x: F0Val, s: s)
    }
    F0Val = MLX.swappedAxes(F0Val, 2, 1)
    F0Val = F0Proj(F0Val)
    F0Val = MLX.swappedAxes(F0Val, 2, 1)

    // N prediction
    var NVal = x1.transposed(0, 2, 1)
    for block in N {
      NVal = block(x: NVal, s: s)
    }
    NVal = MLX.swappedAxes(NVal, 2, 1)
    NVal = NProj(NVal)
    NVal = MLX.swappedAxes(NVal, 2, 1)

    return (F0Val.squeezed(axis: 1), NVal.squeezed(axis: 1))
  }
}

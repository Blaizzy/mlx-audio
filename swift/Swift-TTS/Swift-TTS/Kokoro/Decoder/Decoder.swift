//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Decoder {
  private let encode: AdainResBlk1d
  private var decode: [AdainResBlk1d] = []
  private let F0Conv: ConvWeighted
  private let NConv: ConvWeighted
  private let asrRes: [ConvWeighted]
  private let generator: Generator

  init(
    weights: [String: MLXArray],
    dimIn: Int,
    styleDim: Int,
    dimOut _: Int,
    resblockKernelSizes: [Int],
    upsampleRates: [Int],
    upsampleInitialChannel: Int,
    resblockDilationSizes: [[Int]],
    upsampleKernelSizes: [Int],
    genIstftNFft: Int,
    genIstftHopSize: Int
  ) {
    encode = AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.encode", dimIn: dimIn + 2, dimOut: 1024, styleDim: styleDim)

    decode.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.0", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim))
    decode.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.1", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim))
    decode.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.2", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim))
    decode.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.3", dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: styleDim, upsample: "true"))

    F0Conv = ConvWeighted(
      weightG: weights["decoder.F0_conv.weight_g"]!,
      weightV: weights["decoder.F0_conv.weight_v"]!,
      bias: weights["decoder.F0_conv.bias"]!,
      stride: 2,
      padding: 1,
      groups: 1
    )
    NConv = ConvWeighted(
      weightG: weights["decoder.N_conv.weight_g"]!,
      weightV: weights["decoder.N_conv.weight_v"]!,
      bias: weights["decoder.N_conv.bias"]!,
      stride: 2,
      padding: 1,
      groups: 1
    )

    asrRes = [ConvWeighted(
      weightG: weights["decoder.asr_res.0.weight_g"]!,
      weightV: weights["decoder.asr_res.0.weight_v"]!,
      bias: weights["decoder.asr_res.0.bias"]!,
      padding: 0
    )]

    generator = Generator(
      weights: weights,
      styleDim: styleDim,
      resblockKernelSizes: resblockKernelSizes,
      upsampleRates: upsampleRates,
      upsampleInitialChannel: upsampleInitialChannel,
      resblockDilationSizes: resblockDilationSizes,
      upsampleKernelSizes: upsampleKernelSizes,
      genIstftNFft: genIstftNFft,
      genIstftHopSize: genIstftHopSize
    )
  }

  func callAsFunction(asr: MLXArray, F0Curve: MLXArray, N: MLXArray, s: MLXArray) -> MLXArray {
    BenchmarkTimer.shared.create(id: "Encode", parent: "Decoder")

    let F0CurveSwapped = MLX.swappedAxes(F0Curve.reshaped([F0Curve.shape[0], 1, F0Curve.shape[1]]), 2, 1)
    let F0 = MLX.swappedAxes(F0Conv(F0CurveSwapped, conv: MLX.conv1d), 2, 1)

    let NSwapped = MLX.swappedAxes(N.reshaped([N.shape[0], 1, N.shape[1]]), 2, 1)
    let NProcessed = MLX.swappedAxes(NConv(NSwapped, conv: MLX.conv1d), 2, 1)

    var x = MLX.concatenated([asr, F0, NProcessed], axis: 1)
    x = encode(x: x, s: s)

    let asrResidual = MLX.swappedAxes(asrRes[0](MLX.swappedAxes(asr, 2, 1), conv: MLX.conv1d), 2, 1)
    var res = true

    x.eval()
    BenchmarkTimer.shared.stop(id: "Encode")

    BenchmarkTimer.shared.create(id: "Blocks", parent: "Decoder")

    for block in decode {
      if res {
        x = MLX.concatenated([x, asrResidual, F0, NProcessed], axis: 1)
      }
      x = block(x: x, s: s)

      if block.upsampleType != "none" {
        res = false
      }
    }

    x.eval()
    BenchmarkTimer.shared.stop(id: "Blocks")

    return generator(x, s, F0Curve)
  }
}

//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXRandom

class SourceModuleHnNSF: Module {
  private let sineAmp: Float
  private let noiseStd: Float
  private let lSinGen: SineGen
  private let lLinear: Linear

  init(
    weights: [String: MLXArray],
    samplingRate: Int,
    upsampleScale: Float,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    addNoiseStd: Float = 0.003,
    voicedThreshold: Float = 0
  ) {
    self.sineAmp = sineAmp
    noiseStd = addNoiseStd

    // To produce sine waveforms
    lSinGen = SineGen(
      sampRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: harmonicNum,
      sineAmp: sineAmp,
      noiseStd: addNoiseStd,
      voicedThreshold: voicedThreshold
    )

    // To merge source harmonics into a single excitation
    lLinear = Linear(
      weight: weights["decoder.generator.m_source.l_linear.weight"]!,
      bias: weights["decoder.generator.m_source.l_linear.bias"]!
    )

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let (sineWavs, uv, _) = lSinGen(x)
    let sineMerge = tanh(lLinear(sineWavs))

    let noise = MLXRandom.normal(uv.shape) * (sineAmp / 3)

    // Note: Turns out we don't need noise or uv for that matter
    return (sineMerge, noise, uv)
  }
}

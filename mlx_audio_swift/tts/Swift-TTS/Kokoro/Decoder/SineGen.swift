//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXRandom

class SineGen {
  private let sineAmp: Float
  private let noiseStd: Float
  private let harmonicNum: Int
  private let dim: Int
  private let samplingRate: Int
  private let voicedThreshold: Float
  private let upsampleScale: Float

  init(
    sampRate: Int,
    upsampleScale: Float,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    noiseStd: Float = 0.003,
    voicedThreshold: Float = 0
  ) {
    self.sineAmp = sineAmp
    self.noiseStd = noiseStd
    self.harmonicNum = harmonicNum
    dim = harmonicNum + 1
    samplingRate = sampRate
    self.voicedThreshold = voicedThreshold
    self.upsampleScale = upsampleScale
  }

  private func _f02uv(_ f0: MLXArray) -> MLXArray {
    let arr = f0 .> voicedThreshold
    return arr.asType(.float32)
  }

  private func _f02sine(_ f0Values: MLXArray) -> MLXArray {
    // Ignore integer part (% 1 is there for a purpose :)
    var radValues = (f0Values / Float(samplingRate)) % 1

    // Random phase noise
    let randIni = MLXRandom.normal([f0Values.shape[0], f0Values.shape[2]])
    randIni[0..., 0] = MLXArray(0.0)
    radValues[0 ..< radValues.shape[0], 0, 0 ..< radValues.shape[2]] = radValues[0 ..< radValues.shape[0], 0, 0 ..< radValues.shape[2]] + randIni

    radValues = interpolate(
      input: radValues.transposed(0, 2, 1),
      scaleFactor: [1 / Float(upsampleScale)],
      mode: "linear"
    ).transposed(0, 2, 1)

    var phase = MLX.cumsum(radValues, axis: 1) * 2 * Float.pi
    phase = interpolate(
      input: phase.transposed(0, 2, 1) * Float(upsampleScale),
      scaleFactor: [Float(upsampleScale)],
      mode: "linear"
    ).transposed(0, 2, 1)

    return MLX.sin(phase)
  }

  func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    // let f0Buf = MLX.zeros([f0.shape[0], f0.shape[1], dim])

    // Fundamental component
    let range = MLXArray(1 ... harmonicNum + 1).asType(.float32)
    let fn = f0 * range.reshaped([1, 1, range.shape[0]])

    // Generate sine waveforms
    let sineWaves = _f02sine(fn) * sineAmp

    // Generate UV signal
    let uv = _f02uv(f0)

    // Generate noise
    let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
    let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)

    let result = sineWaves * uv + noise
    return (result, uv, noise)
  }
}

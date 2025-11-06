//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Generator {
  let numKernels: Int
  let numUpsamples: Int
  let mSource: SourceModuleHnNSF
  let f0Upsample: Upsample
  let postNFFt: Int
  var noiseConvs: [Conv1dInference]
  var noiseRes: [AdaINResBlock1]
  var ups: [ConvWeighted]
  var resBlocks: [AdaINResBlock1]
  let convPost: ConvWeighted
  let reflectionPad: ReflectionPad1d
  let stft: MLXSTFT

  init(weights: [String: MLXArray],
       styleDim: Int,
       resblockKernelSizes: [Int],
       upsampleRates: [Int],
       upsampleInitialChannel: Int,
       resblockDilationSizes: [[Int]],
       upsampleKernelSizes: [Int],
       genIstftNFft: Int,
       genIstftHopSize: Int)
  {
    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count

    let upsampleScaleNum = MLX.product(MLXArray(upsampleRates)) * genIstftHopSize
    let upsampleScaleNumVal: Int = upsampleScaleNum.item()

    mSource = SourceModuleHnNSF(
      weights: weights,
      samplingRate: 24000,
      upsampleScale: upsampleScaleNum.item(),
      harmonicNum: 8,
      voicedThreshold: 10
    )

    f0Upsample = Upsample(scaleFactor: .float(Float(upsampleScaleNumVal)))

    noiseConvs = []
    noiseRes = []
    ups = []

    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      ups.append(
        ConvWeighted(
          weightG: weights["decoder.generator.ups.\(i).weight_g"]!,
          weightV: weights["decoder.generator.ups.\(i).weight_v"]!,
          bias: weights["decoder.generator.ups.\(i).bias"]!,
          stride: u,
          padding: (k - u) / 2
        )
      )
    }

    resBlocks = []
    for i in 0 ..< ups.count {
      let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
      for (j, (k, d)) in zip(resblockKernelSizes, resblockDilationSizes).enumerated() {
        resBlocks.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.resblocks.\((i * resblockKernelSizes.count) + j)",
            channels: ch,
            kernelSize: k,
            dilation: d,
            styleDim: styleDim
          )
        )
      }

      let cCur = ch
      if i + 1 < upsampleRates.count {
        let strideF0: Int = MLX.product(MLXArray(upsampleRates)[(i + 1)...]).item()
        noiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: strideF0 * 2,
            stride: strideF0,
            padding: (strideF0 + 1) / 2,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )

        noiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 7,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      } else {
        noiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: 1,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )
        noiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 11,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      }
    }

    postNFFt = genIstftNFft

    convPost = ConvWeighted(
      weightG: weights["decoder.generator.conv_post.weight_g"]!,
      weightV: weights["decoder.generator.conv_post.weight_v"]!,
      bias: weights["decoder.generator.conv_post.bias"]!,
      stride: 1,
      padding: 3
    )

    reflectionPad = ReflectionPad1d(padding: (1, 0))

    stft = MLXSTFT(
      filterLength: genIstftNFft,
      hopLength: genIstftHopSize,
      winLength: genIstftNFft
    )
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ F0Curve: MLXArray) -> MLXArray {
    BenchmarkTimer.shared.create(id: "GeneratorStart", parent: "Decoder")

    var f0New = F0Curve[.newAxis, 0..., 0...].transposed(0, 2, 1)
    f0New = f0Upsample(f0New)

    var (harSource, _, _) = mSource(f0New)

    harSource = MLX.squeezed(harSource.transposed(0, 2, 1), axis: 1)
    let (harSpec, harPhase) = stft.transform(inputData: harSource)
    var har = MLX.concatenated([harSpec, harPhase], axis: 1)
    har = MLX.swappedAxes(har, 2, 1)

    var newX = x
    for i in 0 ..< numUpsamples {
      newX = LeakyReLU(negativeSlope: 0.1)(newX)
      var xSource = noiseConvs[i](har)
      xSource = MLX.swappedAxes(xSource, 2, 1)
      xSource = noiseRes[i](xSource, s)

      newX = MLX.swappedAxes(newX, 2, 1)
    let upsi = ups[i]
        newX = upsi.callAsFunction(newX, conv: { a, b, c, d, e, f, g in
             MLX.convTransposed1d(a, b, stride: c, padding: d, dilation: e, outputPadding: 0, groups: f, stream: g)
        })
        newX = MLX.swappedAxes(newX, 2, 1)

      if i == numUpsamples - 1 {
        newX = reflectionPad(newX)
      }
      newX = newX + xSource

      var xs: MLXArray?
      for j in 0 ..< numKernels {
        if xs == nil {
          xs = resBlocks[i * numKernels + j](newX, s)
        } else {
          let temp = resBlocks[i * numKernels + j](newX, s)
          xs = xs! + temp
        }
      }
      newX = xs! / numKernels
    }

    newX = LeakyReLU(negativeSlope: 0.01)(newX)

    newX = MLX.swappedAxes(newX, 2, 1)
    newX = convPost(newX, conv: MLX.conv1d)
    newX = MLX.swappedAxes(newX, 2, 1)

    let spec = MLX.exp(newX[0..., 0 ..< (postNFFt / 2 + 1), 0...])
    let phase = MLX.sin(newX[0..., (postNFFt / 2 + 1)..., 0...])

    spec.eval()
    phase.eval()

    BenchmarkTimer.shared.stop(id: "GeneratorStart")

    BenchmarkTimer.shared.create(id: "InverseSTFT", parent: "Decoder")
    let result = stft.inverse(magnitude: spec, phase: phase)
    result.eval()
    BenchmarkTimer.shared.stop(id: "InverseSTFT")

    return result
  }
}

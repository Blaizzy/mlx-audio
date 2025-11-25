//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Hanning window implementation to replace np.hanning
func hanning(length: Int) -> MLXArray {
  if length == 1 {
    return MLXArray(1.0)
  }

  let n = MLXArray(Array(stride(from: Float(1 - length), to: Float(length), by: 2.0)))
  let factor = .pi / Float(length - 1)
  return 0.5 + 0.5 * cos(n * factor)
}

// Unwrap implementation to replace np.unwrap
func unwrap(p: MLXArray) -> MLXArray {
  let period: Float = 2.0 * .pi
  let discont: Float = period / 2.0

  let pDiff1 = p[0..., 0 ..< p.shape[1] - 1]
  let pDiff2 = p[0..., 1 ..< p.shape[1]]

  let pDiff = pDiff2 - pDiff1

  let intervalHigh: Float = period / 2.0
  let intervalLow: Float = -intervalHigh

  var pDiffMod = pDiff - intervalLow
  pDiffMod = (((pDiffMod % period) + period) % period) + intervalLow

  let ddSignArray = MLX.where(pDiff .> 0, intervalHigh, pDiffMod)

  pDiffMod = MLX.where(pDiffMod .== intervalLow, ddSignArray, pDiffMod)

  var phCorrect = pDiffMod - pDiff
  phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)

  return MLX.concatenated([p[0..., 0 ..< 1], p[0..., 1...] + phCorrect.cumsum(axis: 1)], axis: 1)
}

func getWindow(window: Any, winLen: Int, nFft: Int) -> MLXArray {
    var w: MLXArray
    if let windowStr = window as? String {
        if windowStr.lowercased() == "hann" {
            w = hanning(length: winLen + 1)[0 ..< winLen]
        } else {
            fatalError("Only hanning is supported for window, not \(windowStr)")
        }
    } else if let windowArray = window as? MLXArray {
        w = windowArray
    } else {
        fatalError("Window must be a string or MLXArray")
    }

    if w.shape[0] < nFft {
        let padSize = nFft - w.shape[0]
        w = MLX.concatenated([w, MLXArray.zeros([padSize])], axis: 0)
    }
    return w
}

func mlxStft(
  x: MLXArray,
  nFft: Int = 800,
  hopLength: Int? = nil,
  winLength: Int? = nil,
  window: Any = "hann",
  center: Bool = true,
  padMode: String = "reflect"
) -> MLXArray {
  let hopLen = hopLength ?? nFft / 4
  let winLen = winLength ?? nFft

  let w = getWindow(window: window, winLen: winLen, nFft: nFft)

  func pad(_ x: MLXArray, padding: Int, padMode: String = "reflect") -> MLXArray {
    if padMode == "constant" {
      return MLX.padded(x, width: [padding, padding])
    } else if padMode == "reflect" {
      let prefix = x[1 ..< padding + 1][.stride(by: -1)]
      let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
      return MLX.concatenated([prefix, x, suffix])
    } else {
      fatalError("Invalid pad mode \(padMode)")
    }
  }

  var xArray = x

  if center {
    xArray = pad(xArray, padding: nFft / 2, padMode: padMode)
  }

  let numFrames = 1 + (xArray.shape[0] - nFft) / hopLen
  if numFrames <= 0 {
    fatalError("Input is too short")
  }

  let shape: [Int] = [numFrames, nFft]
  let strides: [Int] = [hopLen, 1]

  let frames = MLX.asStrided(xArray, shape, strides: strides)

  let spec = MLXFFT.rfft(frames * w)
  return spec.transposed(1, 0)
}


func mlxIstft(
  x: MLXArray,
  hopLength: Int? = nil,
  winLength: Int? = nil,
  window: Any = "hann",
  center: Bool = true
) -> MLXArray {
  let winLen = winLength ?? ((x.shape[1] - 1) * 2)
  let hopLen = hopLength ?? (winLen / 4)

  let w = getWindow(window: window, winLen: winLen, nFft: winLen)

  let xTransposed = x.transposed(1, 0)
  let numFrames = xTransposed.shape[0]
  let t = (numFrames - 1) * hopLen + winLen

  // Inverse FFT of each frame
  let framesTime = MLXFFT.irfft(xTransposed, axis: 1)

  // Compute frame offsets and indices for overlap-add
  let frameOffsets = MLXArray(Array(stride(from: 0, to: numFrames * hopLen, by: hopLen)))
  let winIndices = MLXArray(Array(0..<winLen))

  // Create indices matrix: [numFrames, winLen] where each row is frameOffset + [0, 1, ..., winLen-1]
  let indices = frameOffsets.expandedDimensions(axis: 1) + winIndices.expandedDimensions(axis: 0)
  let indicesFlat = indices.reshaped([-1]).asType(.int32)

  // Prepare updates
  let updatesReconstructed = (framesTime * w).reshaped([-1])
  let updatesWindow = MLX.tiled(w.expandedDimensions(axis: 0), repetitions: [numFrames, 1]).reshaped([-1])

  // Initialize output buffers
  var reconstructed = MLXArray.zeros([t])
  var windowSum = MLXArray.zeros([t])

  // Scatter-add using indexing
  reconstructed = reconstructed.at[indicesFlat].add(updatesReconstructed)
  windowSum = windowSum.at[indicesFlat].add(updatesWindow)

  // Normalize by window sum (avoid division by zero)
  reconstructed = MLX.where(windowSum .!= 0, reconstructed / windowSum, reconstructed)

  // Remove padding if centered
  if center {
    reconstructed = reconstructed[winLen / 2 ..< (reconstructed.shape[0] - winLen / 2)]
  }

  return reconstructed
}

class MLXSTFT {
  let filterLength: Int
  let hopLength: Int
  let winLength: Int
  let window: String

  var magnitude: MLXArray?
  var phase: MLXArray?

  init(filterLength: Int = 800, hopLength: Int = 200, winLength: Int = 800, window: String = "hann") {
    self.filterLength = filterLength
    self.hopLength = hopLength
    self.winLength = winLength
    self.window = window
  }

  func transform(inputData: MLXArray) -> (MLXArray, MLXArray) {
    var audioArray = inputData
    if audioArray.ndim == 1 {
        audioArray = audioArray.expandedDimensions(axis: 0)
    }

    var magnitudes: [MLXArray] = []
    var phases: [MLXArray] = []

    // Process sequentially - MLX handles GPU parallelism internally
    for batchIdx in 0..<audioArray.shape[0] {
      let stft = mlxStft(
        x: audioArray[batchIdx],
        nFft: filterLength,
        hopLength: hopLength,
        winLength: winLength,
        window: window,
        center: true,
        padMode: "reflect"
      )
      magnitudes.append(MLX.abs(stft))
      phases.append(MLX.atan2(stft.imaginaryPart(), stft.realPart()))
    }

    let magnitudesStacked = MLX.stacked(magnitudes, axis: 0)
    let phasesStacked = MLX.stacked(phases, axis: 0)

    return (magnitudesStacked, phasesStacked)
  }

  func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
    var reconstructed: [MLXArray] = []

//    let unwrapTimer = BenchmarkTimer.shared.create(id: "Unwrap", parent: "InverseSTFT")!
//    let fftTimer = BenchmarkTimer.shared.create(id: "Istft", parent: "InverseSTFT")!

    for batchIdx in 0 ..< magnitude.shape[0] {
//      unwrapTimer.startTimer()
      let phaseCont = unwrap(p: phase[batchIdx])

      // Combine magnitude and phase
      let stft = magnitude[batchIdx] * MLX.exp(MLXArray(real: 0, imaginary: 1) * phaseCont)
      stft.eval()
//      unwrapTimer.stop()

//      fftTimer.startTimer()
      // Inverse STFT
      let audio = mlxIstft(
        x: stft,
        hopLength: hopLength,
        winLength: winLength,
        window: window,
        center: true
      )
      audio.eval()
//      fftTimer.stop()
      reconstructed.append(audio)
    }

    let reconstructedStacked = MLX.stacked(reconstructed, axis: 0)
    return reconstructedStacked.expandedDimensions(axis: 1)
  }

  func callAsFunction(inputData: MLXArray) -> MLXArray {
    let (mag, ph) = transform(inputData: inputData)
    magnitude = mag
    phase = ph
    let reconstruction = inverse(magnitude: mag, phase: ph)
    return reconstruction.expandedDimensions(axis: -2)
  }
}

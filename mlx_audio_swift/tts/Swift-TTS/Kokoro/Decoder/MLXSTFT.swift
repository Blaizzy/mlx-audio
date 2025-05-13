//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXFFT
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

class ThreadPool {
  private let semaphore: DispatchSemaphore
  private let queue: DispatchQueue
  private let dispatchGroup: DispatchGroup

  init(maxConcurrentTasks: Int) {
    semaphore = DispatchSemaphore(value: maxConcurrentTasks)
    queue = DispatchQueue(label: "com.threadpool.queue", qos: .userInteractive, attributes: .concurrent)
    dispatchGroup = DispatchGroup()
  }

  func addTask(_ task: @escaping () -> Void) {
    dispatchGroup.enter() // Register the task with the DispatchGroup
    queue.async {
      self.semaphore.wait()
      task() // Run the task
      self.semaphore.signal()
      self.dispatchGroup.leave() // Mark this task as finished
    }
  }

  func waitForAllTasks() {
    dispatchGroup.wait() // Blocks until all tasks are complete
    print("✅ All tasks are complete")
  }
}

func mlxIstft(
  x: MLXArray,
  hopLength: Int? = nil,
  winLength: Int? = nil,
  window: Any = "hann"
) -> MLXArray {
  BenchmarkTimer.shared.create(id: "StartProcessing", parent: "Istft")

  let winLen = winLength ?? ((x.shape[1] - 1) * 2)
  let hopLen = hopLength ?? (winLen / 4)

  let w = getWindow(window: window, winLen: winLen, nFft: winLen)

  let xTransposed = x.transposed(1, 0)
  let t = (xTransposed.shape[0] - 1) * hopLen + winLen
  let windowModLen = 20 / 5

  let wSquared = w * w
  w.eval()
  wSquared.eval()

  let totalWsquared = MLX.concatenated(Array(repeating: wSquared, count: t / winLen))

  xTransposed.eval()
  BenchmarkTimer.shared.stop(id: "StartProcessing")

//  let fft = BenchmarkTimer.shared.create(id: "FFT", parent: "Istft")!
//  let overlap = BenchmarkTimer.shared.create(id: "Overlap", parent: "Istft")!

//  fft.startTimer()
  let output = MLXFFT.irfft(xTransposed, axis: 1) * w
  output.eval()

  var outputs: [MLXArray] = []
  var windowSums: [MLXArray] = []

  for i in 0 ..< windowModLen {
    let outputStride = output[.stride(from: i, by: windowModLen), .ellipsis].reshaped([-1])
    let windowSumArray = totalWsquared[0 ..< outputStride.shape[0]]

    outputs.append(MLX.concatenated([
      MLXArray.zeros([i * hopLen]),
      outputStride,
      MLXArray.zeros([max(0, t - i * hopLen - outputStride.shape[0])]),
    ]))

    windowSums.append(MLX.concatenated([
      MLXArray.zeros([i * hopLen]),
      windowSumArray,
      MLXArray.zeros([max(0, t - i * hopLen - windowSumArray.shape[0])]),
    ]))
  }
//  fft.stop()
//
//  overlap.startTimer()

  var reconstructed = outputs[0]
  var windowSum = windowSums[0]
  for i in 1 ..< windowModLen {
    reconstructed += outputs[i]
    windowSum += windowSums[i]
  }
  reconstructed.eval()
  windowSum.eval()

//  overlap.stop()

  BenchmarkTimer.shared.create(id: "EndProcessing", parent: "Istft")

  reconstructed =
    reconstructed[winLen / 2 ..< (reconstructed.shape[0] - winLen / 2)] /
    windowSum[winLen / 2 ..< (reconstructed.shape[0] - winLen / 2)]
  reconstructed.eval()

  BenchmarkTimer.shared.stop(id: "EndProcessing")

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

    let threadPool = ThreadPool(maxConcurrentTasks: 4)
    var magnitudes: [MLXArray] = Array(repeating: MLXArray.zeros([1]), count: audioArray.shape[0])
    var phases: [MLXArray] = Array(repeating: MLXArray.zeros([1]), count: audioArray.shape[0])

    for batchIdx in 0 ..< audioArray.shape[0] {
        threadPool.addTask {
            let stft = mlxStft(
                x: audioArray[batchIdx],
                nFft: self.filterLength,
                hopLength: self.hopLength,
                winLength: self.winLength,
                window: self.window,
                center: true,
                padMode: "reflect"
            )
            magnitudes[batchIdx] = MLX.abs(stft)
            phases[batchIdx] = MLX.atan2(stft.imaginaryPart(), stft.realPart())
        }
    }

    threadPool.waitForAllTasks()
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
        window: window
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

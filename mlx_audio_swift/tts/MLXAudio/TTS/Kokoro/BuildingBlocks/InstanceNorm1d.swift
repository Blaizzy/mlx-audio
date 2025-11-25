//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class _InstanceNorm {
  let numFeatures: Int
  let eps: Float
  let momentum: Float
  let affine: Bool
  let trackRunningStats: Bool

  var weight: MLXArray?
  var bias: MLXArray?
  var runningMean: MLXArray?
  var runningVar: MLXArray?
  var training: Bool = true

  init(
    numFeatures: Int,
    eps: Float = 1e-5,
    momentum: Float = 0.1,
    affine: Bool = false,
    trackRunningStats: Bool = false
  ) {
    self.numFeatures = numFeatures
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.trackRunningStats = trackRunningStats

    if self.affine {
      weight = MLXArray.ones([numFeatures])
      bias = MLXArray.zeros([numFeatures])
    }

    if self.trackRunningStats {
      runningMean = MLXArray.zeros([numFeatures])
      runningVar = MLXArray.ones([numFeatures])
    }
  }

  func checkInputDim(_: MLXArray) {
    fatalError("Subclass must implement checkInputDim")
  }

  func getNoBatchDim() -> Int {
    fatalError("Subclass must implement getNoBatchDim")
  }

  func handleNoBatchInput(_ input: MLXArray) -> MLXArray {
    let expanded = input.expandedDimensions(axis: 0)
    let result = applyInstanceNorm(expanded)
    return result.squeezed(axes: [0])
  }

  func applyInstanceNorm(_ input: MLXArray) -> MLXArray {
    let dims = Array(0 ..< input.ndim)
    let featureDim = dims[dims.count - getNoBatchDim()]

    let reduceDims = dims.filter { $0 != 0 && $0 != featureDim }

    var mean: MLXArray
    var variance: MLXArray

    if training || !trackRunningStats {
      mean = MLX.mean(input, axes: reduceDims, keepDims: true)
      variance = MLX.variance(input, axes: reduceDims, keepDims: true)

      if trackRunningStats && training, let runningMean = runningMean, let runningVar = runningVar {
        let overallMean = MLX.mean(mean, axes: [0])
        let overallVar = MLX.mean(variance, axes: [0])

        self.runningMean = (1 - momentum) * runningMean + momentum * overallMean
        self.runningVar = (1 - momentum) * runningVar + momentum * overallVar
      }
    } else if let runningMean = runningMean, let runningVar = runningVar {
      var meanShape = Array(repeating: 1, count: input.ndim)
      meanShape[featureDim] = numFeatures
      let varShape = meanShape

      mean = runningMean.reshaped(meanShape)
      variance = runningVar.reshaped(varShape)
    } else {
      fatalError("Running statistics not available")
    }

    // Normalize
    let xNorm = (input - mean) / MLX.sqrt(variance + eps)

    // Apply bias if needed
    if affine, let weight = weight, let bias = bias {
      var weightShape = Array(repeating: 1, count: input.ndim)
      weightShape[featureDim] = numFeatures
      let biasShape = weightShape

      let reshapedWeight = weight.reshaped(weightShape)
      let reshapedBias = bias.reshaped(biasShape)

      return xNorm * reshapedWeight + reshapedBias
    } else {
      return xNorm
    }
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    checkInputDim(input)

    let featureDim = input.ndim - getNoBatchDim()
    if input.shape[featureDim] != numFeatures {
      if affine {
        fatalError("Expected input's size at dim=\(featureDim) to match numFeatures (\(numFeatures)), but got: \(input.shape[featureDim]).")
      } else {
        Log.model.warning("Input's size at dim=\(featureDim) does not match numFeatures. You can silence this warning by not passing in numFeatures, which is not used because affine=False")
      }
    }

    if input.ndim == getNoBatchDim() {
      return handleNoBatchInput(input)
    }

    return applyInstanceNorm(input)
  }
}

class InstanceNorm1d: _InstanceNorm {
  override func getNoBatchDim() -> Int {
    return 2
  }

  override func checkInputDim(_ input: MLXArray) {
    if input.ndim != 2, input.ndim != 3 {
      fatalError("Expected 2D or 3D input (got \(input.ndim)D input)")
    }
  }
}

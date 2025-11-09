//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class LSTM: Module {
  let inputSize: Int
  let hiddenSize: Int
  let hasBias: Bool
  let batchFirst: Bool

  // Forward direction weights and biases
  var wxForward: MLXArray
  var whForward: MLXArray
  var biasIhForward: MLXArray?
  var biasHhForward: MLXArray?

  // Backward direction weights and biases
  var wxBackward: MLXArray
  var whBackward: MLXArray
  var biasIhBackward: MLXArray?
  var biasHhBackward: MLXArray?

  init(
    inputSize: Int,
    hiddenSize: Int,
    bias: Bool = true,
    batchFirst: Bool = true,
    wxForward: MLXArray,
    whForward: MLXArray,
    biasIhForward: MLXArray? = nil,
    biasHhForward: MLXArray? = nil,
    wxBackward: MLXArray,
    whBackward: MLXArray,
    biasIhBackward: MLXArray? = nil,
    biasHhBackward: MLXArray? = nil
  ) {
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    hasBias = bias
    self.batchFirst = batchFirst

    // Forward direction weights and biases
    self.wxForward = wxForward
    self.whForward = whForward
    self.biasIhForward = biasIhForward
    self.biasHhForward = biasHhForward

    // Backward direction weights and biases
    self.wxBackward = wxBackward
    self.whBackward = whBackward
    self.biasIhBackward = biasIhBackward
    self.biasHhBackward = biasHhBackward

    super.init()
  }

  /// Process sequence in forward direction
  private func forwardDirection(
    _ x: MLXArray,
    hidden: MLXArray? = nil,
    cell: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    // Pre-compute input projections
    let xProj: MLXArray
    if let biasIhForward = biasIhForward, let biasHhForward = biasHhForward {
      xProj = MLX.addMM(
        biasIhForward + biasHhForward,
        x,
        wxForward.transposed()
      )
    } else {
      xProj = MLX.matmul(x, wxForward.transposed())
    }

    var allHidden: [MLXArray] = []
    var allCell: [MLXArray] = []

    let seqLen = x.shape[x.shape.count - 2]

    var currentHidden = hidden ?? MLXArray.zeros([x.shape[0], hiddenSize])
    var currentCell = cell ?? MLXArray.zeros([x.shape[0], hiddenSize])

    // Process sequence in forward direction (0 to seqLen-1)
    for idx in 0 ..< seqLen {
      var ifgo = xProj[0..., idx, 0...]
      ifgo = ifgo + MLX.matmul(currentHidden, whForward.transposed())

      // Split gates
      let gates = MLX.split(ifgo, parts: 4, axis: -1)
      let i = MLX.sigmoid(gates[0])
      let f = MLX.sigmoid(gates[1])
      let g = MLX.tanh(gates[2])
      let o = MLX.sigmoid(gates[3])

      // Update cell and hidden states
      currentCell = f * currentCell + i * g
      currentHidden = o * MLX.tanh(currentCell)

      allCell.append(currentCell)
      allHidden.append(currentHidden)
    }

    return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
  }

  /// Process sequence in backward direction
  private func backwardDirection(
    _ x: MLXArray,
    hidden: MLXArray? = nil,
    cell: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    let xProj: MLXArray
    if let biasIhBackward = biasIhBackward, let biasHhBackward = biasHhBackward {
      xProj = MLX.addMM(
        biasIhBackward + biasHhBackward,
        x,
        wxBackward.transposed()
      )
    } else {
      xProj = MLX.matmul(x, wxBackward.transposed())
    }

    var allHidden: [MLXArray] = []
    var allCell: [MLXArray] = []

    let seqLen = x.shape[x.shape.count - 2]

    var currentHidden = hidden ?? MLXArray.zeros([x.shape[0], hiddenSize])
    var currentCell = cell ?? MLXArray.zeros([x.shape[0], hiddenSize])

    // Process sequence in backward direction (seqLen-1 to 0)
    for idx in stride(from: seqLen - 1, through: 0, by: -1) {
      var ifgo = xProj[0..., idx, 0...]
      ifgo = ifgo + MLX.matmul(currentHidden, whBackward.transposed())

      // Split gates
      let gates = MLX.split(ifgo, parts: 4, axis: -1)
      let i = MLX.sigmoid(gates[0])
      let f = MLX.sigmoid(gates[1])
      let g = MLX.tanh(gates[2])
      let o = MLX.sigmoid(gates[3])

      // Update cell and hidden states
      currentCell = f * currentCell + i * g
      currentHidden = o * MLX.tanh(currentCell)

      // Insert at beginning to maintain original sequence order
      allCell.insert(currentCell, at: 0)
      allHidden.insert(currentHidden, at: 0)
    }

    return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
  }

  func callAsFunction(
    _ x: MLXArray,
    hiddenForward: MLXArray? = nil,
    cellForward: MLXArray? = nil,
    hiddenBackward: MLXArray? = nil,
    cellBackward: MLXArray? = nil
  ) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
    let input: MLXArray
    if x.ndim == 2 {
      input = x.expandedDimensions(axis: 0) // (1, seq_len, input_size)
    } else {
      input = x
    }

    let (forwardHidden, forwardCell) = forwardDirection(
      input,
      hidden: hiddenForward,
      cell: cellForward
    )

    let (backwardHidden, backwardCell) = backwardDirection(
      input,
      hidden: hiddenBackward,
      cell: cellBackward
    )

    let output = MLX.concatenated([forwardHidden, backwardHidden], axis: -1)

    return (
      output,
      (
        (forwardHidden[0..., -1, 0...], forwardCell[0..., -1, 0...]),
        (backwardHidden[0..., 0, 0...], backwardCell[0..., 0, 0...])
      )
    )
  }
}

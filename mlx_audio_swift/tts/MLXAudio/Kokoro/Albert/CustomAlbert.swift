//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Custom Albert Model
class CustomAlbert {
  let config: AlbertModelArgs
  let embeddings: AlbertEmbeddings
  let encoder: AlbertEncoder
  let pooler: Linear

  init(weights: [String: MLXArray], config: AlbertModelArgs) {
    self.config = config
    embeddings = AlbertEmbeddings(weights: weights, config: config)
    encoder = AlbertEncoder(weights: weights, config: config)
    pooler = Linear(weight: weights["bert.pooler.weight"]!, bias: weights["bert.pooler.bias"]!)
  }

  func callAsFunction(
    _ inputIds: MLXArray,
    tokenTypeIds: MLXArray? = nil,
    attentionMask: MLXArray? = nil
  ) -> (sequenceOutput: MLXArray, pooledOutput: MLXArray) {
    let embeddingOutput = embeddings(inputIds, tokenTypeIds: tokenTypeIds)

    var attentionMaskProcessed: MLXArray?
    if let attentionMask = attentionMask {
      let shape = attentionMask.shape
      let newDims = [shape[0], 1, 1, shape[1]]
      attentionMaskProcessed = attentionMask.reshaped(newDims)
      attentionMaskProcessed = (1.0 - attentionMaskProcessed!) * -10000.0
    }

    let sequenceOutput = encoder(embeddingOutput, attentionMask: attentionMaskProcessed)
    let firstTokenReshaped = sequenceOutput[0..., 0, 0...]
    let pooledOutput = MLX.tanh(pooler(firstTokenReshaped))

    return (sequenceOutput, pooledOutput)
  }
}

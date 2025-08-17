//
//  Kokoro-tts-lib
//
import Foundation

struct AlbertModelArgs {
  let numHiddenLayers: Int
  let numAttentionHeads: Int
  let hiddenSize: Int
  let intermediateSize: Int
  let maxPositionEmbeddings: Int
  let modelType: String
  let embeddingSize: Int
  let innerGroupNum: Int
  let numHiddenGroups: Int
  let hiddenDropoutProb: Float
  let attentionProbsDropoutProb: Float
  let typeVocabSize: Int
  let initializerRange: Float
  let layerNormEps: Float
  let vocabSize: Int
  let dropout: Float

  init(
    numHiddenLayers: Int = 12,
    numAttentionHeads: Int = 12,
    hiddenSize: Int = 768,
    intermediateSize: Int = 2048,
    maxPositionEmbeddings: Int = 512,
    modelType: String = "albert",
    embeddingSize: Int = 128,
    innerGroupNum: Int = 1,
    numHiddenGroups: Int = 1,
    hiddenDropoutProb: Float = 0.1,
    attentionProbsDropoutProb: Float = 0.1,
    typeVocabSize: Int = 2,
    initializerRange: Float = 0.02,
    layerNormEps: Float = 1e-12,
    vocabSize: Int = 178,
    dropout: Float = 0.1
  ) {
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.hiddenSize = hiddenSize
    self.intermediateSize = intermediateSize
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.modelType = modelType
    self.embeddingSize = embeddingSize
    self.innerGroupNum = innerGroupNum
    self.numHiddenGroups = numHiddenGroups
    self.hiddenDropoutProb = hiddenDropoutProb
    self.attentionProbsDropoutProb = attentionProbsDropoutProb
    self.typeVocabSize = typeVocabSize
    self.initializerRange = initializerRange
    self.layerNormEps = layerNormEps
    self.vocabSize = vocabSize
    self.dropout = dropout
  }
}

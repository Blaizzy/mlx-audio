//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import Synchronization

public enum TTSVoice: String, CaseIterable, Sendable {
  case afAlloy
  case afAoede
  case afBella
  case afHeart
  case afJessica
  case afKore
  case afNicole
  case afNova
  case afRiver
  case afSarah
  case afSky
  case amAdam
  case amEcho
  case amEric
  case amFenrir
  case amLiam
  case amMichael
  case amOnyx
  case amPuck
  case amSanta
  case bfAlice
  case bfEmma
  case bfIsabella
  case bfLily
  case bmDaniel
  case bmFable
  case bmGeorge
  case bmLewis
  case efDora
  case emAlex
  case ffSiwis
  case hfAlpha
  case hfBeta
  case hfOmega
  case hmPsi
  case ifSara
  case imNicola
  case jfAlpha
  case jfGongitsune
  case jfNezumi
  case jfTebukuro
  case jmKumo
  case pfDora
  case pmSanta
  case zfXiaobei
  case zfXiaoni
  case zfXiaoxiao
  case zfXiaoyi
  case zmYunjian
  case zmYunxi
  case zmYunxia
  case zmYunyang
}

public actor KokoroTTS {
  enum KokoroTTSError: LocalizedError {
    case tooManyTokens
    case sentenceSplitError
    case modelNotInitialized
    case audioGenerationError

    var errorDescription: String? {
      switch self {
      case .tooManyTokens:
        return "Input text exceeds maximum token limit"
      case .sentenceSplitError:
        return "Failed to split text into sentences"
      case .modelNotInitialized:
        return "Model has not been initialized"
      case .audioGenerationError:
        return "Failed to generate audio"
      }
    }
  }

  // MARK: - Constants

  private static let maxTokenCount = 510
  private static let sampleRate = 24000

  // MARK: - Properties

  private var bert: CustomAlbert!
  private var bertEncoder: Linear!
  private var durationEncoder: DurationEncoder!
  private var predictorLSTM: LSTM!
  private var durationProj: Linear!
  private var prosodyPredictor: ProsodyPredictor!
  private var textEncoder: TextEncoder!
  private var decoder: Decoder!
  private var eSpeakEngine: ESpeakNGEngine!
  private var kokoroTokenizer: KokoroTokenizer!
  private var chosenVoice: TTSVoice?
  private var voice: MLXArray!

  // Flag to indicate if model components are initialized
  private var isModelInitialized = false

  // Hugging Face repo configuration
  private var repoId: String
  private var progressHandler: @Sendable (Progress) -> Void

  // Callback type for streaming audio generation
  public typealias AudioChunkCallback = @Sendable ([Float]) -> Void

  /// Initializes with optional Hugging Face repo configuration.
  ///
  /// Models are downloaded from Hugging Face Hub on first use.
  public init(
    repoId: String = KokoroWeightLoader.defaultRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) {
    self.repoId = repoId
    self.progressHandler = progressHandler
  }

  // Reset the model to free up memory
  public func resetModel(preserveTextProcessing: Bool = true) {
    // Reset heavy ML model components
    bert = nil
    bertEncoder = nil
    durationEncoder = nil
    predictorLSTM = nil
    durationProj = nil
    prosodyPredictor = nil
    textEncoder = nil
    decoder = nil
    voice = nil
    chosenVoice = nil
    isModelInitialized = false

    // Optionally preserve text processing components for faster restart
    if !preserveTextProcessing {
      if let _ = eSpeakEngine {
        // Ensure eSpeakEngine is terminated properly
        eSpeakEngine = nil
      }
      kokoroTokenizer = nil
    }

  }

  // Initialize model on demand
  private func ensureModelInitialized() async throws {
    if isModelInitialized {
      return
    }

    // Initialize text processing components first (less expensive)
    if eSpeakEngine == nil {
      eSpeakEngine = try ESpeakNGEngine()
    }

    if kokoroTokenizer == nil {
      kokoroTokenizer = KokoroTokenizer(engine: eSpeakEngine)
    }

    // Load lexicons from GitHub (cached on disk)
    if !kokoroTokenizer.lexiconsLoaded {
      async let usLexicon = LexiconLoader.loadUSLexicon()
      async let gbLexicon = LexiconLoader.loadGBLexicon()
      kokoroTokenizer.setLexicons(us: try await usLexicon, gb: try await gbLexicon)
    }

    let sanitizedWeights = try await KokoroWeightLoader.loadWeights(
      repoId: repoId,
      progressHandler: progressHandler
    )

    bert = CustomAlbert(weights: sanitizedWeights, config: AlbertModelArgs())
    bertEncoder = Linear(weight: sanitizedWeights["bert_encoder.weight"]!, bias: sanitizedWeights["bert_encoder.bias"]!)
    durationEncoder = DurationEncoder(weights: sanitizedWeights, dModel: 512, styDim: 128, nlayers: 3)

    predictorLSTM = LSTM(
      inputSize: 512 + 128,
      hiddenSize: 512 / 2,
      wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
      whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
      biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
      biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
      wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
      whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
    )

    durationProj = Linear(
      weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
      bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
    )

    prosodyPredictor = ProsodyPredictor(
      weights: sanitizedWeights,
      styleDim: 128,
      dHid: 512
    )

    textEncoder = TextEncoder(
      weights: sanitizedWeights,
      channels: 512,
      kernelSize: 5,
      depth: 3,
      nSymbols: 178
    )

    decoder = Decoder(
      weights: sanitizedWeights,
      dimIn: 512,
      styleDim: 128,
      dimOut: 80,
      resblockKernelSizes: [3, 7, 11],
      upsampleRates: [10, 6],
      upsampleInitialChannel: 512,
      resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      upsampleKernelSizes: [20, 12],
      genIstftNFft: 20,
      genIstftHopSize: 5
    )

    isModelInitialized = true
  }

  private func generateAudioForTokens(
    inputIds: [Int],
    speed: Float
  ) throws -> [Float] {
    let paddedInputIdsBase = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])
    paddedInputIds.eval()

    let inputLengths = MLXArray(paddedInputIds.dim(-1))
    inputLengths.eval()

    let inputLengthMax: Int = MLX.max(inputLengths).item()
    var textMask = MLXArray(0 ..< inputLengthMax)
    textMask.eval()

    textMask = textMask + 1 .> inputLengths
    textMask.eval()

    textMask = textMask.expandedDimensions(axes: [0])
    textMask.eval()

    let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
    let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
    let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)
    attentionMask.eval()

    // Ensure model is initialized
    guard let bert = bert,
          let bertEncoder = bertEncoder else {
      throw KokoroTTSError.modelNotInitialized
    }

    let (bertDur, _) = bert(paddedInputIds, attentionMask: attentionMask)
    bertDur.eval()

    let dEn = bertEncoder(bertDur).transposed(0, 2, 1)
    dEn.eval()

    guard let voice = voice else {
      throw KokoroTTSError.modelNotInitialized
    }
    // Voice shape is [510, 1, 256], index by phoneme length to get [1, 256]
    let voiceIdx = min(inputIds.count - 1, voice.shape[0] - 1)
    let refS = voice[voiceIdx]
    refS.eval()

    // Extract style vector: columns 128+ for duration/prosody prediction
    let s = refS[0..., 128...]
    s.eval()

    // Ensure all components are initialized
    guard let durationEncoder = durationEncoder,
          let predictorLSTM = predictorLSTM,
          let durationProj = durationProj else {
      throw KokoroTTSError.modelNotInitialized
    }

    let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
    d.eval()

    let (x, _) = predictorLSTM(d)
    x.eval()

    let duration = durationProj(x)
    duration.eval()

    let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
    durationSigmoid.eval()

    let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
    predDur.eval()

    // Index and matrix generation
    // Build indices in chunks to reduce memory
    var allIndices: [MLXArray] = []
    let chunkSize = 50

    for startIdx in stride(from: 0, to: predDur.shape[0], by: chunkSize) {
      let endIdx = min(startIdx + chunkSize, predDur.shape[0])
      let chunkIndices = predDur[startIdx..<endIdx]

      let indices = MLX.concatenated(
        chunkIndices.enumerated().map { i, n in
          let nSize: Int = n.item()
          let arrayIndex = MLXArray([i + startIdx])
          arrayIndex.eval()
          let repeated = MLX.repeated(arrayIndex, count: nSize)
          repeated.eval()
          return repeated
        }
      )
      indices.eval()
      allIndices.append(indices)
    }

    let indices = MLX.concatenated(allIndices)
    indices.eval()

    allIndices.removeAll()

    let indicesShape = indices.shape[0]
    let inputIdsShape = paddedInputIds.shape[1]

    // Create sparse matrix using COO format
    var rowIndices: [Int] = []
    var colIndices: [Int] = []

    // Reserve capacity to avoid reallocations
    let estimatedNonZeros = min(indicesShape, inputIdsShape * 5)
    rowIndices.reserveCapacity(estimatedNonZeros)
    colIndices.reserveCapacity(estimatedNonZeros)

    // Process in batches
    let batchSize = 256
    for startIdx in stride(from: 0, to: indicesShape, by: batchSize) {
      let endIdx = min(startIdx + batchSize, indicesShape)
      for i in startIdx..<endIdx {
        let indiceValue: Int = indices[i].item()
        if indiceValue < inputIdsShape {
          rowIndices.append(indiceValue)
          colIndices.append(i)
        }
      }
    }

    // Create dense matrix from COO data
    var swiftPredAlnTrg = [Float](repeating: 0.0, count: inputIdsShape * indicesShape)
    let matrixBatchSize = 1000
    for startIdx in stride(from: 0, to: rowIndices.count, by: matrixBatchSize) {
      let endIdx = min(startIdx + matrixBatchSize, rowIndices.count)
      for i in startIdx..<endIdx {
        let row = rowIndices[i]
        let col = colIndices[i]
        if row < inputIdsShape && col < indicesShape {
          swiftPredAlnTrg[row * indicesShape + col] = 1.0
        }
      }
    }

    // Create MLXArray from the dense matrix
    let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([inputIdsShape, indicesShape])
    predAlnTrg.eval()

    // Clear Swift arrays
    swiftPredAlnTrg = []
    rowIndices = []
    colIndices = []

    let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
    predAlnTrgBatched.eval()

    let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)
    en.eval()

    // Ensure components are initialized
    guard let prosodyPredictor = prosodyPredictor,
          let textEncoder = textEncoder,
          let decoder = decoder else {
      throw KokoroTTSError.modelNotInitialized
    }

    let (F0Pred, NPred) = prosodyPredictor.F0NTrain(x: en, s: s)
    F0Pred.eval()
    NPred.eval()

    let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    tEn.eval()

    let asr = MLX.matmul(tEn, predAlnTrg)
    asr.eval()

    // Extract style vector: columns 0-127 for decoder
    let voiceS = refS[0..., ..<128]
    voiceS.eval()

    let audio = decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: voiceS)[0]
    audio.eval()

    let audioShape = audio.shape

    // Check if the audio shape is valid
    let totalSamples: Int
    if audioShape.count == 1 {
      totalSamples = audioShape[0]
    } else if audioShape.count == 2 {
      totalSamples = audioShape[1]
    } else {
      totalSamples = 0
    }

    if totalSamples <= 1 {
      Log.tts.error("KokoroTTS: Invalid audio shape - totalSamples: \(totalSamples), shape: \(audioShape)")
      throw KokoroTTSError.audioGenerationError
    }

    return audio.asArray(Float.self)
  }

  public func generateAudio(voice: TTSVoice, text: String, speed: Float = 1.0, chunkCallback: @escaping AudioChunkCallback) async throws {
    try await ensureModelInitialized()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    self.voice = nil

    for sentence in sentences {
      let audio = try await self.generateAudioForSentence(voice: voice, text: sentence, speed: speed)
      chunkCallback(audio)
      MLX.GPU.clearCache()
    }

    // Reset model after completing a long text to free memory
    if sentences.count > 5 {
      self.resetModel()
    }
  }

  public func generateAudioStream(voice: TTSVoice, text: String, speed: Float = 1.0) async throws -> AsyncThrowingStream<[Float], Error> {
    try await ensureModelInitialized()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    self.voice = nil
    let index = Atomic<Int>(0)

    return AsyncThrowingStream {
      let i = index.wrappingAdd(1, ordering: .relaxed).oldValue
      guard i < sentences.count else { return nil }

      let audio = try await self.generateAudioForSentence(voice: voice, text: sentences[i], speed: speed)
      MLX.GPU.clearCache()
      return audio
    }
  }

  private func generateAudioForSentence(voice: TTSVoice, text: String, speed: Float) async throws -> [Float] {
    try await ensureModelInitialized()

    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      return [0.0]
    }

    // Load voice if it changed or if it was cleared
    if chosenVoice != voice || self.voice == nil {
      self.voice = try await VoiceLoader.loadVoice(
        voice,
        repoId: repoId,
        progressHandler: progressHandler
      )
      self.voice?.eval() // Force immediate evaluation

      try kokoroTokenizer.setLanguage(for: voice)
      chosenVoice = voice
    }

    do {
      let phonemizedResult = try kokoroTokenizer.phonemize(text)

      let inputIds = PhonemeTokenizer.tokenize(phonemizedText: phonemizedResult.phonemes)
      guard inputIds.count <= Self.maxTokenCount else {
        throw KokoroTTSError.tooManyTokens
      }

      // Continue with normal audio generation
      return try self.processTokensToAudio(inputIds: inputIds, speed: speed)
    } catch {
      // Re-throw the error instead of silently returning a beep
      // This allows proper error handling up the call stack
      Log.tts.error("KokoroTTS: Error generating audio for sentence - \(error)")
      throw error
    }
  }

  // Common processing method to convert tokens to audio - used by streaming methods
  private func processTokensToAudio(inputIds: [Int], speed: Float) throws -> [Float] {
    // Use the token processing method
    return try generateAudioForTokens(
      inputIds: inputIds,
      speed: speed
    )
  }
}

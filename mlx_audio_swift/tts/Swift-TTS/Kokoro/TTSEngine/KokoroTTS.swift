//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Available voices
public enum TTSVoice: String, CaseIterable {
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
  case zfZiaobei
  case zfXiaoni
  case zfXiaoxiao
  case zfZiaoyi
  case zmYunjian
  case zmYunxi
  case zmYunxia
  case zmYunyang
}

// Main class, encapsulates the whole Kokoro text-to-speech pipeline
public class KokoroTTS {
  enum KokoroTTSError: Error {
    case tooManyTokens
    case sentenceSplitError
    case modelNotInitialized
  }

  private var bert: CustomAlbert!
  private var bertEncoder: Linear!
  private var durationEncoder: DurationEncoder!
  private var predictorLSTM: LSTM!
  private var durationProj: Linear!
  private var prosodyPredictor: ProsodyPredictor!
  private var textEncoder: TextEncoder!
  private var decoder: Decoder!
  private var eSpeakEngine: ESpeakNGEngine!
  private var chosenVoice: TTSVoice?
  private var voice: MLXArray!

  // Flag to indicate if model components are initialized
  private var isModelInitialized = false

  // Callback type for streaming audio generation
  public typealias AudioChunkCallback = (MLXArray) -> Void

  init() {}

  // Reset the model to free up memory
  public func resetModel() {
    // First nil out the eSpeakEngine to ensure proper cleanup
    // Important: This must be done first before clearing other objects
    if let _ = eSpeakEngine {
      // Ensure eSpeakEngine is terminated properly
      eSpeakEngine = nil
    }

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

    // Use plain autoreleasepool to encourage memory release
    autoreleasepool { }
  }

  // Initialize model on demand
  private func ensureModelInitialized() {
    if isModelInitialized {
      return
    }

    autoreleasepool {
      let sanitizedWeights = KokoroWeightLoader.loadWeights()

      bert = CustomAlbert(weights: sanitizedWeights, config: AlbertModelArgs())
      bertEncoder = Linear(weight: sanitizedWeights["bert_encoder.weight"]!, bias: sanitizedWeights["bert_encoder.bias"]!)
      durationEncoder = DurationEncoder(weights: sanitizedWeights, dModel: 512, styDim: 128, nlayers: 6)

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
    }

    eSpeakEngine = try! ESpeakNGEngine()
    isModelInitialized = true
  }

  private func generateAudioForTokens(
    inputIds: [Int],
    speed: Float
  ) throws -> MLXArray {
    // Create a fresh autorelease pool for the entire process
    return try autoreleasepool { () -> MLXArray in
      // Start with the standard processing
      try autoreleasepool {
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

        return try autoreleasepool { () -> MLXArray in
          // Ensure model is initialized
          guard let bert = bert,
                let bertEncoder = bertEncoder else {
            throw KokoroTTSError.modelNotInitialized
          }

          let (bertDur, _) = bert(paddedInputIds, attentionMask: attentionMask)
          bertDur.eval()

          autoreleasepool {
            _ = attentionMask
          }

          let dEn = bertEncoder(bertDur).transposed(0, 2, 1)
          dEn.eval()

          autoreleasepool {
            _ = bertDur
          }

          var refS: MLXArray
          do {
            guard let voice = voice else {
              throw KokoroTTSError.modelNotInitialized
            }
            refS = voice[min(inputIds.count - 1, voice.shape[0] - 1), 0 ... 1, 0...]
          } catch {
            // Use a fallback slice from start of the voice array
            guard let voice = voice else {
              throw KokoroTTSError.modelNotInitialized
            }
            refS = voice[0, 0 ... 1, 0...]
          }
          refS.eval()

          let s = refS[0 ... 1, 128...]
          s.eval()

          return try autoreleasepool { () -> MLXArray in
            // Ensure all components are initialized
            guard let durationEncoder = durationEncoder,
                  let predictorLSTM = predictorLSTM,
                  let durationProj = durationProj else {
              throw KokoroTTSError.modelNotInitialized
            }

            let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
            d.eval()

            autoreleasepool {
              _ = dEn
              _ = textMask
            }

            let (x, _) = predictorLSTM(d)
            x.eval()

            let duration = durationProj(x)
            duration.eval()

            autoreleasepool {
              _ = x
            }

            let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
            durationSigmoid.eval()

            autoreleasepool {
              _ = duration
            }

            let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
            predDur.eval()

            autoreleasepool {
              _ = durationSigmoid
            }

            // Index and matrix generation - high memory usage
            // Build indices in chunks to reduce memory
            var allIndices: [MLXArray] = []
            let chunkSize = 50 // Process 50 items at a time

            for startIdx in stride(from: 0, to: predDur.shape[0], by: chunkSize) {
              autoreleasepool {
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
            }

            let indices = MLX.concatenated(allIndices)
            indices.eval()

            allIndices.removeAll()

            let indicesShape = indices.shape[0]
            let inputIdsShape = paddedInputIds.shape[1]

            // Create sparse matrix more efficiently using COO format
            // This drastically reduces memory usage compared to dense matrix
            var rowIndices: [Int] = []
            var colIndices: [Int] = []
            var values: [Float] = []

            // Reserve capacity to avoid reallocations
            let estimatedNonZeros = min(indicesShape, inputIdsShape * 5)
            rowIndices.reserveCapacity(estimatedNonZeros)
            colIndices.reserveCapacity(estimatedNonZeros)
            values.reserveCapacity(estimatedNonZeros)

            // Process in batches to reduce cache misses
            let batchSize = 256
            for startIdx in stride(from: 0, to: indicesShape, by: batchSize) {
              autoreleasepool {
                let endIdx = min(startIdx + batchSize, indicesShape)
                for i in startIdx..<endIdx {
                  let indiceValue: Int = indices[i].item()
                  if indiceValue < inputIdsShape {
                    rowIndices.append(indiceValue)
                    colIndices.append(i)
                    values.append(1.0)
                  }
                }
              }
            }

            autoreleasepool {
              _ = indices
              _ = predDur
            }

            // Create MLXArray from COO data
            let rowIndicesArray = MLXArray(rowIndices)
            let colIndicesArray = MLXArray(colIndices)
            let coo_indices = MLX.stacked([rowIndicesArray, colIndicesArray], axis: 0).transposed(1, 0)
            let coo_values = MLXArray(values)
            rowIndicesArray.eval()
            colIndicesArray.eval()
            coo_indices.eval()
            coo_values.eval()

            // Go back to the original dense matrix approach but with better memory management
            // Create sparse matrix efficiently using Swift arrays first
            var swiftPredAlnTrg = [Float](repeating: 0.0, count: inputIdsShape * indicesShape)
            // Process in batches to reduce cache misses
            let matrixBatchSize = 1000
            for startIdx in stride(from: 0, to: rowIndices.count, by: matrixBatchSize) {
              autoreleasepool {
                let endIdx = min(startIdx + matrixBatchSize, rowIndices.count)
                for i in startIdx..<endIdx {
                  let row = rowIndices[i]
                  let col = colIndices[i]
                  if row < inputIdsShape && col < indicesShape {
                    swiftPredAlnTrg[row * indicesShape + col] = 1.0
                  }
                }
              }
            }

            // Create MLXArray from the dense matrix
            let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([inputIdsShape, indicesShape])
            predAlnTrg.eval()

            // Clear Swift array immediately
            swiftPredAlnTrg = []

            autoreleasepool {
              rowIndices = []
              colIndices = []
              values = []
            }

            let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
            predAlnTrgBatched.eval()

            let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)
            en.eval()

            autoreleasepool {
              _ = d
              _ = predAlnTrgBatched
            }

            return try autoreleasepool { () -> MLXArray in
              // Ensure components are initialized
              guard let prosodyPredictor = prosodyPredictor,
                    let textEncoder = textEncoder,
                    let decoder = decoder else {
                throw KokoroTTSError.modelNotInitialized
              }

              let (F0Pred, NPred) = prosodyPredictor.F0NTrain(x: en, s: s)
              F0Pred.eval()
              NPred.eval()

              autoreleasepool {
                _ = en
              }

              let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
              tEn.eval()

              autoreleasepool {
                _ = paddedInputIds
                _ = inputLengths
              }

              let asr = MLX.matmul(tEn, predAlnTrg)
              asr.eval()

              autoreleasepool {
                _ = tEn
                _ = predAlnTrg
              }

              let voiceS = refS[0 ... 1, 0 ... 127]
              voiceS.eval()

              autoreleasepool {
                _ = refS
              }

              let audio = decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: voiceS)[0]
              audio.eval()

              autoreleasepool {
                _ = asr
                _ = F0Pred
                _ = NPred
                _ = voiceS
                _ = s
              }

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
                // Return an error tone
                var errorAudioData = [Float](repeating: 0.0, count: 24000)

                // Create a simple repeating beep pattern to indicate error
                for i in stride(from: 0, to: 24000, by: 100) {
                  let endIdx = min(i + 100, 24000)
                  for j in i..<endIdx {
                    let t = Float(j) / Float(Constants.sampleRate)
                    let freq = (Int(t * 2) % 2 == 0) ? 500.0 : 800.0
                    errorAudioData[j] = sin(Float(2.0 * .pi * freq) * t) * 0.5
                  }
                }

                let fallbackAudio = MLXArray(errorAudioData)
                fallbackAudio.eval()
                return fallbackAudio
              }

              return audio
            }
          }
        }
      }
    }
  }

  public func generateAudio(voice: TTSVoice, text: String, speed: Float = 1.0, chunkCallback: @escaping AudioChunkCallback) throws {
    ensureModelInitialized()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    // Process each sentence in sequence with better performance
    DispatchQueue.global(qos: .userInitiated).async {

      // Clear any previous voice and weight cache to start fresh
      autoreleasepool {
        self.voice = nil
      }

      // Use a separate autorelease pool for each sentence to release memory faster
      for (index, sentence) in sentences.enumerated() {
        autoreleasepool {
          do {
            // Generate audio for this sentence
            let audio = try self.generateAudioForSentence(voice: voice, text: sentence, speed: speed)

            // Force evaluation to ensure tensor is computed before sending
            audio.eval()

            // Send this chunk to the callback immediately on the main thread
            // Dispatch to main thread to avoid threading issues with UI updates
            DispatchQueue.main.async {
              chunkCallback(audio)
            }

            // Explicitly release large tensors
            autoreleasepool {
              _ = audio
            }
          } catch {
            // Handle error silently
          }
        }
        MLX.GPU.clearCache()
      }

      // Reset model after completing a long text to free memory
      if sentences.count > 5 {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
          self.resetModel()
        }
      }
    }
  }

  private func generateAudioForSentence(voice: TTSVoice, text: String, speed: Float) throws -> MLXArray {
    ensureModelInitialized()

    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      return MLXArray.zeros([1])
    }

    return autoreleasepool { () -> MLXArray in
      if chosenVoice != voice {
        autoreleasepool {
          self.voice = VoiceLoader.loadVoice(voice)
          self.voice?.eval() // Force immediate evaluation
        }

        // Safely initialize or re-initialize ESpeakNG engine if needed
        if eSpeakEngine == nil {
          // Recreate if we lost our reference
          eSpeakEngine = try? ESpeakNGEngine()
        }

        // Only attempt to set language if we have a valid engine
        if let engine = eSpeakEngine {
          try? engine.setLanguage(for: voice)
        }

        chosenVoice = voice
      }

      do {
        let outputStr = try eSpeakEngine.phonemize(text: text)

        let inputIds = Tokenizer.tokenize(phonemizedText: outputStr)
        guard inputIds.count <= Constants.maxTokenCount else {
          throw KokoroTTSError.tooManyTokens
        }

        // Continue with normal audio generation
        return try self.processTokensToAudio(inputIds: inputIds, speed: speed)
      } catch {
        // Return a short error tone instead of crashing
        var errorAudioData = [Float](repeating: 0.0, count: 4800) // 0.2s at 24kHz

        // Simple error beep
        for i in 0..<4800 {
          let t = Float(i) / Float(Constants.sampleRate)
          let freq: Float = 880.0 // High-pitched error tone
          errorAudioData[i] = sin(Float(2.0 * .pi * freq) * t) * 0.3
        }

        return MLXArray(errorAudioData)
      }
    }
  }

  // Common processing method to convert tokens to audio - used by streaming methods
  private func processTokensToAudio(inputIds: [Int], speed: Float) throws -> MLXArray {
    // Use the token processing method
    return try generateAudioForTokens(
      inputIds: inputIds,
      speed: speed
    )
  }

  struct Constants {
    static let maxTokenCount = 510
    static let sampleRate = 24000
  }
}

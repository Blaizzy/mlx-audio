import Foundation
@preconcurrency import MLX
@preconcurrency import MLXFast
@preconcurrency import MLXNN
import MLXRandom
@preconcurrency import MLXLMCommon

// Orpheus TTS - Swift implementation of the Orpheus 3B model
// Python equivalent: mlx_audio/tts/models/llama/llama.py

public enum OrpheusVoice: String, CaseIterable, Sendable {
    case tara = "tara" // Female, conversational, clear
    case leah = "leah" // Female, warm, gentle
    case jess = "jess" // Female, energetic, youthful
    case leo = "leo" // Male, authoritative, deep
    case dan = "dan" // Male, friendly, casual
    case mia = "mia" // Female, professional, articulate
    case zac = "zac" // Male, enthusiastic, dynamic
    case zoe = "zoe" // Female, calm, soothing
}

// MARK: - Profiling Helper
struct Profiler {
    static let enabled: Bool = false

    static func time<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
        guard enabled else { return try block() }

        let start = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let end = CFAbsoluteTimeGetCurrent()
        let duration = (end - start) * 1000 // Convert to milliseconds
        Log.perf.debug("⏱️ [PROFILE] \(label): \(String(format: "%.2f", duration))ms")
        return result
    }

    static func timeAsync<T>(_ label: String, _ block: () async throws -> T) async rethrows -> T {
        guard enabled else { return try await block() }

        let start = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let end = CFAbsoluteTimeGetCurrent()
        let duration = (end - start) * 1000 // Convert to milliseconds
        Log.perf.debug("⏱️ [PROFILE] \(label): \(String(format: "%.2f", duration))ms")
        return result
    }
}

/// Wrapper for inference state that contains non-Sendable MLX types.
/// Marked @unchecked Sendable because all access is controlled within the actor.
private struct InferenceState: @unchecked Sendable {
    var cache: [KVCache]
    var logits: MLXArray
    var currentIds: MLXArray
}

public actor OrpheusTTS {
    enum OrpheusTTSError: LocalizedError {
        case tooManyTokens
        case weightsNotAvailable
        case modelNotInitialized
        case weightLoadingFailed(String)

        var errorDescription: String? {
            switch self {
            case .tooManyTokens:
                return "Input text exceeds maximum token limit"
            case .weightsNotAvailable:
                return "Model weights not available"
            case .modelNotInitialized:
                return "Model has not been initialized"
            case .weightLoadingFailed(let message):
                return "Failed to load model weights: \(message)"
            }
        }
    }

    // MARK: - Constants

    private static let maxTokenCount = 1200
    private static let sampleRate = 24000
    private static let startToken = 128259
    private static let endToken = 128258
    private static let padToken = 128263
    private static let audioStartToken = 128261
    private static let audioEndToken = 128262
    private static let voicePrefixToken = 128260
    private static let repetitionContextSize = 20
    private static let codeOffset = 128266
    private static let audioCodeDataStartMarker = 128257

    // MARK: - Properties

    // Model components are nonisolated(unsafe) because they contain non-Sendable types (MLXArray)
    // but are only accessed within the actor's methods
    private nonisolated(unsafe) let model: OrpheusLMHeadModel
    private nonisolated(unsafe) let snacDecoder: SNACDecoder
    private var chosenVoice: OrpheusVoice?
    private let tokenizer: OrpheusTokenizer

    private init(model: OrpheusLMHeadModel, snacDecoder: SNACDecoder, tokenizer: OrpheusTokenizer) {
        self.model = model
        self.snacDecoder = snacDecoder
        self.tokenizer = tokenizer
    }

    public static func load(
        repoId: String = OrpheusWeightLoader.defaultRepoId,
        snacRepoId: String = SNACDecoder.defaultRepoId,
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> OrpheusTTS {
        // Load model weights from Hub
        let loadedWeights = try await Profiler.timeAsync("Weight loading") {
            try await OrpheusWeightLoader.loadWeights(repoId: repoId, progressHandler: progressHandler)
        }

        // Load SNAC decoder weights and config
        let snacConfig = try await Profiler.timeAsync("SNAC config loading") {
            try await SNACDecoder.loadConfig(repoId: snacRepoId, progressHandler: progressHandler)
        }
        let snacWeights = try await Profiler.timeAsync("SNAC weights loading") {
            try await SNACDecoder.loadWeights(repoId: snacRepoId, progressHandler: progressHandler)
        }
        let snacDecoder = Profiler.time("SNAC decoder init") {
            SNACDecoder(config: snacConfig, weights: snacWeights)
        }

        // Download and load tokenizer files from Hub
        let tokenizerURLs = try await Profiler.timeAsync("Tokenizer download") {
            try await OrpheusTokenizer.download(repoId: repoId, progressHandler: progressHandler)
        }
        let tokenizer = try Profiler.time("Tokenizer init") {
            try OrpheusTokenizer(tokenizerURL: tokenizerURLs.tokenizerURL, configURL: tokenizerURLs.configURL)
        }

        // Initialize the model using Module pattern
        let model = Profiler.time("Model initialization") {
            OrpheusLMHeadModel()
        }

        // Check if model is quantized (has .scales weights)
        let isQuantized = loadedWeights.keys.contains { $0.contains(".scales") }
        if isQuantized {
            Log.model.info("Detected quantized model weights")
            // Apply quantization to convert Linear/Embedding -> Quantized versions where needed
            Profiler.time("Apply quantization") {
                quantize(model: model) { path, module in
                    // Quantize any layer that has corresponding .scales in weights
                    if loadedWeights["\(path).scales"] != nil {
                        // Return (groupSize, bits, mode) - 64 group size, 4-bit quantization, affine mode
                        return (64, 4, .affine)
                    }
                    return nil
                }
            }
        }

        // Load weights into model using Module system
        try Profiler.time("Weight loading into model") {
            let parameters = ModuleParameters.unflattened(loadedWeights)
            try model.update(parameters: parameters, verify: [.all])
        }

        // Evaluate model to ensure weights are loaded
        Profiler.time("Model evaluation") {
            eval(model)
        }

        return OrpheusTTS(model: model, snacDecoder: snacDecoder, tokenizer: tokenizer)
    }

    public func generateAudio(voice: OrpheusVoice, text: String, temperature: Float = 0.6, topP: Float = 0.8) async throws -> [Float] {
        let totalGenerationStart = CFAbsoluteTimeGetCurrent()

        // Prepare input with voice prefix
        let prompt = "\(voice.rawValue): \(text)"
        Log.tts.debug("Orpheus prompt: \(prompt)")

        let input_ids_tuple = Profiler.time("Tokenizer preparation") {
            tokenizer.prepareInputIds(prompts: [prompt])
        }

        // Convert the tokenizer output to a Swift [Int32]
        let current_ids = Profiler.time("Input IDs conversion") {
            let array = MLXArray(input_ids_tuple.0[0].asArray(Int32.self))
            return array.ndim == 1 ? array.reshaped([1, -1]) : array
        }

        Log.tts.debug("Input IDs: \(current_ids.shape) = \(current_ids.asArray(Int32.self))")

        // Initialize inference state with KV caches and initial forward pass
        // Using InferenceState wrapper to satisfy Swift concurrency checker
        var state = Profiler.time("Initial forward pass") {
            let cache = model.newCache()
            var logits = model(current_ids, cache: cache)
            // Get logits for the last token only
            logits = logits[0, -1].expandDims(at: 0)
            return InferenceState(cache: cache, logits: logits, currentIds: current_ids)
        }

        // Generate audio tokens
        var generatedTokensForPenalty: [Int32] = [] // For repetition penalty
        var i = 0

        let maxOutputTokens = Self.maxTokenCount // Define how many tokens to generate at most

        while i < maxOutputTokens {
            let iterationStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0

            let historyForRepetition = Profiler.time("History preparation") {
                MLXArray(generatedTokensForPenalty)
            }

            let samplingStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let nextTokenArray = sampleNextToken(
                logits: state.logits,
                history: historyForRepetition,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: 1.3
            )
            let samplingEnd = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let samplingDuration = Profiler.enabled ? (samplingEnd - samplingStart) * 1000 : 0
            if Profiler.enabled {
                Log.perf.debug("⏱️ [PROFILE] Token sampling (iter \(i)): \(String(format: "%.2f", samplingDuration))ms")
            }

            // Only extract the Int32 value when we absolutely need it for CPU operations
            let next_token: Int32 = Profiler.time("Token extraction") {
                // This operation forces GPU->CPU transfer and might be a sync point
                let result: Int32 = nextTokenArray[0].item()
                return result
            }

            // Stop generation only at the general end-of-text token
            if next_token == Self.endToken {
                let endArr = MLXArray([Self.endToken]).reshaped([1,1])
                state.currentIds = MLX.concatenated([state.currentIds, endArr], axis: 1)
                if Profiler.enabled {
                    Log.tts.debug("End token \(Self.endToken) encountered. Appending and breaking.")
                }
                break
            }

            // Add next token to the sequence for parsing and for model input
            Profiler.time("Token concatenation (iter \(i))") {
                let nextTokenForConcat = nextTokenArray.reshaped([1, 1])
                state.currentIds = MLX.concatenated([state.currentIds, nextTokenForConcat], axis: 1)
            }

            // Add to history for repetition penalty *after* it's been sampled
            Profiler.time("History update") {
                generatedTokensForPenalty.append(next_token)
                if generatedTokensForPenalty.count > Self.repetitionContextSize { // Keep history to context size
                    generatedTokensForPenalty.removeFirst()
                }
            }

            // Prepare for the next iteration - only process the last token
            let forwardPassStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let nextInput = nextTokenArray.reshaped([1, 1])
            state.logits = model(nextInput, cache: state.cache)
            // Squeeze the sequence dimension since we're processing single tokens
            state.logits = state.logits.squeezed(axis: 1)
            let forwardPassEnd = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let forwardPassDuration = Profiler.enabled ? (forwardPassEnd - forwardPassStart) * 1000 : 0
            if Profiler.enabled {
                Log.perf.debug("⏱️ [PROFILE] Forward pass (iter \(i)): \(String(format: "%.2f", forwardPassDuration))ms")
            }

            // Clear GPU cache periodically
            if (i + 1) % 50 == 0 {
                Profiler.time("GPU cache clear") {
                    MLX.GPU.clearCache()
                }
            }

            if Profiler.enabled {
                let iterationEnd = CFAbsoluteTimeGetCurrent()
                let iterationDuration = (iterationEnd - iterationStart) * 1000

                // Print detailed timing every 10 iterations or for first 5
                if i < 5 || i % 10 == 0 {
                    Log.perf.debug("  🔀 Iteration \(i): \(String(format: "%.2f", iterationDuration))ms total")
                    Log.perf.debug("    📊 Forward: \(String(format: "%.2f", forwardPassDuration))ms")
                    Log.perf.debug("    🎯 Token: \(next_token)")
                }
            }

            i += 1
        }

        if i >= maxOutputTokens {
            Log.tts.warning("Reached max token count (\(maxOutputTokens)) during generation.")
        }

        // Parse the output into code lists
        let code_lists = Profiler.time("Output parsing") {
            parseOutput(tokens: state.currentIds.asArray(Int32.self).map { Int($0) })
        }

        // Generate audio using SNAC decoder
        let waveform = Profiler.time("SNAC decoding") {
            snacDecoder.decode(codes: code_lists)
        }

        let totalGenerationEnd = CFAbsoluteTimeGetCurrent()
        let totalDuration = (totalGenerationEnd - totalGenerationStart) * 1000
        Log.perf.info("🏁 [PROFILE] Total audio generation: \(String(format: "%.2f", totalDuration))ms")

        waveform.eval()
        return waveform.asArray(Float.self)
    }

    private func sampleNextToken(
        logits: MLXArray,
        history: MLXArray,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float = 1.3
    ) -> MLXArray {
        let samplingStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0

        // Start with raw logits
        var currentLogits = logits

        // 1. Apply repetition penalty if needed
        if repetitionPenalty != 1.0 && history.size > 0 {
            currentLogits = Profiler.time("Repetition penalty") {
                // Vectorised implementation to keep data on GPU/Metal.
                let indices = history // Int32 tensor with shape [K]
                let logits1D = currentLogits[0] // Shape [V]

                // Gather the logits corresponding to the history tokens.
                let gathered = MLX.take(logits1D, indices)

                // Compute updated logits according to the repetition penalty.
                let negMask   = gathered .< 0
                let updated   = MLX.where(
                    negMask,
                    gathered * repetitionPenalty,
                    gathered / repetitionPenalty
                )

                // Scatter the updated values back into the logits tensor using native subscript.
                logits1D[indices] = updated

                // Restore the [1, V] shape expected downstream.
                return logits1D.expandDims(at: 0)
            }
        }

        // 2. Apply temperature scaling
        let scaledLogits = Profiler.time("Temperature scaling") {
            currentLogits / max(temperature, 1e-6)
        }

        // 3. Apply top-p filtering
        var filteredLogits = scaledLogits
        if topP > 0.0 && topP < 1.0 {
            filteredLogits = Profiler.time("Top-p filtering") {
                let vocabSize = scaledLogits.shape[1]
                if vocabSize > 1 {
                    // Vectorised top-p filtering (no host round-trips).

                    // 1. Probabilities.
                    let probs = MLX.softmax(scaledLogits[0], axis: -1)        // [V]

                    // 2. Sort (descending).
                    let sortedIdx   = MLX.argSort(MLX.negative(probs))         // [V] Int32
                    let sortedProbs = MLX.take(probs, sortedIdx)               // [V]

                    // 3. Cumulative sum.
                    let cumProbs = sortedProbs.cumsum(axis: -1)                // [V]

                    // 4. Mask tokens occurring strictly after the cut-off.
                    let gtMask        = cumProbs .> topP                      // Bool [V]
                    let gtMaskInt     = gtMask.asType(.int32)                 // Int32 [V]
                    let prefix        = gtMaskInt.cumsum(axis: -1)            // Int32 [V]
                    let removeMaskSorted = prefix .> 1                        // Bool [V]

                    // 5. Bring mask back to original vocab order.
                    let invIdx          = MLX.argSort(sortedIdx)              // [V]
                    let removeMask      = MLX.take(removeMaskSorted, invIdx)  // Bool [V]

                    // 6. Apply mask: set filtered logits to -inf.
                    let negInfScalar    = MLXArray(-Float.infinity)           // scalar
                    let logits1D        = scaledLogits[0]
                    let filtered1D      = MLX.where(removeMask, negInfScalar, logits1D)

                    // 7. Restore [1, V] shape expected downstream.
                    return filtered1D.expandDims(at: 0)
                }
                return scaledLogits
            }
        }

        // 4. Sample from filtered distribution
        let nextTokenIdArray = Profiler.time("Categorical sampling") {
            MLXRandom.categorical(filteredLogits, count: 1)
        }

        if Profiler.enabled {
            let samplingEnd = CFAbsoluteTimeGetCurrent()
            let samplingDuration = (samplingEnd - samplingStart) * 1000
            Log.perf.debug("  🎲 Sampling total: \(String(format: "%.2f", samplingDuration))ms")
        }

        return nextTokenIdArray
    }

    private func parseOutput(tokens: [Int]) -> [[Int]] {
        // Find the last occurrence of the audio start token as defined in Constants
        let lastStartIndex = tokens.lastIndex(of: Self.audioCodeDataStartMarker) ?? -1

        // Get tokens after the last start token
        let relevantTokens = lastStartIndex >= 0 ? Array(tokens[(lastStartIndex + 1)...]) : tokens

        // Filter out the general end token (128258) and ensure codes are valid (>= codeOffset)
        // Python's llama.py uses token_to_remove = 128258 and does not filter a separate audioEndToken.
        let filteredTokens = relevantTokens.filter { $0 != Self.endToken && $0 >= Self.codeOffset }

        // Ensure length is multiple of 7 by trimming
        let newLength = (filteredTokens.count / 7) * 7
        let trimmedTokens = Array(filteredTokens[..<newLength])

        // Subtract offset from all tokens
        let adjustedTokens = trimmedTokens.map { $0 - Self.codeOffset }

        // Split into layers based on the stride pattern
        var layer1: [Int] = []
        var layer2: [Int] = []
        var layer3: [Int] = []

        // Process codes in groups of 7
        for i in 0..<(adjustedTokens.count / 7) {
            let base = 7 * i
            layer1.append(adjustedTokens[base])
            layer2.append(adjustedTokens[base + 1] - 4096)
            layer3.append(adjustedTokens[base + 2] - 2 * 4096)
            layer3.append(adjustedTokens[base + 3] - 3 * 4096)
            layer2.append(adjustedTokens[base + 4] - 4 * 4096)
            layer3.append(adjustedTokens[base + 5] - 5 * 4096)
            layer3.append(adjustedTokens[base + 6] - 6 * 4096)
        }

        return [layer1, layer2, layer3]
    }
}

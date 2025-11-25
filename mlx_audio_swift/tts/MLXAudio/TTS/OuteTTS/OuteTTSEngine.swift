//
//  OuteTTSEngine.swift
//  MLXAudio
//
//  Ported from: mlx_audio/tts/models/outetts/outetts.py
//

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXNN
@preconcurrency import MLXLMCommon
import MLXRandom
import Hub
import Tokenizers

// MARK: - OuteTTS Configuration

public struct OuteTTSConfig: Sendable {
    public let modelId: String
    public let sampleRate: Int
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let minP: Float
    public let repetitionPenalty: Float
    public let repetitionContextSize: Int

    public static let `default` = OuteTTSConfig(
        modelId: "mlx-community/Llama-OuteTTS-1.0-1B-4bit",
        sampleRate: 24000,
        maxTokens: 4096,
        temperature: 0.4,
        topP: 0.9,
        topK: 40,
        minP: 0.05,
        repetitionPenalty: 1.1,
        repetitionContextSize: 64
    )

    public init(
        modelId: String = "mlx-community/Llama-OuteTTS-1.0-1B-4bit",
        sampleRate: Int = 24000,
        maxTokens: Int = 4096,
        temperature: Float = 0.4,
        topP: Float = 0.9,
        topK: Int = 40,
        minP: Float = 0.05,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 64
    ) {
        self.modelId = modelId
        self.sampleRate = sampleRate
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}

// MARK: - Generation Result

public struct OuteTTSGenerationResult: Sendable {
    public let audio: [Float]
    public let sampleRate: Int
    public let duration: TimeInterval
    public let tokenCount: Int
    public let processingTime: TimeInterval
    public let realTimeFactor: Double

    public init(audio: [Float], sampleRate: Int, duration: TimeInterval, tokenCount: Int, processingTime: TimeInterval) {
        self.audio = audio
        self.sampleRate = sampleRate
        self.duration = duration
        self.tokenCount = tokenCount
        self.processingTime = processingTime
        self.realTimeFactor = duration / processingTime
    }
}

// MARK: - OuteTTS Engine

public actor OuteTTSEngine {
    private let config: OuteTTSConfig

    private nonisolated(unsafe) var model: OuteTTSLlamaModel?
    private nonisolated(unsafe) var tokenizer: (any Tokenizer)?
    private nonisolated(unsafe) var audioProcessor: OuteTTSAudioProcessor?
    private nonisolated(unsafe) var promptProcessor: OuteTTSPromptProcessor?
    private var defaultSpeaker: OuteTTSSpeakerProfile?
    private var eosTokenId: Int = 151645  // Default for OuteTTS <|im_end|>
    private var isLoaded: Bool = false

    public init(config: OuteTTSConfig = .default) {
        self.config = config
    }

    /// Load all model components
    public func load(progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }) async throws {
        let (model, tokenizer) = try await Self.loadOuteTTSModel(
            modelId: config.modelId,
            progressHandler: progressHandler
        )
        self.model = model
        self.tokenizer = tokenizer

        // Get EOS token ID from tokenizer
        if let eosId = tokenizer.convertTokenToId("<|im_end|>") {
            self.eosTokenId = eosId
        }

        // Load audio codec
        let audioProcessor = OuteTTSAudioProcessor(sampleRate: config.sampleRate)
        try await audioProcessor.loadCodec(progressHandler: progressHandler)
        self.audioProcessor = audioProcessor

        // buildTokenMaps caches special token IDs for fast prompt building
        // This avoids slow tokenizer.encode() calls on the full prompt string
        let promptProcessor = OuteTTSPromptProcessor()
        promptProcessor.buildTokenMaps(
            convertTokenToId: { token in
                tokenizer.convertTokenToId(token)
            },
            encode: { text in
                tokenizer.encode(text: text, addSpecialTokens: false)
            }
        )
        self.promptProcessor = promptProcessor

        // Load default speaker profile from bundle
        self.defaultSpeaker = loadDefaultSpeaker()
        self.isLoaded = true
    }

    /// Load OuteTTS model and tokenizer directly (like Orpheus approach)
    private static func loadOuteTTSModel(
        modelId: String,
        progressHandler: @escaping @Sendable (Progress) -> Void
    ) async throws -> (OuteTTSLlamaModel, any Tokenizer) {
        // Download model files using MLXLMCommon's download helper
        let configuration = ModelConfiguration(id: modelId, extraEOSTokens: ["<|im_end|>"])
        let modelDirectory = try await downloadModel(
            hub: HubApi.shared,
            configuration: configuration,
            progressHandler: progressHandler
        )

        // Load config
        let configURL = modelDirectory.appending(component: "config.json")
        var configData = try Data(contentsOf: configURL)

        // Fix rope_scaling values that may be integers instead of floats
        if var configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
           var ropeScaling = configDict["rope_scaling"] as? [String: Any] {
            for (key, value) in ropeScaling {
                if let intValue = value as? Int {
                    ropeScaling[key] = Double(intValue)
                }
            }
            configDict["rope_scaling"] = ropeScaling
            configData = try JSONSerialization.data(withJSONObject: configDict)
        }

        struct BaseConfig: Codable {
            let modelType: String
            let quantization: QuantizationConfig?

            enum CodingKeys: String, CodingKey {
                case modelType = "model_type"
                case quantization
            }

            struct QuantizationConfig: Codable {
                let groupSize: Int
                let bits: Int

                enum CodingKeys: String, CodingKey {
                    case groupSize = "group_size"
                    case bits
                }
            }
        }

        let baseConfig = try JSONDecoder().decode(BaseConfig.self, from: configData)

        guard baseConfig.modelType == "llama" else {
            throw OuteTTSEngineError.generationFailed("Unsupported model type: \(baseConfig.modelType)")
        }

        let llamaConfig = try JSONDecoder().decode(OuteTTSLlamaConfiguration.self, from: configData)
        let model = OuteTTSLlamaModel(llamaConfig)

        // Load weights from safetensor files
        var weights = [String: MLXArray]()
        let contents = try FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }
        for url in safetensorFiles {
            let w = try MLX.loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }

        // Remap keys: strip "model." prefix from OuteTTS weights
        var remappedWeights = [String: MLXArray]()
        for (key, value) in weights {
            let newKey: String
            if key.hasPrefix("model.model.") {
                newKey = String(key.dropFirst(6))
            } else if key.hasPrefix("model.lm_head.") {
                newKey = String(key.dropFirst(6))
            } else {
                newKey = key
            }
            remappedWeights[newKey] = value
        }

        // Apply sanitize (removes rotary embeddings)
        remappedWeights = model.sanitize(weights: remappedWeights)

        // Apply quantization if needed
        if let quant = baseConfig.quantization {
            quantize(model: model, groupSize: quant.groupSize, bits: quant.bits) { path, _ in
                remappedWeights["\(path).scales"] != nil
            }
        } else if remappedWeights.keys.contains(where: { $0.contains(".scales") }) {
            // Fallback: weights have .scales but no quantization config
            quantize(model: model, groupSize: 64, bits: 4) { path, _ in
                remappedWeights["\(path).scales"] != nil
            }
        }

        // Apply weights to model
        let parameters = ModuleParameters.unflattened(remappedWeights)
        try model.update(parameters: parameters, verify: [.all])
        eval(model)

        // Load tokenizer
        let tokenizerConfigURL = modelDirectory.appending(component: "tokenizer_config.json")
        let tokenizerDataURL = modelDirectory.appending(component: "tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw OuteTTSEngineError.generationFailed("tokenizer_config.json not found")
        }
        guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
            throw OuteTTSEngineError.generationFailed("tokenizer.json not found")
        }

        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerDataData = try Data(contentsOf: tokenizerDataURL)

        let decoder = JSONDecoder()
        let tokenizerConfig = try decoder.decode(Config.self, from: tokenizerConfigData)
        let tokenizerData = try decoder.decode(Config.self, from: tokenizerDataData)

        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )

        return (model, tokenizer)
    }

    /// Load default speaker profile from bundle
    private func loadDefaultSpeaker() -> OuteTTSSpeakerProfile? {
        guard let url = Bundle.main.url(forResource: "default_speaker", withExtension: "json") else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(OuteTTSSpeakerProfile.self, from: data)
        } catch {
            return nil
        }
    }

    /// Get speaker profile (from file, reference audio, or default)
    public func getSpeaker(
        voicePath: String? = nil,
        referenceAudio: MLXArray? = nil,
        referenceText: String? = nil,
        referenceWords: [(word: String, start: Double, end: Double)]? = nil
    ) async throws -> OuteTTSSpeakerProfile? {
        // Load from file
        if let path = voicePath {
            return try audioProcessor?.loadSpeaker(from: path)
        }

        // Create from reference audio
        if let audio = referenceAudio, let text = referenceText, let words = referenceWords {
            return try await audioProcessor?.createSpeakerFromTranscription(
                audio: audio,
                text: text,
                words: words
            )
        }

        // Return default speaker
        return defaultSpeaker
    }

    /// Generate audio from text
    public func generate(
        text: String,
        speaker: OuteTTSSpeakerProfile? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        maxTokens: Int? = nil
    ) async throws -> OuteTTSGenerationResult {
        guard let model = model,
              let promptProcessor = promptProcessor,
              let audioProcessor = audioProcessor else {
            throw OuteTTSEngineError.modelNotLoaded
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // Use provided speaker or default
        let speakerProfile = speaker ?? defaultSpeaker

        // getCompletionPromptTokens builds token IDs directly instead of
        // building a string and tokenizing it. This avoids 38s of BPE tokenization
        // on the ~18,000 character prompt with thousands of special tokens.
        // Audio code tokens are computed as (baseId + codeValue) instead of looked up.
        let inputTokens = promptProcessor.getCompletionPromptTokens(text: text, speaker: speakerProfile)

        // Generation parameters
        let temp = temperature ?? config.temperature
        let top = topP ?? config.topP
        let maxToks = maxTokens ?? config.maxTokens
        let eosToken = self.eosTokenId

        // Initialize with input tokens
        let inputIds = MLXArray(inputTokens.map { Int32($0) }).reshaped([1, -1])

        let cache = model.newCache()

        // Initial forward pass to populate cache (prefill)
        var logits = model(inputIds, cache: cache)
        logits = logits[0, -1].expandedDimensions(axis: 0)

        eval(logits)

        var generatedTokensHistory: [Int32] = []
        generatedTokensHistory.reserveCapacity(maxToks)
        let repetitionContextSize = config.repetitionContextSize

        // Generation loop
        for _ in 0..<maxToks {
            // Apply temperature
            var scaledLogits = logits / max(temp, 1e-6)

            // Apply repetition penalty
            if config.repetitionPenalty != 1.0 && !generatedTokensHistory.isEmpty {
                let history = MLXArray(generatedTokensHistory.suffix(repetitionContextSize))
                let logits1D = scaledLogits[0]
                let gathered = MLX.take(logits1D, history)
                let negMask = gathered .< 0
                let updated = MLX.where(
                    negMask,
                    gathered * config.repetitionPenalty,
                    gathered / config.repetitionPenalty
                )
                logits1D[history] = updated
                scaledLogits = logits1D.expandedDimensions(axis: 0)
            }

            // Apply top-p (nucleus) filtering
            if top > 0.0 && top < 1.0 {
                let probs = MLX.softmax(scaledLogits[0], axis: -1)
                let sortedIdx = MLX.argSort(MLX.negative(probs))
                let sortedProbs = MLX.take(probs, sortedIdx)
                let cumProbs = sortedProbs.cumsum(axis: -1)
                let gtMask = cumProbs .> top
                let gtMaskInt = gtMask.asType(.int32)
                let prefix = gtMaskInt.cumsum(axis: -1)
                let removeMaskSorted = prefix .> 1
                let invIdx = MLX.argSort(sortedIdx)
                let removeMask = MLX.take(removeMaskSorted, invIdx)
                let negInf = MLXArray(-Float.infinity)
                let filtered1D = MLX.where(removeMask, negInf, scaledLogits[0])
                scaledLogits = filtered1D.expandedDimensions(axis: 0)
            }

            eval(scaledLogits)

            // Sample next token
            let nextTokenArray = MLXRandom.categorical(scaledLogits, count: 1)
            let nextToken: Int32 = nextTokenArray[0].item()

            // Check for EOS
            if nextToken == Int32(eosToken) {
                break
            }

            generatedTokensHistory.append(nextToken)

            // Forward pass with single token
            let nextInput = nextTokenArray.reshaped([1, 1])
            logits = model(nextInput, cache: cache)
            logits = logits.squeezed(axis: 1)
            
            eval(logits)
        }

        let generatedTokens = generatedTokensHistory.map { Int($0) }

        // Extract audio codes from generated tokens
        let audioCodes = promptProcessor.extractAudioFromTokens(generatedTokens)

        guard !audioCodes[0].isEmpty && !audioCodes[1].isEmpty else {
            throw OuteTTSEngineError.generationFailed("No audio codes found in generated tokens")
        }

        // Decode audio using DAC codec
        guard let codec = audioProcessor.audioCodec else {
            throw OuteTTSEngineError.codecNotLoaded
        }

        let c1Array = MLXArray(audioCodes[0].map { Int32($0) })
        let c2Array = MLXArray(audioCodes[1].map { Int32($0) })
        let codesArray = MLX.stacked([c1Array, c2Array], axis: 0).reshaped([1, 2, -1])

        let audio = codec.decodeFromCodes(codesArray)
        eval(audio)

        let audioFlat = audio.reshaped([-1])
        let audioData = audioFlat.asArray(Float.self)

        let endTime = CFAbsoluteTimeGetCurrent()
        let processingTime = endTime - startTime
        let duration = Double(audioData.count) / Double(config.sampleRate)

        return OuteTTSGenerationResult(
            audio: audioData,
            sampleRate: config.sampleRate,
            duration: duration,
            tokenCount: generatedTokens.count,
            processingTime: processingTime
        )
    }

    /// Chunk text into smaller segments
    public func chunkText(_ text: String, maxWords: Int = 30) -> [String] {
        let pattern = "[.!?。！？︕︖]+"
        let regex = try? NSRegularExpression(pattern: pattern)
        let range = NSRange(text.startIndex..., in: text)

        var sentences: [String] = []
        var lastEnd = text.startIndex

        regex?.enumerateMatches(in: text, range: range) { match, _, _ in
            if let matchRange = match?.range, let swiftRange = Range(matchRange, in: text) {
                let sentence = String(text[lastEnd..<swiftRange.upperBound])
                    .trimmingCharacters(in: .whitespaces)
                if !sentence.isEmpty {
                    sentences.append(sentence)
                }
                lastEnd = swiftRange.upperBound
            }
        }

        if lastEnd < text.endIndex {
            let remaining = String(text[lastEnd...]).trimmingCharacters(in: .whitespaces)
            if !remaining.isEmpty {
                sentences.append(remaining)
            }
        }

        var chunks: [String] = []
        var currentChunk: [String] = []
        var currentLength = 0

        for sentence in sentences {
            let words = sentence.split(separator: " ")
            if currentLength + words.count > maxWords && !currentChunk.isEmpty {
                chunks.append(currentChunk.joined(separator: " "))
                currentChunk = []
                currentLength = 0
            }
            currentChunk.append(sentence)
            currentLength += words.count
        }

        if !currentChunk.isEmpty {
            chunks.append(currentChunk.joined(separator: " "))
        }

        return chunks.isEmpty ? [text] : chunks
    }

    /// Check if the engine is loaded
    public var loaded: Bool {
        isLoaded
    }
}

// MARK: - Errors

public enum OuteTTSEngineError: Error, LocalizedError {
    case modelNotLoaded
    case codecNotLoaded
    case invalidInput
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded. Call load() first."
        case .codecNotLoaded:
            return "Audio codec not loaded."
        case .invalidInput:
            return "Invalid input text or parameters."
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        }
    }
}

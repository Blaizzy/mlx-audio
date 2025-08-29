//
// SesameModelWrapper for Sesame TTS
// Main Model wrapper class with generation pipeline
// Based on Python mlx_audio/tts/models/sesame/sesame.py Model class
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Sesame TTS Error Types
enum SesameTTSError: Error {
    case inputTooLong(maxLength: Int, actualLength: Int)
    case modelNotInitialized
    case tokenizationFailed(reason: String)
    case generationFailed(reason: String)
    case invalidConfiguration(reason: String)

    var localizedDescription: String {
        switch self {
        case .inputTooLong(let max, let actual):
            return "Input too long: maximum \(max) tokens, got \(actual) tokens"
        case .modelNotInitialized:
            return "Model not initialized. Call ensureInitialized() first."
        case .tokenizationFailed(let reason):
            return "Tokenization failed: \(reason)"
        case .generationFailed(let reason):
            return "Audio generation failed: \(reason)"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        }
    }
}

/// Segment representing a piece of audio with text and speaker information
/// Equivalent to Python's Segment dataclass
struct Segment {
    let speaker: Int
    let text: String
    let audio: MLXArray
}

/// Generation result containing audio and metadata
/// Equivalent to Python's GenerationResult
struct GenerationResult {
    let audio: MLXArray
    let samples: Int
    let sampleRate: Int
    let segmentIdx: Int
    let tokenCount: Int
    let audioDuration: String
    let realTimeFactor: Float
    let prompt: [String: Any]
    let audioSamples: [String: Any]
    let processingTimeSeconds: Double
    let peakMemoryUsage: Float
}

/// Main Model wrapper for Sesame TTS
/// Equivalent to Python's Model class with Swift optimizations
class SesameModelWrapper: Module {
    @ModuleInfo var model: SesameModel?
    @ModuleInfo var audioTokenizer: Mimi?

    private let config: LlamaModelArgs
    private var textTokenizer: Any? // We'll implement tokenizer later
    private var streamingDecoder: MimiStreamingDecoder?
    private var watermarker: Any? // We'll implement watermarking later
    private let sampleRate: Int

    // Swift optimization flags
    private var isInitialized = false
    private var lastMemoryUsage: Float = 0.0

    /// Initialize the Model wrapper
    /// - Parameter config: Model configuration
    init(_ config: LlamaModelArgs) {
        self.config = config
        self.sampleRate = 24000 // Default sample rate, should match Mimi

        super.init()
    }

    /// Ensure model is initialized (lazy initialization)
    /// Following Kokoro's pattern for memory efficiency
    private func ensureInitialized() {
        guard !isInitialized else { return }

        autoreleasepool {
            // Initialize heavy ML components
            let sesameModel = SesameModel(config)
            sesameModel.setupCaches(maxBatchSize: 1)
            self._model.wrappedValue = sesameModel

            // Initialize Mimi codec for audio tokenization
            let mimiConfig = MimiConfig.mimi202407(numCodebooks: 8) // Default 8 codebooks
            let mimi = Mimi(mimiConfig)
            self._audioTokenizer.wrappedValue = mimi

            // Initialize streaming decoder
            self.streamingDecoder = MimiStreamingDecoder(mimi)

            // Initialize text tokenizer (Llama-3)
            do {
                self.textTokenizer = try SesameTokenizer()
            } catch {
                print("Warning: Could not initialize SesameTokenizer: \(error)")
                // Continue without tokenizer - will use fallback
            }

            // TODO: Initialize watermarker

            isInitialized = true
        }
    }

    /// Get the sample rate
    var sampleRateProperty: Int {
        return sampleRate
    }

    /// Get model layers (for quantization predicate)
    var layers: [LlamaTransformerLayer] {
        guard let model = model else { return [] }
        return model.backbone.layers
    }

    /// Reset model to free up memory (Kokoro-inspired)
    /// - Parameter preserveTextProcessing: Whether to keep tokenizer components
    func resetModel(preserveTextProcessing: Bool = true) {
        // Clear GPU cache first
        MLX.GPU.clearCache()

        // Reset heavy ML components using autoreleasepool
        autoreleasepool {
            self._model.wrappedValue = nil
            self._audioTokenizer.wrappedValue = nil
            self.streamingDecoder = nil
        }

        // Reset flags
        isInitialized = false
        lastMemoryUsage = 0.0
    }

    /// Create a Model wrapper with default Llama-1B configuration
    /// - Returns: Configured Model wrapper
    static func createDefault() -> SesameModelWrapper {
        let config = LlamaModelArgs.llama1B()
        return SesameModelWrapper(config)
    }

    /// Create a Model wrapper with custom configuration
    /// - Parameter flavor: Model flavor ("llama-1B" or "llama-100M")
    /// - Returns: Configured Model wrapper
    static func create(withFlavor flavor: String) -> SesameModelWrapper {
        let config = flavor == "llama-100M" ?
            LlamaModelArgs.llama100M() :
            LlamaModelArgs.llama1B()
        return SesameModelWrapper(config)
    }

    /// Tokenize text segment with speaker information
    /// - Parameters:
    ///   - text: Text to tokenize
    ///   - speaker: Speaker ID
    /// - Returns: Tuple of (tokens, mask) arrays
    private func tokenizeTextSegment(_ text: String, speaker: Int) -> (MLXArray, MLXArray) {
        guard let tokenizer = textTokenizer as? SesameTokenizer else {
            // Fallback: simple tokenization if tokenizer not available
            let tokens = text.split(separator: " ").map { String($0) }
            let tokenIds = tokens.enumerated().map { Int32($0.offset + 1) }
            let tokenArray = MLXArray(tokenIds).reshaped([1, -1])
            let maskArray = MLXArray.ones(tokenArray.shape, dtype: .bool)
            return (tokenArray, maskArray)
        }

        return tokenizer.prepareInputIds(text: text, speaker: speaker)
    }

    /// Tokenize audio into tokens
    /// - Parameters:
    ///   - audio: Audio array
    ///   - addEOS: Whether to add end-of-sequence token
    /// - Returns: Tuple of (tokens, mask) arrays
    private func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        guard let audioTokenizer = audioTokenizer else {
            fatalError("Audio tokenizer not initialized")
        }
        // Encode audio using Mimi codec
        let audioTokens = audioTokenizer.encode(audio) // [batch, codebooks, time]

        var tokens = audioTokens
        var mask = MLXArray.ones(like: tokens)

        // Add EOS frame if requested
        if addEOS {
            let eosFrame = MLXArray.zeros([tokens.shape[0], tokens.shape[1], 1])
            tokens = MLX.concatenated([tokens, eosFrame], axis: 2)
            let eosMask = MLXArray.zeros([mask.shape[0], mask.shape[1], 1])
            mask = MLX.concatenated([mask, eosMask], axis: 2)
        }

        // Reshape to [time, codebooks] format
        let reshapedTokens = tokens.swappedAxes(1, 2) // [batch, time, codebooks]
        let reshapedMask = mask.swappedAxes(1, 2)

        return (reshapedTokens, reshapedMask)
    }

    /// Tokenize a complete segment (text + audio)
    /// - Parameters:
    ///   - segment: Segment to tokenize
    ///   - addEOS: Whether to add end-of-sequence token
    /// - Returns: Tuple of (tokens, mask) arrays
    private func tokenizeSegment(_ segment: Segment, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        let (textTokens, textMask) = tokenizeTextSegment(segment.text, speaker: segment.speaker)
        let (audioTokens, audioMask) = tokenizeAudio(segment.audio, addEOS: addEOS)

        let combinedTokens = MLX.concatenated([textTokens, audioTokens], axis: 0)
        let combinedMask = MLX.concatenated([textMask, audioMask], axis: 0)

        return (combinedTokens, combinedMask)
    }

    /// Generate audio from text with optional voice prompt
    /// - Parameters:
    ///   - text: Text to generate audio for
    ///   - voice: Voice/prompt to use (optional)
    ///   - speaker: Speaker ID
    ///   - context: Context segments
    ///   - maxAudioLengthMs: Maximum audio length in milliseconds
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling parameter
    /// - Returns: GenerationResult with audio and metadata
    func generate(
        text: String,
        voice: String? = nil,
        speaker: Int = 0,
        context: [Segment] = [],
        maxAudioLengthMs: Float = 90000,
        temperature: Float = 0.9,
        topK: Int = 50
    ) throws -> GenerationResult {
        // Ensure model is initialized (lazy loading)
        ensureInitialized()

        // Use autoreleasepool for memory management (Kokoro pattern)
        return try autoreleasepool {
            let startTime = Date().timeIntervalSince1970

            // Prepare context (use provided or default voice)
            var currentContext = context
            if currentContext.isEmpty {
                if let voice = voice {
                    currentContext = defaultSpeakerPrompt(voice: voice)
                } else {
                    currentContext = defaultSpeakerPrompt(voice: "conversational_a")
                }
            }

            // Create sampler
            let sampler = { (logits: MLXArray) -> MLXArray in
                self.sampleTopK(logits: logits, temperature: temperature, topK: topK)
            }

            let maxAudioFrames = Int(maxAudioLengthMs / 80) // 80ms per frame

            // Tokenize context
            var allTokens: [MLXArray] = []
            var allMasks: [MLXArray] = []

            for segment in currentContext {
                let (tokens, mask) = tokenizeSegment(segment, addEOS: false)
                allTokens.append(tokens)
                allMasks.append(mask)
            }

            // Tokenize generation text
            let (genTokens, genMask) = tokenizeTextSegment(text, speaker: speaker)
            allTokens.append(genTokens)
            allMasks.append(genMask)

            // Concatenate all tokens
            let promptTokens = MLX.concatenated(allTokens, axis: 0)
            let promptMask = MLX.concatenated(allMasks, axis: 0)

            // Prepare for generation
            var currTokens = promptTokens.expandedDimensions(axis: 0)
            var currMask = promptMask.expandedDimensions(axis: 0)
            let currPos = MLXArray.arange(start: 0, stop: promptTokens.shape[0], dtype: .int32)
                .expandedDimensions(axis: 0)

            var samples: [MLXArray] = []
            var generatedFrameCount = 0

            // Maximum sequence length check
            let maxSeqLen = 2048 - maxAudioFrames
            if currTokens.shape[1] >= maxSeqLen {
                throw SesameTTSError.inputTooLong(
                    maxLength: maxSeqLen,
                    actualLength: Int(currTokens.shape[1])
                )
            }

            // Reset caches
            guard let model = model else {
                throw SesameTTSError.modelNotInitialized
            }
            model.resetCaches()
            streamingDecoder?.reset()

            // Generate audio frames
            for _ in 0..<maxAudioFrames {
                let sample = model.generateFrame(
                    tokens: currTokens,
                    tokensMask: currMask,
                    inputPos: currPos,
                    sampler: sampler
                )

                // Check for EOS (all zeros) - convert MLXArray to Bool
                let isAllZeros = MLX.all(sample .== 0).item(Bool.self)
                if isAllZeros {
                    break
                }

                samples.append(sample)

                // Prepare next frame
                let nextTokens = MLX.concatenated([
                    sample,
                    MLXArray.zeros([1, 1], dtype: .int32)
                ], axis: 1).expandedDimensions(axis: 0)

                let nextMask = MLX.concatenated([
                    MLXArray.ones(like: sample),
                    MLXArray.zeros([1, 1], dtype: .bool)
                ], axis: 1).expandedDimensions(axis: 0)

                currTokens = nextTokens
                currMask = nextMask
                generatedFrameCount += 1
            }

            // Decode audio tokens to audio
            let audioTokens = MLX.stacked(samples, axis: 0)
            let transposedTokens = audioTokens.swappedAxes(1, 2)

            let audio = streamingDecoder?.decodeFrames(transposedTokens) ?? MLXArray.zeros([1, 1])

            // Force evaluation and memory cleanup (Kokoro pattern)
            audio.eval()
            MLX.GPU.clearCache()

            // Calculate metadata
            let endTime = Date().timeIntervalSince1970
            let processingTime = endTime - startTime
            let tokenCount = generatedFrameCount * config.audioNumCodebooks
            let sampleCount = Int(audio.shape[1])
            let audioDurationSeconds = Double(sampleCount) / Double(sampleRate)
            let rtf = processingTime / audioDurationSeconds

            // Format duration
            let durationStr = formatDuration(audioDurationSeconds)

            return GenerationResult(
                audio: audio,
                samples: sampleCount,
                sampleRate: sampleRate,
                segmentIdx: 0,
                tokenCount: tokenCount,
                audioDuration: durationStr,
                realTimeFactor: Float(rtf),
                prompt: [
                    "tokens": tokenCount,
                    "tokens-per-sec": Double(tokenCount) / processingTime
                ],
                audioSamples: [
                    "samples": sampleCount,
                    "samples-per-sec": Double(sampleCount) / processingTime
                ],
                processingTimeSeconds: processingTime,
                peakMemoryUsage: lastMemoryUsage
            )
        }
    }

    /// Sample from logits using top-k sampling
    /// - Parameters:
    ///   - logits: Logits array
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k parameter
    /// - Returns: Sampled token indices
    private func sampleTopK(logits: MLXArray, temperature: Float, topK: Int) -> MLXArray {
        let scaledLogits = logits / temperature

        // Get top-k using argSort (following Orpheus implementation)
        let sortedIndices = MLX.argSort(MLX.negative(scaledLogits), axis: -1)
        let topKIndices = sortedIndices[0..., 0..<topK]

        // Get corresponding values
        let topKValues = MLX.take(scaledLogits, topKIndices, axis: -1)

        // Sample from top-k
        let probs = MLX.softmax(topKValues, axis: -1)
        let sampleIdx = MLXRandom.categorical(MLX.log(probs), count: 1)[0]

        return topKIndices[0..., sampleIdx]
    }

    /// Get default speaker prompt for a voice
    /// - Parameter voice: Voice name
    /// - Returns: Array of segments for the voice prompt
    private func defaultSpeakerPrompt(voice: String) -> [Segment] {
        // TODO: Implement voice prompt loading
        // For now, return empty segments
        return []
    }

    /// Format duration as HH:MM:SS.mmm
    /// - Parameter durationSeconds: Duration in seconds
    /// - Returns: Formatted duration string
    private func formatDuration(_ durationSeconds: Double) -> String {
        let hours = Int(durationSeconds / 3600)
        let minutes = Int((durationSeconds.truncatingRemainder(dividingBy: 3600)) / 60)
        let seconds = Int(durationSeconds.truncatingRemainder(dividingBy: 60))
        let milliseconds = Int((durationSeconds.truncatingRemainder(dividingBy: 1)) * 1000)

        return String(format: "%02d:%02d:%02d.%03d", hours, minutes, seconds, milliseconds)
    }
}

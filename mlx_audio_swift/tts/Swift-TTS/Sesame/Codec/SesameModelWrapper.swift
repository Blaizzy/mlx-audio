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
public struct Segment {
    let speaker: Int
    let text: String
    let audio: MLXArray?

    public init(speaker: Int, text: String, audio: MLXArray? = nil) {
        self.speaker = speaker
        self.text = text
        self.audio = audio
    }
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
    private var textTokenizer: SesameTokenizer? // Llama-3 tokenizer
    private var voiceManager: SesameVoiceManager? // Voice management system
    private var streamingDecoder: MimiStreamingDecoder?
    private var watermarker: Any? // We'll implement watermarking later
    private var sampleRate: Int

    // Swift optimization flags
    private var isInitialized = false
    private var lastMemoryUsage: Float = 0.0

    /// Initialize the Model wrapper
    /// - Parameter config: Model configuration
    init(_ config: LlamaModelArgs) {
        self.config = config
        // Sample rate will be set when Mimi is initialized
        self.sampleRate = 24000 // Default, will be updated

        super.init()
    }

    /// Ensure model is initialized (lazy initialization)
    /// Following Kokoro's pattern for memory efficiency
    private func ensureInitialized() {
        guard !isInitialized else { return }
        
        print("🚀 DEBUG ensureInitialized: Starting model initialization...")

        autoreleasepool {
            print("🚀 DEBUG ensureInitialized: Creating SesameModel...")
            // Initialize heavy ML components
            let sesameModel = SesameModel(config)
            print("🚀 DEBUG ensureInitialized: SesameModel created, setting up caches...")
            
            sesameModel.setupCaches(maxBatchSize: 1)
            print("🚀 DEBUG ensureInitialized: Caches set up, assigning to wrapper...")
            
            self._model.wrappedValue = sesameModel
            print("🚀 DEBUG ensureInitialized: SesameModel assigned successfully")

            print("🚀 DEBUG ensureInitialized: Creating Mimi codec...")
            // CRITICAL: We need pre-trained Mimi weights to work
            
            // DEBUG: Print what's in the Resources folder
            if let resourcesPath = Bundle.main.path(forResource: "", ofType: "", inDirectory: "Sesame/Resources") {
                print("🔍 DEBUG: Sesame/Resources folder found at: \(resourcesPath)")
                do {
                    let contents = try FileManager.default.contentsOfDirectory(atPath: resourcesPath)
                    print("🔍 DEBUG: Contents of Sesame/Resources folder:")
                    for item in contents {
                        print("  - \(item)")
                    }
                } catch {
                    print("🔍 DEBUG: Error reading Sesame/Resources: \(error)")
                }
            } else {
                print("🔍 DEBUG: Sesame/Resources folder not found")
            }
            
            // Try different paths to find the Mimi weights
            let possibleFiles = [
                "tokenizer-e351c8d8-checkpoint125.safetensors",  // Original filename
                "sesame-mimi.safetensors",  // Renamed version
                "sesame-mini.safetensors"   // Alternative naming
            ]
            
            var weightsPath: String?
            for fileName in possibleFiles {
                let fileNameWithoutExt = String(fileName.dropLast(12)) // Remove .safetensors
                if let path = Bundle.main.path(forResource: fileNameWithoutExt, ofType: "safetensors", inDirectory: "Sesame/Resources") {
                    print("🔍 DEBUG: Found weights file: \(fileName) at \(path)")
                    weightsPath = path
                    break
                } else if let path = Bundle.main.path(forResource: fileNameWithoutExt, ofType: "safetensors") {
                    print("🔍 DEBUG: Found weights file: \(fileName) at root: \(path)")
                    weightsPath = path
                    break
                }
            }
            
            guard let foundWeightsPath = weightsPath else {
                fatalError("""
                🚨 MISSING MIMI WEIGHTS! 🚨
                
                Sesame TTS requires pre-trained Mimi weights to function.
                
                Could not find any of these files in bundle:
                - sesame-mimi.safetensors
                - sesame-mini.safetensors
                - tokenizer-e351c8d8-checkpoint125.safetensors
                
                TO FIX THIS:
                1. Download the Mimi weights from HuggingFace:
                   Repository: kyutai/moshiko-pytorch-bf16
                   File: tokenizer-e351c8d8-checkpoint125.safetensors
                   
                2. Rename the file to: sesame-mimi.safetensors
                
                3. Add it to your Xcode project in:
                   Swift-TTS/Sesame/Resources/sesame-mimi.safetensors
                
                4. Make sure it's added to the target's bundle resources
                """)
            }
            
            print("🚀 DEBUG ensureInitialized: Found Mimi weights at: \(foundWeightsPath)")
            
            // Initialize Mimi codec with pre-trained weights
            let mimiConfig = MimiConfig.mimi202407(numCodebooks: config.audioNumCodebooks)
            let mimi = Mimi(mimiConfig)
            
            // Load the pre-trained weights
            print("🚀 DEBUG ensureInitialized: Loading Mimi weights...")
            let weightsURL = URL(fileURLWithPath: foundWeightsPath)
            let mimiWithWeights = mimi.loadPytorchWeights(url: weightsURL, strict: false)
            
            print("🚀 DEBUG ensureInitialized: Mimi weights loaded successfully, created with \(config.audioNumCodebooks) codebooks, assigning...")
            
            self._audioTokenizer.wrappedValue = mimiWithWeights
            print("🚀 DEBUG ensureInitialized: Mimi assigned successfully")

            // Update sample rate from Mimi codec
            self.sampleRate = Int(mimi.sampleRate)
            print("🚀 DEBUG ensureInitialized: Sample rate set to \(self.sampleRate)")

            print("🚀 DEBUG ensureInitialized: Creating streaming decoder...")
            // Initialize streaming decoder
            self.streamingDecoder = MimiStreamingDecoder(mimi)
            print("🚀 DEBUG ensureInitialized: Streaming decoder created")

            print("🚀 DEBUG ensureInitialized: Initializing text tokenizer...")
            // Initialize text tokenizer (Llama-3)
            do {
                let tokenizer = try SesameTokenizer()
                print("🚀 DEBUG ensureInitialized: SesameTokenizer created successfully")
                self.textTokenizer = tokenizer
                // Initialize voice manager with tokenizer
                self.voiceManager = SesameVoiceManager(tokenizer: tokenizer)
                print("🚀 DEBUG ensureInitialized: Voice manager created")
            } catch {
                print("🚀 WARNING: Could not initialize SesameTokenizer: \(error)")
                // Continue without tokenizer - will use fallback
                self.voiceManager = SesameVoiceManager(tokenizer: nil)
                print("🚀 DEBUG ensureInitialized: Using fallback voice manager")
            }

            // TODO: Initialize watermarker

            isInitialized = true
            print("🚀 DEBUG ensureInitialized: Initialization complete!")
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

    /// Create a Model wrapper with sesame_config.json configuration
    /// - Returns: Configured Model wrapper
    static func createDefault() throws -> SesameModelWrapper {
        // Load configuration from sesame_config.json
        guard let configPath = Bundle.main.path(forResource: "sesame_config", ofType: "json") else {
            throw SesameTTSError.invalidConfiguration(reason: "Could not find sesame_config.json in bundle")
        }

        let config = try LlamaModelArgs.fromSesameConfig(configPath: configPath)
        return SesameModelWrapper(config)
    }

    /// Create a Model wrapper with custom configuration
    /// - Parameter flavor: Model flavor ("llama-1B" or "llama-100M")
    /// - Returns: Configured Model wrapper
    static func create(withFlavor flavor: String) throws -> SesameModelWrapper {
        // For backward compatibility, still support flavor-based creation
        // But this will use the dynamic config loading internally
        guard let configPath = Bundle.main.path(forResource: "sesame_config", ofType: "json") else {
            throw SesameTTSError.invalidConfiguration(reason: "Could not find sesame_config.json in bundle")
        }

        let config = try LlamaModelArgs.fromSesameConfig(configPath: configPath)
        return SesameModelWrapper(config)
    }

    /// Create a Model wrapper with custom configuration file
    /// - Parameter configPath: Path to sesame_config.json file
    /// - Returns: Configured Model wrapper
    static func create(withConfigPath configPath: String) throws -> SesameModelWrapper {
        let config = try LlamaModelArgs.fromSesameConfig(configPath: configPath)
        return SesameModelWrapper(config)
    }

    /// Validate the current configuration for dimension compatibility
    /// - Returns: Validation results or throws error if invalid
    func validateConfiguration() throws -> (backboneHiddenSize: Int, decoderHiddenSize: Int, projectionShape: (Int, Int)) {
        let backboneHiddenSize = config.hiddenSize
        let decoderHiddenSize = config.depthDecoderConfig?.hiddenSize ?? config.hiddenSize
        let projectionShape = (backboneHiddenSize, decoderHiddenSize)

        print("✅ Configuration Validation:")
        print("  - Backbone hidden size: \(backboneHiddenSize)")
        print("  - Decoder hidden size: \(decoderHiddenSize)")
        print("  - Projection shape: \(projectionShape.0) -> \(projectionShape.1)")

        // Verify dimensions are reasonable (backbone should be larger than decoder)
        guard backboneHiddenSize >= decoderHiddenSize else {
            throw SesameTTSError.invalidConfiguration(reason: "Backbone hidden size (\(backboneHiddenSize)) should be >= decoder hidden size (\(decoderHiddenSize))")
        }

        // Verify projection makes sense (should reduce dimensions)
        if projectionShape.0 <= projectionShape.1 {
            print("⚠️  WARNING: Projection does not reduce dimensions - this might not be optimal")
        }

        return (backboneHiddenSize, decoderHiddenSize, projectionShape)
    }

    /// Get available voices
    /// - Returns: Array of available voice names
    public func getAvailableVoices() -> [String] {
        guard let voiceManager = voiceManager else { return [] }
        return voiceManager.getAvailableVoices()
    }

    /// Validate if a voice exists
    /// - Parameter voiceName: Name of the voice to validate
    /// - Returns: True if voice exists
    public func validateVoice(voiceName: String) -> Bool {
        guard let voiceManager = voiceManager else { return false }
        return voiceManager.validateVoice(voiceName: voiceName)
    }

    /// Get voice description
    /// - Parameter voiceName: Name of the voice
    /// - Returns: Voice description
    public func getVoiceDescription(voiceName: String) -> String {
        guard let voiceManager = voiceManager else { return "Voice manager not initialized" }
        return voiceManager.getVoiceDescription(voiceName: voiceName)
    }

    /// Add custom voice configuration
    /// - Parameters:
    ///   - config: Voice configuration
    ///   - prompts: Voice prompts (optional)
    public func addCustomVoice(config: VoiceConfig, prompts: [VoicePrompt] = []) {
        voiceManager?.addVoice(config: config, prompts: prompts)
    }

    /// Tokenize text segment with speaker information
    /// - Parameters:
    ///   - text: Text to tokenize
    ///   - speaker: Speaker ID
    /// - Returns: Tuple of (tokens, mask) arrays with shape (seq_len, 33)
    private func tokenizeTextSegment(_ text: String, speaker: Int) -> (MLXArray, MLXArray) {
        print("🔤 DEBUG tokenizeTextSegment: Input text='\(text)', speaker=\(speaker)")

        guard let tokenizer = textTokenizer else {
            print("🔤 DEBUG tokenizeTextSegment: Using fallback tokenizer")
            // Fallback: simple tokenization if tokenizer not available
            let tokens = text.split(separator: " ").map { String($0) }
            let tokenIds = tokens.enumerated().map { Int32($0.offset + 1) }
            let tokenArray = MLXArray(tokenIds).reshaped([-1, 1]) // (seq_len, 1)
            print("🔤 DEBUG tokenizeTextSegment: tokenArray shape=\(tokenArray.shape)")

            // Create frame with shape (seq_len, 33) like Python
            let textFrame = MLXArray.zeros([tokenArray.shape[0], 33], dtype: .int32)
            print("🔤 DEBUG tokenizeTextSegment: textFrame shape=\(textFrame.shape)")

            // Put text tokens in the last column (dynamically calculated)
            let lastColIndex = textFrame.shape[1] - 1  // Should be 32 for 33 columns
            print("🔤 DEBUG tokenizeTextSegment: fallback lastColIndex=\(lastColIndex)")

            // Squeeze tokenArray to remove the extra dimension for proper broadcasting
            let squeezedTokenArray = tokenArray.squeezed()  // [seq_len, 1] -> [seq_len]
            print("🔤 DEBUG tokenizeTextSegment: squeezedTokenArray shape=\(squeezedTokenArray.shape)")

            textFrame[MLXArray(0..<tokenArray.shape[0]), MLXArray([lastColIndex])] = squeezedTokenArray
            // Set mask to all 1s for simplicity - don't mask anything in fallback mode
            let textFrameMask = MLXArray.ones([tokenArray.shape[0], 33], dtype: .bool)

            print("🔤 DEBUG tokenizeTextSegment: returning fallback result")
            return (textFrame, textFrameMask)
        }

        print("🔤 DEBUG tokenizeTextSegment: Using SesameTokenizer")
        
        // FIXED: Match Python format exactly: f"[{speaker}]{text}"
        let formattedText = "[\(speaker)]\(text)"
        print("🔤 DEBUG tokenizeTextSegment: formattedText='\(formattedText)'")
        
        // FIXED: Use the tokenizer's encode method without parameter label
        let tokenIds = tokenizer.encode(formattedText)
        let tokens = MLXArray(tokenIds.map { Int32($0) })
        print("🔤 DEBUG tokenizeTextSegment: tokenizer returned tokens shape=\(tokens.shape)")

        // tokens should be [seq_len], we need [seq_len, 33]
        let seqLen = tokens.shape[0]
        print("🔤 DEBUG tokenizeTextSegment: seqLen=\(seqLen)")

        let textFrame = MLXArray.zeros([seqLen, 33], dtype: .int32)
        let textFrameMask = MLXArray.zeros([seqLen, 33], dtype: .bool)
        print("🔤 DEBUG tokenizeTextSegment: created textFrame shape=\(textFrame.shape)")

        // Put text tokens in the last column (column 32) - match Python exactly
        let lastColumnIndex = 32  // Fixed: Python uses [:, -1] which is column 32 for 33 columns
        print("🔤 DEBUG tokenizeTextSegment: lastColumnIndex=\(lastColumnIndex)")

        textFrame[MLXArray(0..<seqLen), MLXArray([lastColumnIndex])] = tokens
        textFrameMask[MLXArray(0..<seqLen), MLXArray([lastColumnIndex])] = MLXArray.ones([seqLen], dtype: .bool)

        print("🔤 DEBUG tokenizeTextSegment: Successfully set column \(lastColumnIndex)")
        print("🔤 DEBUG tokenizeTextSegment: returning result with shapes: tokens=\(textFrame.shape), mask=\(textFrameMask.shape)")
        return (textFrame, textFrameMask)
    }

    /// Tokenize audio into tokens
    /// - Parameters:
    ///   - audio: Audio array
    ///   - addEOS: Whether to add end-of-sequence token
    /// - Returns: Tuple of (tokens, mask) arrays with shape (seq_len, 33)
    private func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        guard let audioTokenizer = audioTokenizer else {
            fatalError("Audio tokenizer not initialized")
        }

        // FIXED: Match Python format exactly: audio[None, None, ...]
        // Add batch and channel dimensions like Python
        let audioWithDims = audio.expandedDimensions(axis: 0).expandedDimensions(axis: 0) // [1, 1, samples]
        
        // Encode audio using Mimi codec - returns (K, T) = (codebooks, time)
        let audioTokens = audioTokenizer.encode(audioWithDims)[0] // [0] like Python to remove batch dim

        // Add EOS frame if requested
        var processedTokens = audioTokens
        if addEOS {
            let eosFrame = MLXArray.zeros([audioTokens.shape[0], 1])
            processedTokens = MLX.concatenated([audioTokens, eosFrame], axis: 1)
        }

        // Create frame with shape (seq_len, 33) like Python
        let seqLen = processedTokens.shape[1] // time dimension

        let audioFrame = MLXArray.zeros([seqLen, 33], dtype: .int32)
        let audioFrameMask = MLXArray.zeros([seqLen, 33], dtype: .bool)

        // FIXED: Match Python exactly: audio_frame[:, :-1] = audio_tokens.swapaxes(0, 1)
        // Put audio tokens in all columns EXCEPT the last one (columns 0-31, not 0-32)
        let audioTokensTransposed = processedTokens.swappedAxes(0, 1) // [time, codebooks]

        // Python uses [:, :-1] which means all columns except the last
        audioFrame[MLXArray(0..<seqLen), MLXArray(0..<32)] = audioTokensTransposed
        audioFrameMask[MLXArray(0..<seqLen), MLXArray(0..<32)] = MLXArray.ones([seqLen, 32], dtype: .bool)
        return (audioFrame, audioFrameMask)
    }

    /// Tokenize a complete segment (text + audio)
    /// - Parameters:
    ///   - segment: Segment to tokenize
    ///   - addEOS: Whether to add end-of-sequence token
    /// - Returns: Tuple of (tokens, mask) arrays with shape (total_seq_len, 33)
    private func tokenizeSegment(_ segment: Segment, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        print("🔗 DEBUG tokenizeSegment: segment.text='\(segment.text)', speaker=\(segment.speaker), audio=\(segment.audio != nil ? "present" : "nil"), addEOS=\(addEOS)")

        let (textTokens, textMask) = tokenizeTextSegment(segment.text, speaker: segment.speaker)
        print("🔗 DEBUG tokenizeSegment: textTokens shape=\(textTokens.shape), textMask shape=\(textMask.shape)")

        // FIXED: Match Python logic exactly - Python assumes audio is always present
        // Only handle optional audio if we're in a context where it might be missing
        if let audio = segment.audio {
            print("🔗 DEBUG tokenizeSegment: Processing audio")
            let (audioTokens, audioMask) = tokenizeAudio(audio, addEOS: addEOS)
            print("🔗 DEBUG tokenizeSegment: audioTokens shape=\(audioTokens.shape), audioMask shape=\(audioMask.shape)")

            // Concatenate along axis 0 (sequence dimension) like Python
            print("🔗 DEBUG tokenizeSegment: Concatenating along axis 0")
            let combinedTokens = MLX.concatenated([textTokens, audioTokens], axis: 0)
            let combinedMask = MLX.concatenated([textMask, audioMask], axis: 0)

            print("🔗 DEBUG tokenizeSegment: returning combined result with shapes: tokens=\(combinedTokens.shape), mask=\(combinedMask.shape)")
            return (combinedTokens, combinedMask)
        } else {
            print("🔗 DEBUG tokenizeSegment: No audio, returning text tokens only")
            return (textTokens, textMask)
        }
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
    ///   - stream: Whether to use streaming decoder for real-time generation
    ///   - voiceMatch: Whether to use voice matching (append text to voice prompt)
    /// - Returns: GenerationResult with audio and metadata
    func generate(
        text: String,
        voice: String? = nil,
        speaker: Int = 0,
        context: [Segment] = [],
        maxAudioLengthMs: Float = 90000,
        temperature: Float = 0.9,
        topK: Int = 50,
        stream: Bool = false,
        voiceMatch: Bool = true
    ) throws -> GenerationResult {
        // Ensure model is initialized (lazy loading)
        ensureInitialized()

        // Use autoreleasepool for memory management (Kokoro pattern)
        return try autoreleasepool {
            let startTime = Date().timeIntervalSince1970

            // Prepare context (use provided or default voice)
            var currentContext = context
            if currentContext.isEmpty {
                if voiceMatch {
                    // Use voice matching - will append text to voice prompt later
                    if let voice = voice {
                        currentContext = defaultSpeakerPrompt(voice: voice)
                    } else {
                        currentContext = defaultSpeakerPrompt(voice: "conversational_a")
                    }
                } else {
                    // Don't use voice matching - use default voice
                    if let voice = voice {
                        currentContext = defaultSpeakerPrompt(voice: voice)
                    } else {
                        currentContext = defaultSpeakerPrompt(voice: "conversational_a")
                    }
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

            // Handle voice matching like Python
            var tokens: [MLXArray] = []
            var tokensMask: [MLXArray] = []

            print("🎯 DEBUG generate: Starting voice matching logic...")
            
            if voiceMatch && !currentContext.isEmpty {
                print("🎯 DEBUG generate: Voice matching enabled, combining context with text")
                // FIXED: Match Python logic exactly
                let generationText = (currentContext[0].text + " " + text).trimmingCharacters(in: .whitespaces)
                print("🎯 DEBUG generate: Combined text length: \(generationText.count)")
                
                let currentContextUpdated = [
                    Segment(
                        speaker: speaker,
                        text: generationText,
                        audio: currentContext[0].audio
                    )
                ]
                
                print("🎯 DEBUG generate: Processing combined segment...")
                // Process the single combined segment
                for (index, segment) in currentContextUpdated.enumerated() {
                    print("🎯 DEBUG generate: Processing segment \(index)")
                    let (segmentTokens, segmentTokensMask) = tokenizeSegment(segment, addEOS: false)
                    print("🎯 DEBUG generate: Segment \(index) tokenized, shapes: tokens=\(segmentTokens.shape), mask=\(segmentTokensMask.shape)")
                    
                    tokens.append(segmentTokens)
                    tokensMask.append(segmentTokensMask)
                    print("🎯 DEBUG generate: Segment \(index) added to arrays")
                }
            } else {
                print("🎯 DEBUG generate: Voice matching disabled, processing context and text separately")
                // FIXED: Process context segments without voice matching
                for (index, segment) in currentContext.enumerated() {
                    print("🎯 DEBUG generate: Processing context segment \(index)")
                    let (segmentTokens, segmentTokensMask) = tokenizeSegment(segment, addEOS: false)
                    print("🎯 DEBUG generate: Context segment \(index) tokenized")
                    
                    tokens.append(segmentTokens)
                    tokensMask.append(segmentTokensMask)
                    print("🎯 DEBUG generate: Context segment \(index) added")
                }
                
                print("🎯 DEBUG generate: Processing generation text as separate segment")
                let (genTokens, genMask) = tokenizeTextSegment(text, speaker: speaker)
                print("🎯 DEBUG generate: Generation text tokenized, shapes: tokens=\(genTokens.shape), mask=\(genMask.shape)")
                
                tokens.append(genTokens)
                tokensMask.append(genMask)
                print("🎯 DEBUG generate: Generation text added to arrays")
            }

            print("🎯 DEBUG generate: All segments processed, total segments: \(tokens.count)")
            print("🎯 DEBUG generate: About to concatenate tokens...")

            // Add safety check before concatenation
            guard !tokens.isEmpty else {
                throw SesameTTSError.tokenizationFailed(reason: "No tokens to process")
            }

            // Log shapes before concatenation
            for (i, tokenArray) in tokens.enumerated() {
                print("🎯 DEBUG generate: Token array \(i) shape: \(tokenArray.shape)")
            }

            // Concatenate all tokens along sequence axis (axis 0) like Python
            print("🎯 DEBUG generate: Starting token concatenation...")
            let promptTokens = MLX.concatenated(tokens, axis: 0)
            print("🎯 DEBUG generate: Tokens concatenated successfully, shape: \(promptTokens.shape)")
            
            print("🎯 DEBUG generate: Starting mask concatenation...")
            let promptMask = MLX.concatenated(tokensMask, axis: 0)
            print("🎯 DEBUG generate: Masks concatenated successfully, shape: \(promptMask.shape)")

            print("🎯 DEBUG generate: Adding batch dimensions...")
            // Prepare for generation - add batch dimension like Python
            var currTokens = promptTokens.expandedDimensions(axis: 0)  // [1, seq_len, 33]
            print("🎯 DEBUG generate: currTokens shape after batch dim: \(currTokens.shape)")
            
            var currMask = promptMask.expandedDimensions(axis: 0)      // [1, seq_len, 33]
            print("🎯 DEBUG generate: currMask shape after batch dim: \(currMask.shape)")
            
            print("🎯 DEBUG generate: Creating position array...")
            var currPos = MLXArray.arange(start: 0, stop: promptTokens.shape[0], dtype: .int32)
                .expandedDimensions(axis: 0)  // [1, seq_len]
            print("🎯 DEBUG generate: currPos shape: \(currPos.shape)")

            var samples: [MLXArray] = []
            var generatedFrameCount = 0

            print("🎯 DEBUG generate: Checking sequence length limits...")
            // Maximum sequence length check
            let maxSeqLen = 2048 - maxAudioFrames
            if currTokens.shape[1] >= maxSeqLen {
                throw SesameTTSError.inputTooLong(
                    maxLength: maxSeqLen,
                    actualLength: Int(currTokens.shape[1])
                )
            }
            print("🎯 DEBUG generate: Sequence length check passed")

            print("🎯 DEBUG generate: Resetting caches...")
            // Reset caches
            guard let model = model else {
                throw SesameTTSError.modelNotInitialized
            }
            model.resetCaches()
            streamingDecoder?.reset()
            print("🎯 DEBUG generate: Caches reset complete")

            print("🎯 DEBUG generate: About to start frame generation loop...")
            // Generate audio frames
            print("🎵 DEBUG generate: Starting audio frame generation...")
            var loopCount = 0
            let maxLoopIterations = 3 // Very small limit for debugging
            
            // Force garbage collection before starting
            MLX.GPU.clearCache()
            
            for i in 0..<min(maxAudioFrames, maxLoopIterations) {
                print("🎵 DEBUG generate: === Frame \(i+1)/\(maxLoopIterations) ===")
                print("🎵 DEBUG generate: currTokens shape: \(currTokens.shape)")
                print("🎵 DEBUG generate: currMask shape: \(currMask.shape)")
                print("🎵 DEBUG generate: currPos shape: \(currPos.shape)")
                
                print("🎵 DEBUG generate: Calling generateFrame...")
                
                let sample = model.generateFrame(
                    tokens: currTokens,
                    tokensMask: currMask,
                    inputPos: currPos,
                    sampler: sampler
                )
                
                print("🎵 DEBUG generate: generateFrame returned, sample shape: \(sample.shape)")

                // Check for EOS (all zeros) - convert MLXArray to Bool
                let isAllZeros = MLX.all(sample .== 0).item(Bool.self)
                print("🎵 DEBUG generate: EOS check: \(isAllZeros)")
                
                if isAllZeros {
                    print("🎵 DEBUG generate: EOS detected, breaking")
                    break
                }

                samples.append(sample)
                print("🎵 DEBUG generate: Sample appended, total samples: \(samples.count)")

                print("🎵 DEBUG generate: Preparing next frame...")
                // Prepare next frame like Python: [sample, zeros] then expand dims
                let sampleWithPadding = MLX.concatenated([
                    sample,
                    MLXArray.zeros([1, 1], dtype: .int32)
                ], axis: 1)  // [1, 2]
                print("🎵 DEBUG generate: sampleWithPadding shape: \(sampleWithPadding.shape)")

                let maskWithPadding = MLX.concatenated([
                    MLXArray.ones(like: sample),
                    MLXArray.zeros([1, 1], dtype: .bool)
                ], axis: 1)  // [1, 2]
                print("🎵 DEBUG generate: maskWithPadding shape: \(maskWithPadding.shape)")

                let nextTokens = sampleWithPadding.expandedDimensions(axis: 1)  // [1, 1, 2]
                let nextMask = maskWithPadding.expandedDimensions(axis: 1)      // [1, 1, 2]
                print("🎵 DEBUG generate: nextTokens shape: \(nextTokens.shape), nextMask shape: \(nextMask.shape)")

                currTokens = nextTokens
                currMask = nextMask
                
                print("🎵 DEBUG generate: Updating currPos...")
                // CRITICAL: Update currPos like Python does: curr_pos = curr_pos[:, -1:] + 1
                let lastIndex = currPos.shape[1] - 1
                let lastPos = currPos[0..., lastIndex..<currPos.shape[1]]  // Get [:, -1:]
                currPos = lastPos + 1
                
                print("🎵 DEBUG generate: Updated currPos shape: \(currPos.shape)")
                
                generatedFrameCount += 1
                loopCount += 1
                
                print("🎵 DEBUG generate: Frame \(i+1) complete")
                
                // Force evaluation and cleanup after each frame
                MLX.eval([currTokens, currMask, currPos])
                MLX.GPU.clearCache()
            }
            
            print("🎵 DEBUG generate: Frame generation complete after \(loopCount) iterations")
            
            // Early exit during debugging - don't try to decode audio yet
            if samples.isEmpty {
                print("🎵 DEBUG generate: No samples generated, returning dummy audio")
                let dummyAudio = MLXArray.zeros([1, 1000])  // 1000 samples of silence
                
                let endTime = Date().timeIntervalSince1970
                let processingTime = endTime - startTime
                
                return GenerationResult(
                    audio: dummyAudio,
                    samples: 1000,
                    sampleRate: sampleRate,
                    segmentIdx: 0,
                    tokenCount: 0,
                    audioDuration: "00:00:00.042",
                    realTimeFactor: Float(processingTime),
                    prompt: ["tokens": 0],
                    audioSamples: ["samples": 1000],
                    processingTimeSeconds: processingTime,
                    peakMemoryUsage: 0.0
                )
            }

            print("🎵 DEBUG generate: Preparing tokens for decoding...")
            print("🎵 DEBUG generate: samples count: \(samples.count)")
            for (i, sample) in samples.enumerated() {
                print("🎵 DEBUG generate: sample \(i) shape: \(sample.shape)")
            }

            // Decode audio tokens to audio - MATCH Python exactly
            print("🎵 DEBUG generate: Stacking samples...")
            let audioTokens = MLX.stacked(samples, axis: 0)
            print("🎵 DEBUG generate: audioTokens shape after stacking: \(audioTokens.shape)")
            
            print("🎵 DEBUG generate: Transposing tokens...")
            // CRITICAL: Match Python exactly: mx.transpose(mx.stack(samples), axes=[1, 2, 0])
            // samples are [1, 32] each (1 batch, 32 codebooks), stacked to [3, 1, 32]
            // transpose with axes=[1, 2, 0] means: [3, 1, 32] -> [1, 32, 3]
            // This gives us [batch, num_codebooks, num_frames] which is what Mimi expects
            let transposedTokens = audioTokens.transposed(1, 2, 0)
            print("🎵 DEBUG generate: transposedTokens shape for decoder: \(transposedTokens.shape)")
            
            // Verify shape is correct for Mimi decoder: [batch, num_codebooks, num_frames]
            print("🎵 DEBUG generate: Expected shape: [1, 32, 3], actual: \(transposedTokens.shape)")

            print("🎵 DEBUG generate: Calling audio decoder...")
            var audio: MLXArray
            if stream, let streamingDecoder = streamingDecoder {
                print("🎵 DEBUG generate: Using streaming decoder...")
                // Use streaming decoder for real-time generation
                audio = streamingDecoder.decodeFrames(transposedTokens).squeezed(axis: 0).squeezed(axis: 0)
            } else if let audioTokenizer = audioTokenizer {
                print("🎵 DEBUG generate: Using regular Mimi decoder...")
                // Use regular decoder - match Python exactly with squeeze operations
                audio = audioTokenizer.decode(transposedTokens).squeezed(axis: 0).squeezed(axis: 0)
            } else {
                throw SesameTTSError.modelNotInitialized
            }
            
            print("🎵 DEBUG generate: Audio decoded successfully, shape: \(audio.shape)")

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

        // Get the sampled token index from top-k
        let sampledTokenIndex = topKIndices[0..., sampleIdx]

        // Return as [batch_size] - ensure it's 1D for expansion in generateFrame
        return sampledTokenIndex.reshaped([logits.shape[0]])
    }

    /// Get default speaker prompt for a voice
    /// - Parameter voice: Voice name
    /// - Returns: Array of segments for the voice prompt
    private func defaultSpeakerPrompt(voice: String) -> [Segment] {
        guard let voiceManager = voiceManager else {
            // Fallback if voice manager not available
            return [
                Segment(speaker: 0, text: "Hello, I'm ready to help.", audio: nil),
                Segment(speaker: 0, text: "What would you like to discuss?", audio: nil)
            ]
        }

        // Use voice manager to get voice segments
        if voiceManager.validateVoice(voiceName: voice) {
            return voiceManager.getVoiceSegments(voiceName: voice)
        } else {
            // Use default voice if requested voice doesn't exist
            return voiceManager.getDefaultSegments()
        }
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
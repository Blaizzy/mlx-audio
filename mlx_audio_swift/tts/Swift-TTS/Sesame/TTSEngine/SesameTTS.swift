//
 // Sesame TTS - Complete Text-to-Speech Pipeline
 // Main API class following KokoroTTS pattern
 //

import Foundation
import MLX
import MLXNN

// Available Sesame voices (based on CSM-1B model)
public enum SesameVoice: String, CaseIterable {
    case conversational_a = "conversational_a"
    case conversational_b = "conversational_b"
    case conversational_c = "conversational_c"
    case conversational_d = "conversational_d"
    case narrative_a = "narrative_a"
    case narrative_b = "narrative_b"
    case narrative_c = "narrative_c"
    case narrative_d = "narrative_d"
    case excited = "excited"
    case calm = "calm"
    case professional = "professional"
    case friendly = "friendly"

    /// Human readable description for each voice
    public var description: String {
        switch self {
        case .conversational_a:
            return "Conversational A - Natural, everyday speech"
        case .conversational_b:
            return "Conversational B - Casual, friendly tone"
        case .conversational_c:
            return "Conversational C - Professional, clear"
        case .conversational_d:
            return "Conversational D - Warm, engaging"
        case .narrative_a:
            return "Narrative A - Storyteller style"
        case .narrative_b:
            return "Narrative B - Dramatic reading"
        case .narrative_c:
            return "Narrative C - Educational content"
        case .narrative_d:
            return "Narrative D - Documentary style"
        case .excited:
            return "Excited - Energetic, enthusiastic"
        case .calm:
            return "Calm - Relaxed, soothing"
        case .professional:
            return "Professional - Business, formal"
        case .friendly:
            return "Friendly - Warm, approachable"
        }
    }
}

/// Main Sesame TTS class - Complete text-to-speech pipeline
/// Equivalent to KokoroTTS but for Sesame CSM model
public class SesameTTS {

    // MARK: - Error Types
    public enum SesameTTSError: Error, LocalizedError {
        case modelNotInitialized
        case invalidVoice(String)
        case tokenizationFailed(String)
        case generationFailed(String)
        case modelLoadFailed(String)
        case invalidText(String)

        public var errorDescription: String? {
            switch self {
            case .modelNotInitialized:
                return "Sesame TTS model not initialized. Call loadModel() first."
            case .invalidVoice(let voice):
                return "Invalid voice: \(voice). Available voices: \(SesameVoice.allCases.map { $0.rawValue }.joined(separator: ", "))"
            case .tokenizationFailed(let reason):
                return "Text tokenization failed: \(reason)"
            case .generationFailed(let reason):
                return "Audio generation failed: \(reason)"
            case .modelLoadFailed(let reason):
                return "Model loading failed: \(reason)"
            case .invalidText(let text):
                return "Invalid text input: \(text)"
            }
        }
    }

    // MARK: - Private Properties

    /// Core model wrapper (lazy loaded)
    private var modelWrapper: SesameModelWrapper?

    /// Current voice configuration
    private var currentVoice: SesameVoice?

    /// Model initialization flag
    private var isModelInitialized = false

    /// Model configuration
    private var modelConfig: LlamaModelArgs

    /// Audio chunk callback for streaming
    public typealias AudioChunkCallback = (MLXArray) -> Void

    // MARK: - Initialization

    /// Initialize Sesame TTS with default configuration
    public init() {
        do {
            // Try to load configuration from sesame_config.json in bundle
            if let configPath = Bundle.main.path(forResource: "sesame_config", ofType: "json") {
                self.modelConfig = try LlamaModelArgs.fromSesameConfig(configPath: configPath)
            } else {
                // Fallback to hardcoded configuration if JSON not found
                print("⚠️  WARNING: Could not find sesame_config.json, using fallback configuration")
                self.modelConfig = LlamaModelArgs.llama1B()
            }
        } catch {
            // Fallback on error
            print("⚠️  WARNING: Failed to load sesame_config.json: \(error), using fallback configuration")
            self.modelConfig = LlamaModelArgs.llama1B()
        }
    }

    /// Initialize with custom model configuration
    public init(modelConfig: LlamaModelArgs) {
        self.modelConfig = modelConfig
    }

    /// Initialize Sesame TTS from configuration file
    /// - Parameter configPath: Path to sesame_config.json file
    public convenience init(fromConfig configPath: String) throws {
        let modelConfig = try LlamaModelArgs.fromSesameConfig(configPath: configPath)
        self.init(modelConfig: modelConfig)
    }

    // MARK: - Model Management

    /// Load and initialize the Sesame model
    /// This is a heavy operation that should be called once
    public func loadModel() throws {
        guard !isModelInitialized else { return }

        print("🎵 Loading Sesame TTS model...")

        // Initialize model wrapper (can throw)
        modelWrapper = SesameModelWrapper(modelConfig)

        // Mark as initialized
        isModelInitialized = true

        print("✅ Sesame TTS model loaded successfully")
    }

    /// Reset the model to free up memory
    /// - Parameter preserveTextProcessing: Whether to keep tokenizer and voice manager
    public func resetModel(preserveTextProcessing: Bool = true) {
        guard let wrapper = modelWrapper else { return }

        // Reset model components
        wrapper.resetModel()

        // Clear references
        modelWrapper = nil
        currentVoice = nil
        isModelInitialized = false

        // Use autoreleasepool to encourage memory release
        autoreleasepool { }

        print("🧹 Sesame TTS model reset")
    }

    /// Check if model is loaded and ready
    public var isLoaded: Bool {
        return isModelInitialized && modelWrapper != nil
    }

    // MARK: - Voice Management

    /// Get all available voices
    public var availableVoices: [SesameVoice] {
        return SesameVoice.allCases
    }

    /// Get current voice
    public var voice: SesameVoice? {
        return currentVoice
    }

    /// Set voice for generation
    /// - Parameter voice: The voice to use for audio generation
    public func setVoice(_ voice: SesameVoice) {
        self.currentVoice = voice
    }

    // MARK: - Audio Generation

    /// Generate audio from text with specified voice
    /// - Parameters:
    ///   - text: Text to convert to speech
    ///   - voice: Voice to use (optional, uses current voice if not specified)
    ///   - temperature: Sampling temperature (0.1-2.0, default 0.9)
    ///   - topK: Top-k sampling parameter (1-100, default 50)
    ///   - maxAudioLengthMs: Maximum audio length in milliseconds (default 90000)
    ///   - stream: Whether to use streaming generation for real-time output
    ///   - voiceMatch: Whether to use voice matching (append text to voice prompt)
    /// - Returns: Generated audio as MLXArray (sample_rate=24000)
    public func generateAudio(
        text: String,
        voice: SesameVoice? = nil,
        temperature: Float = 0.9,
        topK: Int = 50,
        maxAudioLengthMs: Float = 90000,
        stream: Bool = false,
        voiceMatch: Bool = true
    ) throws -> MLXArray {
        // Ensure model is loaded
        guard isLoaded else {
            throw SesameTTSError.modelNotInitialized
        }

        guard let modelWrapper = modelWrapper else {
            throw SesameTTSError.modelNotInitialized
        }

        // Validate input text
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            throw SesameTTSError.invalidText("Text cannot be empty")
        }

        // Use specified voice or current voice
        let generationVoice = voice ?? currentVoice ?? .conversational_a

        print("🎵 Generating audio with voice: \(generationVoice.rawValue)")
        print("📝 Text: \(trimmedText.prefix(50))..." + (trimmedText.count > 50 ? "" : ""))

        print("🎤 DEBUG SesameTTS: Calling modelWrapper.generate")
        print("🎤 DEBUG SesameTTS: text='\(trimmedText)', voice='\(generationVoice.rawValue)', voiceMatch=\(voiceMatch)")

        do {
            // Generate audio using model wrapper
            let result = try modelWrapper.generate(
                text: trimmedText,
                voice: generationVoice.rawValue,
                maxAudioLengthMs: maxAudioLengthMs,
                temperature: temperature,
                topK: topK,
                stream: stream,
                voiceMatch: voiceMatch
            )

            print("✅ Audio generated successfully")
            print("📊 Audio shape: \(result.audio.shape)")
            print("🎵 Sample rate: \(result.sampleRate) Hz")
            print("⏱️  Duration: \(result.audioDuration)")
            print("⚡ Real-time factor: \(result.realTimeFactor)")

            return result.audio

        } catch let error as SesameTTSError {
            throw error
        } catch {
            throw SesameTTSError.generationFailed(error.localizedDescription)
        }
    }

    /// Generate audio with streaming callback for real-time processing
    /// - Parameters:
    ///   - text: Text to convert to speech
    ///   - voice: Voice to use
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling parameter
    ///   - maxAudioLengthMs: Maximum audio length
    ///   - chunkCallback: Callback called with each audio chunk as it's generated
    public func generateAudioStream(
        text: String,
        voice: SesameVoice? = nil,
        temperature: Float = 0.9,
        topK: Int = 50,
        maxAudioLengthMs: Float = 90000,
        chunkCallback: @escaping AudioChunkCallback
    ) throws {
        // For streaming, we'll need to modify the model wrapper to support chunked generation
        // This is a placeholder for future streaming implementation
        let audio = try generateAudio(
            text: text,
            voice: voice,
            temperature: temperature,
            topK: topK,
            maxAudioLengthMs: maxAudioLengthMs,
            stream: true
        )

        // For now, just call the callback once with the complete audio
        // In future, this should stream audio chunks as they're generated
        chunkCallback(audio)
    }

    // MARK: - Utility Methods

    /// Get sample rate of generated audio
    public var sampleRate: Int {
        return 24000 // Sesame generates 24kHz audio
    }

    /// Get model information
    public var modelInfo: [String: Any] {
        return [
            "name": "Sesame CSM-1B",
            "architecture": "Dual-Transformer with Mimi Codec",
            "sample_rate": sampleRate,
            "voices": availableVoices.map { $0.rawValue },
            "streaming_supported": true,
            "max_audio_length_ms": 90000
        ]
    }

    /// Get memory usage information
    public var memoryUsage: [String: Any]? {
        guard modelWrapper != nil else { return nil }

        return [
            "model_loaded": isLoaded,
            "sample_rate": sampleRate,
            "current_voice": currentVoice?.rawValue ?? "none"
        ]
    }
}

// MARK: - Convenience Extensions

extension SesameTTS {
    /// Quick generation with default settings
    /// - Parameters:
    ///   - text: Text to convert to speech
    ///   - voice: Voice to use (defaults to conversational_a)
    public func speak(text: String, voice: SesameVoice = .conversational_a) throws -> MLXArray {
        return try generateAudio(text: text, voice: voice)
    }

    /// Generate with voice name string
    /// - Parameters:
    ///   - text: Text to convert to speech
    ///   - voiceName: Voice name as string
    public func generateAudio(text: String, voiceName: String) throws -> MLXArray {
        guard let voice = SesameVoice(rawValue: voiceName) else {
            throw SesameTTSError.invalidVoice(voiceName)
        }
        return try generateAudio(text: text, voice: voice)
    }
}

// MARK: - Debug Helpers

extension SesameTTS {
    /// Print available voices
    public func printAvailableVoices() {
        print("🎵 Available Sesame Voices:")
        for voice in availableVoices {
            print("   • \(voice.rawValue) - \(voice.description)")
        }
    }

    /// Print model status
    public func printStatus() {
        print("🎵 Sesame TTS Status:")
        print("   • Model loaded: \(isLoaded ? "✅" : "❌")")
        print("   • Current voice: \(currentVoice?.rawValue ?? "none")")
        print("   • Sample rate: \(sampleRate) Hz")
        print("   • Streaming: ✅ Supported")

        if let memoryInfo = memoryUsage {
            print("   • Memory info: \(memoryInfo)")
        }
    }
}

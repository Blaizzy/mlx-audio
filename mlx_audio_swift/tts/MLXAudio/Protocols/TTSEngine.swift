//
//  TTSEngine.swift
//  MLXAudio
//
//  Protocol-oriented foundation for all TTS engines.
//

import Foundation

/// Core protocol that all TTS engines must conform to.
///
/// Note: We avoid `associatedtype` to allow using `any TTSEngine` as an existential type.
/// Voice selection uses string IDs instead.
@MainActor
public protocol TTSEngine: Observable {
    /// The provider type for this engine
    var provider: TTSProvider { get }

    // MARK: - State Properties

    /// Whether the model is loaded and ready for generation
    var isLoaded: Bool { get }

    /// Whether audio generation is currently in progress
    var isGenerating: Bool { get }

    /// Whether audio playback is currently in progress
    var isPlaying: Bool { get }

    /// URL of the last generated audio file (for sharing/export)
    var lastGeneratedAudioURL: URL? { get }

    /// Time taken for the last generation (seconds)
    var generationTime: TimeInterval { get }

    // MARK: - Voice Management

    /// Available voices for this engine
    var availableVoices: [Voice] { get }

    /// Currently selected voice ID
    var selectedVoiceID: String { get set }

    // MARK: - Lifecycle Methods

    /// Load the model with optional progress reporting
    /// - Parameter progressHandler: Optional callback for download/load progress
    func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws

    /// Generate audio from text
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - speed: Playback speed multiplier (1.0 = normal)
    /// - Returns: The generated audio result
    func generate(text: String, speed: Float) async throws -> AudioResult

    /// Play the last generated audio
    func play() async throws

    /// Stop any ongoing generation or playback
    func stop() async

    /// Clean up resources (model weights, audio buffers, etc.)
    func cleanup() async throws
}

// MARK: - Streaming Support

/// Optional streaming support - only engines with real-time streaming capability conform.
///
/// Currently only MarvisEngine supports streaming.
@MainActor
public protocol StreamingTTSEngine: TTSEngine {
    /// Generate audio as a stream of chunks
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - speed: Playback speed multiplier (1.0 = normal)
    /// - Returns: An async stream of audio chunks
    func generateStreaming(text: String, speed: Float) -> AsyncThrowingStream<AudioChunk, Error>
}

/// A chunk of audio data for streaming playback
public struct AudioChunk: Sendable {
    /// Raw audio samples
    public let samples: [Float]

    /// Sample rate in Hz (e.g., 24000)
    public let sampleRate: Int

    /// Whether this is the final chunk in the stream
    public let isLast: Bool

    /// Processing time for this chunk
    public let processingTime: TimeInterval

    public init(samples: [Float], sampleRate: Int, isLast: Bool, processingTime: TimeInterval) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.isLast = isLast
        self.processingTime = processingTime
    }
}

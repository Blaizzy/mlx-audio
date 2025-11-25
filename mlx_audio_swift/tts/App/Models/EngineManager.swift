//
//  EngineManager.swift
//  MLX Audio Demo
//
//  Manages TTS engine lifecycle: creation, loading, switching, and cleanup.
//

import Foundation
import MLX

/// Manages TTS engine lifecycle and state
@MainActor
@Observable
final class EngineManager {
    // MARK: - State

    /// The current TTS engine instance
    private(set) var currentEngine: (any TTSEngine)

    /// Whether a model is currently being loaded
    private(set) var isLoading: Bool = false

    /// Model loading progress (0.0 to 1.0)
    private(set) var loadingProgress: Double = 0

    /// Last error that occurred
    private(set) var error: TTSError?

    // MARK: - Computed Properties

    var selectedProvider: TTSProvider { currentEngine.provider }
    var isLoaded: Bool { currentEngine.isLoaded }
    var isGenerating: Bool { currentEngine.isGenerating }
    var isPlaying: Bool { currentEngine.isPlaying }
    var generationTime: TimeInterval { currentEngine.generationTime }
    var lastGeneratedAudioURL: URL? { currentEngine.lastGeneratedAudioURL }
    var availableVoices: [Voice] { currentEngine.availableVoices }
    var supportsStreaming: Bool { currentEngine is StreamingTTSEngine }

    var selectedVoiceID: String {
        get { currentEngine.selectedVoiceID }
        set { currentEngine.selectedVoiceID = newValue }
    }

    // MARK: - Initialization

    init(initialProvider: TTSProvider = .kokoro) {
        self.currentEngine = Self.createEngine(for: initialProvider)
    }

    // MARK: - Engine Lifecycle

    /// Switch to a different TTS provider
    func selectProvider(_ provider: TTSProvider) async {
        guard provider != selectedProvider else { return }

        Log.ui.info("Switching provider from \(self.selectedProvider.displayName) to \(provider.displayName)")

        try? await currentEngine.cleanup()
        currentEngine = Self.createEngine(for: provider)
        error = nil
    }

    /// Load the current engine's model
    func loadEngine() async throws {
        guard !currentEngine.isLoaded else {
            Log.model.debug("Engine already loaded")
            return
        }

        isLoading = true
        loadingProgress = 0
        error = nil

        MLX.GPU.set(cacheLimit: TTSConstants.Memory.gpuCacheLimit)

        do {
            try await currentEngine.load { [weak self] progress in
                Task { @MainActor in
                    self?.loadingProgress = progress.fractionCompleted
                }
            }
            isLoading = false
            loadingProgress = 1.0
        } catch {
            isLoading = false
            loadingProgress = 0
            let ttsError = TTSError.modelLoadFailed(underlying: error)
            self.error = ttsError
            throw ttsError
        }
    }

    /// Generate audio from text
    func generate(text: String, speed: Float) async throws -> AudioResult {
        guard currentEngine.isLoaded else {
            throw TTSError.modelNotLoaded
        }

        error = nil
        MLX.GPU.set(cacheLimit: TTSConstants.Memory.gpuCacheLimit)

        do {
            return try await currentEngine.generate(text: text, speed: speed)
        } catch let e as TTSError {
            error = e
            throw e
        } catch {
            let ttsError = TTSError.generationFailed(underlying: error)
            self.error = ttsError
            throw ttsError
        }
    }

    /// Generate with streaming (Marvis only)
    func generateStreaming(text: String, speed: Float) -> AsyncThrowingStream<AudioChunk, Error> {
        guard let streamingEngine = currentEngine as? StreamingTTSEngine else {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: TTSError.invalidArgument("Streaming not supported"))
            }
        }
        return streamingEngine.generateStreaming(text: text, speed: speed)
    }

    /// Play the last generated audio
    func play() async throws {
        try await currentEngine.play()
    }

    /// Stop generation and playback
    func stop() async {
        await currentEngine.stop()
    }

    // MARK: - Private

    private static func createEngine(for provider: TTSProvider) -> any TTSEngine {
        switch provider {
        case .kokoro: return KokoroEngine()
        case .orpheus: return OrpheusEngine()
        case .marvis: return MarvisEngine()
        case .outetts: return OuteTTSEngineWrapper()
        }
    }
}

//
//  AppState.swift
//  MLX Audio Demo
//
//  UI state and coordination for the TTS application.
//  Engine lifecycle is managed by EngineManager.
//

import SwiftUI

/// Central state management for the TTS application
@MainActor
@Observable
final class AppState {
    // MARK: - Dependencies

    /// Engine lifecycle manager
    let engineManager: EngineManager

    // MARK: - User Input

    /// Text to synthesize
    var inputText: String = "How are you doing today?"

    /// Speech speed multiplier (Kokoro only)
    var speed: Float = TTSConstants.Speed.default

    // MARK: - UI State

    /// Whether to show the inspector panel (macOS/iPad)
    var showInspector: Bool = true

    /// Whether to auto-play generated audio
    var autoPlay: Bool = true

    /// Status message for display
    var statusMessage: String = ""

    // MARK: - Generated Output

    /// Last generated audio result
    private(set) var lastResult: AudioResult?

    /// Flag to prevent auto-play after user stops
    private var stopRequested: Bool = false

    // MARK: - Delegated State (from EngineManager)

    var selectedProvider: TTSProvider { engineManager.selectedProvider }
    var isLoaded: Bool { engineManager.isLoaded }
    var isGenerating: Bool { engineManager.isGenerating }
    var isPlaying: Bool { engineManager.isPlaying }
    var isModelLoading: Bool { engineManager.isLoading }
    var loadingProgress: Double { engineManager.loadingProgress }
    var error: TTSError? { engineManager.error }
    var generationTime: TimeInterval { engineManager.generationTime }
    var lastGeneratedAudioURL: URL? { engineManager.lastGeneratedAudioURL }
    var availableVoices: [Voice] { engineManager.availableVoices }
    var supportsStreaming: Bool { engineManager.supportsStreaming }

    var selectedVoiceID: String {
        get { engineManager.selectedVoiceID }
        set { engineManager.selectedVoiceID = newValue }
    }

    var canGenerate: Bool {
        isLoaded && !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating
    }

    // MARK: - Initialization

    init() {
        self.engineManager = EngineManager()
    }

    // MARK: - Provider Management

    /// Switch to a different TTS provider
    func selectProvider(_ provider: TTSProvider) async {
        guard provider != selectedProvider else { return }

        await engineManager.selectProvider(provider)
        lastResult = nil
        statusMessage = provider.statusMessage
    }

    // MARK: - Engine Operations

    /// Load the current engine's model
    func loadEngine() async throws {
        try await engineManager.loadEngine()
    }

    /// Generate audio from the current input text
    func generate() async {
        guard canGenerate else { return }

        stopRequested = false
        statusMessage = "Generating..."

        do {
            lastResult = try await engineManager.generate(text: inputText, speed: speed)

            guard !stopRequested else { return }

            if let result = lastResult {
                statusMessage = formatResultStatus(result)
            }

            if autoPlay {
                try await engineManager.play()
            }
        } catch let e as TTSError {
            statusMessage = e.localizedDescription
        } catch {
            statusMessage = TTSError.generationFailed(underlying: error).localizedDescription
        }
    }

    /// Generate with streaming (Marvis only)
    func generateStreaming() async {
        guard canGenerate else { return }
        guard supportsStreaming else {
            statusMessage = "Streaming not supported for \(selectedProvider.displayName)"
            return
        }

        stopRequested = false
        statusMessage = "Streaming..."

        do {
            var allSamples: [Float] = []
            var sampleRate = 0
            var totalProcessingTime: TimeInterval = 0

            for try await chunk in engineManager.generateStreaming(text: inputText, speed: 1.0) {
                if stopRequested { break }
                allSamples.append(contentsOf: chunk.samples)
                sampleRate = chunk.sampleRate
                totalProcessingTime += chunk.processingTime
                statusMessage = "Streaming... \(allSamples.count) samples"
            }

            guard !stopRequested else { return }

            lastResult = .samples(
                data: allSamples,
                sampleRate: sampleRate,
                processingTime: totalProcessingTime
            )

            if let result = lastResult {
                statusMessage = formatResultStatus(result)
            }
        } catch let e as TTSError {
            if !stopRequested { statusMessage = e.localizedDescription }
        } catch {
            if !stopRequested { statusMessage = TTSError.generationFailed(underlying: error).localizedDescription }
        }
    }

    /// Play the last generated audio
    func play() async {
        do {
            try await engineManager.play()
        } catch {
            statusMessage = TTSError.audioPlaybackFailed(underlying: error).localizedDescription
        }
    }

    /// Stop generation and playback
    func stop() async {
        stopRequested = true
        await engineManager.stop()
        statusMessage = "Stopped"
    }

    // MARK: - Private Helpers

    private func formatResultStatus(_ result: AudioResult) -> String {
        let timeStr = String(format: "%.2f", result.processingTime)

        if let duration = result.duration {
            let durationStr = String(format: "%.2f", duration)
            if let rtf = result.realTimeFactor {
                let rtfStr = String(format: "%.2f", rtf)
                return "Generated \(durationStr)s audio in \(timeStr)s (RTF: \(rtfStr)x)"
            }
            return "Generated \(durationStr)s audio in \(timeStr)s"
        }

        return "Generated in \(timeStr)s"
    }
}

// MARK: - Engine-Specific Property Access

extension AppState {
    /// Quality level for Marvis
    var marvisQualityLevel: MarvisTTS.QualityLevel {
        get { (engineManager.currentEngine as? MarvisEngine)?.qualityLevel ?? .maximum }
        set { (engineManager.currentEngine as? MarvisEngine)?.qualityLevel = newValue }
    }

    /// Whether streaming playback is enabled for Marvis
    var useStreaming: Bool {
        get { (engineManager.currentEngine as? MarvisEngine)?.playbackEnabled ?? false }
        set { (engineManager.currentEngine as? MarvisEngine)?.playbackEnabled = newValue }
    }

    /// Streaming interval for Marvis
    var streamingInterval: Double {
        get { (engineManager.currentEngine as? MarvisEngine)?.streamingInterval ?? TTSConstants.Timing.defaultStreamingInterval }
        set { (engineManager.currentEngine as? MarvisEngine)?.streamingInterval = newValue }
    }

    /// Temperature for Orpheus
    var orpheusTemperature: Float {
        get { (engineManager.currentEngine as? OrpheusEngine)?.temperature ?? 0.6 }
        set { (engineManager.currentEngine as? OrpheusEngine)?.temperature = newValue }
    }

    /// Top-P for Orpheus
    var orpheusTopP: Float {
        get { (engineManager.currentEngine as? OrpheusEngine)?.topP ?? 0.8 }
        set { (engineManager.currentEngine as? OrpheusEngine)?.topP = newValue }
    }
}

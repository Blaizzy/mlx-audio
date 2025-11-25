//
//  KokoroEngine.swift
//  MLXAudio
//
//  Kokoro TTS engine conforming to TTSEngine protocol.
//  Wraps the existing KokoroTTS implementation.
//

import AVFoundation
import Foundation

/// Kokoro TTS engine - fast, lightweight TTS with many voice options
@Observable
@MainActor
public final class KokoroEngine: TTSEngine {
    // MARK: - TTSEngine Protocol Properties

    public let provider: TTSProvider = .kokoro
    public private(set) var isLoaded: Bool = false
    public private(set) var isGenerating: Bool = false
    public private(set) var isPlaying: Bool = false
    public private(set) var lastGeneratedAudioURL: URL?
    public private(set) var generationTime: TimeInterval = 0

    public var availableVoices: [Voice] {
        TTSProvider.kokoro.availableVoices
    }

    public var selectedVoiceID: String = TTSProvider.kokoro.defaultVoiceID

    // MARK: - Private Properties

    @ObservationIgnored private var kokoroTTS: KokoroTTS?
    @ObservationIgnored private var audioPlayer: AudioSamplePlayer?
    @ObservationIgnored private var audioBuffers: [AVAudioPCMBuffer] = []
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var lastGeneratedSamples: [Float] = []

    // MARK: - Initialization

    public init() {
        Log.tts.debug("KokoroEngine initialized")
    }

    deinit {
        generationTask?.cancel()
    }

    // MARK: - TTSEngine Protocol Methods

    public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
        guard !isLoaded else {
            Log.tts.debug("KokoroEngine already loaded")
            return
        }

        Log.model.info("Loading Kokoro TTS model...")

        // Create KokoroTTS instance with progress handler
        kokoroTTS = KokoroTTS(
            repoId: KokoroWeightLoader.defaultRepoId,
            progressHandler: progressHandler ?? { _ in }
        )

        // Create audio player
        audioPlayer = AudioSamplePlayer(sampleRate: TTSConstants.Audio.kokoroSampleRate)

        isLoaded = true
        Log.model.info("Kokoro TTS model loaded successfully")
    }

    public func generate(text: String, speed: Float) async throws -> AudioResult {
        guard isLoaded, let kokoroTTS = kokoroTTS else {
            throw TTSError.modelNotLoaded
        }

        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            throw TTSError.invalidArgument("Text cannot be empty")
        }

        // Cancel any existing generation
        generationTask?.cancel()
        isGenerating = true
        generationTime = 0
        lastGeneratedSamples = []

        let startTime = Date()
        var allSamples: [Float] = []
        var firstChunkTime: TimeInterval = 0

        // Resolve voice
        guard let ttsVoice = resolveVoice(selectedVoiceID) else {
            isGenerating = false
            throw TTSError.invalidVoice(selectedVoiceID)
        }

        do {
            for try await samples in try await kokoroTTS.generateAudioStream(
                voice: ttsVoice,
                text: trimmedText,
                speed: speed
            ) {
                // Record time to first chunk
                if firstChunkTime == 0 {
                    firstChunkTime = Date().timeIntervalSince(startTime)
                    generationTime = firstChunkTime
                }

                allSamples.append(contentsOf: samples)
            }

            isGenerating = false
            lastGeneratedSamples = allSamples

            let totalTime = Date().timeIntervalSince(startTime)
            Log.tts.timing("Kokoro generation", duration: totalTime)

            // Save to file
            do {
                let fileURL = try AudioFileWriter.save(
                    samples: allSamples,
                    sampleRate: TTSConstants.Audio.kokoroSampleRate,
                    filename: TTSConstants.FileNames.kokoroOutput.replacingOccurrences(of: ".wav", with: "")
                )
                lastGeneratedAudioURL = fileURL
            } catch {
                Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
            }

            return .samples(
                data: allSamples,
                sampleRate: TTSConstants.Audio.kokoroSampleRate,
                processingTime: generationTime
            )

        } catch {
            isGenerating = false
            Log.tts.error("Kokoro generation failed: \(error.localizedDescription)")
            throw TTSError.generationFailed(underlying: error)
        }
    }

    public func play() async throws {
        guard let audioPlayer = audioPlayer else {
            throw TTSError.modelNotLoaded
        }

        guard !lastGeneratedSamples.isEmpty else {
            Log.audio.warning("No audio to play")
            return
        }

        isPlaying = true
        await audioPlayer.play(samples: lastGeneratedSamples)
        isPlaying = false
    }

    public func stop() async {
        // Cancel generation
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false

        // Stop playback
        await audioPlayer?.stop()
        isPlaying = false

        Log.tts.debug("KokoroEngine stopped")
    }

    public func cleanup() async throws {
        await stop()

        await kokoroTTS?.resetModel(preserveTextProcessing: false)
        kokoroTTS = nil
        audioPlayer = nil
        audioBuffers.removeAll()
        lastGeneratedSamples.removeAll()
        isLoaded = false

        Log.tts.debug("KokoroEngine cleaned up")
    }

    // MARK: - Extended API

    /// Generate and immediately play audio
    public func say(_ text: String, speed: Float = 1.0) async throws {
        _ = try await generate(text: text, speed: speed)
        try await play()
    }

    /// Generate audio with streaming playback (audio plays as it generates)
    public func generateWithStreaming(text: String, speed: Float = 1.0) async throws -> AudioResult {
        guard isLoaded, let kokoroTTS = kokoroTTS, let audioPlayer = audioPlayer else {
            throw TTSError.modelNotLoaded
        }

        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            throw TTSError.invalidArgument("Text cannot be empty")
        }

        generationTask?.cancel()
        isGenerating = true
        isPlaying = true
        generationTime = 0
        lastGeneratedSamples = []

        let startTime = Date()
        var allSamples: [Float] = []
        var firstChunkTime: TimeInterval = 0

        guard let ttsVoice = resolveVoice(selectedVoiceID) else {
            isGenerating = false
            isPlaying = false
            throw TTSError.invalidVoice(selectedVoiceID)
        }

        do {
            for try await samples in try await kokoroTTS.generateAudioStream(
                voice: ttsVoice,
                text: trimmedText,
                speed: speed
            ) {
                if firstChunkTime == 0 {
                    firstChunkTime = Date().timeIntervalSince(startTime)
                    generationTime = firstChunkTime
                }

                allSamples.append(contentsOf: samples)

                // Stream to audio player
                audioPlayer.enqueue(samples: samples, prebufferSeconds: 0)
            }

            isGenerating = false
            lastGeneratedSamples = allSamples

            // Wait for playback to complete
            await audioPlayer.awaitCompletion()
            isPlaying = false

            // Save to file
            do {
                let fileURL = try AudioFileWriter.save(
                    samples: allSamples,
                    sampleRate: TTSConstants.Audio.kokoroSampleRate,
                    filename: TTSConstants.FileNames.kokoroOutput.replacingOccurrences(of: ".wav", with: "")
                )
                lastGeneratedAudioURL = fileURL
            } catch {
                Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
            }

            return .samples(
                data: allSamples,
                sampleRate: TTSConstants.Audio.kokoroSampleRate,
                processingTime: generationTime
            )

        } catch {
            isGenerating = false
            isPlaying = false
            throw TTSError.generationFailed(underlying: error)
        }
    }

    // MARK: - Private Helpers

    private func resolveVoice(_ voiceID: String) -> TTSVoice? {
        // Try direct rawValue match first
        if let voice = TTSVoice(rawValue: voiceID) {
            return voice
        }

        // Try identifier mapping (e.g., "af_heart" -> .afHeart)
        if let voice = TTSVoice.fromIdentifier(voiceID) {
            return voice
        }

        return nil
    }

}

// MARK: - Voice ID Helpers

extension KokoroEngine {
    /// Get the Voice object for the currently selected voice
    var selectedVoice: Voice? {
        availableVoices.first { $0.id == selectedVoiceID }
    }

    /// Select a voice by Voice object
    func selectVoice(_ voice: Voice) {
        selectedVoiceID = voice.id
    }
}

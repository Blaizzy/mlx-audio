//
//  OuteTTSEngineWrapper.swift
//  MLXAudio
//
//  OuteTTS engine conforming to TTSEngine protocol.
//  Wraps the OuteTTSEngine actor implementation.
//

import AVFoundation
import Foundation

/// OuteTTS engine - TTS with speaker profiles
@Observable
@MainActor
public final class OuteTTSEngineWrapper: TTSEngine {
    // MARK: - TTSEngine Protocol Properties

    public let provider: TTSProvider = .outetts
    public private(set) var isLoaded: Bool = false
    public private(set) var isGenerating: Bool = false
    public private(set) var isPlaying: Bool = false
    public private(set) var lastGeneratedAudioURL: URL?
    public private(set) var generationTime: TimeInterval = 0

    public var availableVoices: [Voice] {
        TTSProvider.outetts.availableVoices
    }

    public var selectedVoiceID: String = TTSProvider.outetts.defaultVoiceID

    // MARK: - OuteTTS-Specific Properties

    /// Temperature for sampling (higher = more variation)
    public var temperature: Float = 0.4

    /// Top-p (nucleus) sampling threshold
    public var topP: Float = 0.9

    /// Path to custom speaker profile
    public var customSpeakerPath: String?

    // MARK: - Private Properties

    @ObservationIgnored private var outeTTS: OuteTTSEngine?
    @ObservationIgnored private var audioPlayer: AudioSamplePlayer?
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var lastGeneratedSamples: [Float] = []

    private static let sampleRate = 24000

    // MARK: - Initialization

    public init() {
        Log.tts.debug("OuteTTSEngineWrapper initialized")
    }

    deinit {
        generationTask?.cancel()
    }

    // MARK: - TTSEngine Protocol Methods

    public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
        guard !isLoaded else {
            Log.tts.debug("OuteTTSEngineWrapper already loaded")
            return
        }

        Log.model.info("Loading OuteTTS model...")

        do {
            let engine = OuteTTSEngine()
            try await engine.load(progressHandler: progressHandler ?? { _ in })
            outeTTS = engine

            audioPlayer = AudioSamplePlayer(sampleRate: Self.sampleRate)

            isLoaded = true
            Log.model.info("OuteTTS model loaded successfully")
        } catch {
            Log.model.error("Failed to load OuteTTS model: \(error.localizedDescription)")
            throw TTSError.modelLoadFailed(underlying: error)
        }
    }

    public func generate(text: String, speed: Float) async throws -> AudioResult {
        // Note: OuteTTS doesn't support speed adjustment, parameter is ignored
        guard isLoaded, let outeTTS = outeTTS else {
            throw TTSError.modelNotLoaded
        }

        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            throw TTSError.invalidArgument("Text cannot be empty")
        }

        generationTask?.cancel()
        isGenerating = true
        generationTime = 0
        lastGeneratedSamples = []

        let startTime = Date()

        do {
            // Get speaker profile based on selected voice
            let speaker = resolveSpeaker()

            let result = try await outeTTS.generate(
                text: trimmedText,
                speaker: speaker,
                temperature: temperature,
                topP: topP
            )

            generationTime = Date().timeIntervalSince(startTime)
            Log.tts.timing("OuteTTS generation", duration: generationTime)

            lastGeneratedSamples = result.audio

            isGenerating = false

            // Calculate audio duration for RTF
            let audioDuration = result.duration
            let rtf = generationTime / audioDuration
            Log.tts.rtf("OuteTTS", rtf: rtf)

            // Save to file
            do {
                let fileURL = try AudioFileWriter.save(
                    samples: result.audio,
                    sampleRate: Self.sampleRate,
                    filename: "outetts_output"
                )
                lastGeneratedAudioURL = fileURL
            } catch {
                Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
            }

            return .samples(
                data: result.audio,
                sampleRate: Self.sampleRate,
                processingTime: generationTime
            )

        } catch {
            isGenerating = false
            Log.tts.error("OuteTTS generation failed: \(error.localizedDescription)")
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
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false

        await audioPlayer?.stop()
        isPlaying = false

        Log.tts.debug("OuteTTSEngineWrapper stopped")
    }

    public func cleanup() async throws {
        await stop()

        outeTTS = nil
        audioPlayer = nil
        lastGeneratedSamples.removeAll()
        isLoaded = false

        Log.tts.debug("OuteTTSEngineWrapper cleaned up")
    }

    // MARK: - Extended API

    /// Generate and immediately play audio
    public func say(_ text: String) async throws {
        _ = try await generate(text: text, speed: 1.0)
        try await play()
    }

    /// Load a custom speaker profile
    public func loadCustomSpeaker(from path: String) throws {
        customSpeakerPath = path
        selectedVoiceID = "custom"
    }

    // MARK: - Private Helpers

    private func resolveSpeaker() -> OuteTTSSpeakerProfile? {
        switch selectedVoiceID {
        case "default":
            // Default speaker is loaded by OuteTTSEngine from Hugging Face
            return nil
        case "custom":
            if let path = customSpeakerPath {
                return try? OuteTTSSpeakerProfile.load(from: path)
            }
            return nil
        default:
            return nil
        }
    }
}

// MARK: - Voice Helpers

extension OuteTTSEngineWrapper {
    /// Get the Voice object for the currently selected voice
    var selectedVoice: Voice? {
        availableVoices.first { $0.id == selectedVoiceID }
    }

    /// Select a voice by Voice object
    func selectVoice(_ voice: Voice) {
        selectedVoiceID = voice.id
    }
}

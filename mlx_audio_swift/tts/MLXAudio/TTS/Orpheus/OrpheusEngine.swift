//
//  OrpheusEngine.swift
//  MLXAudio
//
//  Orpheus TTS engine conforming to TTSEngine protocol.
//  Wraps the existing OrpheusTTS implementation.
//

import AVFoundation
import Foundation

/// Orpheus TTS engine - high quality with emotional expressions
///
/// Supports expressions: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`,
/// `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
@Observable
@MainActor
public final class OrpheusEngine: TTSEngine {
    // MARK: - TTSEngine Protocol Properties

    public let provider: TTSProvider = .orpheus
    public private(set) var isLoaded: Bool = false
    public private(set) var isGenerating: Bool = false
    public private(set) var isPlaying: Bool = false
    public private(set) var lastGeneratedAudioURL: URL?
    public private(set) var generationTime: TimeInterval = 0

    public var availableVoices: [Voice] {
        TTSProvider.orpheus.availableVoices
    }

    public var selectedVoiceID: String = TTSProvider.orpheus.defaultVoiceID

    // MARK: - Orpheus-Specific Properties

    /// Temperature for sampling (higher = more variation)
    public var temperature: Float = 0.6

    /// Top-p (nucleus) sampling threshold
    public var topP: Float = 0.8

    // MARK: - Private Properties

    @ObservationIgnored private var orpheusTTS: OrpheusTTS?
    @ObservationIgnored private var audioPlayer: AudioSamplePlayer?
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var lastGeneratedSamples: [Float] = []

    // MARK: - Initialization

    public init() {
        Log.tts.debug("OrpheusEngine initialized")
    }

    deinit {
        generationTask?.cancel()
    }

    // MARK: - TTSEngine Protocol Methods

    public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
        guard !isLoaded else {
            Log.tts.debug("OrpheusEngine already loaded")
            return
        }

        Log.model.info("Loading Orpheus TTS model...")

        do {
            orpheusTTS = try await OrpheusTTS.load(
                progressHandler: progressHandler ?? { _ in }
            )

            audioPlayer = AudioSamplePlayer(sampleRate: TTSConstants.Audio.orpheusSampleRate)

            isLoaded = true
            Log.model.info("Orpheus TTS model loaded successfully")
        } catch {
            Log.model.error("Failed to load Orpheus model: \(error.localizedDescription)")
            throw TTSError.modelLoadFailed(underlying: error)
        }
    }

    public func generate(text: String, speed: Float) async throws -> AudioResult {
        // Note: Orpheus doesn't support speed adjustment, parameter is ignored
        guard isLoaded, let orpheusTTS = orpheusTTS else {
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

        guard let voice = resolveVoice(selectedVoiceID) else {
            isGenerating = false
            throw TTSError.invalidVoice(selectedVoiceID)
        }

        do {
            let samples = try await orpheusTTS.generateAudio(
                voice: voice,
                text: trimmedText,
                temperature: temperature,
                topP: topP
            )

            generationTime = Date().timeIntervalSince(startTime)
            Log.tts.timing("Orpheus generation", duration: generationTime)

            lastGeneratedSamples = samples

            isGenerating = false

            // Calculate audio duration for RTF
            let audioDuration = Double(samples.count) / Double(TTSConstants.Audio.orpheusSampleRate)
            let rtf = generationTime / audioDuration
            Log.tts.rtf("Orpheus", rtf: rtf)

            // Save to file
            do {
                let fileURL = try AudioFileWriter.save(
                    samples: samples,
                    sampleRate: TTSConstants.Audio.orpheusSampleRate,
                    filename: TTSConstants.FileNames.orpheusOutput.replacingOccurrences(of: ".wav", with: "")
                )
                lastGeneratedAudioURL = fileURL
            } catch {
                Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
            }

            return .samples(
                data: samples,
                sampleRate: TTSConstants.Audio.orpheusSampleRate,
                processingTime: generationTime
            )

        } catch {
            isGenerating = false
            Log.tts.error("Orpheus generation failed: \(error.localizedDescription)")
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

        Log.tts.debug("OrpheusEngine stopped")
    }

    public func cleanup() async throws {
        await stop()

        orpheusTTS = nil
        audioPlayer = nil
        lastGeneratedSamples.removeAll()
        isLoaded = false

        Log.tts.debug("OrpheusEngine cleaned up")
    }

    // MARK: - Extended API

    /// Generate and immediately play audio
    public func say(_ text: String) async throws {
        _ = try await generate(text: text, speed: 1.0)
        try await play()
    }

    // MARK: - Private Helpers

    private func resolveVoice(_ voiceID: String) -> OrpheusVoice? {
        OrpheusVoice(rawValue: voiceID)
    }
}

// MARK: - Voice Helpers

extension OrpheusEngine {
    /// Get the Voice object for the currently selected voice
    var selectedVoice: Voice? {
        availableVoices.first { $0.id == selectedVoiceID }
    }

    /// Select a voice by Voice object
    func selectVoice(_ voice: Voice) {
        selectedVoiceID = voice.id
    }
}

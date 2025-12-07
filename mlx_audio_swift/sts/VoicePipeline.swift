//
//  VoicePipeline.swift
//  MLXAudioSTS
//
//  Created by MLX Audio Team on 2025-12-06.
//

import Foundation
import AVFoundation
import MLXAudioCore
import MLXAudioSTT

// MARK: - Voice Pipeline Configuration

/// Configuration for VoicePipeline
///
/// - Note: Some fields are reserved for future use and not yet wired up:
///   - `outputSampleRate`: TTS output rate (handled by native TTS implementation)
///   - `ttsVoice`: Voice selection (passed to TTS externally)
///   - `streamingInterval`: For future streaming response support
public struct VoicePipelineConfig: Sendable {
    // Audio settings
    public var inputSampleRate: Int
    /// Reserved: TTS output sample rate (handled by native TTS implementation)
    public var outputSampleRate: Int

    // VAD settings
    public var silenceThreshold: Float
    public var silenceDuration: TimeInterval
    public var frameDurationMs: Int

    // Model settings
    public var sttModel: STTModelVariant
    /// Reserved: Voice selection (passed to external TTS)
    public var ttsVoice: String?

    /// Reserved: Streaming response interval
    public var streamingInterval: TimeInterval

    public init(
        inputSampleRate: Int = 16_000,
        outputSampleRate: Int = 24_000,
        silenceThreshold: Float = 0.03,
        silenceDuration: TimeInterval = 1.5,
        frameDurationMs: Int = 30,
        sttModel: STTModelVariant = .whisperLargeV3Turbo,
        ttsVoice: String? = nil,
        streamingInterval: TimeInterval = 0.5
    ) {
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate
        self.silenceThreshold = silenceThreshold
        self.silenceDuration = silenceDuration
        self.frameDurationMs = frameDurationMs
        self.sttModel = sttModel
        self.ttsVoice = ttsVoice
        self.streamingInterval = streamingInterval
    }
}

// MARK: - Voice Pipeline Events

public enum VoicePipelineEvent: Sendable {
    case listening
    case speechDetected
    case processing
    case transcription(String)
    case generatingResponse
    case speaking
    case speakingComplete
    case idle
    case error(String)
}

// MARK: - Voice Pipeline Errors

public enum VoicePipelineError: Error, LocalizedError {
    case notInitialized
    case audioInputFailed(String)
    case sttFailed(String)
    case ttsFailed(String)
    case alreadyRunning
    case notRunning

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Voice pipeline not initialized"
        case .audioInputFailed(let reason):
            return "Audio input failed: \(reason)"
        case .sttFailed(let reason):
            return "Speech-to-text failed: \(reason)"
        case .ttsFailed(let reason):
            return "Text-to-speech failed: \(reason)"
        case .alreadyRunning:
            return "Voice pipeline is already running"
        case .notRunning:
            return "Voice pipeline is not running"
        }
    }
}

// MARK: - Response Generator Protocol

/// Protocol for generating responses from transcribed text
public protocol ResponseGenerator: Sendable {
    func generateResponse(for input: String) async throws -> String
}

/// Simple closure-based response generator
public struct ClosureResponseGenerator: ResponseGenerator {
    private let generator: @Sendable (String) async throws -> String

    public init(_ generator: @escaping @Sendable (String) async throws -> String) {
        self.generator = generator
    }

    public func generateResponse(for input: String) async throws -> String {
        try await generator(input)
    }
}

// MARK: - Voice Pipeline

/// Speech-to-Speech pipeline that orchestrates STT → LLM → TTS
///
/// Usage:
/// ```swift
/// // Create response generator (e.g., LLM)
/// let responseGenerator = ClosureResponseGenerator { input in
///     return "You said: \(input)"
/// }
///
/// // Create pipeline
/// let pipeline = try await VoicePipeline(
///     config: VoicePipelineConfig(),
///     responseGenerator: responseGenerator,
///     ttsSession: marvisSession
/// )
///
/// // Listen to events
/// Task {
///     for await event in pipeline.events {
///         switch event {
///         case .transcription(let text):
///             print("You said: \(text)")
///         case .speaking:
///             print("Speaking response...")
///         default:
///             break
///         }
///     }
/// }
///
/// // Start pipeline
/// try await pipeline.start()
/// ```
public actor VoicePipeline {
    // MARK: - Properties

    private let config: VoicePipelineConfig
    private let responseGenerator: ResponseGenerator
    private let stt: STTBridge

    // TTS is provided externally (native Swift implementation)
    private let ttsGenerate: @Sendable (String) async throws -> Void

    private var isRunning = false
    private var audioEngine: AVAudioEngine?
    private var eventContinuation: AsyncStream<VoicePipelineEvent>.Continuation?

    // Audio buffering
    private var audioBuffer: [Float] = []
    private var silentFrameCount = 0
    private var speechDetected = false
    private var deviceSampleRate: Double = 0

    // MARK: - Public Event Stream

    public nonisolated var events: AsyncStream<VoicePipelineEvent> {
        AsyncStream { continuation in
            Task { await self.setEventContinuation(continuation) }
        }
    }

    private func setEventContinuation(_ continuation: AsyncStream<VoicePipelineEvent>.Continuation) {
        self.eventContinuation = continuation
    }

    // MARK: - Initialization

    /// Initialize voice pipeline
    ///
    /// - Parameters:
    ///   - config: Pipeline configuration
    ///   - responseGenerator: Generator for creating responses (e.g., LLM)
    ///   - ttsGenerate: Closure to generate and play TTS audio
    ///   - progressHandler: Progress callback during model loading
    public init(
        config: VoicePipelineConfig = VoicePipelineConfig(),
        responseGenerator: ResponseGenerator,
        ttsGenerate: @escaping @Sendable (String) async throws -> Void,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws {
        self.config = config
        self.responseGenerator = responseGenerator
        self.ttsGenerate = ttsGenerate

        // Initialize STT bridge
        self.stt = try await STTBridge(
            model: config.sttModel,
            config: STTConfig(sampleRate: config.inputSampleRate)
        )
    }

    // MARK: - Public API

    /// Start the voice pipeline (begins listening for speech)
    public func start() async throws {
        guard !isRunning else {
            throw VoicePipelineError.alreadyRunning
        }

        // Setup audio engine
        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else {
            throw VoicePipelineError.audioInputFailed("Failed to create audio engine")
        }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        // Store device sample rate for resampling
        deviceSampleRate = format.sampleRate

        // Calculate frame size based on device sample rate (not target rate)
        let frameSize = Int(deviceSampleRate * Double(config.frameDurationMs) / 1000.0)

        inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(frameSize), format: format) { [weak self] buffer, _ in
            guard let self else { return }
            Task { await self.processAudioBuffer(buffer) }
        }

        do {
            try audioEngine.start()
            isRunning = true
            emit(.listening)
        } catch {
            // Clean up on failure
            inputNode.removeTap(onBus: 0)
            self.audioEngine = nil
            throw VoicePipelineError.audioInputFailed(error.localizedDescription)
        }
    }

    /// Stop the voice pipeline
    public func stop() {
        guard isRunning else { return }

        isRunning = false
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        audioBuffer.removeAll()
        speechDetected = false
        silentFrameCount = 0

        emit(.idle)
    }

    /// Manually inject text (bypass STT)
    public func injectText(_ text: String) async throws {
        guard isRunning else {
            throw VoicePipelineError.notRunning
        }

        await processTranscription(text)
    }

    /// Interrupt current speech output
    public func interrupt() {
        // Signal TTS to stop
        // This would need to be implemented in the TTS layer
        emit(.idle)
    }

    // MARK: - Private Methods

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard isRunning else { return }

        // Convert buffer to float array
        guard let floatData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: floatData[0], count: frameCount))

        // Check for speech using energy-based VAD
        let energy = calculateEnergy(samples)
        let isSpeech = energy > config.silenceThreshold

        if isSpeech {
            if !speechDetected {
                speechDetected = true
                emit(.speechDetected)
            }
            silentFrameCount = 0
            audioBuffer.append(contentsOf: samples)
        } else if speechDetected {
            silentFrameCount += 1
            audioBuffer.append(contentsOf: samples)

            let framesUntilSilence = Int(config.silenceDuration * 1000.0 / Double(config.frameDurationMs))

            if silentFrameCount >= framesUntilSilence {
                // End of speech detected - process the audio
                let capturedAudio = audioBuffer
                audioBuffer.removeAll()
                speechDetected = false
                silentFrameCount = 0

                Task {
                    await processCapture(samples: capturedAudio)
                }
            }
        }
    }

    private func calculateEnergy(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumSquares = samples.reduce(0) { $0 + $1 * $1 }
        return sqrt(sumSquares / Float(samples.count))
    }

    private func processCapture(samples: [Float]) async {
        emit(.processing)

        do {
            // Resample from device rate to STT rate if needed
            let sttSamples: [Float]
            if Int(deviceSampleRate) != config.inputSampleRate {
                let buffer = AudioBuffer(samples: samples, sampleRate: Int(deviceSampleRate))
                sttSamples = buffer.resampled(to: config.inputSampleRate).samples
            } else {
                sttSamples = samples
            }

            // Run STT
            let transcription = try await stt.transcribe(
                samples: sttSamples,
                sampleRate: config.inputSampleRate
            )

            let text = transcription.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else {
                emit(.listening)
                return
            }

            emit(.transcription(text))
            await processTranscription(text)

        } catch {
            emit(.error("STT failed: \(error.localizedDescription)"))
            emit(.listening)
        }
    }

    private func processTranscription(_ text: String) async {
        emit(.generatingResponse)

        do {
            // Generate response using provided generator (e.g., LLM)
            let response = try await responseGenerator.generateResponse(for: text)

            guard !response.isEmpty else {
                emit(.listening)
                return
            }

            // Generate and play TTS
            emit(.speaking)
            try await ttsGenerate(response)
            emit(.speakingComplete)

        } catch {
            emit(.error("Response generation failed: \(error.localizedDescription)"))
        }

        emit(.listening)
    }

    private func emit(_ event: VoicePipelineEvent) {
        eventContinuation?.yield(event)
    }
}

// MARK: - Convenience Initializer

extension VoicePipeline {
    /// Initialize with a simple closure-based response generator
    public convenience init(
        config: VoicePipelineConfig = VoicePipelineConfig(),
        responseGenerator: @escaping @Sendable (String) async throws -> String,
        ttsGenerate: @escaping @Sendable (String) async throws -> Void,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws {
        try await self.init(
            config: config,
            responseGenerator: ClosureResponseGenerator(responseGenerator),
            ttsGenerate: ttsGenerate,
            progressHandler: progressHandler
        )
    }
}

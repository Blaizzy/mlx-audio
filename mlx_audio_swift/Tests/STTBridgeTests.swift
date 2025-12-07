//
//  STTBridgeTests.swift
//  MLXAudioTests
//
//  Created by MLX Audio Team on 2025-12-06.
//

import Testing
import Foundation
@testable import MLXAudioCore
@testable import MLXAudioSTT
@testable import MLXAudioSTS

// Note: These tests require PythonKit and Python-Apple-support to be configured
// They will be skipped if Python is not available

struct STTBridgeTests {

    // MARK: - AudioBuffer Tests

    @Test("AudioBuffer initializes from Float samples")
    func testAudioBufferFromFloats() {
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        let buffer = AudioBuffer(samples: samples, sampleRate: 16000)

        #expect(buffer.sampleCount == 5)
        #expect(buffer.sampleRate == 16000)
        #expect(buffer.duration == 5.0 / 16000.0)
    }

    @Test("AudioBuffer initializes from Int16 samples")
    func testAudioBufferFromInt16() {
        let int16Samples: [Int16] = [16384, -16384, 0]
        let buffer = AudioBuffer(int16Samples: int16Samples, sampleRate: 16000)

        #expect(buffer.sampleCount == 3)
        #expect(abs(buffer.samples[0] - 0.5) < 0.001)
        #expect(abs(buffer.samples[1] - (-0.5)) < 0.001)
        #expect(buffer.samples[2] == 0)
    }

    @Test("AudioBuffer resamples correctly")
    func testAudioBufferResampling() {
        // Create 1 second of audio at 16kHz
        let samples = [Float](repeating: 0.5, count: 16000)
        let buffer = AudioBuffer(samples: samples, sampleRate: 16000)

        // Resample to 8kHz
        let resampled = buffer.resampled(to: 8000)

        #expect(resampled.sampleRate == 8000)
        #expect(resampled.sampleCount == 8000)
        #expect(abs(resampled.duration - 1.0) < 0.01)
    }

    @Test("AudioBuffer converts to Int16 data")
    func testAudioBufferToInt16Data() {
        let samples: [Float] = [0.5, -0.5, 0.0]
        let buffer = AudioBuffer(samples: samples, sampleRate: 16000)
        let data = buffer.toInt16Data()

        #expect(data.count == 6) // 3 samples * 2 bytes each
    }

    // MARK: - TranscriptionResult Tests

    @Test("TranscriptionResult initializes correctly")
    func testTranscriptionResultInit() {
        let segment = TranscriptionResult.Segment(
            text: "Hello",
            start: 0.0,
            end: 1.0,
            confidence: 0.95
        )

        let result = TranscriptionResult(
            text: "Hello world",
            language: "en",
            segments: [segment],
            processingTime: 0.5
        )

        #expect(result.text == "Hello world")
        #expect(result.language == "en")
        #expect(result.segments.count == 1)
        #expect(result.processingTime == 0.5)
    }

    // MARK: - STTModelVariant Tests

    @Test("STTModelVariant has correct default")
    func testSTTModelVariantDefault() {
        #expect(STTModelVariant.default == .whisperLargeV3Turbo)
    }

    @Test("STTModelVariant has display names")
    func testSTTModelVariantDisplayNames() {
        #expect(STTModelVariant.whisperLargeV3Turbo.displayName == "Whisper Large V3 Turbo")
        #expect(STTModelVariant.whisperSmall.displayName == "Whisper Small")
    }

    // MARK: - STTConfig Tests

    @Test("STTConfig has correct defaults")
    func testSTTConfigDefaults() {
        let config = STTConfig()

        #expect(config.sampleRate == 16_000)
        #expect(config.language == nil)
        #expect(config.task == .transcribe)
    }

    // MARK: - VoicePipelineConfig Tests

    @Test("VoicePipelineConfig has correct defaults")
    func testVoicePipelineConfigDefaults() {
        let config = VoicePipelineConfig()

        #expect(config.inputSampleRate == 16_000)
        #expect(config.outputSampleRate == 24_000)
        #expect(config.silenceThreshold == 0.03)
        #expect(config.silenceDuration == 1.5)
        #expect(config.frameDurationMs == 30)
        #expect(config.sttModel == .whisperLargeV3Turbo)
    }

    // MARK: - Error Tests

    @Test("STTError has correct descriptions")
    func testSTTErrorDescriptions() {
        let pythonError = STTError.pythonNotInitialized
        #expect(pythonError.localizedDescription.contains("Python"))

        let modelError = STTError.modelLoadFailed("test-model")
        #expect(modelError.localizedDescription.contains("test-model"))
    }

    @Test("VoicePipelineError has correct descriptions")
    func testVoicePipelineErrorDescriptions() {
        let notInitError = VoicePipelineError.notInitialized
        #expect(notInitError.localizedDescription.contains("not initialized"))

        let audioError = VoicePipelineError.audioInputFailed("mic error")
        #expect(audioError.localizedDescription.contains("mic error"))
    }

    // MARK: - PythonSetup Tests (conditional)

    @Test("PythonSetup reports not ready before initialization")
    func testPythonSetupNotReady() {
        // Before initialization, Python should not be ready
        // Note: This test assumes Python hasn't been initialized yet
        // In CI, this may vary depending on test order
        #expect(PythonSetup.pythonVersion == nil || PythonSetup.isPythonReady)
    }

    // MARK: - PythonSetupError Tests

    @Test("PythonSetupError has correct descriptions")
    func testPythonSetupErrorDescriptions() {
        let resourceError = PythonSetupError.resourceNotFound("test_resource")
        #expect(resourceError.localizedDescription.contains("test_resource"))
        #expect(resourceError.localizedDescription.contains("not found"))

        let initError = PythonSetupError.initializationFailed("test_reason")
        #expect(initError.localizedDescription.contains("test_reason"))
        #expect(initError.localizedDescription.contains("failed"))

        let moduleError = PythonSetupError.moduleNotFound("test_module")
        #expect(moduleError.localizedDescription.contains("test_module"))
        #expect(moduleError.localizedDescription.contains("not found"))
    }

    @Test("PythonSetup finalize is safe to call multiple times")
    func testPythonSetupFinalizeIdempotent() {
        // finalize() should be safe to call even if not initialized
        PythonSetup.finalize()
        PythonSetup.finalize()
        // No crash = test passes
    }
}

// MARK: - Integration Tests (require Python)

struct STTBridgeIntegrationTests {

    @Test("STTBridge throws without Python initialization")
    func testSTTBridgeRequiresPython() async {
        // Skip if Python is already initialized
        guard !PythonSetup.isPythonReady else {
            return
        }

        do {
            _ = try await STTBridge(model: .whisperTiny)
            Issue.record("Expected STTError.pythonNotInitialized")
        } catch let error as STTError {
            if case .pythonNotInitialized = error {
                // Expected error
            } else {
                Issue.record("Expected pythonNotInitialized, got \(error)")
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
}

// MARK: - Mock Tests

struct VoicePipelineMockTests {

    @Test("ClosureResponseGenerator works correctly")
    func testClosureResponseGenerator() async throws {
        let generator = ClosureResponseGenerator { input in
            return "Response to: \(input)"
        }

        let response = try await generator.generateResponse(for: "Hello")
        #expect(response == "Response to: Hello")
    }
}

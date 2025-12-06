//
//  STTBridge.swift
//  MLXAudioSTT
//
//  Created by MLX Audio Team on 2025-12-06.
//

import Foundation

#if canImport(PythonKit)
import PythonKit
#endif

// MARK: - STT Errors

public enum STTError: Error, LocalizedError {
    case pythonNotInitialized
    case modelLoadFailed(String)
    case transcriptionFailed(String)
    case invalidAudio(String)
    case deallocated

    public var errorDescription: String? {
        switch self {
        case .pythonNotInitialized:
            return "Python interpreter not initialized. Call PythonSetup.initialize() first."
        case .modelLoadFailed(let model):
            return "Failed to load STT model: \(model)"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        case .invalidAudio(let reason):
            return "Invalid audio: \(reason)"
        case .deallocated:
            return "STT bridge was deallocated during operation"
        }
    }
}

// MARK: - Transcription Result

public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String?
    public let segments: [Segment]
    public let processingTime: TimeInterval

    public struct Segment: Sendable {
        public let text: String
        public let start: TimeInterval
        public let end: TimeInterval
        public let confidence: Float?

        public init(text: String, start: TimeInterval, end: TimeInterval, confidence: Float? = nil) {
            self.text = text
            self.start = start
            self.end = end
            self.confidence = confidence
        }
    }

    public init(
        text: String,
        language: String? = nil,
        segments: [Segment] = [],
        processingTime: TimeInterval = 0
    ) {
        self.text = text
        self.language = language
        self.segments = segments
        self.processingTime = processingTime
    }
}

// MARK: - STT Model Variants

public enum STTModelVariant: String, CaseIterable, Sendable {
    case whisperLargeV3Turbo = "mlx-community/whisper-large-v3-turbo"
    case whisperLargeV3 = "mlx-community/whisper-large-v3"
    case whisperMedium = "mlx-community/whisper-medium"
    case whisperSmall = "mlx-community/whisper-small"
    case whisperBase = "mlx-community/whisper-base"
    case whisperTiny = "mlx-community/whisper-tiny"

    public static let `default`: STTModelVariant = .whisperLargeV3Turbo

    public var displayName: String {
        switch self {
        case .whisperLargeV3Turbo: return "Whisper Large V3 Turbo"
        case .whisperLargeV3: return "Whisper Large V3"
        case .whisperMedium: return "Whisper Medium"
        case .whisperSmall: return "Whisper Small"
        case .whisperBase: return "Whisper Base"
        case .whisperTiny: return "Whisper Tiny"
        }
    }
}

// MARK: - STT Configuration

public struct STTConfig: Sendable {
    public var sampleRate: Int
    public var language: String?
    public var task: Task

    public enum Task: String, Sendable {
        case transcribe
        case translate
    }

    public init(
        sampleRate: Int = 16_000,
        language: String? = nil,
        task: Task = .transcribe
    ) {
        self.sampleRate = sampleRate
        self.language = language
        self.task = task
    }
}

// MARK: - STT Bridge

/// Bridge to mlx_audio.stt via PythonKit
///
/// Usage:
/// ```swift
/// // Initialize Python first
/// try PythonSetup.initialize()
///
/// // Create STT bridge
/// let stt = try await STTBridge(model: .whisperLargeV3Turbo)
///
/// // Transcribe audio file
/// let result = try await stt.transcribe(audioURL: fileURL)
/// print(result.text)
/// ```
///
/// - Note: Marked `@unchecked Sendable` because `PythonObject` is not Sendable.
///   Thread safety is ensured by serializing all Python calls through a dedicated
///   `DispatchQueue` (`queue`), which prevents concurrent access to Python objects
///   and respects Python's GIL.
public final class STTBridge: @unchecked Sendable {
    #if canImport(PythonKit)
    private let model: PythonObject
    private let sttUtils: PythonObject
    #endif

    private let config: STTConfig
    private let modelVariant: STTModelVariant
    private let queue = DispatchQueue(label: "com.mlxaudio.stt", qos: .userInitiated)

    /// Initialize STT bridge with specified model
    ///
    /// - Parameters:
    ///   - model: Model variant to load
    ///   - config: STT configuration
    ///   - progressHandler: Progress callback during model download
    public init(
        model: STTModelVariant = .default,
        config: STTConfig = STTConfig(),
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws {
        self.modelVariant = model
        self.config = config

        #if canImport(PythonKit)
        guard PythonSetup.isPythonReady else {
            throw STTError.pythonNotInitialized
        }

        // Load on background thread to avoid blocking
        let (loadedModel, loadedUtils) = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<(PythonObject, PythonObject), Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let utils = Python.import("mlx_audio.stt.utils")
                    let loadedModel = utils.load_model(model.rawValue)
                    continuation.resume(returning: (loadedModel, utils))
                } catch {
                    continuation.resume(throwing: STTError.modelLoadFailed(model.rawValue))
                }
            }
        }

        self.model = loadedModel
        self.sttUtils = loadedUtils
        #else
        throw STTError.pythonNotInitialized
        #endif
    }

    /// Transcribe audio from file URL
    public func transcribe(audioURL: URL) async throws -> TranscriptionResult {
        #if canImport(PythonKit)
        return try await withCheckedThrowingContinuation { continuation in
            queue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: STTError.deallocated)
                    return
                }

                let startTime = CFAbsoluteTimeGetCurrent()

                do {
                    // Load audio using Python
                    let audio = self.sttUtils.load_audio(audioURL.path, sr: self.config.sampleRate)

                    // Run transcription with language/task options
                    let result: PythonObject
                    if let language = self.config.language {
                        result = self.model.generate(
                            audio,
                            language: PythonObject(language),
                            task: PythonObject(self.config.task.rawValue)
                        )
                    } else {
                        result = self.model.generate(
                            audio,
                            task: PythonObject(self.config.task.rawValue)
                        )
                    }

                    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

                    // Extract results
                    let text = String(result.text) ?? ""
                    let language = String(result.language)

                    // Parse segments if available
                    var segments: [TranscriptionResult.Segment] = []
                    if let pySegments = result.segments,
                       let segmentCount = Int(Python.len(pySegments)),
                       segmentCount > 0 {
                        for i in 0..<segmentCount {
                            let seg = pySegments[i]
                            segments.append(TranscriptionResult.Segment(
                                text: String(seg.text) ?? "",
                                start: Double(seg.start) ?? 0,
                                end: Double(seg.end) ?? 0,
                                confidence: nil
                            ))
                        }
                    }

                    continuation.resume(returning: TranscriptionResult(
                        text: text,
                        language: language,
                        segments: segments,
                        processingTime: processingTime
                    ))
                } catch {
                    continuation.resume(throwing: STTError.transcriptionFailed(String(describing: error)))
                }
            }
        }
        #else
        throw STTError.pythonNotInitialized
        #endif
    }

    /// Transcribe audio from AudioBuffer
    public func transcribe(buffer: AudioBuffer) async throws -> TranscriptionResult {
        #if canImport(PythonKit)
        return try await withCheckedThrowingContinuation { continuation in
            queue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: STTError.deallocated)
                    return
                }

                let startTime = CFAbsoluteTimeGetCurrent()

                do {
                    // Resample if needed
                    let targetBuffer = buffer.sampleRate == self.config.sampleRate
                        ? buffer
                        : buffer.resampled(to: self.config.sampleRate)

                    // Convert to numpy array via PythonKit
                    let np = Python.import("numpy")
                    let mx = Python.import("mlx.core")

                    let pyArray = np.array(targetBuffer.samples, dtype: np.float32)
                    let mxArray = mx.array(pyArray)

                    // Run transcription with language/task options
                    let result: PythonObject
                    if let language = self.config.language {
                        result = self.model.generate(
                            mxArray,
                            language: PythonObject(language),
                            task: PythonObject(self.config.task.rawValue)
                        )
                    } else {
                        result = self.model.generate(
                            mxArray,
                            task: PythonObject(self.config.task.rawValue)
                        )
                    }

                    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

                    let text = String(result.text) ?? ""
                    let language = String(result.language)

                    continuation.resume(returning: TranscriptionResult(
                        text: text,
                        language: language,
                        segments: [],
                        processingTime: processingTime
                    ))
                } catch {
                    continuation.resume(throwing: STTError.transcriptionFailed(String(describing: error)))
                }
            }
        }
        #else
        throw STTError.pythonNotInitialized
        #endif
    }

    /// Transcribe audio from raw Float samples
    public func transcribe(samples: [Float], sampleRate: Int) async throws -> TranscriptionResult {
        let buffer = AudioBuffer(samples: samples, sampleRate: sampleRate)
        return try await transcribe(buffer: buffer)
    }
}

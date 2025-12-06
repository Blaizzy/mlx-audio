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
/// // Create STT bridge with progress
/// let stt = try await STTBridge(model: .whisperLargeV3Turbo) { progress in
///     print("Loading: \(Int(progress.fractionCompleted * 100))%")
/// }
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

                    // Download model with progress if handler provided
                    let localPath: PythonObject
                    if let progressHandler = progressHandler {
                        do {
                            localPath = try Self.downloadWithProgress(
                                repoId: model.rawValue,
                                progressHandler: progressHandler
                            )
                        } catch {
                            continuation.resume(throwing: STTError.modelLoadFailed(model.rawValue))
                            return
                        }
                    } else {
                        // Use default download (no progress)
                        localPath = utils.get_model_path(model.rawValue)
                    }

                    // Load model from local path
                    let loadedModel = utils.load_model(localPath)
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

    #if canImport(PythonKit)
    /// Unique ID generator for concurrent downloads
    private static var downloadIdCounter: UInt64 = 0
    private static let downloadIdLock = NSLock()

    private static func nextDownloadId() -> String {
        downloadIdLock.lock()
        defer { downloadIdLock.unlock() }
        downloadIdCounter += 1
        return "_swift_progress_\(downloadIdCounter)"
    }

    /// Download model with progress callback using custom tqdm class
    /// - Throws: Python exception if download fails
    private static func downloadWithProgress(
        repoId: String,
        progressHandler: @escaping (Progress) -> Void
    ) throws -> PythonObject {
        let hfHub = Python.import("huggingface_hub")

        // Generate unique callback key for this download (reentrancy safe)
        let callbackKey = nextDownloadId()
        let tqdmClassName = "_SwiftProgressTqdm\(callbackKey)"

        // Create Swift callback as Python-callable function
        let swiftCallback = PythonFunction { (args: [PythonObject]) -> PythonObject in
            guard args.count >= 2,
                  let current = Double(args[0]),
                  let total = Double(args[1]),
                  total > 0 else {
                return Python.None
            }
            let progress = Progress(totalUnitCount: Int64(total))
            progress.completedUnitCount = Int64(current)
            DispatchQueue.main.async {
                progressHandler(progress)
            }
            return Python.None
        }

        // Store callback with unique key (reentrancy safe)
        Python.globals()[callbackKey] = swiftCallback.pythonObject

        // Cleanup helper - ensures callback is removed even on error
        func cleanup() {
            Python.globals()[callbackKey] = Python.None
            Python.globals()[tqdmClassName] = Python.None
        }

        // Define tqdm-compatible class with unique name referencing unique callback
        let tqdmCode = """
class \(tqdmClassName):
    def __init__(self, iterable=None, total=None, **kwargs):
        self.iterable = iterable
        self.total = total
        if self.total is None and iterable is not None:
            try:
                self.total = len(iterable)
            except TypeError:
                self.total = 0
        self.n = 0

    def __iter__(self):
        if self.iterable is None:
            return
        for item in self.iterable:
            yield item
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def update(self, n=1):
        self.n += n
        cb = globals().get('\(callbackKey)')
        if cb and self.total and self.total > 0:
            cb(self.n, self.total)

    def close(self):
        pass

    def set_description(self, desc=None, refresh=True):
        pass

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        pass

    def refresh(self):
        pass

    def clear(self):
        pass

    def reset(self, total=None):
        self.n = 0
        if total is not None:
            self.total = total
"""
        // Execute class definition in Python
        let builtins = Python.import("builtins")
        builtins.exec(tqdmCode, Python.globals())
        let SwiftProgressTqdm = Python.globals()[tqdmClassName]

        // Download with custom tqdm class (cleanup on success or failure)
        let localPath: PythonObject
        do {
            localPath = try hfHub.snapshot_download.throwing
                .dynamicallyCall(withKeywordArguments: [
                    "": repoId,
                    "allow_patterns": ["*.json", "*.safetensors", "*.py", "*.model", "*.tiktoken", "*.txt"],
                    "tqdm_class": SwiftProgressTqdm
                ])
        } catch {
            cleanup()
            throw error
        }

        cleanup()
        return localPath
    }
    #endif

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

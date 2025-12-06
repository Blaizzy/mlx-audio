# Swift Bridge Design: STT, STS, TTS Modules

**Date**: 2025-12-06
**Version**: 1.2
**Status**: ✅ Completed
**Merged**: PR #2 → `main`

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Python Runtime | [Python-Apple-support](https://github.com/beeware/Python-Apple-support) | Embedded Python.xcframework |
| Swift-Python Bridge | [PythonKit](https://github.com/pvieito/PythonKit) | Call Python from Swift |
| ML Framework | MLX (native Swift + Python) | Model inference |
| TTS | Native Swift (existing) | Kokoro, Marvis, Orpheus |
| STT | PythonKit → mlx_audio.stt | Whisper via Python bridge |

## Executive Summary

This document proposes a unified Swift bridge architecture to expose MLX-Audio's STT (Speech-to-Text), STS (Speech-to-Speech), and TTS (Text-to-Speech) capabilities for iOS/macOS applications. The design follows Apple's Swift conventions and builds upon the existing `MLXAudio` Swift package patterns.

---

## 1. Current State Analysis

### 1.1 Existing Swift Implementation (TTS Only)

```
mlx_audio_swift/tts/MLXAudio/
├── Kokoro/          # Kokoro TTS model
├── Orpheus/         # Orpheus TTS model
├── Marvis/          # Marvis TTS model + Mimi codec
├── TTSProvider.swift
└── Utils/
```

**Current TTS Pattern** (MarvisSession):
```swift
public final class MarvisSession: Module {
    public func generate(for text: String) async throws -> GenerationResult
    public func stream(_ text: String) -> AsyncThrowingStream<GenerationResult, Error>
}
```

### 1.2 Python Modules to Bridge

| Module | Python Path | Key Classes |
|--------|-------------|-------------|
| **STT** | `mlx_audio.stt` | Whisper, Parakeet, Wav2Vec, Voxtral |
| **STS** | `mlx_audio.sts` | VoicePipeline (STT + LLM + TTS) |
| **TTS** | `mlx_audio.tts` | Kokoro, Sesame/CSM, Spark (existing) |

---

## 2. Architecture Design

### 2.1 Module Hierarchy

```
MLXAudio/
├── Core/
│   ├── AudioSession.swift          # Shared audio configuration
│   ├── AudioBuffer.swift           # Common audio buffer type
│   └── ModelProvider.swift         # Unified model loading
│
├── STT/                            # NEW: Speech-to-Text
│   ├── STTSession.swift            # Main STT interface
│   ├── Models/
│   │   ├── WhisperModel.swift
│   │   ├── ParakeetModel.swift
│   │   └── VoxtralModel.swift
│   └── TranscriptionResult.swift
│
├── TTS/                            # EXISTING: Text-to-Speech
│   ├── TTSSession.swift            # Unified TTS interface (new)
│   ├── Kokoro/
│   ├── Marvis/
│   └── Orpheus/
│
├── STS/                            # NEW: Speech-to-Speech
│   ├── VoicePipeline.swift         # Main STS interface
│   ├── VADProcessor.swift          # Voice Activity Detection
│   └── PipelineConfig.swift
│
└── DSP/                            # NEW: Audio Processing
    ├── AudioResampler.swift
    ├── STFTProcessor.swift
    └── MelFilterbank.swift
```

### 2.2 Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                       VoicePipeline                          │
│                          (STS)                               │
└─────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │   STTSession  │   │   LLMBridge   │   │   TTSSession  │
    │               │   │  (mlx-lm)     │   │               │
    └───────────────┘   └───────────────┘   └───────────────┘
            │                                        │
            ▼                                        ▼
    ┌───────────────┐                       ┌───────────────┐
    │ WhisperModel  │                       │ MarvisSession │
    │ ParakeetModel │                       │ KokoroTTS     │
    │ VoxtralModel  │                       │ OrpheusTTS    │
    └───────────────┘                       └───────────────┘
            │                                        │
            └────────────────────┬───────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Core / DSP Layer     │
                    │  AudioBuffer, Resampler │
                    └─────────────────────────┘
```

---

## 3. API Specifications

### 3.1 STTSession (Speech-to-Text)

```swift
// MARK: - STT Session

public final class STTSession {

    // MARK: - Model Variants

    public enum ModelVariant: String, CaseIterable {
        case whisperLargeV3Turbo = "mlx-community/whisper-large-v3-turbo"
        case whisperSmall = "mlx-community/whisper-small"
        case parakeetCTC = "mlx-community/parakeet-ctc-1.1b"
        case voxtral = "mlx-community/voxtral"

        public static let `default`: ModelVariant = .whisperLargeV3Turbo
    }

    // MARK: - Configuration

    public struct Config {
        public var sampleRate: Int = 16_000
        public var language: String? = nil  // nil = auto-detect
        public var task: Task = .transcribe

        public enum Task: String {
            case transcribe
            case translate
        }
    }

    // MARK: - Results

    public struct TranscriptionResult {
        public let text: String
        public let language: String?
        public let segments: [Segment]
        public let processingTime: TimeInterval

        public struct Segment {
            public let text: String
            public let start: TimeInterval
            public let end: TimeInterval
            public let confidence: Float?
        }
    }

    // MARK: - Initialization

    public init(
        model: ModelVariant = .default,
        config: Config = Config(),
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws

    // MARK: - Public API

    /// Transcribe audio from file URL
    public func transcribe(audioURL: URL) async throws -> TranscriptionResult

    /// Transcribe audio from raw samples
    public func transcribe(audio: [Float], sampleRate: Int) async throws -> TranscriptionResult

    /// Transcribe audio from MLXArray
    public func transcribe(audio: MLXArray) async throws -> TranscriptionResult

    /// Stream transcription (for real-time)
    public func streamTranscribe(
        from stream: AsyncStream<[Float]>
    ) -> AsyncThrowingStream<TranscriptionResult, Error>
}
```

### 3.2 TTSSession (Unified TTS Interface)

```swift
// MARK: - Unified TTS Session

public final class TTSSession {

    // MARK: - Provider Selection

    public enum Provider: String, CaseIterable {
        case kokoro
        case marvis
        case orpheus

        public static let `default`: Provider = .marvis
    }

    // MARK: - Unified Result

    public struct SpeechResult {
        public let audio: [Float]
        public let sampleRate: Int
        public let sampleCount: Int
        public let audioDuration: TimeInterval
        public let realTimeFactor: Double
        public let processingTime: TimeInterval
    }

    // MARK: - Initialization

    public init(
        provider: Provider = .default,
        voice: String? = nil,
        progressHandler: @escaping (Progress) -> Void = { _ in },
        playbackEnabled: Bool = true
    ) async throws

    // MARK: - Public API

    /// Generate speech (one-shot)
    public func generate(for text: String) async throws -> SpeechResult

    /// Generate speech without playback
    public func generateRaw(for text: String) async throws -> SpeechResult

    /// Stream speech generation
    public func stream(
        text: String,
        interval: TimeInterval = 0.5
    ) -> AsyncThrowingStream<SpeechResult, Error>

    /// Stop current playback
    public func stopPlayback()

    /// Available voices for current provider
    public var availableVoices: [String] { get }

    /// Change voice
    public func setVoice(_ voice: String) throws
}
```

### 3.3 VoicePipeline (Speech-to-Speech)

```swift
// MARK: - Voice Pipeline (STS)

public final class VoicePipeline {

    // MARK: - Configuration

    public struct Config {
        // Audio settings
        public var inputSampleRate: Int = 16_000
        public var outputSampleRate: Int = 24_000

        // VAD settings
        public var silenceThreshold: Float = 0.03
        public var silenceDuration: TimeInterval = 1.5
        public var frameDurationMs: Int = 30
        public var vadMode: Int = 3  // 0-3, higher = more aggressive

        // Model settings
        public var sttModel: STTSession.ModelVariant = .whisperLargeV3Turbo
        public var ttsProvider: TTSSession.Provider = .marvis
        public var ttsVoice: String? = nil

        // Streaming
        public var streamingInterval: Int = 3

        public init() {}
    }

    // MARK: - Events

    public enum Event {
        case listening
        case speechDetected
        case processing
        case transcription(String)
        case generating
        case speaking(TTSSession.SpeechResult)
        case idle
        case error(Error)
    }

    // MARK: - Initialization

    public init(
        config: Config = Config(),
        responseGenerator: @escaping (String) async throws -> String,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws

    // MARK: - Public API

    /// Start the voice pipeline
    public func start() async throws

    /// Stop the voice pipeline
    public func stop()

    /// Event stream for UI updates
    public var events: AsyncStream<Event> { get }

    /// Manually inject text (bypass STT)
    public func injectText(_ text: String) async throws

    /// Interrupt current speech
    public func interrupt()
}
```

---

## 4. Data Flow Diagrams

### 4.1 STT Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Audio Input │ ──▶ │  Resampler   │ ──▶ │   Whisper   │ ──▶ │   Result   │
│ (URL/Array) │     │  (16kHz)     │     │   Model     │     │   (Text)   │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
```

### 4.2 TTS Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│    Text     │ ──▶ │  Tokenizer   │ ──▶ │  TTS Model  │ ──▶ │   Audio    │
│   Input     │     │              │     │  (Marvis)   │     │  Playback  │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
```

### 4.3 STS (VoicePipeline) Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VoicePipeline                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌─────┐    ┌──────────┐    ┌─────┐    ┌──────────┐    │
│  │   Mic    │───▶│ VAD │───▶│   STT    │───▶│ LLM │───▶│   TTS    │    │
│  │  Input   │    │     │    │ (Whisper)│    │     │    │ (Marvis) │    │
│  └──────────┘    └─────┘    └──────────┘    └─────┘    └──────────┘    │
│       │                          │              │            │          │
│       │                          │              │            │          │
│       ▼                          ▼              ▼            ▼          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Event Stream → UI                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Plan

### Phase 1: Core Infrastructure
- [ ] `AudioBuffer` - Unified audio data type
- [ ] `AudioResampler` - Sample rate conversion
- [ ] `ModelProvider` - Centralized model loading

### Phase 2: STT Module
- [ ] `STTSession` - Main interface
- [ ] `WhisperModel` - Port from Python
- [ ] `TranscriptionResult` - Result type

### Phase 3: Unified TTS Interface
- [ ] `TTSSession` - Unified wrapper
- [ ] Refactor existing Kokoro/Marvis/Orpheus

### Phase 4: STS Pipeline
- [ ] `VADProcessor` - Voice activity detection
- [ ] `VoicePipeline` - Main orchestrator
- [ ] Event streaming

### Phase 5: Integration
- [ ] Package.swift updates
- [ ] iOS/macOS demo apps
- [ ] Documentation

---

## 6. Package.swift Updates

```swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "mlx-audio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        // Full library
        .library(name: "MLXAudio", targets: ["MLXAudio"]),

        // Modular imports (matching Python)
        .library(name: "MLXAudioSTT", targets: ["MLXAudioSTT"]),
        .library(name: "MLXAudioTTS", targets: ["MLXAudioTTS"]),
        .library(name: "MLXAudioSTS", targets: ["MLXAudioSTS"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
    ],
    targets: [
        // Core (shared)
        .target(
            name: "MLXAudioCore",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "mlx_audio_swift/Core"
        ),

        // STT (new)
        .target(
            name: "MLXAudioSTT",
            dependencies: ["MLXAudioCore"],
            path: "mlx_audio_swift/stt"
        ),

        // TTS (existing + unified interface)
        .target(
            name: "MLXAudioTTS",
            dependencies: [
                "MLXAudioCore",
                "ESpeakNG",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "mlx_audio_swift/tts/MLXAudio",
            exclude: ["Preview Content", "Assets.xcassets", "MLXAudioApp.swift"]
        ),

        // STS (new, depends on STT + TTS)
        .target(
            name: "MLXAudioSTS",
            dependencies: ["MLXAudioSTT", "MLXAudioTTS"],
            path: "mlx_audio_swift/sts"
        ),

        // Full library
        .target(
            name: "MLXAudio",
            dependencies: ["MLXAudioSTT", "MLXAudioTTS", "MLXAudioSTS"]
        ),

        // Binary targets
        .binaryTarget(
            name: "ESpeakNG",
            path: "mlx_audio_swift/tts/MLXAudio/Kokoro/Frameworks/ESpeakNG.xcframework"
        ),

        // Tests
        .testTarget(
            name: "MLXAudioTests",
            dependencies: ["MLXAudio"],
            path: "mlx_audio_swift/Tests"
        ),
    ]
)
```

---

## 7. Usage Examples

### 7.1 STT Only

```swift
import MLXAudioSTT

// Initialize
let stt = try await STTSession(model: .whisperLargeV3Turbo)

// Transcribe file
let result = try await stt.transcribe(audioURL: fileURL)
print(result.text)

// Transcribe raw audio
let audio: [Float] = loadAudio()
let result = try await stt.transcribe(audio: audio, sampleRate: 16000)
```

### 7.2 TTS Only (Unified)

```swift
import MLXAudioTTS

// Initialize with provider
let tts = try await TTSSession(provider: .marvis, voice: "conversational_a")

// Generate
let result = try await tts.generate(for: "Hello, world!")

// Stream
for try await chunk in tts.stream(text: "Long text here...") {
    print("Chunk: \(chunk.sampleCount) samples")
}
```

### 7.3 STS Pipeline

```swift
import MLXAudioSTS

// Configure pipeline
var config = VoicePipeline.Config()
config.ttsProvider = .marvis
config.sttModel = .whisperLargeV3Turbo

// Response generator (LLM integration)
func generateResponse(_ input: String) async throws -> String {
    // Call your LLM here
    return "This is the response to: \(input)"
}

// Create pipeline
let pipeline = try await VoicePipeline(
    config: config,
    responseGenerator: generateResponse
)

// Listen to events
Task {
    for await event in pipeline.events {
        switch event {
        case .transcription(let text):
            print("You said: \(text)")
        case .speaking(let audio):
            print("Speaking: \(audio.audioDuration)s")
        case .error(let error):
            print("Error: \(error)")
        default:
            break
        }
    }
}

// Start
try await pipeline.start()
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Async/await first** | Matches existing MarvisSession pattern; modern Swift concurrency |
| **Protocol-based providers** | Allows swapping TTS/STT implementations without changing API |
| **Event streaming for STS** | UI responsiveness; decouples state from pipeline |
| **Modular targets** | Mirrors Python's modular imports; reduces app bundle size |
| **MLXArray as internal type** | Use `[Float]` at API boundary for simplicity |

---

## 9. Open Questions

1. **LLM Integration**: Should `VoicePipeline` include mlx-lm directly, or accept a closure?
   - **Recommendation**: Accept closure for flexibility

2. **VAD Implementation**: Use WebRTC VAD port or energy-based?
   - **Recommendation**: Energy-based initially (simpler), add WebRTC later

3. **Real-time STT**: Should streaming be frame-based or segment-based?
   - **Recommendation**: Segment-based (matches Whisper's design)

---

## 10. Hybrid Architecture (PythonKit + Native Swift)

Since TTS is already native Swift and STT requires Python, we use a **hybrid approach**:

```
┌─────────────────────────────────────────────────────────────┐
│                     SwiftUI App                              │
├─────────────────────────────────────────────────────────────┤
│  VoicePipeline (Swift)                                       │
│  ├── STTBridge (PythonKit → mlx_audio.stt)                  │
│  ├── LLM (closure / mlx-swift-lm)                           │
│  └── TTS (Native Swift: Marvis/Kokoro/Orpheus)              │
├─────────────────────────────────────────────────────────────┤
│  PythonKit                    │  MLX Swift (native)          │
├─────────────────────────────────────────────────────────────┤
│  Python.xcframework (BeeWare Python-Apple-support)          │
│  ├── python-stdlib                                           │
│  └── site-packages/                                          │
│      ├── mlx/                                                │
│      ├── mlx_audio/                                          │
│      └── numpy/                                              │
└─────────────────────────────────────────────────────────────┘
```

### STT Bridge Implementation

```swift
// MLXAudioSTTBridge.swift
import Foundation
import PythonKit

public final class STTBridge {
    private let whisperModel: PythonObject
    private let sttUtils: PythonObject

    public init(model: String = "mlx-community/whisper-large-v3-turbo") throws {
        // Ensure Python is initialized (call once at app start)
        guard PythonLibrary.isInitialized else {
            throw STTError.pythonNotInitialized
        }

        sttUtils = Python.import("mlx_audio.stt.utils")
        whisperModel = sttUtils.load_model(model)
    }

    public func transcribe(audioURL: URL) async throws -> TranscriptionResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: STTError.deallocated)
                    return
                }

                do {
                    let result = self.whisperModel.generate(audioURL.path)
                    let text = String(result.text) ?? ""
                    continuation.resume(returning: TranscriptionResult(
                        text: text,
                        language: String(result.language),
                        processingTime: 0
                    ))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
```

### Python Environment Setup

```swift
// PythonSetup.swift
import Foundation
import PythonKit

public enum PythonSetup {
    public static func initialize() throws {
        guard let pythonHome = Bundle.main.path(forResource: "python", ofType: nil),
              let appPath = Bundle.main.path(forResource: "app", ofType: nil) else {
            throw PythonSetupError.resourceNotFound
        }

        setenv("PYTHONHOME", pythonHome, 1)
        setenv("PYTHONPATH", "\(pythonHome)/lib/python3.12/site-packages:\(appPath)", 1)

        // Initialize Python interpreter
        Py_Initialize()

        // Verify mlx_audio is available
        let mlxAudio = Python.import("mlx_audio")
        print("mlx_audio version: \(mlxAudio.__version__)")
    }
}
```

---

## 11. References

- Existing Python implementation: `mlx_audio/sts/voice_pipeline.py`
- Existing Swift TTS: `mlx_audio_swift/tts/MLXAudio/`
- MLX Swift: https://github.com/ml-explore/mlx-swift
- mlx-swift-lm: https://github.com/ml-explore/mlx-swift-lm
- Python-Apple-support: https://github.com/beeware/Python-Apple-support
- PythonKit: https://github.com/pvieito/PythonKit
- BeeSwift example: https://github.com/radcli14/BeeSwift

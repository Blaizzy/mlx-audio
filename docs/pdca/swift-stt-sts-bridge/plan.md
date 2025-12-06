# Plan: Swift STT/STS Bridge

**Date**: 2025-12-06
**Feature**: Swift bridge for STT and STS modules using PythonKit + Python-Apple-support
**Branch**: `feat/swift-stt-sts-bridge`

## Hypothesis

We can create a Swift bridge for mlx-audio's STT (Speech-to-Text) and STS (Speech-to-Speech) functionality by:

1. **STT**: Using PythonKit to call `mlx_audio.stt` Python module (Whisper model)
2. **STS**: Building a Swift `VoicePipeline` that orchestrates:
   - STT via PythonKit bridge
   - LLM via user-provided closure
   - TTS via existing native Swift implementations (Marvis/Kokoro/Orpheus)

### Why This Approach

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Pure Swift port of Whisper | Best performance, no Python | Large effort (~2 weeks) | Future phase |
| PythonKit bridge | Quick, reuse Python code | Python dependency | **Selected** |
| Subprocess IPC | Clean separation | Slow, complex serialization | Rejected |

## Expected Outcomes

| Metric | Target |
|--------|--------|
| STT latency | < 2x Python baseline |
| Memory overhead | < 200MB additional |
| API compatibility | Match existing MarvisSession pattern |
| Test coverage | > 80% for new code |

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `mlx_audio_swift/Core/` directory
- [ ] Implement `PythonSetup.swift` - Python environment initialization
- [ ] Implement `AudioBuffer.swift` - Shared audio data type
- [ ] Add PythonKit dependency to Package.swift

### Phase 2: STT Bridge
- [ ] Implement `STTBridge.swift` - PythonKit wrapper for mlx_audio.stt
- [ ] Implement `TranscriptionResult.swift` - Result type
- [ ] Add async/await support for transcription
- [ ] Test with Whisper model

### Phase 3: STS Pipeline
- [ ] Implement `VoicePipeline.swift` - Main orchestrator
- [ ] Implement `VADProcessor.swift` - Voice activity detection
- [ ] Implement event streaming with `AsyncStream<Event>`
- [ ] Integrate with existing TTS (MarvisSession)

### Phase 4: Integration & Testing
- [ ] Update Package.swift with new targets
- [ ] Create example SwiftUI app
- [ ] Write unit tests
- [ ] Documentation

## Risks & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| PythonKit initialization fails | High | Medium | Test early, add detailed error handling |
| Python GIL blocks UI | Medium | High | Run all Python calls on background thread |
| Memory leak from Python objects | Medium | Medium | Explicit cleanup, autoreleasepool |
| Model download stalls | Low | Low | Progress callbacks, timeout handling |

## Dependencies

- Python-Apple-support (already integrated)
- PythonKit (Swift Package)
- mlx_audio Python package (bundled)
- Existing Swift TTS (Marvis, Kokoro, Orpheus)

## Success Criteria

1. `STTBridge` successfully transcribes audio files
2. `VoicePipeline` completes full STT → LLM → TTS cycle
3. No UI thread blocking during inference
4. Memory usage stable over multiple cycles

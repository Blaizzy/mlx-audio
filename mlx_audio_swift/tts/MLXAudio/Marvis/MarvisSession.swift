import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import AVFoundation

public final class MarvisSession: Module {
    public enum Voice: String, CaseIterable {
        case conversationalA = "conversational_a"
        case conversationalB = "conversational_b"
    }

    public enum QualityLevel: String, CaseIterable {
        case low
        case medium
        case high
        case maximum

        public var codebookCount: Int {
            switch self {
            case .low: return 8
            case .medium: return 16
            case .high: return 24
            case .maximum: return 32
            }
        }
    }

    public let sampleRate: Double

    private let model: MarvisModel
    private let _promptURLs: [URL]?
    // Renamed underscored members to Swiftier names
    private let textTokenizer: any Tokenizer
    private let audioTokenizer: MimiTokenizer
    private let streamingDecoder: MimiStreamingDecoder
    
    // Audio playback
    private var playback: AudioPlayback?
    private let playbackEnabled: Bool
    // Bound configuration for session-like ergonomics
    private var boundVoice: Voice? = .conversationalA
    private var boundRefAudio: MLXArray? = nil
    private var boundRefText: String? = nil
    private var boundQuality: QualityLevel = .maximum

    public init(
        config: MarvisModelArgs,
        repoId: String,
        promptURLs: [URL]? = nil,
        progressHandler: @escaping (Progress) -> Void,
        playbackEnabled: Bool = true
    ) async throws {
        self.model = MarvisModel(config: config)

        self._promptURLs = promptURLs
        self.playbackEnabled = playbackEnabled

        self.textTokenizer = try await loadTokenizer(configuration: ModelConfiguration(id: repoId), hub: HubApi.shared)

        self.audioTokenizer = try await MimiTokenizer(Mimi.fromPretrained(progressHandler: progressHandler))

        self.streamingDecoder = MimiStreamingDecoder(audioTokenizer.codec)
        self.sampleRate = audioTokenizer.codec.cfg.sampleRate
        super.init()
        model.resetCaches()

        if playbackEnabled {
            playback = AudioPlayback(sampleRate: sampleRate)
        } else {
            playback = nil
        }

    }
    
    deinit { playback?.stop() }

    private func tokenizeTextSegment(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let K = model.args.audioNumCodebooks
        let frameW = K + 1

        let prompt = "[\(speaker)]" + text
        let ids = MLXArray(textTokenizer.encode(text: prompt))

        let T = ids.shape[0]
        var frame = MLXArray.zeros([T, frameW], type: Int32.self)
        var mask = MLXArray.zeros([T, frameW], type: Bool.self)

        let lastCol = frameW - 1
        do {
            let left = split(frame, indices: [lastCol], axis: 1)[0]
            let right = split(frame, indices: [lastCol], axis: 1)[1]
            let tail = split(right, indices: [1], axis: 1)
            let after = (tail.count > 1) ? tail[1] : MLXArray.zeros([T, 0], type: Int32.self)
            frame = concatenated([left, ids.reshaped([T, 1]), after], axis: 1)
        }

        do {
            let ones = MLXArray.ones([T, 1], type: Bool.self)
            let left = split(mask, indices: [lastCol], axis: 1)[0]
            let right = split(mask, indices: [lastCol], axis: 1)[1]
            let tail = split(right, indices: [1], axis: 1)
            let after = (tail.count > 1) ? tail[1] : MLXArray.zeros([T, 0], type: Bool.self)
            mask = concatenated([left, ones, after], axis: 1)
        }

        return (frame, mask)
    }

    private func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        let K = model.args.audioNumCodebooks
        let frameW = K + 1

        let x = audio.reshaped([1, 1, audio.shape[0]])
        var codes = audioTokenizer.codec.encode(x) // [1, K, Tq]
        codes = split(codes, indices: [1], axis: 0)[0].reshaped([K, codes.shape[2]])

        if addEOS {
            let eos = MLXArray.zeros([K, 1], type: Int32.self)
            codes = concatenated([codes, eos], axis: 1) // [K, Tq+1]
        }

        let T = codes.shape[1]
        var frame = MLXArray.zeros([T, frameW], type: Int32.self) // [T, K+1]
        var mask = MLXArray.zeros([T, frameW], type: Bool.self)

        let codesT = swappedAxes(codes, 0, 1) // [T, K]
        if K > 0 {
            let leftLen = K
            let right = split(frame, indices: [leftLen], axis: 1)[1] // [T, 1]
            frame = concatenated([codesT, right], axis: 1)
        }
        if K > 0 {
            let ones = MLXArray.ones([T, K], type: Bool.self)
            let right = MLXArray.zeros([T, 1], type: Bool.self)
            mask = concatenated([ones, right], axis: 1)
        }

        return (frame, mask)
    }

    private func tokenizeSegment(_ segment: Segment, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        let (txt, txtMask) = tokenizeTextSegment(text: segment.text, speaker: segment.speaker)
        let (aud, audMask) = tokenizeAudio(segment.audio, addEOS: addEOS)
        return (concatenated([txt, aud], axis: 0), concatenated([txtMask, audMask], axis: 0))
    }
}

public extension MarvisSession {
    // MARK: - Shared model loading helpers
    // MARK: - Shared model loading helpers

    private static func snapshotAndConfig(
        repoId: String,
        progressHandler: @escaping (Progress) -> Void
    ) async throws -> (args: MarvisModelArgs, promptURLs: [URL], weightFileURL: URL) {
        let modelDirectoryURL = try await Hub.snapshot(from: repoId, progressHandler: progressHandler)
        let weightFileURL = modelDirectoryURL.appending(path: "model.safetensors")
        let promptDir = modelDirectoryURL.appending(path: "prompts", directoryHint: .isDirectory)
        var audioPromptURLs: [URL] = []
        for url in try FileManager.default.contentsOfDirectory(at: promptDir, includingPropertiesForKeys: nil) where url.pathExtension == "wav" {
            audioPromptURLs.append(url)
        }
        let configFileURL = modelDirectoryURL.appending(path: "config.json")
        let args = try JSONDecoder().decode(MarvisModelArgs.self, from: Data(contentsOf: configFileURL))
        return (args, audioPromptURLs, weightFileURL)
    }

    private func installWeights(args: MarvisModelArgs, weightFileURL: URL) throws {
        var weights: [String: MLXArray] = [:]
        let w = try loadArrays(url: weightFileURL)
        for (k, v) in w { weights[k] = v }

        if let quantization = args.quantization, let groupSize = quantization["group_size"], let bits = quantization["bits"] {
            quantize(model: self, groupSize: groupSize, bits: bits) { path, _ in
                weights["\(path).scales"] != nil
            }
        } else {
            weights = Self.sanitize(weights: weights)
        }

        let parameters = ModuleParameters.unflattened(weights)
        try update(parameters: parameters, verify: [.all])
        eval(self)
    }

    // MARK: - Generation helpers

    /// Builds the generation context from either a bound voice or reference.
    private func makeContext(voice: Voice?, refAudio: MLXArray?, refText: String?) throws -> Segment {
        if let refAudio, let refText {
            return Segment(speaker: 0, text: refText, audio: refAudio)
        } else if let voice {
            var refAudioURL: URL?
            for promptURL in _promptURLs ?? [] {
                if promptURL.lastPathComponent == "\(voice.rawValue).wav" {
                    refAudioURL = promptURL
                    break
                }
            }
            guard let refAudioURL else { throw MarvisTTSError.voiceNotFound }

            let (sampleRate, audio) = try loadAudioArray(from: refAudioURL)
            guard abs(sampleRate - 24000) < .leastNonzeroMagnitude else {
                throw MarvisTTSError.invalidRefAudio("Reference audio must be single-channel (mono) 24kHz, in WAV format.")
            }
            let refTextURL = refAudioURL.deletingPathExtension().appendingPathExtension("txt")
            let text = try String(data: Data(contentsOf: refTextURL), encoding: .utf8)
            guard let text else { throw MarvisTTSError.voiceNotFound }
            return Segment(speaker: 0, text: text, audio: audio)
        }
        throw MarvisTTSError.voiceNotFound
    }

    /// Tokenizes a single segment and returns initial token state.
    private func tokenizeStart(for segment: Segment) -> (tokens: MLXArray, mask: MLXArray, pos: MLXArray) {
        let (st, sm) = tokenizeSegment(segment, addEOS: false)
        let promptTokens = concatenated([st], axis: 0).asType(Int32.self) // [T, K+1]
        let promptMask = concatenated([sm], axis: 0).asType(Bool.self)   // [T, K+1]
        let currTokens = expandedDimensions(promptTokens, axis: 0) // [1, T, K+1]
        let currMask = expandedDimensions(promptMask, axis: 0)     // [1, T, K+1]
        let currPos = expandedDimensions(MLXArray.arange(promptTokens.shape[0]), axis: 0) // [1, T]
        return (currTokens, currMask, currPos)
    }

    /// Decodes audio frames for one prompt, optionally streaming.
    private func decodePrompt(
        currTokens startTokens: MLXArray,
        currMask startMask: MLXArray,
        currPos startPos: MLXArray,
        qualityLevel: QualityLevel,
        stream: Bool,
        streamingIntervalTokens: Int,
        sampler sampleFn: (MLXArray) -> MLXArray,
        onStreamingResult: ((GenerationResult) -> Void)?,
        enqueuePlayback: Bool
    ) -> [GenerationResult] {
        var results: [GenerationResult] = []

        var samplesFrames: [MLXArray] = [] // each is [B=1, K]
        var currTokens = startTokens
        var currMask = startMask
        var currPos = startPos

        var generatedCount = 0
        var yieldedCount = 0
        let maxAudioFrames = Int(60000 / 80.0) // 12.5 fps, 80 ms per frame
        let maxSeqLen = 2048 - maxAudioFrames
        precondition(currTokens.shape[1] < maxSeqLen, "Inputs too long, must be below max_seq_len - max_audio_frames: \(maxSeqLen)")

        var startTime = CFAbsoluteTimeGetCurrent()
        var frameCount = 0

        for frameIdx in 0 ..< maxAudioFrames {
            let frame = model.generateFrame(
                maxCodebooks: qualityLevel.codebookCount,
                tokens: currTokens,
                tokensMask: currMask,
                sampler: sampleFn
            ) // [1, K]

            // EOS if every codebook is 0
            if frame.sum().item(Int32.self) == 0 { break }

            samplesFrames.append(frame)
            frameCount += 1

            autoreleasepool {
                let zerosText = MLXArray.zeros([1, 1], type: Int32.self)
                let nextFrame = concatenated([frame, zerosText], axis: 1) // [1, K+1]
                currTokens = expandedDimensions(nextFrame, axis: 1)       // [1, 1, K+1]

                let onesK = ones([1, frame.shape[1]], type: Bool.self)
                let zero1 = zeros([1, 1], type: Bool.self)
                let nextMask = concatenated([onesK, zero1], axis: 1) // [1, K+1]
                currMask = expandedDimensions(nextMask, axis: 1)     // [1, 1, K+1]

                currPos = split(currPos, indices: [currPos.shape[1] - 1], axis: 1)[1] + MLXArray(1)
            }

            generatedCount += 1

            // Periodic cleanup
            if frameIdx % 50 == 0 && frameIdx > 0 {
                autoreleasepool { }
            }

            if stream, (generatedCount - yieldedCount) >= streamingIntervalTokens {
                yieldedCount = generatedCount
                let gr = generateResultChunk(samplesFrames, start: startTime, streaming: true, enqueuePlayback: enqueuePlayback)
                results.append(gr)
                onStreamingResult?(gr)
                samplesFrames.removeAll(keepingCapacity: true)
                startTime = CFAbsoluteTimeGetCurrent()
            }
        }

        if !samplesFrames.isEmpty {
            let gr = generateResultChunk(samplesFrames, start: startTime, streaming: stream, enqueuePlayback: enqueuePlayback)
            if stream { onStreamingResult?(gr) } else { results.append(gr) }
        }

        return results
    }
    // MARK: - Async convenience initializers (initializer-based instead of factories)

    /// Initializes and loads the Marvis model, binding a default voice.
    /// Mirrors factory-style `make(voice:)` but as an initializer for ergonomics.
    convenience init(
        voice: Voice = .conversationalA,
        repoId: String = "Marvis-AI/marvis-tts-250m-v0.1",
        progressHandler: @escaping (Progress) -> Void = { _ in },
        playbackEnabled: Bool = true
    ) async throws {
        let (args, prompts, weightFileURL) = try await Self.snapshotAndConfig(repoId: repoId, progressHandler: progressHandler)
        try await self.init(config: args, repoId: repoId, promptURLs: prompts, progressHandler: progressHandler, playbackEnabled: playbackEnabled)
        try self.installWeights(args: args, weightFileURL: weightFileURL)

        // Bind configuration
        self.boundVoice = voice
        self.boundRefAudio = nil
        self.boundRefText = nil
    }

    /// Initializes and loads the Marvis model, binding a custom reference (24 kHz mono).
    convenience init(
        refAudio: MLXArray,
        refText: String,
        repoId: String = "Marvis-AI/marvis-tts-250m-v0.1",
        progressHandler: @escaping (Progress) -> Void = { _ in },
        playbackEnabled: Bool = true
    ) async throws {
        let (args, prompts, weightFileURL) = try await Self.snapshotAndConfig(repoId: repoId, progressHandler: progressHandler)
        try await self.init(config: args, repoId: repoId, promptURLs: prompts, progressHandler: progressHandler, playbackEnabled: playbackEnabled)
        try self.installWeights(args: args, weightFileURL: weightFileURL)

        // Bind configuration
        self.boundVoice = nil
        self.boundRefAudio = refAudio
        self.boundRefText = refText
    }
    // MARK: - Factories (Apple-style ergonomics)

    /// Creates a Marvis session and binds a default voice.
    static func make(
        voice: Voice = .conversationalA,
        repoId: String = "Marvis-AI/marvis-tts-250m-v0.1",
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> MarvisSession {
        let engine = try await fromPretrained(repoId: repoId, progressHandler: progressHandler)
        engine.boundVoice = voice
        engine.boundRefAudio = nil
        engine.boundRefText = nil
        return engine
    }

    /// Creates a Marvis session and binds a custom reference voice.
    static func make(
        refAudio: MLXArray,
        refText: String,
        repoId: String = "Marvis-AI/marvis-tts-250m-v0.1",
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> MarvisSession {
        let engine = try await fromPretrained(repoId: repoId, progressHandler: progressHandler)
        engine.boundVoice = nil
        engine.boundRefAudio = refAudio
        engine.boundRefText = refText
        return engine
    }

    static func fromPretrained(repoId: String = "Marvis-AI/marvis-tts-250m-v0.1", progressHandler: @escaping (Progress) -> Void) async throws -> MarvisSession {
        let (args, prompts, weightFileURL) = try await snapshotAndConfig(repoId: repoId, progressHandler: progressHandler)
        let model = try await MarvisSession(config: args, repoId: repoId, promptURLs: prompts, progressHandler: progressHandler)
        try model.installWeights(args: args, weightFileURL: weightFileURL)
        return model
    }

    private static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (rawKey, v) in weights {
            var k = rawKey

            if !k.hasPrefix("model.") {
                k = "model." + k
            }

            if k.contains("attn") && !k.contains("self_attn") {
                k = k.replacingOccurrences(of: "attn", with: "self_attn")
                k = k.replacingOccurrences(of: "output_proj", with: "o_proj")
            }

            if k.contains("mlp") {
                k = k.replacingOccurrences(of: "w1", with: "gate_proj")
                k = k.replacingOccurrences(of: "w2", with: "down_proj")
                k = k.replacingOccurrences(of: "w3", with: "up_proj")
            }

            if k.contains("sa_norm") || k.contains("mlp_norm") {
                k = k.replacingOccurrences(of: "sa_norm", with: "input_layernorm")
                k = k.replacingOccurrences(of: "scale", with: "weight")
                k = k.replacingOccurrences(of: "mlp_norm", with: "post_attention_layernorm")
                k = k.replacingOccurrences(of: "scale", with: "weight")
            }

            if k.contains("decoder.norm") || k.contains("backbone.norm") {
                k = k.replacingOccurrences(of: "scale", with: "weight")
            }

            out[k] = v
        }

        return out
    }
}

private struct Segment {
    public let speaker: Int
    public let text: String
    public let audio: MLXArray

    public init(speaker: Int, text: String, audio: MLXArray) {
        self.speaker = speaker
        self.text = text
        self.audio = audio
    }
}

// MARK: -

public enum MarvisTTSError: Error, LocalizedError {
    case invalidArgument(String)
    case voiceNotFound
    case invalidRefAudio(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArgument(let msg):
            return msg
        case .voiceNotFound:
            return "Requested voice not found or missing reference assets."
        case .invalidRefAudio(let msg):
            return msg
        }
    }
}

public extension MarvisSession {
    struct GenerationResult {
        public let audio: [Float]
        public let sampleRate: Int
        public let sampleCount: Int
        public let frameCount: Int
        public let audioDuration: TimeInterval
        public let realTimeFactor: Double
        public let processingTime: Double
    }

    // Internal core generation routine (sync), used by async and streaming APIs.
    private func generateCore(
        text: [String],
        voice: Voice?,
        refAudio: MLXArray?,
        refText: String?,
        qualityLevel: QualityLevel,
        stream: Bool,
        streamingInterval: Double,
        onStreamingResult: ((GenerationResult) -> Void)?,
        enqueuePlayback: Bool
    ) throws -> [GenerationResult] {
        guard voice != nil || refAudio != nil else {
            throw MarvisTTSError.invalidArgument("`voice` or `refAudio`/`refText` must be specified.")
        }

        let base = try makeContext(voice: voice, refAudio: refAudio, refText: refText)
        let sampleFn = TopPSampler(temperature: 0.9, topP: 0.8).sample
        let intervalTokens = Int(streamingInterval * 12.5)
        var results: [GenerationResult] = []

        for prompt in text {
            let generationText = (base.text + " " + prompt).trimmingCharacters(in: .whitespaces)
            let seg = Segment(speaker: 0, text: generationText, audio: base.audio)

            model.resetCaches()
            if stream { streamingDecoder.reset() }

            let (tok, msk, pos) = tokenizeStart(for: seg)
            let r = decodePrompt(
                currTokens: tok,
                currMask: msk,
                currPos: pos,
                qualityLevel: qualityLevel,
                stream: stream,
                streamingIntervalTokens: intervalTokens,
                sampler: sampleFn,
                onStreamingResult: onStreamingResult,
                enqueuePlayback: enqueuePlayback
            )
            results.append(contentsOf: r)
        }

        model.resetCaches()
        if stream { streamingDecoder.reset() }
        autoreleasepool { }
        return results
    }

    /// Manually triggers memory cleanup for this TTS instance
    func cleanupMemory() {
        model.resetCaches()
        streamingDecoder.reset()
        
        // Stop audio engine
        playback?.stop()

        autoreleasepool {
            // Allow cleanup of any cached arrays
        }

    }

    private func generateResultChunk(_ frames: [MLXArray], start: CFTimeInterval, streaming: Bool, enqueuePlayback: Bool) -> GenerationResult {

        let frameCount = frames.count

        var stacked = stacked(frames, axis: 0) // [F, 1, K]
        stacked = swappedAxes(stacked, 0, 1) // [1, F, K]
        stacked = swappedAxes(stacked, 1, 2) // [1, K, F]

        let audio1x1x = streaming
            ? streamingDecoder.decodeFrames(stacked) // [1, 1, S]
            : audioTokenizer.codec.decode(stacked) // [1, 1, S]

        let sampleCount = audio1x1x.shape[2]
        let audio = audio1x1x.reshaped([sampleCount]) // [S]

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let sr = Int(sampleRate)
        let audioSeconds = Double(sampleCount) / Double(sr)
        let rtf = (audioSeconds > 0) ? elapsed / audioSeconds : 0.0


        // Create result with proper memory management
        let result = GenerationResult(
            audio: audio.asArray(Float32.self),
            sampleRate: sr,
            sampleCount: sampleCount,
            frameCount: frameCount,
            audioDuration: audioSeconds,
            realTimeFactor: (rtf * 100).rounded() / 100,
            processingTime: elapsed,
        )

        // Play the generated audio (if enabled)
        if enqueuePlayback {
            playback?.enqueue(result.audio, prebufferSeconds: streaming ? 2.0 : 0.0)
        }

        // Force cleanup of large intermediate arrays
        autoreleasepool {
            // The stacked array and audio1x1x are large and should be released
            _ = stacked  // Keep reference until autoreleasepool exits
            _ = audio1x1x  // Keep reference until autoreleasepool exits
            _ = audio  // Keep reference until autoreleasepool exits
        }

        return result
    }

    // MARK: - Async/Await convenience

    /// Non-blocking async variant for a single text string.
    func generateAsync(
        text: String,
        voice: Voice? = .conversationalA,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        qualityLevel: QualityLevel = .maximum,
        splitPattern: String? = #"(\n+)"#
    ) async throws -> [GenerationResult] {
        let pieces: [String]
        if let pat = splitPattern, let re = try? NSRegularExpression(pattern: pat) {
            let full = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let range = NSRange(full.startIndex ..< full.endIndex, in: full)
            let splits = re.split(full, range: range)
            pieces = splits.isEmpty ? [full] : splits
        } else {
            pieces = [text]
        }

        return try await Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return [] }
            return try self.generateCore(
                text: pieces,
                voice: voice,
                refAudio: refAudio,
                refText: refText,
                qualityLevel: qualityLevel,
                stream: false,
                streamingInterval: 0.5,
                onStreamingResult: nil,
                enqueuePlayback: self.playbackEnabled
            )
        }.value
    }

    // MARK: - Streaming API (AsyncThrowingStream)

    /// Streams generated audio chunks for the given text as an AsyncThrowingStream.
    /// Each yielded `GenerationResult` represents a chunk of decoded audio.
    func stream(
        text: String,
        voice: Voice? = .conversationalA,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        qualityLevel: QualityLevel = .maximum,
        splitPattern: String? = #"(\n+)"#,
        streamingInterval: Double = 0.5
    ) -> AsyncThrowingStream<GenerationResult, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [weak self] in
                guard let self else { return }
                do {
                    let pieces: [String]
                    if let pat = splitPattern, let re = try? NSRegularExpression(pattern: pat) {
                        let full = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        let range = NSRange(full.startIndex ..< full.endIndex, in: full)
                        let splits = re.split(full, range: range)
                        pieces = splits.isEmpty ? [full] : splits
                    } else {
                        pieces = [text]
                    }

                    _ = try self.generateCore(
                        text: pieces,
                        voice: voice,
                        refAudio: refAudio,
                        refText: refText,
                        qualityLevel: qualityLevel,
                        stream: true,
                        streamingInterval: streamingInterval,
                        onStreamingResult: { gr in
                            continuation.yield(gr)
                        },
                        enqueuePlayback: self.playbackEnabled
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Apple-style shorthand using bound configuration

    /// Synthesizes speech using the bound voice or reference; returns a single merged result.
    /// Shorthand 'generate(for:)' mirrors Python 'generate_audio' semantics while staying Swifty.
    func generate(for text: String, quality: QualityLevel? = nil) async throws -> GenerationResult {
        let results = try await generateAsync(
            text: text,
            voice: boundVoice,
            refAudio: boundRefAudio,
            refText: boundRefText,
            qualityLevel: quality ?? boundQuality
        )
        return Self.mergeResults(results)
    }

    /// Generates speech without enqueuing playback; returns one merged result.
    func generateRaw(for text: String, quality: QualityLevel? = nil) async throws -> GenerationResult {
        let pieces = [text]
        let qualityToUse = quality ?? boundQuality
        let results: [GenerationResult] = try await Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return [] as [GenerationResult] }
            return try self.generateCore(
                text: pieces,
                voice: self.boundVoice,
                refAudio: self.boundRefAudio,
                refText: self.boundRefText,
                qualityLevel: qualityToUse,
                stream: false,
                streamingInterval: 0.5,
                onStreamingResult: nil,
                enqueuePlayback: false
            )
        }.value
        return Self.mergeResults(results)
    }

    /// Streams speech using the bound voice or reference.
    func stream(_ text: String, quality: QualityLevel? = nil, interval: Double = 0.5) -> AsyncThrowingStream<GenerationResult, Error> {
        stream(
            text: text,
            voice: boundVoice,
            refAudio: boundRefAudio,
            refText: boundRefText,
            qualityLevel: quality ?? boundQuality,
            streamingInterval: interval
        )
    }

    /// Merges multiple chunk results into a single aggregated result.
    private static func mergeResults(_ parts: [GenerationResult]) -> GenerationResult {
        guard let first = parts.first else {
            return GenerationResult(
                audio: [], sampleRate: 24_000, sampleCount: 0,
                frameCount: 0, audioDuration: 0, realTimeFactor: 0, processingTime: 0
            )
        }
        if parts.count == 1 { return first }

        var samples: [Float] = []
        samples.reserveCapacity(parts.reduce(0) { $0 + $1.sampleCount })
        var sampleCount = 0
        var frameCount = 0
        var audioDuration: Double = 0
        var processingTime: Double = 0

        for r in parts {
            samples += r.audio
            sampleCount += r.sampleCount
            frameCount += r.frameCount
            audioDuration += r.audioDuration
            processingTime += r.processingTime
        }

        let rtf = audioDuration > 0 ? processingTime / audioDuration : 0
        return GenerationResult(
            audio: samples,
            sampleRate: first.sampleRate,
            sampleCount: sampleCount,
            frameCount: frameCount,
            audioDuration: audioDuration,
            realTimeFactor: (rtf * 100).rounded() / 100,
            processingTime: processingTime
        )
    }
}

// MARK: -

private extension NSRegularExpression {
    func split(_ s: String, range: NSRange) -> [String] {
        var last = 0
        var parts: [String] = []
        enumerateMatches(in: s, options: [], range: range) { m, _, _ in
            guard let m else { return }
            let r = NSRange(location: last, length: m.range.location - last)
            if let rr = Range(r, in: s) {
                let piece = String(s[rr]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !piece.isEmpty { parts.append(piece) }
            }
            last = m.range.upperBound
        }
        let tailR = NSRange(location: last, length: range.upperBound - last)
        if let rr = Range(tailR, in: s) {
            let piece = String(s[rr]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { parts.append(piece) }
        }
        return parts
    }
}

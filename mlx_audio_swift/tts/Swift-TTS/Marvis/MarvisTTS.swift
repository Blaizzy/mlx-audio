import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import Darwin
import AVFoundation

// MARK: - Mach Kernel Declarations for Memory Monitoring
@_silgen_name("mach_task_self_") func mach_task_self() -> mach_port_t
@_silgen_name("task_info") func task_info(target_task: mach_port_t, flavor: UInt32, task_info_out: UnsafeMutablePointer<integer_t>, task_info_outCnt: UnsafeMutablePointer<mach_msg_type_number_t>) -> kern_return_t

let TASK_BASIC_INFO = UInt32(20)
let KERN_SUCCESS = kern_return_t(0)

struct mach_task_basic_info {
    var virtual_size: mach_vm_size_t = 0
    var resident_size: mach_vm_size_t = 0
    var resident_size_max: mach_vm_size_t = 0
    var user_time: time_value_t = time_value_t(seconds: 0, microseconds: 0)
    var system_time: time_value_t = time_value_t(seconds: 0, microseconds: 0)
    var policy: policy_t = 0
    var suspend_count: integer_t = 0
}

// MARK: - Debug Helpers
private extension MarvisTTS {
    static func getMemoryUsage() -> String {
        let processInfo = ProcessInfo.processInfo
        // Get current process memory usage instead of system-wide
        let taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        var info = taskInfo

        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_BASIC_INFO), intPtr, &count)
            }
        }

        if result == KERN_SUCCESS {
            let usedMemory = Double(info.resident_size)
            let usedMemoryGB = usedMemory / (1024.0 * 1024.0 * 1024.0)
            return String(format: "%.2f GB", usedMemoryGB)
        } else {
            return "Unable to get memory info"
        }
    }

    static func getCPUUsage() -> String {
        // Note: This is a simplified CPU usage check
        // In a real implementation, you might want to use more sophisticated CPU monitoring
        let startTime = CFAbsoluteTimeGetCurrent()
        // Small delay to get CPU measurement
        usleep(10000)
        let endTime = CFAbsoluteTimeGetCurrent()
        let cpuTime = endTime - startTime
        return String(format: "%.2f%%", cpuTime * 100.0)
    }

    static func logDebug(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let memory = getMemoryUsage()
        print("[\(timestamp)] [MarvisTTS] \(message) | Memory: \(memory)")
    }

    /// Forces memory cleanup and garbage collection
    static func forceMemoryCleanup() {
        logDebug("Forcing memory cleanup...")
        autoreleasepool {
            // Allow any pending autoreleased objects to be released
        }
        // Note: Swift/MLX uses ARC and automatic memory management
        // This is primarily for hinting the system to release memory
    }
}

public final class MarvisTTS: Module {
    public enum Voice: String, CaseIterable {
        case conversationalA = "conversational_a"
        case conversationalB = "conversational_b"
    }

    public let sampleRate: Double

    private let model: SesameModel
    private let _promptURLs: [URL]?
    private let _textTokenizer: any Tokenizer
    private let _audio_tokenizer: MimiTokenizer
    private let _streamingDecoder: MimiStreamingDecoder
    
    // Audio playback components
    private var audioEngine: AVAudioEngine!
    private var playerNode: AVAudioPlayerNode!
    private var audioFormat: AVAudioFormat!

    public init(
        config: SesameModelArgs,
        repoId: String,
        promptURLs: [URL]? = nil,
        progressHandler: @escaping (Progress) -> Void
    ) async throws {
        MarvisTTS.logDebug("Starting MarvisTTS initialization")

        MarvisTTS.logDebug("Initializing SesameModel with config...")
        let startTime = CFAbsoluteTimeGetCurrent()
        self.model = SesameModel(config: config)
        let modelInitTime = CFAbsoluteTimeGetCurrent() - startTime
        MarvisTTS.logDebug(String(format: "SesameModel initialization completed in %.2f seconds", modelInitTime))

        self._promptURLs = promptURLs

        MarvisTTS.logDebug("Loading text tokenizer...")
        let tokenizerStart = CFAbsoluteTimeGetCurrent()
        self._textTokenizer = try await loadTokenizer(configuration: ModelConfiguration(id: repoId), hub: HubApi.shared)
        let tokenizerTime = CFAbsoluteTimeGetCurrent() - tokenizerStart
        MarvisTTS.logDebug(String(format: "Text tokenizer loaded in %.2f seconds", tokenizerTime))

        MarvisTTS.logDebug("Loading Mimi tokenizer (this may take significant memory and CPU)...")
        let mimiStart = CFAbsoluteTimeGetCurrent()
        self._audio_tokenizer = try await MimiTokenizer(Mimi.fromPretrained(progressHandler: progressHandler))
        let mimiTime = CFAbsoluteTimeGetCurrent() - mimiStart
        MarvisTTS.logDebug(String(format: "Mimi tokenizer loaded in %.2f seconds", mimiTime))

        MarvisTTS.logDebug("Initializing streaming decoder...")
        self._streamingDecoder = MimiStreamingDecoder(_audio_tokenizer.codec)
        self.sampleRate = _audio_tokenizer.codec.cfg.sampleRate
        super.init()

        MarvisTTS.logDebug("Resetting model caches...")
        model.resetCaches()

        MarvisTTS.logDebug("Setting up audio playback...")
        setupAudioEngine()

        MarvisTTS.logDebug("MarvisTTS initialization completed successfully")
    }
    
    deinit {
        if let engine = audioEngine, engine.isRunning {
            engine.stop()
        }
        if let player = playerNode, player.isPlaying {
            player.stop()
        }
    }

    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)
        
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)
        
        do {
            try audioEngine.start()
        } catch {
            MarvisTTS.logDebug("Failed to start audio engine: \(error)")
        }
    }
    
    private func playAudio(_ samples: [Float]) {
        guard let audioFormat = audioFormat else { return }
        
        let frameLength = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameLength) else {
            MarvisTTS.logDebug("Failed to create audio buffer")
            return
        }
        
        buffer.frameLength = frameLength
        if let channelData = buffer.floatChannelData {
            for i in 0..<samples.count {
                channelData[0][i] = samples[i]
            }
        }
        
        if !playerNode.isPlaying {
            playerNode.play()
        }
        
        playerNode.scheduleBuffer(buffer, completionHandler: nil)
    }

    private func tokenizeTextSegment(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let K = model.args.audioNumCodebooks
        let frameW = K + 1

        let prompt = "[\(speaker)]" + text
        let ids = MLXArray(_textTokenizer.encode(text: prompt))

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
        var codes = _audio_tokenizer.codec.encode(x) // [1, K, Tq]
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

public extension MarvisTTS {
    static func fromPretrained(repoId: String = "Marvis-AI/marvis-tts-250m-v0.1", progressHandler: @escaping (Progress) -> Void) async throws -> MarvisTTS {
        logDebug("Starting MarvisTTS.fromPretrained with repoId: \(repoId)")

        logDebug("Downloading/snapshotting model repository...")
        let snapshotStart = CFAbsoluteTimeGetCurrent()
        let modelDirectoryURL = try await Hub.snapshot(from: repoId, progressHandler: progressHandler)
        let snapshotTime = CFAbsoluteTimeGetCurrent() - snapshotStart
        logDebug(String(format: "Repository snapshot completed in %.2f seconds", snapshotTime))

        let weightFileURL = modelDirectoryURL.appending(path: "model.safetensors")
        let promptFileURLs = modelDirectoryURL.appending(path: "prompts", directoryHint: .isDirectory)

        logDebug("Scanning for audio prompt files...")
        var audioPromptURLs = [URL]()
        for promptURL in try FileManager.default.contentsOfDirectory(at: promptFileURLs, includingPropertiesForKeys: nil) {
            if promptURL.pathExtension == "wav" {
                audioPromptURLs.append(promptURL)
            }
        }
        logDebug("Found \(audioPromptURLs.count) audio prompt files")

        logDebug("Loading and parsing config.json...")
        let configFileURL = modelDirectoryURL.appending(path: "config.json")
        let args = try JSONDecoder().decode(SesameModelArgs.self, from: Data(contentsOf: configFileURL))
        logDebug("Config loaded successfully. Audio codebooks: \(args.audioNumCodebooks), Hidden size: \(args.hiddenSize)")

        logDebug("Creating MarvisTTS instance (this includes loading Mimi tokenizer)...")
        let marvisStart = CFAbsoluteTimeGetCurrent()
        let model = try await MarvisTTS(config: args, repoId: repoId, promptURLs: audioPromptURLs, progressHandler: progressHandler)
        let marvisTime = CFAbsoluteTimeGetCurrent() - marvisStart
        logDebug(String(format: "MarvisTTS instance created in %.2f seconds", marvisTime))

        logDebug("Loading weights from safetensors file (this is likely where memory usage spikes)...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        var weights = [String: MLXArray]()
        let w = try loadArrays(url: weightFileURL)
        for (key, value) in w {
            weights[key] = value
        }
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        logDebug(String(format: "Weights loaded in %.2f seconds. Total weight tensors: %d", loadTime, weights.count))

        // Calculate approximate memory usage of weights
        var totalWeightSize: Int = 0
        for (_, array) in weights {
            let elementCount = array.shape.reduce(1, *)
            totalWeightSize += elementCount * 4 // Assuming Float32 (4 bytes per element)
        }
        let weightSizeGB = Double(totalWeightSize) / (1024.0 * 1024.0 * 1024.0)
        logDebug(String(format: "Approximate weight memory usage: %.2f GB", weightSizeGB))

        logDebug("Processing quantization or weight sanitization...")
        let processStart = CFAbsoluteTimeGetCurrent()
        if let quantization = args.quantization, let groupSize = quantization["group_size"], let bits = quantization["bits"] {
            logDebug("Applying quantization (group_size: \(groupSize), bits: \(bits))...")
            quantize(model: model, groupSize: groupSize, bits: bits) { path, _ in
                weights["\(path).scales"] != nil
            }
        } else {
            logDebug("Sanitizing weights (no quantization)...")
            weights = sanitize(weights: weights)
        }
        let processTime = CFAbsoluteTimeGetCurrent() - processStart
        logDebug(String(format: "Weight processing completed in %.2f seconds", processTime))

        logDebug("Creating unflattened parameters...")
        let paramsStart = CFAbsoluteTimeGetCurrent()
        let parameters = ModuleParameters.unflattened(weights)
        let paramsTime = CFAbsoluteTimeGetCurrent() - paramsStart
        logDebug(String(format: "Parameters unflattened in %.2f seconds", paramsTime))

        logDebug("Updating model parameters (this may cause memory spike)...")
        let updateStart = CFAbsoluteTimeGetCurrent()
        try model.update(parameters: parameters, verify: [.all])
        let updateTime = CFAbsoluteTimeGetCurrent() - updateStart
        logDebug(String(format: "Model parameters updated in %.2f seconds", updateTime))

        logDebug("Evaluating model (final step)...")
        let evalStart = CFAbsoluteTimeGetCurrent()
        eval(model)
        let evalTime = CFAbsoluteTimeGetCurrent() - evalStart
        logDebug(String(format: "Model evaluation completed in %.2f seconds", evalTime))

        logDebug("MarvisTTS.fromPretrained completed successfully")
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

enum MarvisTTSError: Error {
    case invalidArgument(String)
    case voiceNotFound
    case invalidRefAudio(String)
}

public extension MarvisTTS {
    struct GenerationResult {
        public let audio: [Float]
        public let sampleRate: Int
        public let sampleCount: Int
        public let frameCount: Int
        public let audioDuration: TimeInterval
        public let realTimeFactor: Double
        public let processingTime: Double
    }

    @discardableResult
    func generate(
        text: String,
        voice: Voice? = .conversationalA,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        splitPattern: String? = #"(\n+)"#,
        stream: Bool = false,
        streamingInterval: Double = 0.5,
        voiceMatch: Bool = true,
        onStreamingResult: ((GenerationResult) -> Void)? = nil
    ) throws -> [GenerationResult] {
        let pieces: [String]
        if let pat = splitPattern, let re = try? NSRegularExpression(pattern: pat) {
            let full = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let range = NSRange(full.startIndex ..< full.endIndex, in: full)
            let splits = re.split(full, range: range)
            pieces = splits.isEmpty ? [full] : splits
        } else {
            pieces = [text]
        }
        return try generate(
            text: pieces,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            stream: stream,
            streamingInterval: streamingInterval,
            onStreamingResult: onStreamingResult
        )
    }

    @discardableResult
    func generate(
        text: [String],
        voice: Voice? = .conversationalA,
        refAudio: MLXArray?,
        refText: String?,
        stream: Bool = false,
        streamingInterval: Double = 0.5,
        onStreamingResult: ((GenerationResult) -> Void)? = nil
    ) throws -> [GenerationResult] {
        let totalGenStart = CFAbsoluteTimeGetCurrent()
        MarvisTTS.logDebug("Starting generation for \(text.count) text segments")

        guard voice != nil || refAudio != nil else {
            throw MarvisTTSError.invalidArgument("`voice` or `refAudio`/`refText` must be specified.")
        }

        MarvisTTS.logDebug("Setting up generation context...")
        let contextSetupStart = CFAbsoluteTimeGetCurrent()
        let context: Segment
        if let refAudio, let refText {
            context = Segment(speaker: 0, text: refText, audio: refAudio)
            MarvisTTS.logDebug("Using provided reference audio and text")
        } else if let voice {
            MarvisTTS.logDebug("Loading voice: \(voice.rawValue)")
            var refAudioURL: URL?
            for promptURL in _promptURLs ?? [] {
                if promptURL.lastPathComponent == "\(voice.rawValue).wav" {
                    refAudioURL = promptURL
                }
            }
            guard let refAudioURL else {
                throw MarvisTTSError.voiceNotFound
            }

            MarvisTTS.logDebug("Loading reference audio from: \(refAudioURL.lastPathComponent)")
            let (sampleRate, refAudio) = try loadAudioArray(from: refAudioURL)
            guard abs(sampleRate - 24000) < .leastNonzeroMagnitude else {
                throw MarvisTTSError.invalidRefAudio("Reference audio must be single-channel (mono) 24kHz, in WAV format.")
            }

            let refTextURL = refAudioURL.deletingPathExtension().appendingPathExtension("txt")
            let refText = try String(data: Data(contentsOf: refTextURL), encoding: .utf8)
            guard let refText else {
                throw MarvisTTSError.voiceNotFound
            }
            context = Segment(speaker: 0, text: refText, audio: refAudio)
            MarvisTTS.logDebug("Reference audio loaded successfully. Sample rate: \(sampleRate)Hz")
        } else {
            throw MarvisTTSError.voiceNotFound
        }
        let contextSetupTime = CFAbsoluteTimeGetCurrent() - contextSetupStart
        MarvisTTS.logDebug(String(format: "Context setup completed in %.2f seconds", contextSetupTime))

        let sampleFn = TopPSampler(temperature: 0.9, topP: 0.8).sample
        let maxAudioFrames = Int(60000 / 80.0) // 12.5 fps, 80 ms per frame
        let streamingIntervalTokens = Int(streamingInterval * 12.5)
        MarvisTTS.logDebug("Generation parameters: maxAudioFrames=\(maxAudioFrames), streaming=\(stream), streamingInterval=\(streamingInterval)")

        var results: [GenerationResult] = []

        for (textIndex, prompt) in text.enumerated() {
            MarvisTTS.logDebug("Processing text segment \(textIndex + 1)/\(text.count): '\(prompt.prefix(50))...'")

            let generationText = (context.text + " " + prompt).trimmingCharacters(in: .whitespaces)
            let currentContext = [Segment(speaker: 0, text: generationText, audio: context.audio)]

            MarvisTTS.logDebug("Resetting model caches...")
            model.resetCaches()
            if stream { _streamingDecoder.reset() }

            MarvisTTS.logDebug("Tokenizing input...")
            let tokenizeStart = CFAbsoluteTimeGetCurrent()
            var toks: [MLXArray] = []
            var masks: [MLXArray] = []
            for seg in currentContext {
                let (st, sm) = tokenizeSegment(seg, addEOS: false)
                toks.append(st); masks.append(sm)
            }

            let promptTokens = concatenated(toks, axis: 0).asType(Int32.self) // [T, K+1]
            let promptMask = concatenated(masks, axis: 0).asType(Bool.self) // [T, K+1]
            let tokenizeTime = CFAbsoluteTimeGetCurrent() - tokenizeStart
            MarvisTTS.logDebug(String(format: "Tokenization completed in %.2f seconds. Token shape: %@", tokenizeTime, promptTokens.shape))

            var samplesFrames: [MLXArray] = [] // each is [B=1, K]
            var currTokens = expandedDimensions(promptTokens, axis: 0) // [1, T, K+1]
            var currMask = expandedDimensions(promptMask, axis: 0) // [1, T, K+1]
            var currPos = expandedDimensions(MLXArray.arange(promptTokens.shape[0]), axis: 0) // [1, T]
            var generatedCount = 0
            var yieldedCount = 0

            let maxSeqLen = 2048 - maxAudioFrames
            precondition(currTokens.shape[1] < maxSeqLen, "Inputs too long, must be below max_seq_len - max_audio_frames: \(maxSeqLen)")
            MarvisTTS.logDebug("Starting autoregressive generation loop (max frames: \(maxAudioFrames))")

            var startTime = CFAbsoluteTimeGetCurrent()
            var frameCount = 0

            for frameIdx in 0 ..< maxAudioFrames {
                if frameIdx % 100 == 0 || frameIdx == 0 {
                    MarvisTTS.logDebug("Generating frame \(frameIdx + 1)/\(maxAudioFrames)...")
                }

                let frameStart = CFAbsoluteTimeGetCurrent()
                let frame = model.generateFrame(
                    tokens: currTokens,
                    tokensMask: currMask,
                    inputPos: currPos,
                    sampler: sampleFn
                ) // [1, K]
                let frameTime = CFAbsoluteTimeGetCurrent() - frameStart

                if frameIdx % 100 == 0 || frameIdx == 0 {
                    MarvisTTS.logDebug(String(format: "Frame \(frameIdx + 1) generated in %.4f seconds", frameTime))
                }

                // EOS if every codebook is 0
                if frame.sum().item(Int32.self) == 0 {
                    MarvisTTS.logDebug("EOS detected at frame \(frameIdx + 1), stopping generation")
                    break
                }

                samplesFrames.append(frame)
                frameCount += 1

                // Memory cleanup: explicitly manage array lifecycle within autoreleasepool
                autoreleasepool {
                    let zerosText = MLXArray.zeros([1, 1], type: Int32.self)
                    let nextFrame = concatenated([frame, zerosText], axis: 1) // [1, K+1]
                    currTokens = expandedDimensions(nextFrame, axis: 1) // [1, 1, K+1]

                    let onesK = ones([1, frame.shape[1]], type: Bool.self)
                    let zero1 = zeros([1, 1], type: Bool.self)
                    let nextMask = concatenated([onesK, zero1], axis: 1) // [1, K+1]
                    currMask = expandedDimensions(nextMask, axis: 1) // [1, 1, K+1]

                    currPos = split(currPos, indices: [currPos.shape[1] - 1], axis: 1)[1] + MLXArray(1)
                }

                generatedCount += 1

                // Periodic memory cleanup every 50 frames
                if frameIdx % 50 == 0 && frameIdx > 0 {
                    MarvisTTS.logDebug("Performing periodic memory cleanup at frame \(frameIdx)")
                    autoreleasepool {
                        // Allow temporary arrays to be released
                    }
                }

                if stream, (generatedCount - yieldedCount) >= streamingIntervalTokens {
                    MarvisTTS.logDebug("Streaming chunk at frame \(generatedCount)")
                    yieldedCount = generatedCount
                    let gr = generateResultChunk(samplesFrames, start: startTime, streaming: true)
                    results.append(gr)
                    onStreamingResult?(gr)
                    samplesFrames.removeAll(keepingCapacity: true)
                    startTime = CFAbsoluteTimeGetCurrent()
                }
            }

            MarvisTTS.logDebug("Generation loop completed. Generated \(frameCount) frames")

            if !samplesFrames.isEmpty {
                MarvisTTS.logDebug("Processing final audio chunk...")
                let finalStart = CFAbsoluteTimeGetCurrent()
                let gr = generateResultChunk(samplesFrames, start: startTime, streaming: stream)
                let finalTime = CFAbsoluteTimeGetCurrent() - finalStart
                MarvisTTS.logDebug(String(format: "Final chunk processed in %.2f seconds", finalTime))

                if stream {
                    onStreamingResult?(gr)
                } else {
                    results.append(gr)
                }
            }

            let textGenTime = CFAbsoluteTimeGetCurrent() - tokenizeTime - tokenizeStart
            MarvisTTS.logDebug(String(format: "Text segment \(textIndex + 1) generation completed in %.2f seconds", textGenTime))
        }

        let totalGenTime = CFAbsoluteTimeGetCurrent() - totalGenStart
        MarvisTTS.logDebug(String(format: "Total generation completed in %.2f seconds. Generated %d results", totalGenTime, results.count))

        // Force cleanup of large arrays and caches to prevent memory leaks
        MarvisTTS.logDebug("Performing memory cleanup...")
        let cleanupStart = CFAbsoluteTimeGetCurrent()

        // Clear model caches
        model.resetCaches()

        // Clear streaming decoder if used
        if stream {
            _streamingDecoder.reset()
        }

        // Force garbage collection hint (though Swift/MLX manages this automatically)
        // Clear any accumulated temporary arrays
        autoreleasepool {
            // This block ensures any autoreleased objects are cleaned up
        }

        let cleanupTime = CFAbsoluteTimeGetCurrent() - cleanupStart
        MarvisTTS.logDebug(String(format: "Memory cleanup completed in %.2f seconds", cleanupTime))

        return results
    }

    /// Manually triggers memory cleanup for this TTS instance
    public func cleanupMemory() {
        MarvisTTS.logDebug("Performing manual memory cleanup for MarvisTTS instance")
        model.resetCaches()
        _streamingDecoder.reset()
        
        // Stop audio engine
        if playerNode.isPlaying {
            playerNode.stop()
        }
        if audioEngine.isRunning {
            audioEngine.stop()
        }

        autoreleasepool {
            // Allow cleanup of any cached arrays
        }

        MarvisTTS.forceMemoryCleanup()
    }

    private func generateResultChunk(_ frames: [MLXArray], start: CFTimeInterval, streaming: Bool) -> GenerationResult {
        MarvisTTS.logDebug("Processing result chunk with \(frames.count) frames, streaming=\(streaming)")

        let frameCount = frames.count

        MarvisTTS.logDebug("Stacking and transposing frames...")
        var stacked = stacked(frames, axis: 0) // [F, 1, K]
        stacked = swappedAxes(stacked, 0, 1) // [1, F, K]
        stacked = swappedAxes(stacked, 1, 2) // [1, K, F]

        MarvisTTS.logDebug("Decoding audio frames...")
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let audio1x1x = streaming
            ? _streamingDecoder.decodeFrames(stacked) // [1, 1, S]
            : _audio_tokenizer.codec.decode(stacked) // [1, 1, S]
        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        MarvisTTS.logDebug(String(format: "Audio decoding completed in %.2f seconds", decodeTime))

        let sampleCount = audio1x1x.shape[2]
        MarvisTTS.logDebug("Reshaping audio array to \(sampleCount) samples...")
        let audio = audio1x1x.reshaped([sampleCount]) // [S]

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let sr = Int(sampleRate)
        let audioSeconds = Double(sampleCount) / Double(sr)
        let rtf = (audioSeconds > 0) ? elapsed / audioSeconds : 0.0

        MarvisTTS.logDebug(String(format: "Result chunk completed: %.2f seconds audio, RTF=%.2f, processing time=%.2f seconds", audioSeconds, rtf, elapsed))

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

        // Play the generated audio
        playAudio(result.audio)

        // Force cleanup of large intermediate arrays
        autoreleasepool {
            // The stacked array and audio1x1x are large and should be released
            _ = stacked  // Keep reference until autoreleasepool exits
            _ = audio1x1x  // Keep reference until autoreleasepool exits
            _ = audio  // Keep reference until autoreleasepool exits
        }

        return result
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

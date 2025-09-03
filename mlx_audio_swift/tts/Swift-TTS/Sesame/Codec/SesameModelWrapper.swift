//
// SesameModelWrapper for Sesame TTS
// Main Model wrapper class with generation pipeline
// Based on Python mlx_audio/tts/models/sesame/sesame.py Model class
// Updated to use HuggingFace Tokenizers package following Marvis TTS pattern
//

import Foundation
import AVFoundation
import Hub
import MLX
import MLXNN
import MLXLMCommon
import MLXRandom
import Tokenizers

/// Sesame TTS Error Types
enum SesameTTSError: Error {
    case inputTooLong(maxLength: Int, actualLength: Int)
    case modelNotInitialized
    case tokenizationFailed(reason: String)
    case generationFailed(reason: String)
    case invalidConfiguration(reason: String)

    var localizedDescription: String {
        switch self {
        case .inputTooLong(let max, let actual):
            return "Input too long: maximum \(max) tokens, got \(actual) tokens"
        case .modelNotInitialized:
            return "Model not initialized. Call ensureInitialized() first."
        case .tokenizationFailed(let reason):
            return "Tokenization failed: \(reason)"
        case .generationFailed(let reason):
            return "Audio generation failed: \(reason)"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        }
    }
}

/// Generation result containing audio and metadata
/// Equivalent to Python's GenerationResult
struct GenerationResult {
    let audio: MLXArray
    let samples: Int
    let sampleRate: Int
    let segmentIdx: Int
    let tokenCount: Int
    let audioDuration: String
    let realTimeFactor: Float
    let prompt: [String: Any]
    let audioSamples: [String: Any]
    let processingTimeSeconds: Double
    let peakMemoryUsage: Float
}

/// Main Model wrapper for Sesame TTS
/// Equivalent to Python's Model class with Swift optimizations
class SesameModelWrapper: Module {
    @ModuleInfo var model: SesameModel?
    @ModuleInfo var audioTokenizer: Mimi?

    private let config: LlamaModelArgs
    // Removed external Tokenizer; using simple whitespace tokenization path
    private var mimiTokenizer: MimiTokenizer?
    private var voiceManager: SesameVoiceManager? // Voice management system
    private var streamingDecoder: MimiStreamingDecoder?
    private var watermarker: Any? // We'll implement watermarking later
    private var sampleRate: Int

    // Swift optimization flags
    private var isInitialized = false
    private var lastMemoryUsage: Float = 0.0

    /// Initialize the Model wrapper
    /// - Parameter config: Model configuration
    init(_ config: LlamaModelArgs) {
        self.config = config
        // Sample rate will be set when Mimi is initialized
        self.sampleRate = 24000 // Default, will be updated

        super.init()
    }

    /// Ensure model is initialized (lazy initialization)
    /// Following Kokoro's pattern for memory efficiency
    private func ensureInitialized() {
        guard !isInitialized else { return }
        
        print("🚀 DEBUG ensureInitialized: Starting model initialization...")

        autoreleasepool {
            print("🚀 DEBUG ensureInitialized: Creating SesameModel...")
            // Initialize heavy ML components
            let sesameModel = SesameModel(config)
            print("🚀 DEBUG ensureInitialized: SesameModel created, setting up caches...")
            
            sesameModel.setupCaches(maxBatchSize: 1)
            print("🚀 DEBUG ensureInitialized: Caches set up, assigning to wrapper...")
            
            self._model.wrappedValue = sesameModel
            print("🚀 DEBUG ensureInitialized: SesameModel assigned successfully")

            print("🚀 DEBUG ensureInitialized: Creating Mimi codec...")
            // Initialize Mimi codec with pre-trained weights
            let mimiConfig = MimiConfig.mimi202407(numCodebooks: config.audioNumCodebooks)
            let mimi = Mimi(mimiConfig)
            // Load Mimi weights from bundled resource if available
            if let mimiPath = Bundle.main.path(forResource: "sesame-mimi", ofType: "safetensors", inDirectory: "Sesame/Resources") {
                let url = URL(fileURLWithPath: mimiPath)
                print("🔄 Loading Mimi weights: \(url.lastPathComponent)")
                _ = mimi.loadPytorchWeights(url: url, strict: false)
            } else {
                print("⚠️  WARNING: Mimi weights not found in bundle; codec will be uninitialized")
            }

            self._audioTokenizer.wrappedValue = mimi
            self.mimiTokenizer = MimiTokenizer(mimi)
            print("🚀 DEBUG ensureInitialized: Mimi assigned successfully")

            // Update sample rate from Mimi codec
            self.sampleRate = Int(mimi.sampleRate)
            print("🚀 DEBUG ensureInitialized: Sample rate set to \(self.sampleRate)")

            print("🚀 DEBUG ensureInitialized: Creating streaming decoder...")
            // Initialize streaming decoder
            self.streamingDecoder = MimiStreamingDecoder(mimi)
            print("🚀 DEBUG ensureInitialized: Streaming decoder created")

            print("🚀 DEBUG ensureInitialized: Using built-in simple tokenizer")

            // Initialize voice manager with tokenizer
            self.voiceManager = SesameVoiceManager(tokenizer: nil) // Updated voice manager
            print("🚀 DEBUG ensureInitialized: Voice manager created")

            // TODO: Initialize watermarker

            // Attempt to load Llama backbone/decoder weights from bundled model
            self.loadBackboneDecoderWeights()

            isInitialized = true
            print("🚀 DEBUG ensureInitialized: Initialization complete!")
        }
    }

    /// Get the sample rate
    var sampleRateProperty: Int {
        return sampleRate
    }

    /// Get model layers (for quantization predicate)
    var layers: [LlamaTransformerLayer] {
        guard let model = model else { return [] }
        return model.backbone.layers
    }

    /// Reset model to free up memory (Kokoro-inspired)
    /// - Parameter preserveTextProcessing: Whether to keep tokenizer components
    func resetModel(preserveTextProcessing: Bool = true) {
        // Clear GPU cache first
        MLX.GPU.clearCache()

        // Reset heavy ML components using autoreleasepool
        autoreleasepool {
            self._model.wrappedValue = nil
            self._audioTokenizer.wrappedValue = nil
            self.streamingDecoder = nil
        }

        // Reset flags
        isInitialized = false
        lastMemoryUsage = 0.0
    }

    /// Create a Model wrapper with sesame_config.json configuration
    /// - Returns: Configured Model wrapper
    static func createDefault() throws -> SesameModelWrapper {
        // Load configuration from sesame_config.json
        guard let configPath = Bundle.main.path(forResource: "sesame_config", ofType: "json") else {
            throw SesameTTSError.invalidConfiguration(reason: "Could not find sesame_config.json in bundle")
        }

        let config = try LlamaModelArgs.fromSesameConfig(configPath: configPath)
        return SesameModelWrapper(config)
    }

    /// Tokenize text segment with speaker information (following Marvis TTS pattern)
    /// - Parameters:
    ///   - text: Text to tokenize
    ///   - speaker: Speaker ID
    /// - Returns: Tuple of (tokens, mask) arrays with shape (seq_len, 33)
    private func tokenizeTextSegment(_ text: String, speaker: Int) -> (MLXArray, MLXArray) {
        print("🔤 DEBUG tokenizeTextSegment: Input text='\(text)', speaker=\(speaker)")

        let K = config.audioNumCodebooks
        let frameW = K + 1
        
        let prompt = "[\(speaker)]" + text
        
        // Simple whitespace tokenization (placeholder)
        let tokenIds = prompt.split(separator: " ").enumerated().map { Int32($0.offset + 1) }
        let ids = MLXArray(tokenIds)
        
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

    /// Tokenize audio into tokens (following Marvis TTS pattern)
    /// - Parameters:
    ///   - audio: Audio array
    ///   - addEOS: Whether to add end-of-sequence token
    /// - Returns: Tuple of (tokens, mask) arrays with shape (seq_len, 33)
    private func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        guard let mimiTokenizer = mimiTokenizer else {
            fatalError("Audio tokenizer not initialized")
        }

        let K = config.audioNumCodebooks
        let frameW = K + 1
        
        let x = audio.reshaped([1, 1, audio.shape[0]])
        var codes = mimiTokenizer.codec.encode(x) // [1, K, Tq]
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

    /// Tokenize a complete segment (text + audio) (following Marvis TTS pattern)
    /// - Parameters:
    ///   - segment: Segment to tokenize
    ///   - addEOS: Whether to add end-of-sequence token
    /// - Returns: Tuple of (tokens, mask) arrays with shape (total_seq_len, 33)
    private func tokenizeSegment(_ segment: SesameSegment, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        let (txt, txtMask) = tokenizeTextSegment(segment.text, speaker: segment.speaker)
        let (aud, audMask) = tokenizeAudio(segment.audio, addEOS: addEOS)
        return (concatenated([txt, aud], axis: 0), concatenated([txtMask, audMask], axis: 0))
    }

    /// Generate audio from text with optional voice prompt
    /// - Parameters:
    ///   - text: Text to generate audio for
    ///   - voice: Voice/prompt to use (optional)
    ///   - speaker: Speaker ID
    ///   - context: Context segments
    ///   - maxAudioLengthMs: Maximum audio length in milliseconds
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling parameter
    ///   - stream: Whether to use streaming decoder for real-time generation
    ///   - voiceMatch: Whether to use voice matching (append text to voice prompt)
    /// - Returns: GenerationResult with audio and metadata
    func generate(
        text: String,
        voice: String? = nil,
        speaker: Int = 0,
        context: [SesameSegment] = [],
        maxAudioLengthMs: Float = 90000,
        temperature: Float = 0.9,
        topK: Int = 50,
        stream: Bool = false,
        voiceMatch: Bool = true
    ) throws -> GenerationResult {
        ensureInitialized()

        guard let model = model, let mimi = audioTokenizer else {
            throw SesameTTSError.modelNotInitialized
        }

        let K = config.audioNumCodebooks
        let maxAudioFrames = Int(maxAudioLengthMs / 80.0)
        let startTime = Date().timeIntervalSince1970

        // Build context: if no context and voice provided, load prompt from bundle
        var activeContext: [SesameSegment] = context
        if activeContext.isEmpty {
            if let voiceName = voice,
               let wavPath = Bundle.main.path(forResource: voiceName, ofType: "wav", inDirectory: "Sesame/TTSEngine/prompts"),
               let txtPath = Bundle.main.path(forResource: voiceName, ofType: "txt", inDirectory: "Sesame/TTSEngine/prompts") {
                do {
                    let txt = try String(contentsOfFile: txtPath)
                    let audio = try loadWavMono24k(url: URL(fileURLWithPath: wavPath))
                    activeContext = [SesameSegment(speaker: speaker, text: txt, audio: audio)]
                } catch {
                    print("⚠️  WARNING loading prompt: \(error)")
                }
            }
        }

        // Voice match: prepend prompt text to the new text
        var currentContext = activeContext
        if voiceMatch, let first = activeContext.first {
            let genText = (first.text + " " + text).trimmingCharacters(in: .whitespaces)
            currentContext = [SesameSegment(speaker: speaker, text: genText, audio: first.audio)]
        }

        // Tokenize
        var toks: [MLXArray] = []
        var masks: [MLXArray] = []
        for seg in currentContext {
            let (st, sm) = tokenizeSegment(seg, addEOS: false)
            toks.append(st); masks.append(sm)
        }

        let promptTokens = MLX.concatenated(toks, axis: 0).asType(Int32.self) // [T, K+1]
        let promptMask = MLX.concatenated(masks, axis: 0).asType(Bool.self)   // [T, K+1]

        // Prepare generation state
        var frames: [MLXArray] = [] // each [1, K]
        var currTokens = promptTokens.expandedDimensions(axis: 0) // [1, T, K+1]
        var currMask = promptMask.expandedDimensions(axis: 0)     // [1, T, K+1]
        var currPos = MLXArray.arange(start: 0, stop: promptTokens.shape[0], dtype: .int32).reshaped([1, promptTokens.shape[0]])

        let maxSeqLen = 2048 - maxAudioFrames
        if currTokens.shape[1] >= maxSeqLen {
            throw SesameTTSError.generationFailed(reason: "Inputs too long; max seq len exceeded")
        }

        for _ in 0..<maxAudioFrames {
            let frame = model.generateFrame(tokens: currTokens, tokensMask: currMask, inputPos: currPos, sampler: { logits in
                return self.sampleTopK(logits: logits, temperature: temperature, topK: topK)
            }) // [1, K]

            // EOS if all zeros
            if frame.sum().item(Int32.self) == 0 { break }
            frames.append(frame)

            // Prepare next step input: [K audio tokens, 1 text placeholder]
            let zerosText = MLXArray.zeros([1, 1], type: Int32.self)
            let nextFrame = MLX.concatenated([frame, zerosText], axis: 1) // [1, K+1]
            currTokens = nextFrame.expandedDimensions(axis: 1) // [1, 1, K+1]

            let onesK = MLXArray.ones([1, K], type: Bool.self)
            let zero1 = MLXArray.zeros([1, 1], type: Bool.self)
            currMask = MLX.concatenated([onesK, zero1], axis: 1).expandedDimensions(axis: 1)

            // Keep last position and increment by 1
            let tail = split(currPos, indices: [currPos.shape[1] - 1], axis: 1)[1]
            currPos = tail + MLXArray(1)
        }

        // Decode collected frames
        var stackedF = MLX.stacked(frames, axis: 0) // [F, 1, K]
        stackedF = stackedF.swappedAxes(0, 1)       // [1, F, K]
        stackedF = stackedF.swappedAxes(1, 2)       // [1, K, F]

        let audio1x1x = stream ? (self.streamingDecoder?.decodeFrames(stackedF) ?? mimi.decode(stackedF))
                               : mimi.decode(stackedF)
        let samples = audio1x1x.shape.last!
        let audio = audio1x1x.reshaped([samples])

        let elapsed = Date().timeIntervalSince1970 - startTime
        let sr = sampleRate
        let audioSeconds = Double(samples) / Double(sr)
        let rtf = audioSeconds > 0 ? Float(elapsed / audioSeconds) : 0

        return GenerationResult(
            audio: audio,
            samples: samples,
            sampleRate: sr,
            segmentIdx: 0,
            tokenCount: frames.count,
            audioDuration: formatDuration(audioSeconds),
            realTimeFactor: rtf,
            prompt: ["tokens": frames.count],
            audioSamples: ["samples": samples],
            processingTimeSeconds: elapsed,
            peakMemoryUsage: 0.0
        )
    }

    /// Get default speaker prompt for a voice
    /// - Parameter voice: Voice name
    /// - Returns: Array of segments for the voice prompt
    private func defaultSpeakerPrompt(voice: String) -> [SesameSegment] {
        // Return simple segments for now
        return [
            SesameSegment(speaker: 0, text: "Hello, I'm ready to help.", audio: MLXArray.zeros([1000])),
            SesameSegment(speaker: 0, text: "What would you like to discuss?", audio: MLXArray.zeros([1000]))
        ]
    }

    /// Sample from logits using top-k sampling
    /// - Parameters:
    ///   - logits: Logits array
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k parameter
    /// - Returns: Sampled token indices
    private func sampleTopK(logits: MLXArray, temperature: Float, topK: Int) -> MLXArray {
        let scaledLogits = logits / temperature

        // Get top-k using argSort (following Orpheus implementation)
        let sortedIndices = MLX.argSort(MLX.negative(scaledLogits), axis: -1)
        let topKIndices = sortedIndices[0..., 0..<topK]

        // Get corresponding values
        let topKValues = MLX.take(scaledLogits, topKIndices, axis: -1)

        // Sample from top-k
        let probs = MLX.softmax(topKValues, axis: -1)
        let sampleIdx = MLXRandom.categorical(MLX.log(probs), count: 1)[0]

        // Get the sampled token index from top-k
        let sampledTokenIndex = topKIndices[0..., sampleIdx]

        // Return as [batch_size] - ensure it's 1D for expansion in generateFrame
        return sampledTokenIndex.reshaped([logits.shape[0]])
    }

    // MARK: - Weight loading for backbone/decoder
    private func loadBackboneDecoderWeights() {
        guard let model = self.model else { return }
        guard let path = Bundle.main.path(forResource: "model", ofType: "safetensors", inDirectory: "Sesame/TTSEngine") else {
            print("⚠️  WARNING: model.safetensors not found; backbone/decoder will be uninitialized")
            return
        }
        do {
            let weights = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            let mapped = mapSesameModelWeights(weights)
            let params = ModuleParameters.unflattened(mapped)
            try model.update(parameters: params, verify: [.all])
            print("✅ Loaded backbone/decoder weights (\(mapped.count) tensors)")
        } catch {
            print("⚠️  WARNING: Failed to load model weights: \(error)")
        }
    }

    private func mapSesameModelWeights(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(raw.count)

        for (k0, v) in raw {
            var k = k0
            if !k.hasPrefix("model.") { k = "model." + k }

            // Attn and MLP renames (Python sanitize)
            if k.contains("attn") && !k.contains("self_attn") {
                k = k.replacingOccurrences(of: "attn", with: "self_attn")
                k = k.replacingOccurrences(of: "output_proj", with: "o_proj")
            }
            if k.contains("mlp") {
                k = k.replacingOccurrences(of: ".w1.", with: ".gate_proj.")
                k = k.replacingOccurrences(of: ".w2.", with: ".down_proj.")
                k = k.replacingOccurrences(of: ".w3.", with: ".up_proj.")
            }
            if k.contains("sa_norm") || k.contains("mlp_norm") {
                k = k.replacingOccurrences(of: "sa_norm", with: "input_layernorm")
                k = k.replacingOccurrences(of: "mlp_norm", with: "post_attention_layernorm")
                k = k.replacingOccurrences(of: "scale", with: "weight")
            }
            if k.contains("decoder.norm") || k.contains("backbone.norm") {
                k = k.replacingOccurrences(of: "scale", with: "weight")
            }

            // Map to Swift property names
            // Drop leading "model."
            k.removeFirst("model.".count)

            // Embeddings and heads
            if k == "text_embeddings.weight" { out["textEmbeddings.embeddings"] = v; continue }
            if k == "audio_embeddings.weight" { out["audioEmbeddings.embeddings"] = v; continue }
            if k == "projection.weight" { out["projection.weight"] = v; continue }
            if k == "codebook0_head.weight" { out["codebook0Head.weight"] = v; continue }
            if k.hasPrefix("audio_head") { out["audioHead"] = v; continue }

            // Layers mapping
            // backbone.layers.N.self_attn.q_proj.weight -> backbone.layers.N.selfAttention.qProj.weight
            func mapLayerPath(_ path: String, root: String) -> String? {
                guard path.hasPrefix(root + ".layers.") else { return nil }
                var tail = String(path.dropFirst((root + ".layers.").count))
                // tail: "<i>.<rest>"
                guard let dot = tail.firstIndex(of: ".") else { return nil }
                let idx = String(tail[..<dot])
                tail = String(tail[tail.index(after: dot)...])
                // self_attn.*
                if tail.hasPrefix("self_attn.") {
                    var t = String(tail.dropFirst("self_attn.".count))
                    t = t.replacingOccurrences(of: "q_proj", with: "qProj")
                    t = t.replacingOccurrences(of: "k_proj", with: "kProj")
                    t = t.replacingOccurrences(of: "v_proj", with: "vProj")
                    t = t.replacingOccurrences(of: "o_proj", with: "oProj")
                    return "\(root).layers.\(idx).selfAttention.\(t)"
                }
                if tail.hasPrefix("mlp.") {
                    var t = String(tail.dropFirst("mlp.".count))
                    t = t.replacingOccurrences(of: "gate_proj", with: "gateProj")
                    t = t.replacingOccurrences(of: "up_proj", with: "upProj")
                    t = t.replacingOccurrences(of: "down_proj", with: "downProj")
                    return "\(root).layers.\(idx).mlp.\(t)"
                }
                if tail.hasPrefix("input_layernorm.") {
                    let t = String(tail.dropFirst("input_layernorm.".count))
                    return "\(root).layers.\(idx).inputNorm.\(t)"
                }
                if tail.hasPrefix("post_attention_layernorm.") {
                    let t = String(tail.dropFirst("post_attention_layernorm.".count))
                    return "\(root).layers.\(idx).postNorm.\(t)"
                }
                return nil
            }

            if let mapped = mapLayerPath(k, root: "backbone") { out[mapped] = v; continue }
            if let mapped = mapLayerPath(k, root: "decoder") { out[mapped] = v; continue }

            // Fallback: pass through keys that already align
            out[k] = v
        }
        return out
    }

    // MARK: - WAV loader (mono 24k)
    private func loadWavMono24k(url: URL) throws -> MLXArray {
        let file = try AVAudioFile(forReading: url)
        let inFmt = file.processingFormat
        let total = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: total) else {
            throw NSError(domain: "SesameTTS", code: -1, userInfo: [NSLocalizedDescriptionKey: "Buffer alloc failed"])
        }
        try file.read(into: buffer)

        var mono: [Float] = []
        if inFmt.commonFormat == .pcmFormatFloat32, let chans = buffer.floatChannelData {
            let n = Int(buffer.frameLength)
            if inFmt.channelCount == 1 {
                mono = Array(UnsafeBufferPointer(start: chans[0], count: n))
            } else {
                // take first channel as mono
                mono = Array(UnsafeBufferPointer(start: chans[0], count: n))
            }
        } else {
            // Fallback: convert using AVAudioConverter
            let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inFmt.sampleRate, channels: 1, interleaved: false)!
            let conv = AVAudioConverter(from: inFmt, to: outFmt)!
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: total) else {
                throw NSError(domain: "SesameTTS", code: -2, userInfo: [NSLocalizedDescriptionKey: "Conv buffer alloc failed"])
            }
            var consumed = false
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                if consumed { outStatus.pointee = .endOfStream; return nil }
                consumed = true; outStatus.pointee = .haveData; return buffer
            }
            try conv.convert(to: outBuf, error: nil, withInputFrom: inputBlock)
            let n = Int(outBuf.frameLength)
            mono = Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: n))
        }

        // Resample if needed
        let sr = inFmt.sampleRate
        if abs(sr - 24000) > .ulpOfOne {
            // naive linear resample to 24k
            let oldLen = mono.count
            let newLen = Int(Double(oldLen) * 24000.0 / sr)
            var out = [Float](repeating: 0, count: newLen)
            for i in 0..<newLen {
                let pos = Double(i) * Double(oldLen - 1) / Double(newLen - 1)
                let i0 = Int(floor(pos))
                let i1 = min(i0 + 1, oldLen - 1)
                let w = Float(pos - Double(i0))
                out[i] = mono[i0] * (1 - w) + mono[i1] * w
            }
            mono = out
        }
        return MLXArray(mono)
    }

    /// Format duration as HH:MM:SS.mmm
    /// - Parameter durationSeconds: Duration in seconds
    /// - Returns: Formatted duration string
    private func formatDuration(_ durationSeconds: Double) -> String {
        let hours = Int(durationSeconds / 3600)
        let minutes = Int((durationSeconds.truncatingRemainder(dividingBy: 3600)) / 60)
        let seconds = Int(durationSeconds.truncatingRemainder(dividingBy: 60))
        let milliseconds = Int((durationSeconds.truncatingRemainder(dividingBy: 1)) * 1000)

        return String(format: "%02d:%02d:%02d.%03d", hours, minutes, seconds, milliseconds)
    }

    // MARK: - Voice Management (placeholder methods)
    
    public func getAvailableVoices() -> [String] {
        return ["conversational_a", "conversational_b"]
    }
    
    public func validateVoice(voiceName: String) -> Bool {
        return getAvailableVoices().contains(voiceName)
    }
    
    public func getVoiceDescription(voiceName: String) -> String {
        return "Voice: \(voiceName)"
    }
    
    public func addCustomVoice(config: VoiceConfig, prompts: [VoicePrompt] = []) {
        // TODO: Implement custom voice addition
    }
}

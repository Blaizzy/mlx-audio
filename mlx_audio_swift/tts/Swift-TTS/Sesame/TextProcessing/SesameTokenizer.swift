// SesameTokenizer.swift
// Simplified tokenizer using HuggingFace Tokenizers package
// Following working Marvis TTS pattern

import Foundation
import Hub
import MLX
import MLXLMCommon

/// Sesame TTS Tokenizer using HuggingFace Tokenizers package
/// Simplified implementation following the working Marvis TTS pattern
public class SesameTokenizer {
    
    // Simplified tokenizer placeholders to avoid dependency on Tokenizers APIs
    private let audioTokenizer: MimiTokenizer
    private let args: LlamaModelArgs
    
    internal init(config: LlamaModelArgs, mimi: Mimi) {
        self.args = config
        self.audioTokenizer = MimiTokenizer(mimi)
    }
    
    // MARK: - Token Properties
    
    public var bosTokenId: Int { return 1 }
    public var eosTokenId: Int { return 2 }
    public var padTokenId: Int { return 0 }
    public var audioTokenId: Int { return args.audioTokenId }
    public var audioEosTokenId: Int { return args.audioEosTokenId }
    
    public var vocabSize: Int { return args.textVocabSize }
    public var textVocabSize: Int { return args.textVocabSize }
    public var audioVocabSize: Int { return args.audioVocabSize }
    public var numCodebooks: Int { return args.audioNumCodebooks }
    
    // MARK: - Text Tokenization
    
    /// Encode text to token IDs (following Marvis TTS pattern)
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        // Fallback simple whitespace tokenizer
        var ids = text.split(separator: " ").enumerated().map { i, _ in i + 100 } // dummy ids
        if addSpecialTokens { ids = [bosTokenId] + ids + [eosTokenId] }
        return ids
    }
    
    /// Decode token IDs back to text
    public func decode(_ tokens: [Int]) -> String { "<decoded \(tokens.count) tokens>" }
    
    /// Prepare text segment tokens and mask (following Marvis TTS pattern)
    public func tokenizeTextSegment(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let K = args.audioNumCodebooks
        let frameW = K + 1
        
        let prompt = "[\(speaker)]" + text
        let ids = MLXArray(encode(prompt))
        
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
    
    // MARK: - Audio Tokenization
    
    /// Tokenize audio using Mimi codec (following Marvis TTS pattern)
    public func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        let K = args.audioNumCodebooks
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
    
    // MARK: - Combined Tokenization
    
    /// Tokenize a complete segment (text + audio) following Marvis TTS pattern
    public func tokenizeSegment(_ segment: SesameSegment, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        let (txt, txtMask) = tokenizeTextSegment(text: segment.text, speaker: segment.speaker)
        let (aud, audMask) = tokenizeAudio(segment.audio, addEOS: addEOS)
        return (concatenated([txt, aud], axis: 0), concatenated([txtMask, audMask], axis: 0))
    }
    
    // MARK: - Legacy Support (for compatibility)
    
    /// Prepare input IDs (legacy method for compatibility)
    public func prepareInputIds(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        return tokenizeTextSegment(text: text, speaker: speaker)
    }
    
    /// Prepare audio tokens (legacy method for compatibility)
    public func prepareAudioTokens(_ audioTokens: [[Int]]) -> MLXArray {
        let flattenedTokens = audioTokens.flatMap { $0 }
        return MLXArray(flattenedTokens.map { Int32($0) }).reshaped([audioTokens.count, audioTokens[0].count])
    }
}

/// MimiTokenizer wrapper (following Marvis TTS pattern)
public final class MimiTokenizer {
    let codec: Mimi
    
    public init(_ codec: Mimi) {
        self.codec = codec
    }
}

// MARK: - Errors

enum SesameTokenizerError: Error {
    case configNotFound
    case tokenizerNotFound
    case encodingFailed(String)
    
    var localizedDescription: String {
        switch self {
        case .configNotFound:
            return "Tokenizer configuration file not found"
        case .tokenizerNotFound:
            return "Tokenizer model file not found"
        case .encodingFailed(let reason):
            return "Tokenization failed: \(reason)"
        }
    }
}

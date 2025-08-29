// SesameTokenizer.swift
// Llama-3 tokenizer implementation for Sesame TTS
// Based on OrpheusTokenizer pattern

import Foundation
import MLX

/// Llama-3 tokenizer for Sesame TTS
/// Equivalent to Python's Llama-3 tokenizer
public class SesameTokenizer {
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let continuingSubwordPrefix: String?
    private let endOfWordSuffix: String?
    private let unkToken: String?
    private let bosToken: String?
    private let eosToken: String?
    private let padToken: String?

    // Special tokens for Sesame
    private let audioTokens: [String] = [
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>"
    ]

    // Add vocabSize property
    public var vocabSize: Int {
        return vocab.count
    }

    // BOS and EOS token IDs (following Llama-3 convention)
    public var bosTokenId: Int {
        return vocab["<|begin_of_text|>"] ?? 128000
    }

    public var eosTokenId: Int {
        return vocab["<|end_of_text|>"] ?? 128001
    }

    public var padTokenId: Int {
        return vocab["<|pad|>"] ?? 128003
    }

    // Hashable struct for BPE pairs
    private struct StringPair: Hashable {
        let first: String
        let second: String
    }

    // Build a merge rank dictionary for fast lookup (once)
    private lazy var mergeRanks: [StringPair: Int] = {
        var dict = [StringPair: Int]()
        for (i, pair) in merges.enumerated() {
            dict[StringPair(first: pair.0, second: pair.1)] = i
        }
        return dict
    }()

    private func getPairs(_ symbols: [String]) -> Set<StringPair> {
        var pairs = Set<StringPair>()
        for i in 0..<(symbols.count - 1) {
            pairs.insert(StringPair(first: symbols[i], second: symbols[i + 1]))
        }
        return pairs
    }

    /// Initialize tokenizer from JSON resources
    /// - Throws: TokenizerError if resources are not found
    public init() throws {
        // Load tokenizer configuration
        guard let configPath = Bundle.main.path(forResource: "tokenizer_config", ofType: "json"),
              let configData = try? Data(contentsOf: URL(fileURLWithPath: configPath)),
              let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw TokenizerError.configNotFound
        }

        // Extract BPE-specific config
        continuingSubwordPrefix = config["continuing_subword_prefix"] as? String
        endOfWordSuffix = config["end_of_word_suffix"] as? String
        unkToken = config["unk_token"] as? String
        bosToken = config["bos_token"] as? String
        eosToken = config["eos_token"] as? String
        padToken = config["pad_token"] as? String

        // Load vocabulary and merges from tokenizer.json
        guard let tokenizerPath = Bundle.main.path(forResource: "tokenizer", ofType: "json"),
              let tokenizerData = try? Data(contentsOf: URL(fileURLWithPath: tokenizerPath)),
              let tokenizerDict = try? JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any],
              let model = tokenizerDict["model"] as? [String: Any],
              let vocabDict = model["vocab"] as? [String: Int],
              let mergesArray = model["merges"] as? [[String]] else {
            throw TokenizerError.tokenizerNotFound
        }

        vocab = vocabDict

        // Convert merges to tuples
        merges = mergesArray.map { pair in
            (pair[0], pair[1])
        }
    }

    /// Encode text to token IDs
    /// - Parameters:
    ///   - text: Input text to tokenize
    ///   - addSpecialTokens: Whether to add BOS/EOS tokens
    /// - Returns: Array of token IDs
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        var tokens = bytePairEncode(text)

        if addSpecialTokens {
            tokens.insert(bosTokenId, at: 0)
            tokens.append(eosTokenId)
        }

        return tokens
    }

    /// Decode token IDs back to text
    /// - Parameter tokens: Array of token IDs
    /// - Returns: Decoded text
    public func decode(_ tokens: [Int]) -> String {
        let tokenStrings = tokens.compactMap { tokenId -> String? in
            // Find the token string for this ID
            for (token, id) in vocab where id == tokenId {
                return token
            }
            return nil
        }

        return decodeBPE(tokenStrings)
    }

    /// Prepare input IDs for Sesame model
    /// - Parameters:
    ///   - text: Text to tokenize
    ///   - speaker: Speaker ID
    /// - Returns: Tuple of (tokens, mask) arrays
    public func prepareInputIds(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let tokens = encode(text, addSpecialTokens: true)

        // Convert to MLXArray
        let tokenIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, -1])

        // Create attention mask (all true, indicating all tokens are attended)
        let attentionMask = MLXArray.ones(tokenIds.shape, dtype: .bool)

        return (tokenIds, attentionMask)
    }

    // MARK: - Private Methods

    private func bytePairEncode(_ text: String) -> [Int] {
        // Basic whitespace cleaning and pre-tokenization
        let preTokenized = preTokenize(text)

        var tokens = [Int]()

        for token in preTokenized {
            // Try to find the token in vocab first
            if let tokenId = vocab[token] {
                tokens.append(tokenId)
            } else {
                // Apply BPE
                tokens.append(contentsOf: applyBPE(token))
            }
        }

        return tokens
    }

    private func preTokenize(_ text: String) -> [String] {
        // Basic pre-tokenization: split on whitespace and punctuation
        let pattern = #"(\w+|[^\w\s]+)"#
        let regex = try! NSRegularExpression(pattern: pattern, options: [])
        let matches = regex.matches(in: text, options: [], range: NSRange(text.startIndex..., in: text))

        return matches.map { match in
            String(text[Range(match.range, in: text)!])
        }
    }

    private func applyBPE(_ token: String) -> [Int] {
        var symbols = Array(token).map { String($0) }
        var pairs = getPairs(symbols)

        while !pairs.isEmpty {
            // Find the highest priority pair to merge
            var bestPair: StringPair?
            var bestRank = Int.max

            for pair in pairs {
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }

            guard let pair = bestPair else { break }

            // Merge the pair
            symbols = mergePair(symbols, pair: (pair.first, pair.second))
            pairs = getPairs(symbols)
        }

        // Convert symbols back to token IDs
        return symbols.compactMap { vocab[$0] }
    }

    private func mergePair(_ symbols: [String], pair: (String, String)) -> [String] {
        var result = [String]()
        var i = 0

        while i < symbols.count {
            if i < symbols.count - 1 && symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                result.append(pair.0 + pair.1)
                i += 2
            } else {
                result.append(symbols[i])
                i += 1
            }
        }

        return result
    }

    private func decodeBPE(_ tokens: [String]) -> String {
        var result = ""

        for (i, token) in tokens.enumerated() {
            if i > 0 && !token.hasPrefix("Ġ") && !token.hasPrefix("Ċ") && !token.hasPrefix("Ċ") {
                result += " "
            }

            result += token.replacingOccurrences(of: "Ġ", with: " ")
                                 .replacingOccurrences(of: "Ċ", with: "\n")
                                 .replacingOccurrences(of: "Ċ", with: "\n")
        }

        return result
    }
}

/// Tokenizer errors
enum TokenizerError: Error {
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

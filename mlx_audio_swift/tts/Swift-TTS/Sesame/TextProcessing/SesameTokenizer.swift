// SesameTokenizer.swift
// Llama-3 tokenizer implementation for Sesame TTS
// Based on OrpheusTokenizer pattern

import Foundation
import MLX

/// Llama-3 tokenizer for Sesame TTS
/// Equivalent to Python's Llama-3 tokenizer
public class SesameTokenizer {

    // Unicode handling utilities
    private let unicodeNormalizer: UnicodeNormalizer
    private var vocab: [String: Int] = [:]
    private var merges: [(String, String)] = []
    private var continuingSubwordPrefix: String? = nil
    private var endOfWordSuffix: String? = nil
    private var unkToken: String? = nil
    private var bosToken: String? = nil
    private var eosToken: String? = nil
    private var padToken: String? = nil

    // Special tokens for Sesame
    private let audioTokens: [String] = [
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>"
    ]

    // Special token processor for comprehensive handling
    private let specialTokenProcessor: SpecialTokenProcessor

    // Add vocabSize property
    public var vocabSize: Int {
        return vocab.count
    }

    // Cache the loaded config for property access
    private var cachedConfig: [String: Any]?

    // BOS and EOS token IDs (from actual tokenizer vocab)
    public var bosTokenId: Int {
        return vocab["<|im_start|>"] ?? 1
    }

    public var eosTokenId: Int {
        return vocab["<|endoftext|>"] ?? 0
    }

    public var padTokenId: Int {
        return vocab["<|im_end|>"] ?? 2
    }

    // Audio-specific token IDs (these may need to be added to vocab if not present)
    public var audioTokenId: Int {
        return vocab["<|audio|>"] ?? 128002
    }

    public var audioEosTokenId: Int {
        return vocab["<|audio_end|>"] ?? 128003
    }

    // Sesame-specific properties
    public var textVocabSize: Int {
        return vocab.count
    }

    public var audioVocabSize: Int {
        // Use cached config if available, fallback to default
        if let cachedConfig = self.cachedConfig,
           let audioVocabSize = cachedConfig["audio_vocab_size"] as? Int {
            return audioVocabSize
        }
        return 2051 // Default from config
    }

    public var numCodebooks: Int {
        // Use cached config if available, fallback to default
        if let cachedConfig = self.cachedConfig,
           let numCodebooks = cachedConfig["audio_num_codebooks"] as? Int {
            return numCodebooks
        }
        return 32 // Default from config
    }





    /// Initialize tokenizer from JSON resources
    /// - Throws: TokenizerError if resources are not found
    public init() throws {
        // Initialize Unicode normalizer and special token processor
        self.unicodeNormalizer = UnicodeNormalizer()
        self.specialTokenProcessor = SpecialTokenProcessor()
        // Try to load tokenizer configuration from multiple possible locations
        var configData: Data?
        var config: [String: Any]?

        // Load tokenizer configuration (like Orpheus)
        guard let configPath = Bundle.main.path(forResource: "sesame_config", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: configPath)) else {
            throw TokenizerError.configNotFound
        }
        configData = data
        config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]

        // Third try: Embedded fallback configuration
        if config == nil {
            // Use embedded default configuration
            let defaultConfig: [String: Any] = [
                "bos_token_id": 1,
                "eos_token_id": 0,
                "pad_token_id": 2,
                "audio_token_id": 128002,
                "audio_eos_token_id": 128003,
                "audio_vocab_size": 2051,
                "text_vocab_size": 49152,
                "audio_num_codebooks": 32
            ]
            config = defaultConfig
        }

        guard let finalConfig = config else {
            throw TokenizerError.configNotFound
        }

        // Cache the config for property access
        self.cachedConfig = finalConfig

        // Extract token IDs from config (Sesame-specific format)
        let bosTokenIdFromConfig = finalConfig["bos_token_id"] as? Int ?? 1
        let eosTokenIdFromConfig = finalConfig["eos_token_id"] as? Int ?? 0
        let padTokenIdFromConfig = finalConfig["pad_token_id"] as? Int ?? 2
        let audioTokenId = finalConfig["audio_token_id"] as? Int ?? 128002
        let audioEosTokenId = finalConfig["audio_eos_token_id"] as? Int ?? 128003

        // Set token strings based on IDs (for compatibility)
        bosToken = "<|im_start|>"
        eosToken = "<|endoftext|>"
        padToken = "<|im_end|>"
        unkToken = "<|unk|>"

        // BPE configuration (may not be present in this config)
        continuingSubwordPrefix = finalConfig["continuing_subword_prefix"] as? String
        endOfWordSuffix = finalConfig["end_of_word_suffix"] as? String

        // Try to load vocabulary and merges from sesame_tokenizer.json
        // If loading fails, fall back to minimal vocabulary (like Python implementation)
        if let tokenizerPath = Bundle.main.path(forResource: "sesame_tokenizer", ofType: "json"),
           let tokenizerData = try? Data(contentsOf: URL(fileURLWithPath: tokenizerPath)),
           let tokenizerDict = try? JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any],
           let model = tokenizerDict["model"] as? [String: Any],
           let vocabDict = model["vocab"] as? [String: Int],
           let mergesArray = model["merges"] as? [[String]] {

            // Try to validate, but don't fail if validation fails
            do {
                try validateVocabulary(vocabDict)
                let validMerges = validateAndFilterMerges(mergesArray)
                vocab = vocabDict
                merges = validMerges.map { ($0[0], $0[1]) }
            } catch {
                print("Warning: Tokenizer validation failed, using fallback: \(error)")
                // Fall back to minimal vocabulary
                vocab = SesameTokenizer.createMinimalVocab(
                    bosTokenId: bosTokenId,
                    eosTokenId: eosTokenId,
                    padTokenId: padTokenId,
                    audioTokenId: audioTokenId,
                    audioEosTokenId: audioEosTokenId
                )
                merges = []
            }
        } else {
            // Fallback to minimal vocabulary if file not found or parsing fails
            vocab = SesameTokenizer.createMinimalVocab(
                bosTokenId: bosTokenId,
                eosTokenId: eosTokenId,
                padTokenId: padTokenId,
                audioTokenId: audioTokenId,
                audioEosTokenId: audioEosTokenId
            )
            merges = []
        }

        // Ensure byte fallback tokens are in vocabulary
        // ensureByteFallbackTokens()  // Called when needed during tokenization


    }

    /// Validate vocabulary integrity
    private func validateVocabulary(_ vocabDict: [String: Int]) throws {
        // Check for essential special tokens
        let essentialTokens = ["<|im_start|>", "<|endoftext|>", "<|im_end|>"]
        for token in essentialTokens {
            guard vocabDict[token] != nil else {
                throw TokenizerError.encodingFailed("Missing essential special token: \(token)")
            }
        }

        // Check for duplicate token IDs
        let tokenIds = vocabDict.values
        let uniqueIds = Set(tokenIds)
        guard tokenIds.count == uniqueIds.count else {
            throw TokenizerError.encodingFailed("Duplicate token IDs found in vocabulary")
        }

        // Check for negative token IDs
        for (token, tokenId) in vocabDict {
            guard tokenId >= 0 else {
                throw TokenizerError.encodingFailed("Negative token ID found for token '\(token)': \(tokenId)")
            }
        }

        // Ensure vocabulary is not empty
        guard !vocabDict.isEmpty else {
            throw TokenizerError.encodingFailed("Vocabulary is empty")
        }
    }

    /// Validate and filter merges (returns only valid merges)
    private func validateAndFilterMerges(_ mergesArray: [[String]]) -> [[String]] {
        // Ensure merges array is not empty
        guard !mergesArray.isEmpty else {
            print("Warning: Merges array is empty, using minimal merges")
            return []
        }

        // Validate each merge pair and filter out invalid ones
        var validMerges: [[String]] = []

        for (index, mergePair) in mergesArray.enumerated() {
            guard mergePair.count == 2 else {
                print("Warning: Invalid merge pair at index \(index): expected 2 elements, got \(mergePair.count), skipping")
                continue
            }

            let (first, second) = (mergePair[0], mergePair[1])

            // Ensure merge components are not empty
            guard !first.isEmpty && !second.isEmpty else {
                print("Warning: Empty merge component at index \(index): '\(first)' + '\(second)', skipping")
                continue
            }

            // Ensure merge components are reasonable length
            // Allow longer sequences for repetitive characters and be more permissive overall
            let isRepetitiveSequence = { (component: String) -> Bool in
                // Whitespace characters
                if component.allSatisfy({ $0 == "Ċ" || $0 == "Ġ" || $0 == " " || $0 == "\n" || $0 == "\t" }) {
                    return true
                }
                // Dash/hyphen sequences (common separators)
                if component.allSatisfy({ $0 == "-" || $0 == "_" || $0 == "=" || $0 == "+" || $0 == "|" }) {
                    return true
                }
                // Asterisk sequences (markdown formatting)
                if component.allSatisfy({ $0 == "*" || $0 == "#" || $0 == "`" || $0 == "~" }) {
                    return true
                }
                // Any single character repeated (like "aaa", "111", etc.)
                if component.count > 1 && component.allSatisfy({ $0 == component.first }) {
                    return true
                }
                return false
            }

            // Be more permissive: allow up to 150 chars for repetitive sequences, 50 for others
            let maxLength = (isRepetitiveSequence(first) || isRepetitiveSequence(second)) ? 150 : 50
            guard first.count <= maxLength && second.count <= maxLength else {
                print("Warning: Skipping invalid merge at index \(index): '\(first)' + '\(second)' (length: \(first.count) + \(second.count), max: \(maxLength))")
                continue // Skip invalid merges instead of failing
            }

            // Add valid merge to the list
            validMerges.append(mergePair)
        }

        // Report filtering results
        if validMerges.count != mergesArray.count {
            print("Warning: Filtered out \(mergesArray.count - validMerges.count) invalid merges, kept \(validMerges.count)")
        }

        return validMerges
    }

    /// Ensure byte fallback tokens are available in vocabulary
    private func ensureByteFallbackTokens() {
        // Generate byte fallback tokens for 0x00 to 0xFF
        var maxTokenId = vocab.values.max() ?? 128000

        for byte in 0x00...0xFF {
            let byteToken = String(format: "<0x%02X>", byte)
            if vocab[byteToken] == nil {
                maxTokenId += 1
                vocab[byteToken] = maxTokenId
            }
        }

        // Ensure unknown token is available
        if vocab["<|unk|>"] == nil {
            maxTokenId += 1
            vocab["<|unk|>"] = maxTokenId
        }
    }

    /// Create minimal vocabulary with known token IDs
    private static func createMinimalVocab(bosTokenId: Int, eosTokenId: Int, padTokenId: Int, audioTokenId: Int, audioEosTokenId: Int) -> [String: Int] {
        var vocab: [String: Int] = [:]

        // Add known special tokens from the actual tokenizer
        vocab["<|im_start|>"] = bosTokenId
        vocab["<|endoftext|>"] = eosTokenId
        vocab["<|im_end|>"] = padTokenId
        vocab["<|audio|>"] = audioTokenId
        vocab["<|audio_end|>"] = audioEosTokenId

        // Add fallback tokens in case the tokenizer doesn't have them
        vocab["<|unk|>"] = 128004

        // Add common tokens for basic functionality
        vocab[" "] = 128005
        vocab["\n"] = 128006

        return vocab
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
    ///   - speaker: Speaker ID (currently unused in Sesame)
    /// - Returns: Tuple of (tokens, mask) arrays
    public func prepareInputIds(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let tokens = encode(text, addSpecialTokens: true)

        // Convert to MLXArray
        let tokenIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, -1])

        // Create attention mask (all true, indicating all tokens are attended)
        let attentionMask = MLXArray.ones(tokenIds.shape, dtype: .bool)

        return (tokenIds, attentionMask)
    }

    /// Prepare audio tokens for Sesame model
    /// - Parameter audioTokens: Array of audio token IDs [batch, num_codebooks, seq_len]
    /// - Returns: MLXArray suitable for Sesame model input
    public func prepareAudioTokens(_ audioTokens: [[Int]]) -> MLXArray {
        // Convert to MLXArray and ensure proper shape
        let flattenedTokens = audioTokens.flatMap { $0 }
        return MLXArray(flattenedTokens.map { Int32($0) }).reshaped([audioTokens.count, audioTokens[0].count])
    }

    /// Convert audio token IDs back to token strings
    /// - Parameter tokenIds: Audio token IDs
    /// - Returns: Array of token strings
    public func decodeAudioTokens(_ tokenIds: [Int]) -> [String] {
        return tokenIds.compactMap { tokenId -> String? in
            // Find the token string for this audio token ID
            for (token, id) in vocab where id == tokenId {
                return token
            }
            return nil
        }
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
        // Llama-3 style pre-tokenization with proper Unicode and special token handling
        var tokens = [String]()

        // Apply Unicode normalization and preprocessing
        var remainingText = unicodeNormalizer.preprocessForTokenization(text)

        // First pass: handle special tokens
        let specialTokenTokens = specialTokenProcessor.processSpecialTokensInText(remainingText)
        tokens = specialTokenProcessor.applySpecialTokenRules(tokens: specialTokenTokens)

        // Second pass: handle remaining non-special tokens with regex patterns
        var finalTokens = [String]()
        for token in tokens {
            if specialTokenProcessor.isSpecialToken(token) {
                // Special tokens remain as-is
                finalTokens.append(token)
            } else {
                // Process non-special tokens with regex patterns
                finalTokens.append(contentsOf: tokenizeNonSpecialText(token))
            }
        }

        return tokens
    }

    private func tokenizeNonSpecialText(_ text: String) -> [String] {
        var tokens = [String]()
        var position = text.startIndex

        while position < text.endIndex {
            let remainingRange = position..<text.endIndex
            let remainingSubstring = String(text[remainingRange])

            // Try to match different tokenization patterns in order of priority
            if let (token, newPosition) = matchPattern(remainingSubstring, at: position) {
                tokens.append(token)
                position = newPosition
            } else {
                // Fallback: take single character
                let char = String(text[position])
                tokens.append(char)
                position = text.index(after: position)
            }
        }

        return tokens
    }

    private func matchPattern(_ text: String, at position: String.Index) -> (token: String, newPosition: String.Index)? {
        let patterns = [
            // Numbers (including decimals)
            #"^\d+(?:\.\d+)?"#,

            // Words (Unicode letters and marks)
            #"^\p{L}+(?:\p{M}+)?"#,

            // Punctuation sequences
            #"^[^\p{L}\p{N}\p{M}\s]+"#,

            // Whitespace (preserve as separate tokens)
            #"^\s+"#,

            // Fallback for any single character
            #"^."#
        ]

        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: []),
               let match = regex.firstMatch(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count)),
               match.range.location == 0 {

                let matchedText = String(text[Range(match.range, in: text)!])
                let advance = matchedText.count
                let newPosition = text.index(position, offsetBy: advance)

                return (matchedText, newPosition)
            }
        }

        return nil
    }

    private func applyBPE(_ token: String) -> [Int] {
        // Skip BPE for empty tokens
        guard !token.isEmpty else { return [] }

        // Convert to character array for processing
        var symbols = Array(token).map { String($0) }

        // Skip BPE for single characters
        guard symbols.count > 1 else {
            return symbols.compactMap { vocab[$0] }
        }

        // Apply merges in exact order from the tokenizer configuration
        for mergePair in merges {
            let (first, second) = (mergePair.0, mergePair.1)

            // Continue merging this specific pair until no more occurrences
            var merged = false
            repeat {
                merged = false
                var newSymbols = [String]()
                var i = 0

                while i < symbols.count {
                    if i < symbols.count - 1 && symbols[i] == first && symbols[i + 1] == second {
                        // Merge this pair
                        newSymbols.append(first + second)
                        i += 2
                        merged = true
                    } else {
                        newSymbols.append(symbols[i])
                        i += 1
                    }
                }

                symbols = newSymbols
            } while merged
        }

        // Convert final symbols back to token IDs
        var tokenIds = [Int]()
        for symbol in symbols {
            if let tokenId = vocab[symbol] {
                tokenIds.append(tokenId)
            } else {
                // Handle unknown symbols with byte fallback
                tokenIds.append(contentsOf: handleUnknownSymbol(symbol))
            }
        }

        return tokenIds
    }

    private func handleUnknownSymbol(_ symbol: String) -> [Int] {
        // Use byte fallback for unknown symbols
        let byteTokens = unicodeNormalizer.byteFallback(symbol)

        // Convert byte tokens to token IDs, fallback to unk token if available
        var tokenIds = [Int]()
        for byteToken in byteTokens {
            if let tokenId = vocab[byteToken] {
                tokenIds.append(tokenId)
            } else if let unkId = vocab["<|unk|>"] {
                tokenIds.append(unkId)
            }
        }

        return tokenIds
    }



    private func decodeBPE(_ tokens: [String]) -> String {
        var result = ""

        for (i, token) in tokens.enumerated() {
            // Handle spacing based on token type
            if i > 0 {
                // Don't add space before special tokens
                if !token.hasPrefix("<|") && !token.hasSuffix("|>") {
                    // Check for different spacing conventions
                    if token.hasPrefix("Ġ") || token.hasPrefix("▁") {
                        // Llama/BERT style: Ġ or ▁ prefix indicates space
                        result += " "
                    } else if !token.hasPrefix("Ċ") && !token.hasPrefix("Ċ") {
                        // Add space for regular tokens (unless it's a newline)
                        result += " "
                    }
                }
            }

            // Clean up token
            let cleanedToken = token
                .replacingOccurrences(of: "Ġ", with: "")  // Llama style space prefix
                .replacingOccurrences(of: "▁", with: "")  // BERT style space prefix
                .replacingOccurrences(of: "Ċ", with: "\n") // Newline
                .replacingOccurrences(of: "Ċ", with: "\n") // Alternative newline

            result += cleanedToken
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

/// Special token processor for comprehensive handling of tokenizer special tokens
private class SpecialTokenProcessor {
    // All special tokens from the Llama-3 tokenizer configuration
    private let specialTokens: [String] = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<repo_name>",
        "<reponame>",
        "<file_sep>",
        "<filename>",
        "<gh_stars>",
        "<issue_start>",
        "<issue_comment>",
        "<issue_closed>",
        "<jupyter_start>",
        "<jupyter_text>",
        "<jupyter_code>",
        "<jupyter_output>",
        "<jupyter_script>",
        "<empty_output>",
        "<|audio|>",
        "<|audio_end|>",
        "<|audio_start|>",
        "<|audio_pad|>",
        "<|unk|>"
    ]

    // Set for fast lookup
    private let specialTokenSet: Set<String>

    init() {
        self.specialTokenSet = Set(specialTokens)
    }

    // Check if a token is a special token
    func isSpecialToken(_ token: String) -> Bool {
        return specialTokenSet.contains(token)
    }

    // Get all special tokens
    func getAllSpecialTokens() -> [String] {
        return specialTokens
    }

    // Process text to handle special tokens during pre-tokenization
    func processSpecialTokensInText(_ text: String) -> [String] {
        var tokens = [String]()
        var remainingText = text
        var position = remainingText.startIndex

        while position < remainingText.endIndex {
            var foundSpecialToken = false

            // Try to match special tokens first (longest match)
            for specialToken in specialTokens.sorted(by: { $0.count > $1.count }) {
                if remainingText[position...].hasPrefix(specialToken) {
                    tokens.append(specialToken)
                    position = remainingText.index(position, offsetBy: specialToken.count)
                    foundSpecialToken = true
                    break
                }
            }

            if !foundSpecialToken {
                // No special token found, take one character
                let char = String(remainingText[position])
                tokens.append(char)
                position = remainingText.index(after: position)
            }
        }

        return tokens
    }

    // Handle special token encoding/decoding rules
    func applySpecialTokenRules(tokens: [String]) -> [String] {
        var processedTokens = [String]()

        for token in tokens {
            if isSpecialToken(token) {
                // Special tokens should not be split or modified
                processedTokens.append(token)
            } else {
                // Regular tokens can be processed normally
                processedTokens.append(token)
            }
        }

        return processedTokens
    }

    // Get special token properties (for future enhancement)
    func getSpecialTokenProperties(_ token: String) -> (singleWord: Bool, lstrip: Bool, rstrip: Bool, normalized: Bool)? {
        // This could be extended to store properties from the tokenizer config
        if isSpecialToken(token) {
            return (singleWord: false, lstrip: false, rstrip: false, normalized: false)
        }
        return nil
    }
}

/// Unicode normalization utilities for proper multilingual handling
private class UnicodeNormalizer {
    // NFKC normalization (compatibility decomposition followed by canonical composition)
    func normalize(_ text: String) -> String {
        return text.precomposedStringWithCompatibilityMapping
    }

    // Handle special Unicode cases for tokenization
    func preprocessForTokenization(_ text: String) -> String {
        var processed = normalize(text)

        // Handle specific Unicode character replacements for consistency
        let replacements: [(String, String)] = [
            ("\u{00AD}", ""),  // Remove soft hyphens
            ("\u{200B}", ""),  // Remove zero-width spaces
            ("\u{200C}", ""),  // Remove zero-width non-joiners
            ("\u{200D}", ""),  // Remove zero-width joiners
            ("\u{200E}", ""),  // Remove left-to-right marks
            ("\u{200F}", ""),  // Remove right-to-left marks
            ("\u{2028}", "\n"), // Line separators to newlines
            ("\u{2029}", "\n"), // Paragraph separators to newlines
        ]

        for (from, to) in replacements {
            processed = processed.replacingOccurrences(of: from, with: to)
        }

        return processed
    }

    // Convert string to UTF-8 bytes for byte-level processing
    func utf8Bytes(_ text: String) -> [UInt8] {
        return Array(text.utf8)
    }

    // Convert UTF-8 bytes back to string
    func stringFromUTF8Bytes(_ bytes: [UInt8]) -> String? {
        return String(bytes: bytes, encoding: .utf8)
    }

    // Handle byte fallback for characters not in vocabulary
    func byteFallback(_ text: String) -> [String] {
        let bytes = utf8Bytes(text)
        return bytes.map { byte in
            let byteString = String(format: "<0x%02X>", byte)
            return byteString
        }
    }
}

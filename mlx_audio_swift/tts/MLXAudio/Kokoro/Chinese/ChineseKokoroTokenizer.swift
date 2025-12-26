//
//  ChineseKokoroTokenizer.swift
//  MLXAudio
//
//  Chinese tokenizer for Kokoro TTS that uses native Bopomofo-based G2P
//  instead of eSpeak-ng (which uses Cantonese for Chinese).
//

import Foundation

/// Chinese tokenizer for Kokoro TTS using native Bopomofo G2P.
/// This replaces eSpeak-ng for Chinese text processing.
public final class ChineseKokoroTokenizer {

    // MARK: - Properties

    private let g2p = ChineseG2P.shared

    /// Whether the tokenizer is ready to process text
    public var isReady: Bool {
        g2p.isReady
    }

    // MARK: - Initialization

    /// Initialize with dictionary data
    public func initialize(
        jiebaData: Data,
        pinyinSingleData: Data,
        pinyinPhrasesData: Data? = nil
    ) throws {
        try g2p.initialize(
            jiebaData: jiebaData,
            pinyinSingleData: pinyinSingleData,
            pinyinPhrasesData: pinyinPhrasesData
        )
    }

    /// Initialize with dictionary URLs
    public func initialize(
        jiebaURL: URL,
        pinyinSingleURL: URL,
        pinyinPhrasesURL: URL? = nil
    ) throws {
        try g2p.initialize(
            jiebaURL: jiebaURL,
            pinyinSingleURL: pinyinSingleURL,
            pinyinPhrasesURL: pinyinPhrasesURL
        )
    }

    // MARK: - Public API

    /// Convert Chinese text to phonemes for Kokoro TTS
    /// - Parameter text: Chinese text input
    /// - Returns: Phoneme string in Bopomofo format
    public func phonemize(_ text: String) throws -> String {
        return try g2p.convert(text)
    }

    /// Tokenize text into Kokoro-compatible token IDs
    /// - Parameter text: Chinese text input
    /// - Returns: Array of token IDs for the Kokoro model
    public func tokenize(_ text: String) throws -> [Int] {
        let phonemes = try phonemize(text)
        return tokenizeWithChineseVocab(phonemes)
    }

    /// Tokenize phoneme string using combined base + Chinese vocabulary
    /// - Parameter phonemes: Bopomofo phoneme string
    /// - Returns: Array of token IDs
    public func tokenizeWithChineseVocab(_ phonemes: String) -> [Int] {
        var result: [Int] = []

        for char in phonemes {
            let charStr = String(char)

            // First try Chinese vocabulary
            if let tokenId = ChineseKokoroTokenizer.chineseVocab[charStr] {
                result.append(tokenId)
            }
            // Then try base Kokoro vocabulary
            else if let tokenId = ChineseKokoroTokenizer.baseVocab[charStr] {
                result.append(tokenId)
            }
            // Skip unknown characters
        }

        return result
    }

    // MARK: - Token Structure (Compatible with KokoroTokenizer)

    public struct Token {
        public let text: String
        public var whitespace: String
        public var phonemes: String
        public let stress: Float?
        public let currency: String?
        public var prespace: Bool
        public let alias: String?
        public let isHead: Bool

        public init(
            text: String,
            whitespace: String = " ",
            phonemes: String = "",
            stress: Float? = nil,
            currency: String? = nil,
            prespace: Bool = true,
            alias: String? = nil,
            isHead: Bool = false
        ) {
            self.text = text
            self.whitespace = whitespace
            self.phonemes = phonemes
            self.stress = stress
            self.currency = currency
            self.prespace = prespace
            self.alias = alias
            self.isHead = isHead
        }
    }

    public struct PhonemizerResult {
        public let phonemes: String
        public let tokens: [Token]

        public init(phonemes: String, tokens: [Token]) {
            self.phonemes = phonemes
            self.tokens = tokens
        }
    }

    /// Full phonemization with token information
    /// - Parameter text: Chinese text input
    /// - Returns: PhonemizerResult with phonemes and token details
    public func phonemizeWithTokens(_ text: String) throws -> PhonemizerResult {
        let phonemes = try phonemize(text)

        // Create simple token representation
        let tokens = [Token(
            text: text,
            whitespace: "",
            phonemes: phonemes,
            prespace: false,
            isHead: true
        )]

        return PhonemizerResult(phonemes: phonemes, tokens: tokens)
    }
}

// MARK: - Chinese Phoneme Vocabulary

extension ChineseKokoroTokenizer {

    /// Base Kokoro vocabulary (from Tokenizer.swift)
    public static let baseVocab: [String: Int] = [
        ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11, "(": 12,
        ")": 13, """: 14, """: 15, " ": 16, "\u{0303}": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20,
        "ʨ": 21, "ᵝ": 22, "\u{AB67}": 23, "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35,
        "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47,
        "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56, "o": 57,
        "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65, "x": 66,
        "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77,
        "ç": 78, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90,
        "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113,
        "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125,
        "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135,
        "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148,
        "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172,
        "↘": 173, "ᵻ": 177,
        // Tone numbers
        "1": 240, "2": 241, "3": 242, "4": 243, "5": 244,
    ]

    /// Extended vocabulary for Chinese (Bopomofo + Chinese character finals)
    /// These are added to the base Kokoro vocabulary
    public static let chineseVocab: [String: Int] = [
        // Bopomofo initials (声母)
        "ㄅ": 180, "ㄆ": 181, "ㄇ": 182, "ㄈ": 183,
        "ㄉ": 184, "ㄊ": 185, "ㄋ": 186, "ㄌ": 187,
        "ㄍ": 188, "ㄎ": 189, "ㄏ": 190,
        "ㄐ": 191, "ㄑ": 192, "ㄒ": 193,
        "ㄓ": 194, "ㄔ": 195, "ㄕ": 196, "ㄖ": 197,
        "ㄗ": 198, "ㄘ": 199, "ㄙ": 200,

        // Bopomofo simple vowels (韵母)
        "ㄚ": 201, "ㄛ": 202, "ㄜ": 203, "ㄝ": 204,
        "ㄞ": 205, "ㄟ": 206, "ㄠ": 207, "ㄡ": 208,
        "ㄢ": 209, "ㄣ": 210, "ㄤ": 211, "ㄥ": 212,
        "ㄦ": 213,

        // Bopomofo medials
        "ㄧ": 214, "ㄨ": 215, "ㄩ": 216,

        // Special finals (used for zi/ci/si, zhi/chi/shi/ri)
        "ㄭ": 217,  // zi, ci, si
        "十": 218,  // zhi, chi, shi, ri

        // Chinese character compound finals (matching misaki)
        "压": 219, "言": 220, "阳": 221, "要": 222,
        "阴": 223, "应": 224, "用": 225, "又": 226,
        "穵": 227, "外": 228, "万": 229, "王": 230,
        "为": 231, "文": 232, "瓮": 233, "我": 234,
        "中": 235,
        "月": 236, "元": 237, "云": 238,

        // Erhua marker
        "R": 239,
    ]
}

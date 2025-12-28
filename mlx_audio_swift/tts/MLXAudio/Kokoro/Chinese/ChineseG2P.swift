//
//  ChineseG2P.swift
//  MLXAudio
//
//  Chinese Grapheme-to-Phoneme converter for Kokoro TTS.
//  Converts Chinese text to Bopomofo (注音符号) phoneme sequences.
//
//  Pipeline:
//  1. Text normalization (punctuation, numbers)
//  2. Word segmentation (jieba-style)
//  3. Pinyin conversion with POS tagging
//  4. Tone sandhi application
//  5. Bopomofo mapping
//

import Foundation
import Hub

/// Default HuggingFace repository for Chinese Kokoro TTS (MLX format)
/// MLX conversion of https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
public let chineseKokoroRepo = "FluidInference/kokoro-82m-v1.1-zh-mlx"

/// Chinese Grapheme-to-Phoneme converter for Kokoro TTS.
/// Converts Chinese text to Bopomofo phoneme sequences.
public final class ChineseG2P {

    // MARK: - Components

    private let tokenizer = ChineseTokenizer.shared
    private let pinyinConverter = PinyinConverter.shared
    private let toneSandhi = ToneSandhi()

    /// Unknown token placeholder
    private let unknownToken = ""

    /// Shared instance
    public static let shared = ChineseG2P()

    private var isInitialized = false

    private init() {}

    // MARK: - Initialization

    /// Load all required dictionaries from data
    public func initialize(
        jiebaData: Data,
        pinyinSingleData: Data,
        pinyinPhrasesData: Data? = nil
    ) throws {
        try tokenizer.loadDictionary(data: jiebaData, isCompressed: true)
        try pinyinConverter.loadSinglePinyin(data: pinyinSingleData, isCompressed: true)
        if let phrasesData = pinyinPhrasesData {
            try pinyinConverter.loadPhrasePinyin(data: phrasesData, isCompressed: true)
        }
        isInitialized = true
    }

    /// Load dictionaries from URLs
    public func initialize(
        jiebaURL: URL,
        pinyinSingleURL: URL,
        pinyinPhrasesURL: URL? = nil
    ) throws {
        try tokenizer.loadDictionary(from: jiebaURL)
        try pinyinConverter.loadSinglePinyin(from: pinyinSingleURL)
        if let phrasesURL = pinyinPhrasesURL {
            try pinyinConverter.loadPhrasePinyin(from: phrasesURL)
        }
        isInitialized = true
    }

    /// Download and load dictionaries from HuggingFace
    /// - Parameters:
    ///   - repoId: HuggingFace repository ID (default: FluidInference/kokoro-82m-v1.1-zh-mlx)
    ///   - progressHandler: Optional progress callback
    public func initializeFromHub(
        repoId: String = chineseKokoroRepo,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws {
        print("[ChineseG2P] Downloading dictionaries from \(repoId)...")

        // Download the g2p directory from HuggingFace
        let snapshotURL = try await Hub.snapshot(
            from: repoId,
            matching: ["g2p/*"],
            progressHandler: progressHandler ?? { _ in }
        )

        let jiebaURL = snapshotURL.appending(path: "g2p/jieba.bin.gz")
        let pinyinSingleURL = snapshotURL.appending(path: "g2p/pinyin_single.bin.gz")
        let pinyinPhrasesURL = snapshotURL.appending(path: "g2p/pinyin_phrases.bin.gz")

        // Verify files exist
        let fm = FileManager.default
        guard fm.fileExists(atPath: jiebaURL.path) else {
            throw ChineseG2PError.conversionFailed("jieba.bin.gz not found at \(jiebaURL.path)")
        }
        guard fm.fileExists(atPath: pinyinSingleURL.path) else {
            throw ChineseG2PError.conversionFailed("pinyin_single.bin.gz not found at \(pinyinSingleURL.path)")
        }

        print("[ChineseG2P] Loading dictionaries...")
        try tokenizer.loadDictionary(from: jiebaURL)
        try pinyinConverter.loadSinglePinyin(from: pinyinSingleURL)

        if fm.fileExists(atPath: pinyinPhrasesURL.path) {
            try pinyinConverter.loadPhrasePinyin(from: pinyinPhrasesURL)
        }

        isInitialized = true
        print("[ChineseG2P] Initialization complete")
    }

    /// Check if dictionaries are initialized
    public var isReady: Bool {
        isInitialized || tokenizer.isLoaded
    }

    // MARK: - Public API

    /// Convert Chinese text to Bopomofo phoneme string
    /// - Parameter text: Chinese text input
    /// - Returns: Bopomofo phoneme string suitable for Kokoro TTS
    public func convert(_ text: String) throws -> String {
        guard isInitialized || tokenizer.isLoaded else {
            throw ChineseG2PError.notInitialized
        }

        // 1. Normalize text
        let normalized = normalizeText(text)
        guard !normalized.isEmpty else { return "" }

        // 2. Segment into words
        let words = segmentText(normalized)

        // 3. Convert each word to bopomofo
        var result: [String] = []

        for word in words {
            let bopomofo = convertWord(word)
            result.append(bopomofo)
        }

        // 4. Join without separator (matching Python misaki format)
        return result.joined(separator: "")
    }

    /// Convert Chinese text to array of phoneme tokens
    public func convertToTokens(_ text: String) throws -> [String] {
        let phonemeString = try convert(text)
        return tokenize(phonemeString)
    }

    // MARK: - Text Processing

    /// Digit to Chinese character mapping
    private let digitToChinese: [Character: String] = [
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
    ]

    /// Normalize Chinese text (punctuation, whitespace, numbers)
    private func normalizeText(_ text: String) -> String {
        var result = text

        // Convert numbers to Chinese
        result = convertNumbersToChinese(result)

        // Map Chinese punctuation to ASCII equivalents
        let punctuationMap: [Character: Character] = [
            "、": ",", "，": ",", "。": ".", "．": ".",
            "！": "!", "：": ":", "；": ";", "？": "?",
            "«": "\"", "»": "\"", "《": "\"", "》": "\"",
            "「": "\"", "」": "\"", "【": "\"", "】": "\"",
            "（": "(", "）": ")",
        ]

        result = String(result.map { punctuationMap[$0] ?? $0 })

        // Collapse whitespace
        result = result.components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Convert Arabic numerals to Chinese characters
    private func convertNumbersToChinese(_ text: String) -> String {
        var result = ""
        var i = text.startIndex

        while i < text.endIndex {
            let char = text[i]

            if char.isNumber {
                // Collect the full number (including decimals)
                var numberStr = String(char)
                var j = text.index(after: i)

                while j < text.endIndex {
                    let nextChar = text[j]
                    if nextChar.isNumber || nextChar == "." || nextChar == "%" {
                        numberStr.append(nextChar)
                        j = text.index(after: j)
                    } else {
                        break
                    }
                }

                // Convert the number
                result += convertNumber(numberStr)
                i = j
            } else {
                result.append(char)
                i = text.index(after: i)
            }
        }

        return result
    }

    /// Convert a single number string to Chinese
    private func convertNumber(_ numStr: String) -> String {
        // Handle percentage
        if numStr.hasSuffix("%") {
            let num = String(numStr.dropLast())
            return "百分之" + convertNumber(num)
        }

        // Handle decimal
        if numStr.contains(".") {
            let parts = numStr.split(separator: ".")
            if parts.count == 2 {
                let intPart = convertNumber(String(parts[0]))
                let decPart = String(parts[1]).compactMap { digitToChinese[$0] }.joined()
                return intPart + "点" + decPart
            }
        }

        // Simple digit-by-digit conversion (works for years, phone numbers, etc.)
        return numStr.compactMap { digitToChinese[$0] }.joined()
    }

    /// Segment text into words
    private func segmentText(_ text: String) -> [String] {
        // Split by non-Chinese characters first
        var segments: [String] = []
        var currentSegment = ""
        var isChineseSegment = false

        for char in text {
            let isChinese = isChineseCharacter(char)

            if isChinese != isChineseSegment && !currentSegment.isEmpty {
                segments.append(currentSegment)
                currentSegment = ""
            }

            currentSegment.append(char)
            isChineseSegment = isChinese
        }

        if !currentSegment.isEmpty {
            segments.append(currentSegment)
        }

        // Segment Chinese portions
        var result: [String] = []
        for segment in segments {
            if let first = segment.first, isChineseCharacter(first) {
                let words = tokenizer.segment(segment)
                result.append(contentsOf: words)
            } else {
                result.append(segment)
            }
        }

        return result
    }

    /// Convert a single word to bopomofo
    private func convertWord(_ word: String) -> String {
        // Check if it's punctuation or non-Chinese
        if word.count == 1 {
            let char = word.first!
            if !isChineseCharacter(char) {
                return BopomofoMapper.convert(word)
            }
        }

        // Check for non-Chinese word
        if let first = word.first, !isChineseCharacter(first) {
            // Keep non-Chinese text as-is (could be English mixed in)
            return unknownToken
        }

        // Get pinyin for each character
        let pinyins = pinyinConverter.toPinyin(word, style: .tone3)

        // Apply tone sandhi
        // Note: Simplified - full implementation needs POS tagging
        let modifiedPinyins = toneSandhi.modifiedTone(word: word, pos: "n", finals: pinyins)

        // Convert to bopomofo
        let bopomofo = modifiedPinyins.map { BopomofoMapper.convert($0) }

        return bopomofo.joined()
    }

    /// Check if a character is Chinese
    private func isChineseCharacter(_ char: Character) -> Bool {
        guard let scalar = char.unicodeScalars.first else { return false }
        // CJK Unified Ideographs range
        return scalar.value >= 0x4E00 && scalar.value <= 0x9FFF
    }

    /// Tokenize bopomofo string into individual tokens
    private func tokenize(_ bopomofo: String) -> [String] {
        // Split on slashes (word boundaries)
        let words = bopomofo.split(separator: "/")

        var tokens: [String] = []
        for word in words {
            // Each character/symbol is a token
            for char in word {
                tokens.append(String(char))
            }
        }

        return tokens
    }

    // MARK: - Vocabulary

    /// Get the Bopomofo vocabulary for Kokoro
    public var vocabulary: Set<String> {
        BopomofoMapper.vocabulary
    }
}

// MARK: - Errors

public enum ChineseG2PError: Error, LocalizedError {
    case notInitialized
    case conversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "ChineseG2P not initialized. Call initialize() with dictionary data first."
        case .conversionFailed(let message):
            return "G2P conversion failed: \(message)"
        }
    }
}

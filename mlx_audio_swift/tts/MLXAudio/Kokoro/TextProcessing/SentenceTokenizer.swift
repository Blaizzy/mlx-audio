//
//  SentenceTokenizer.swift
//   Swift-TTS
//

import Foundation
import NaturalLanguage

public final class SentenceTokenizer {
    private init() {}

    // MARK: - Initial Split

    public static func splitIntoSentences(text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        let detectedLanguage = detectLanguage(text: text)
        let initialSentences = performInitialSplit(text: text, language: detectedLanguage)
        let refinedSentences = applyTTSRefinements(sentences: initialSentences, originalText: text)

        return optimizeTTSChunks(sentences: refinedSentences, language: detectedLanguage)
    }

    private static func performInitialSplit(text: String, language: NLLanguage?) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        if let language = language {
            tokenizer.setLanguage(language)
        }

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
            let sentence = String(text[tokenRange])
            sentences.append(sentence)
            return true
        }

        return sentences.isEmpty ? [text] : sentences
    }

    // MARK: - TTS-Specific Refinements

    private static func applyTTSRefinements(sentences: [String], originalText: String) -> [String] {
        var result: [String] = []
        result.reserveCapacity(sentences.count) // Pre-allocate capacity

        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                result.append(trimmed)
            }
        }

        return result
    }

    // MARK: - TTS Chunk Optimization

    private static func optimizeTTSChunks(sentences: [String], language: NLLanguage?) -> [String] {
        guard !sentences.isEmpty else { return [] }

        let scriptType = detectScriptType(language: language)

        switch scriptType {
        case .cjk:
            return optimizeCJKChunks(sentences: sentences)
        case .indic:
            return optimizeIndicChunks(sentences: sentences)
        case .latin, .other:
            return optimizeLatinChunks(sentences: sentences)
        }
    }

    private static func optimizeLatinChunks(sentences: [String]) -> [String] {
        let minLength = 50
        return optimizeChunks(
            sentences: sentences,
            config: ChunkConfig(
                minLength: minLength,
                maxLength: 300,
                separator: " ",
                shouldMerge: { chunk in
                    chunk.count < minLength || !hasStrongSentenceEnding(chunk)
                }
            )
        )
    }

    private static func optimizeCJKChunks(sentences: [String]) -> [String] {
        let minLength = 30
        return optimizeChunks(
            sentences: sentences,
            config: ChunkConfig(
                minLength: minLength,
                maxLength: 200,
                separator: "",
                shouldMerge: { chunk in
                    chunk.count < minLength || !hasCJKSentenceEnding(chunk)
                }
            )
        )
    }

    private static func optimizeIndicChunks(sentences: [String]) -> [String] {
        let minLength = 40
        return optimizeChunks(
            sentences: sentences,
            config: ChunkConfig(
                minLength: minLength,
                maxLength: 250,
                separator: " ",
                shouldMerge: { chunk in
                    chunk.count < minLength || !hasIndicSentenceEnding(chunk)
                }
            )
        )
    }

    private struct ChunkConfig {
        let minLength: Int
        let maxLength: Int
        let separator: String
        let shouldMerge: (String) -> Bool
    }

    private static func optimizeChunks(sentences: [String], config: ChunkConfig) -> [String] {
        guard !sentences.isEmpty else { return [] }

        var result: [String] = []
        result.reserveCapacity(sentences.count) // Pre-allocate capacity
        var currentChunk = ""

        for sentence in sentences {
            if currentChunk.isEmpty {
                currentChunk = sentence
            } else {
                let separatorLength = config.separator.isEmpty ? 0 : config.separator.count
                let potentialLength = currentChunk.count + sentence.count + separatorLength

                if potentialLength <= config.maxLength && config.shouldMerge(currentChunk) {
                    if !config.separator.isEmpty {
                        currentChunk += config.separator
                    }
                    currentChunk += sentence
                } else {
                    result.append(currentChunk)
                    currentChunk = sentence
                }
            }
        }

        if !currentChunk.isEmpty {
            result.append(currentChunk)
        }

        return result
    }

    // MARK: - Helper Methods

    private static let languageRecognizer = NLLanguageRecognizer()

    private static func detectLanguage(text: String) -> NLLanguage? {
        languageRecognizer.reset()
        languageRecognizer.processString(text)
        return languageRecognizer.dominantLanguage
    }

    private enum ScriptType {
        case latin, cjk, indic, other
    }

    private static func detectScriptType(language: NLLanguage?) -> ScriptType {
        guard let language = language else { return .other }

        // The languages supoorted by Kokoro 1.0
        switch language {
        case .simplifiedChinese, .traditionalChinese, .japanese:
            return .cjk
        case .english, .french, .spanish, .italian, .portuguese:
            return .latin
        case .hindi:
            return .indic
        default:
            return .other
        }
    }

    private static func hasSentenceEnding(_ text: String, endings: Set<Character>) -> Bool {
        guard let lastChar = text.last else { return false }
        return endings.contains(lastChar) && !text.hasSuffix(" ")
    }

    private static func hasStrongSentenceEnding(_ text: String) -> Bool {
        return hasSentenceEnding(text, endings: [".", "!", "?"])
    }

    private static func hasCJKSentenceEnding(_ text: String) -> Bool {
        return hasSentenceEnding(text, endings: ["。", "！", "？", "…"])
    }

    private static func hasIndicSentenceEnding(_ text: String) -> Bool {
        return hasSentenceEnding(text, endings: ["।", "॥", ".", "!", "?"])
    }
}

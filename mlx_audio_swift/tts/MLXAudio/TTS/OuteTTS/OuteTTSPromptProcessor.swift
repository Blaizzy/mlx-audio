//
//  OuteTTSPromptProcessor.swift
//  MLXAudio
//
//  Prompt processing for OuteTTS text-to-speech
//  Ported from: mlx_audio/tts/models/outetts/prompt_processor.py
//

import Foundation

/// Processes text prompts and speaker embeddings for OuteTTS
public class OuteTTSPromptProcessor {
    public let specialTokens: OuteTTSSpecialTokens

    // Token ID mappings for audio codes (used for extracting codes from generated tokens)
    private var c1TokenMap: [Int: Int] = [:]  // token_id -> code value
    private var c2TokenMap: [Int: Int] = [:]

    // Base token IDs for fast token building
    // Audio code tokens are sequential: c1_X = c1BaseId + X, c2_X = c2BaseId + X
    // This avoids dictionary lookups for ~1000 audio codes per prompt
    private var c1BaseId: Int = 0
    private var c2BaseId: Int = 0

    // Cached special token IDs for fast prompt building
    // Pre-caching avoids repeated convertTokenToId calls during generation
    private var cachedTokenIds: [String: Int] = [:]
    private var newlineTokenId: Int = 198
    private var tokenEncoder: ((String) -> [Int])?

    // Prompt templates
    private let inputPromptTemplate = "{bos}\n{text_start}{text}{text_end}\n{audio_start}\n"
    private let globalFeaturesTemplate = "{fs}{codes}{fe}\n"

    public init() {
        self.specialTokens = OuteTTSSpecialTokens()
    }

    /// Build token maps from tokenizer (call this after tokenizer is loaded)
    ///
    /// This method caches all special token IDs upfront to enable
    /// getCompletionPromptTokens() to build prompts without slow tokenizer.encode() calls.
    /// - convertTokenToId: Fast single-token lookup (used for special tokens)
    /// - encode: Full BPE encoding (only used for regular text, which is short)
    public func buildTokenMaps(convertTokenToId: (String) -> Int?, encode: @escaping (String) -> [Int]) {
        self.tokenEncoder = encode

        // Get base IDs for audio code tokens (they're sequential)
        let sampleC1 = specialTokens.formatC1(0)
        let sampleC2 = specialTokens.formatC2(0)
        if let c1Id = convertTokenToId(sampleC1) {
            c1BaseId = c1Id
        }
        if let c2Id = convertTokenToId(sampleC2) {
            c2BaseId = c2Id
        }

        // Build reverse maps for extracting audio codes from generated tokens
        for i in 0..<1025 {
            let c1Token = specialTokens.formatC1(i)
            let c2Token = specialTokens.formatC2(i)

            if let c1Id = convertTokenToId(c1Token) {
                c1TokenMap[c1Id] = i
            }
            if let c2Id = convertTokenToId(c2Token) {
                c2TokenMap[c2Id] = i
            }
        }

        // Cache special token IDs
        let specialTokensToCache = [
            specialTokens.bos,
            specialTokens.textStart,
            specialTokens.textEnd,
            specialTokens.audioStart,
            specialTokens.audioEnd,
            specialTokens.wordStart,
            specialTokens.wordEnd,
            specialTokens.features,
            specialTokens.code
        ]
        for token in specialTokensToCache {
            if let id = convertTokenToId(token) {
                cachedTokenIds[token] = id
            }
        }

        // Cache time tokens (0.00 to 30.00 in 0.01 increments)
        for i in 0...3000 {
            let time = Double(i) / 100.0
            let token = specialTokens.formatTime(time)
            if let id = convertTokenToId(token) {
                cachedTokenIds[token] = id
            }
        }

        // Cache feature tokens (energy, spectral_centroid, pitch)
        for i in 0...100 {
            let energyToken = "<|energy_\(i)|>"
            let spectralToken = "<|spectral_centroid_\(i)|>"
            let pitchToken = "<|pitch_\(i)|>"
            if let id = convertTokenToId(energyToken) { cachedTokenIds[energyToken] = id }
            if let id = convertTokenToId(spectralToken) { cachedTokenIds[spectralToken] = id }
            if let id = convertTokenToId(pitchToken) { cachedTokenIds[pitchToken] = id }
        }

        // Get newline token ID
        let newlineTokens = encode("\n")
        if !newlineTokens.isEmpty {
            newlineTokenId = newlineTokens[0]
        }
    }

    /// Get feature tokens from feature dictionary
    public func getFeatures(_ features: OuteTTSAudioFeatures) -> [String] {
        return [
            "<|energy_\(features.energy)|>",
            "<|spectral_centroid_\(features.spectralCentroid)|>",
            "<|pitch_\(features.pitch)|>"
        ]
    }

    /// Get global features string
    public func getGlobalFeatures(_ features: OuteTTSAudioFeatures) -> String {
        let codes = getFeatures(features).joined()
        return globalFeaturesTemplate
            .replacingOccurrences(of: "{fs}", with: specialTokens.globalFeaturesStart)
            .replacingOccurrences(of: "{codes}", with: codes)
            .replacingOccurrences(of: "{fe}", with: specialTokens.globalFeaturesEnd)
    }

    /// Create audio code string from word data
    public func createCodes(_ words: [OuteTTSWordData]) -> String {
        var codes: [String] = []

        for wordData in words {
            var word = wordData.word
            word += specialTokens.features
            word += specialTokens.formatTime(wordData.duration)
            word += getFeatures(wordData.features).joined()

            var pairs: [String] = []
            let pairCount = min(wordData.c1.count, wordData.c2.count)
            for idx in 0..<pairCount {
                let c1 = specialTokens.formatC1(wordData.c1[idx])
                let c2 = specialTokens.formatC2(wordData.c2[idx])
                pairs.append("\(c1)\(c2)")
            }

            word += specialTokens.code + pairs.joined()
            codes.append(specialTokens.wordStart + word + specialTokens.wordEnd)
        }

        return codes.joined(separator: "\n")
    }

    /// Initialize prompt with text
    private func initPrompt(_ text: String) -> String {
        return inputPromptTemplate
            .replacingOccurrences(of: "{bos}", with: specialTokens.bos)
            .replacingOccurrences(of: "{text_start}", with: specialTokens.textStart)
            .replacingOccurrences(of: "{text}", with: text)
            .replacingOccurrences(of: "{text_end}", with: specialTokens.textEnd)
            .replacingOccurrences(of: "{audio_start}", with: specialTokens.audioStart)
    }

    /// Get separator based on text language
    private func getSeparator(for text: String) -> String {
        let hasHiragana = text.unicodeScalars.contains { $0.value >= 0x3040 && $0.value <= 0x309F }
        let hasKatakana = text.unicodeScalars.contains { $0.value >= 0x30A0 && $0.value <= 0x30FF }
        let hasHan = text.unicodeScalars.contains { $0.value >= 0x4E00 && $0.value <= 0x9FFF }
        let hasHangul = text.unicodeScalars.contains { $0.value >= 0xAC00 && $0.value <= 0xD7AF }

        if hasHiragana || hasKatakana || hasHan {
            return "。"
        } else if hasHangul {
            return ". "
        } else {
            return ". "
        }
    }

    /// Merge speaker text with input text
    public func mergeSpeakerText(inputText: String, speakerText: String) -> (merged: String, separator: String) {
        let trimmedSpeaker = speakerText.trimmingCharacters(in: .whitespaces)
        let separator = getSeparator(for: trimmedSpeaker)

        let allowedEnds: [Character]
        if separator == "。" {
            allowedEnds = ["。", "？", "！", "?", "!"]
        } else {
            allowedEnds = [".", "?", "!"]
        }

        var rs = ""
        if !trimmedSpeaker.isEmpty {
            if let lastChar = trimmedSpeaker.last, !allowedEnds.contains(lastChar) {
                rs = separator
            } else if separator != "。" {
                rs = " "
            }
        }

        let output = trimmedSpeaker + rs + inputText.trimmingCharacters(in: .whitespaces)
        return (output, rs.trimmingCharacters(in: .whitespaces))
    }

    /// Normalize text (whitespace, quotes, dashes, control characters)
    public static func normalizeText(_ text: String) -> String {
        var result = text
        result = result.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        result = result.replacingOccurrences(of: "\u{2026}", with: "...")
        result = result.trimmingCharacters(in: .whitespaces)
        result = result.replacingOccurrences(of: "\u{201C}", with: "\"")
        result = result.replacingOccurrences(of: "\u{201D}", with: "\"")
        result = result.replacingOccurrences(of: "\u{2018}", with: "'")
        result = result.replacingOccurrences(of: "\u{2019}", with: "'")
        result = result.replacingOccurrences(of: "\u{2013}", with: "-")
        result = result.replacingOccurrences(of: "\u{2014}", with: "-")
        result = result.unicodeScalars.filter { scalar in
            !(scalar.value >= 0x00 && scalar.value <= 0x1F) &&
            !(scalar.value >= 0x7F && scalar.value <= 0x9F)
        }.map { String($0) }.joined()
        return result
    }

    /// Get completion prompt for generation (string version)
    /// Note: Use getCompletionPromptTokens() instead for better performance
    public func getCompletionPrompt(text: String, speaker: OuteTTSSpeakerProfile?) -> String {
        let normalizedText = Self.normalizeText(text)

        var prompt: String
        var finalWords: [OuteTTSWordData]? = nil

        if let speaker = speaker {
            let (mergedText, separator) = mergeSpeakerText(inputText: normalizedText, speakerText: speaker.text)
            var words = speaker.words
            if !words.isEmpty {
                words[words.count - 1].word += separator
            }
            finalWords = words
            prompt = initPrompt(mergedText)
        } else {
            prompt = initPrompt(normalizedText)
        }

        if let words = finalWords {
            let codes = createCodes(words)
            prompt += codes + "\n" + specialTokens.wordStart
        }

        return prompt
    }

    /// Get completion prompt as token IDs directly
    ///
    /// This method builds token IDs directly instead of building a string
    /// and tokenizing it. This avoids ~38s of BPE tokenization on the ~18,000 character
    /// prompt with thousands of special tokens like <|c1_XXX|> and <|c2_XXX|>.
    ///
    /// Key optimizations:
    /// - Special tokens use cached IDs (instant lookup)
    /// - Audio codes computed as baseId + codeValue (no lookup needed)
    /// - Only regular text (short) goes through the tokenizer
    public func getCompletionPromptTokens(text: String, speaker: OuteTTSSpeakerProfile?) -> [Int] {
        guard let encode = tokenEncoder else {
            return []
        }

        let normalizedText = Self.normalizeText(text)
        var tokens: [Int] = []
        tokens.reserveCapacity(2000)

        func addToken(_ tokenStr: String) {
            if let id = cachedTokenIds[tokenStr] {
                tokens.append(id)
            }
        }

        func addText(_ text: String) {
            let textTokens = encode(text)
            tokens.append(contentsOf: textTokens)
        }

        // Build prompt header
        addToken(specialTokens.bos)
        tokens.append(newlineTokenId)
        addToken(specialTokens.textStart)

        // Add merged text
        if let speaker = speaker {
            let (mergedText, _) = mergeSpeakerText(inputText: normalizedText, speakerText: speaker.text)
            addText(mergedText)
        } else {
            addText(normalizedText)
        }

        // Add text end and audio start
        addToken(specialTokens.textEnd)
        tokens.append(newlineTokenId)
        addToken(specialTokens.audioStart)
        tokens.append(newlineTokenId)

        // Add speaker codes if present
        if let speaker = speaker {
            var words = speaker.words

            if !words.isEmpty {
                let (_, separator) = mergeSpeakerText(inputText: normalizedText, speakerText: speaker.text)
                words[words.count - 1].word += separator
            }

            for wordData in words {
                addToken(specialTokens.wordStart)
                addText(wordData.word)
                addToken(specialTokens.features)

                let timeToken = specialTokens.formatTime(wordData.duration)
                addToken(timeToken)

                let energyToken = "<|energy_\(wordData.features.energy)|>"
                let spectralToken = "<|spectral_centroid_\(wordData.features.spectralCentroid)|>"
                let pitchToken = "<|pitch_\(wordData.features.pitch)|>"
                addToken(energyToken)
                addToken(spectralToken)
                addToken(pitchToken)

                addToken(specialTokens.code)

                // Compute audio code token IDs directly as baseId + codeValue
                // This avoids dictionary lookups for hundreds of codes per word
                let pairCount = min(wordData.c1.count, wordData.c2.count)
                for idx in 0..<pairCount {
                    tokens.append(c1BaseId + wordData.c1[idx])
                    tokens.append(c2BaseId + wordData.c2[idx])
                }

                addToken(specialTokens.wordEnd)
                tokens.append(newlineTokenId)
            }
        }

        addToken(specialTokens.wordStart)

        return tokens
    }

    /// Get training prompt (includes full audio codes)
    public func getTrainingPrompt(speaker: OuteTTSSpeakerProfile) -> String {
        let text = Self.normalizeText(speaker.text)
        let words = speaker.words
        let globalFeatures = speaker.globalFeatures

        var prompt = initPrompt(text)
        prompt += getGlobalFeatures(globalFeatures)
        prompt += createCodes(words)
        prompt += "\n" + specialTokens.audioEnd + "\n" + specialTokens.eos + "\n"

        return prompt
    }

    /// Extract audio codes from generated token IDs
    public func extractAudioFromTokens(_ tokens: [Int]) -> [[Int]] {
        var codebook1: [Int] = []
        var codebook2: [Int] = []

        for token in tokens {
            if let value = c1TokenMap[token] {
                codebook1.append(value)
            }
            if let value = c2TokenMap[token] {
                codebook2.append(value)
            }
        }

        let t = min(codebook1.count, codebook2.count)
        codebook1 = Array(codebook1.prefix(t))
        codebook2 = Array(codebook2.prefix(t))

        return [codebook1, codebook2]
    }
}

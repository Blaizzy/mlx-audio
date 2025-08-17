//
//  KokoroTokenizer.swift
//   Swift-TTS
//

import Foundation
import Accelerate

/// Converts text into phonetic representations with stress markers for on-device text-to-speech
final class KokoroTokenizer {

    // MARK: - Constants

    private static let stresses = "ˌˈ"
    private static let primaryStress = Character("ˈ")
    private static let secondaryStress = Character("ˌ")

    private static let vowels: Set<Character> = [
        "A", "I", "O", "Q", "W", "Y", "a", "i", "u",
        "æ", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ", "ᵻ"
    ]

    private static let subtokenJunks: Set<Character> = [
        "'", ",", "-", ".", "_", "'", "'", "/", " "
    ]

    private static let puncts: Set<String> = [
        "?", ",", ";", "\"", "—", ":", "!", ".", "…", "\""
    ]

    private static let nonQuotePuncts: Set<String> = [
        "?", ",", "—", ".", ":", "!", ";", "…"
    ]

    private static let currencies: [Character: (bill: String, cent: String)] = [
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent")
    ]

    private static let functionWords: Set<String> = [
        "a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "to", "from",
        "and", "or", "but", "nor", "so", "yet", "is", "am", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "must", "can", "could", "that",
        "this", "these", "those", "he", "she", "it", "they", "we", "you", "i", "me",
        "him", "her", "them", "us", "my", "your", "his", "their", "our", "its"
    ]

    private static let sentenceEndingPunct: Set<String> = [".", "!", "?"]

    // MARK: - Regex Patterns (compiled once for performance)

    private static let currencyRegex = try! NSRegularExpression(
        pattern: #"[\$£€]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[\$£€]\d+\.\d\d?\b"#
    )

    private static let timeRegex = try! NSRegularExpression(
        pattern: #"\b(?:[1-9]|1[0-2]):[0-5]\d\b"#
    )

    private static let decimalRegex = try! NSRegularExpression(
        pattern: #"\b\d*\.\d+\b"#
    )

    private static let numberRegex = try! NSRegularExpression(
        pattern: #"\b[0-9]+\b"#
    )

    private static let rangeRegex = try! NSRegularExpression(
        pattern: #"([\$£€]?\d+)-([\$£€]?\d+)"#
    )

    private static let commaInNumberRegex = try! NSRegularExpression(
        pattern: #"(^|[^\d])(\d+(?:,\d+)*)([^\d]|$)"#
    )

    private static let linkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]+)\]\(([^\)]*)\)"#
    )

    private static let alphabeticRegex = try! NSRegularExpression(
        pattern: #"^[a-zA-Z]+$"#
    )

    // MARK: - Instance Properties

    private var cachedUSLexicon: [String: String]?
    private var cachedGBLexicon: [String: String]?
    private var eSpeakEngine: ESpeakNGEngine
    private var currentLanguage: ESpeakNGEngine.LanguageDialect = .none
    private var isLexiconEnabled = false

    // MARK: - Token Structure

    struct Token {
        let text: String
        var whitespace: String
        var phonemes: String
        let stress: Float?
        let currency: String?
        var prespace: Bool
        let alias: String?
        let isHead: Bool

        init(text: String, whitespace: String = " ", phonemes: String = "", stress: Float? = nil,
             currency: String? = nil, prespace: Bool = true, alias: String? = nil, isHead: Bool = false) {
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

    // MARK: - Result Structure

    struct PhonemizerResult {
        let phonemes: String
        let tokens: [Token]
    }

    // MARK: - Initialization

    init(engine: ESpeakNGEngine) {
        self.eSpeakEngine = engine
        loadLexicon()
    }

    // MARK: - Public API

    func setLanguage(for voice: TTSVoice) throws {
        let language = try eSpeakEngine.languageForVoice(voice: voice)
        if currentLanguage != language {
            try eSpeakEngine.setLanguage(for: voice)
            currentLanguage = language

            // Update lexicon usage based on language
            switch language {
            case .enUS, .enGB:
                self.isLexiconEnabled = true
            default:
                self.isLexiconEnabled = false
            }
        }
    }

    /// Main phonemization function
    func phonemize(_ text: String) throws -> PhonemizerResult {
        let (_, tokens, features, nonStringFeatures) = preprocess(text)
        let tokenizedTokens = try tokenize(tokens: tokens, features: features, nonStringFeatures: nonStringFeatures)
        let result = resolveTokens(tokenizedTokens)
        return PhonemizerResult(phonemes: result, tokens: tokenizedTokens)
    }

    /// Load and separate lexicon files for word-to-phoneme mappings
    private func loadLexicon() {
        // Load US lexicons
        var usLexicon: [String: String] = [:]

        // Load us_silver.json first (lower priority)
        if let silverLexicon = loadLexiconFile("us_silver") {
            usLexicon.merge(silverLexicon) { _, new in new }
        }

        // Load us_gold.json second (higher priority, will override us_silver)
        if let goldLexicon = loadLexiconFile("us_gold") {
            usLexicon.merge(goldLexicon) { _, new in new }
        }

        self.cachedUSLexicon = usLexicon.isEmpty ? nil : usLexicon

        // Load GB lexicons
        var gbLexicon: [String: String] = [:]

        // Load gb_silver.json first (lower priority)
        if let silverLexicon = loadLexiconFile("gb_silver") {
            gbLexicon.merge(silverLexicon) { _, new in new }
        }

        // Load gb_gold.json second (higher priority, will override gb_silver)
        if let goldLexicon = loadLexiconFile("gb_gold") {
            gbLexicon.merge(goldLexicon) { _, new in new }
        }

        self.cachedGBLexicon = gbLexicon.isEmpty ? nil : gbLexicon
    }

    /// Helper method to load a single lexicon file
    private func loadLexiconFile(_ filename: String) -> [String: String]? {
        guard let url = Bundle.main.url(forResource: filename, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("Failed to load lexicon file: \(filename).json")
            return nil
        }

        var processedLexicon: [String: String] = [:]

        for (key, value) in json {
            if let stringValue = value as? String {
                processedLexicon[key] = stringValue
            } else if let dictValue = value as? [String: Any],
                      let defaultValue = dictValue["DEFAULT"] as? String {
                processedLexicon[key] = defaultValue
            }
        }

        return processedLexicon
    }

    // MARK: - Preprocessing

    private func preprocess(_ text: String) -> (String, [String], [String: Any], Set<String>) {
        var processedText = text
        var tokens: [String] = []
        var features: [String: Any] = [:]
        var nonStringFeatures: Set<String> = []

        // Remove commas from numbers
        processedText = removeCommasFromNumbers(processedText)

        // Handle ranges (5-10 -> 5 to 10)
        processedText = Self.rangeRegex.stringByReplacingMatches(
            in: processedText,
            range: NSRange(processedText.startIndex..., in: processedText),
            withTemplate: "$1 to $2"
        )

        // Process currencies, times, and decimals
        processedText = flipMoney(processedText)
        processedText = splitNum(processedText)
        processedText = pointNum(processedText)

        // Extract features from link format [text](replacement)
        var lastEnd = processedText.startIndex
        var result = ""

        let matches = Self.linkRegex.matches(
            in: processedText,
            range: NSRange(processedText.startIndex..., in: processedText)
        )

        for match in matches {
            let beforeMatch = String(processedText[lastEnd..<processedText.index(processedText.startIndex, offsetBy: match.range.location)])
            result += beforeMatch
            tokens.append(contentsOf: beforeMatch.components(separatedBy: " ").filter { !$0.isEmpty })

            let originalRange = Range(match.range(at: 1), in: processedText)!
            let replacementRange = Range(match.range(at: 2), in: processedText)!
            let original = String(processedText[originalRange])
            let replacement = String(processedText[replacementRange])

            let (feature, isNonString) = parseFeature(original: original, replacement: replacement)
            let tokenIndex = String(tokens.count)

            if isNonString {
                nonStringFeatures.insert(tokenIndex)
            }
            features[tokenIndex] = feature

            result += original
            tokens.append(original)

            lastEnd = processedText.index(processedText.startIndex, offsetBy: match.range.upperBound)
        }

        if lastEnd < processedText.endIndex {
            let remaining = String(processedText[lastEnd...])
            result += remaining
            tokens.append(contentsOf: remaining.components(separatedBy: " ").filter { !$0.isEmpty })
        }

        return (result, tokens, features, nonStringFeatures)
    }

    private func parseFeature(original: String, replacement: String) -> (Any, Bool) {
        // Direct phoneme specification
        if replacement.hasPrefix("/") && replacement.hasSuffix("/") {
            return (replacement, false)
        }

        // Currency, time, or decimal conversion
        if original.hasPrefix("$") || original.contains(":") || original.contains(".") {
            return ("[\(replacement)]", false)
        }

        // Numeric stress values
        if let value = Float(replacement) {
            return (value, true)
        }

        // Signed integers for stress
        if replacement.hasPrefix("-") || replacement.hasPrefix("+") {
            if let value = Float(replacement) {
                return (value, true)
            }
        }

        // Default - treat as alias
        return ("[\(replacement)]", false)
    }

    // MARK: - Text Processing Utilities

    private func removeCommasFromNumbers(_ text: String) -> String {
        return Self.commaInNumberRegex.stringByReplacingMatches(
            in: text,
            range: NSRange(text.startIndex..., in: text),
            withTemplate: "$1$2$3"
        ).replacingOccurrences(of: ",", with: "")
    }

    private func flipMoney(_ text: String) -> String {
        var result = text
        let matches = Self.currencyRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            let matchRange = Range(match.range, in: text)!
            let matchText = String(text[matchRange])

            guard let currencySymbol = matchText.first,
                  let currency = Self.currencies[currencySymbol] else { continue }

            let value = String(matchText.dropFirst())
            let components = value.components(separatedBy: ".")
            let dollars = components[0]
            let cents = components.count > 1 ? components[1] : "0"

            let transformed: String
            if Int(cents) == 0 {
                transformed = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
            } else {
                let dollarPart = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
                transformed = "\(dollarPart) and \(cents) \(currency.cent)s"
            }

            result = result.replacingCharacters(in: matchRange, with: "\(transformed)")
        }

        return result
    }

    private func splitNum(_ text: String) -> String {
        var result = text
        let matches = Self.timeRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            let matchRange = Range(match.range, in: text)!
            let matchText = String(text[matchRange])

            let components = matchText.components(separatedBy: ":")
            guard components.count == 2,
                  let hour = Int(components[0]),
                  let minute = Int(components[1]) else { continue }

            let transformed: String
            if minute == 0 {
                transformed = "\(hour) o'clock"
            } else if minute < 10 {
                transformed = "\(hour) oh \(minute)"
            } else {
                transformed = "\(hour) \(minute)"
            }

            result = result.replacingCharacters(in: matchRange, with: "\(transformed)")
        }

        return result
    }

    private func pointNum(_ text: String) -> String {
        var result = text
        let decimalMatches = Self.decimalRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        let linkMatches = Self.linkRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        // Create ranges of text inside link parentheses to exclude from processing
        var excludeRanges: [NSRange] = []
        for linkMatch in linkMatches {
            if linkMatch.numberOfRanges >= 3 {
                let replacementRange = linkMatch.range(at: 2)
                excludeRanges.append(replacementRange)
            }
        }

        for match in decimalMatches.reversed() {
            let matchRange = match.range

            // Skip if this decimal is inside a link replacement
            let isInLinkReplacement = excludeRanges.contains { NSIntersectionRange(matchRange, $0).length > 0 }
            if isInLinkReplacement { continue }

            let swiftRange = Range(matchRange, in: text)!
            let matchText = String(text[swiftRange])

            let components = matchText.components(separatedBy: ".")
            guard components.count == 2 else { continue }

            let integerPart = components[0]
            let decimalPart = components[1]
            let decimalDigits = decimalPart.map { String($0) }.joined(separator: " ")
            let transformed = "\(integerPart) point \(decimalDigits)"

            result = result.replacingCharacters(in: swiftRange, with: "\(transformed)")
        }

        return result
    }

    // MARK: - Tokenization

    private func tokenize(tokens: [String], features: [String: Any], nonStringFeatures: Set<String>) throws -> [Token] {
        var result: [Token] = []

        for (index, word) in tokens.enumerated() {
            if word.contains(Self.subtokenJunks) { continue }

            let feature = features[String(index)]

            // Handle direct phoneme specification
            if let featureString = feature as? String,
               featureString.hasPrefix("/") && featureString.hasSuffix("/") {
                let phoneme = String(featureString.dropFirst().dropLast() )
                result.append(Token(text: word, phonemes: phoneme))
                continue
            }

            // Handle stress values - use original word text, not the stress value
            if feature is Float {
                let wordTokens = try tokenizeWord(word, stress: feature as? Float)
                if !wordTokens.isEmpty {
                    result.append(mergeTokens(wordTokens))
                }
                continue
            }

            // Handle alias/replacement text
            var phonemeText = word
            if let featureString = feature as? String,
               featureString.hasPrefix("[") && featureString.hasSuffix("]") {
                phonemeText = String(featureString.dropFirst().dropLast())
            }

            let wordTokens = try tokenizeWord(phonemeText, stress: feature as? Float)
            if !wordTokens.isEmpty {
                result.append(mergeTokens(wordTokens))
            }
        }

        return result
    }

    private func tokenizeWord(_ text: String, stress: Float?) throws -> [Token] {
        let words = text.components(separatedBy: " ").filter { !$0.isEmpty }
        var tokens: [Token] = []

        for (_, word) in words.enumerated() {
            let punctSplit = splitPunctuation(word)

            for (subIndex, token) in punctSplit.enumerated() {
                let phoneme = try generatePhoneme(for: token)
                let isWhitespace = !Self.puncts.contains(token)

                if !isWhitespace && !tokens.isEmpty {
                    tokens[tokens.count - 1].whitespace = ""
                }

                tokens.append(Token(
                    text: token,
                    phonemes: phoneme,
                    stress: stress,
                    prespace: isWhitespace,
                    isHead: subIndex == 0
                ))
            }
        }

        if !tokens.isEmpty {
            tokens[tokens.count - 1].whitespace = ""
            tokens[0].prespace = false
        }

        return tokens
    }

    private func splitPunctuation(_ text: String) -> [String] {
        var result = [text]

        for punct in Self.puncts {
            var newResult: [String] = []
            for part in result {
                if part.contains(punct) {
                    newResult.append(contentsOf: part.components(separatedBy: punct).flatMap { [$0, punct] }.dropLast())
                } else {
                    newResult.append(part)
                }
            }
            result = newResult.filter { !$0.isEmpty }
        }

        return result
    }

    private func generatePhoneme(for token: String) throws -> String {
        if Self.puncts.contains(token) {
            return token
        }

        let lowerToken = token.lowercased()

        // Check lexicon first (only if enabled for current language)
        if isLexiconEnabled {
            let lexicon: [String: String]?
            switch currentLanguage {
            case .enUS:
                lexicon = cachedUSLexicon
            case .enGB:
                lexicon = cachedGBLexicon
            default:
                lexicon = nil
            }

            if let selectedLexicon = lexicon,
               let phoneme = selectedLexicon[token] ?? selectedLexicon[lowerToken] {
                return phoneme
            }
        }

        // Use espeak backend
        return try eSpeakEngine.phonemize(text: lowerToken)
    }

    private func mergeTokens(_ tokens: [Token]) -> Token {
        let stresses = tokens.compactMap { $0.stress }

        var phonemes = ""
        for token in tokens {
            if token.prespace && !phonemes.isEmpty && !phonemes.last!.isWhitespace && !token.phonemes.isEmpty {
                phonemes += " "
            }
            phonemes += token.phonemes
        }

        if phonemes.first?.isWhitespace == true {
            phonemes = String(phonemes.dropFirst())
        }

        let mergedStress = stresses.count == 1 ? stresses[0] : nil
        let text = tokens.dropLast().map { $0.text + $0.whitespace }.joined() + tokens.last!.text

        return Token(
            text: text,
            whitespace: tokens.last?.whitespace ?? "",
            phonemes: phonemes,
            stress: mergedStress,
            prespace: tokens.first?.prespace ?? false,
            isHead: tokens.first?.isHead ?? false
        )
    }

    // MARK: - Stress and Resolution

    private func resolveTokens(_ tokens: [Token]) -> String {
        // Apply G2P phoneme corrections
        let phonemeCorrections: [String: String] = [
            "eɪ": "A",
            "ɹeɪndʒ": "ɹAnʤ",
            "wɪðɪn": "wəðɪn"
        ]

        let wordPhonemeMap: [String: String] = [
            "a": "ɐ",
            "an": "ən"
        ]

        var processedTokens = tokens

        // Process each token
        for (index, token) in processedTokens.enumerated() {
            guard !token.phonemes.isEmpty else { continue }

            // Apply word mapping
            if let mapped = wordPhonemeMap[token.text.lowercased()] {
                processedTokens[index] = Token(
                    text: token.text,
                    whitespace: token.whitespace,
                    phonemes: mapped,
                    stress: token.stress,
                    currency: token.currency,
                    prespace: token.prespace,
                    alias: token.alias,
                    isHead: token.isHead
                )
                continue
            }

            // Apply phoneme corrections
            var correctedPhonemes = token.phonemes
            for (old, new) in phonemeCorrections {
                correctedPhonemes = correctedPhonemes.replacingOccurrences(of: old, with: new)
            }

            // Apply custom stress if specified
            if let customStress = token.stress {
                correctedPhonemes = applyCustomStress(to: correctedPhonemes, stressValue: customStress)
            } else {
                // Apply default stress
                let hasStress = correctedPhonemes.contains(Self.primaryStress) || correctedPhonemes.contains(Self.secondaryStress)

                if !hasStress {
                    if correctedPhonemes.contains(" ") {
                        // Multi-word phonemes
                        let subwords = correctedPhonemes.components(separatedBy: " ")
                        let stressedSubwords = subwords.map { subword -> String in
                            guard !subword.isEmpty && !subword.contains(Self.primaryStress) && !subword.contains(Self.secondaryStress) else {
                                return subword
                            }

                            let hasVowels = subword.contains { Self.vowels.contains($0) }
                            guard hasVowels else { return subword }

                            if ["ænd", "ðə", "ɪn", "ɔn", "æt", "wɪð", "baɪ"].contains(subword) {
                                return subword
                            } else {
                                return addStressBeforeVowel(subword, stress: Self.primaryStress)
                            }
                        }
                        correctedPhonemes = stressedSubwords.joined(separator: " ")
                    } else {
                        // Single word
                        if index == 0 {
                            correctedPhonemes = addStressBeforeVowel(correctedPhonemes, stress: Self.secondaryStress)
                        } else if isContentWord(token.text) && correctedPhonemes.count > 2 {
                            correctedPhonemes = addStressBeforeVowel(correctedPhonemes, stress: Self.primaryStress)
                        }
                    }
                }
            }

            processedTokens[index] = Token(
                text: token.text,
                whitespace: token.whitespace,
                phonemes: correctedPhonemes,
                stress: token.stress,
                currency: token.currency,
                prespace: token.prespace,
                alias: token.alias,
                isHead: token.isHead
            )
        }

        // Build final result
        var result: [String] = []
        var punctuationAdded = false

        for (index, token) in processedTokens.enumerated() {
            let isPunct = Self.puncts.contains(token.text)

            if index > 0 && !isPunct && !punctuationAdded {
                result.append(" ")
            }

            punctuationAdded = false

            if isPunct {
                result.append(token.text)
                punctuationAdded = true
            } else if !token.phonemes.isEmpty {
                result.append(token.phonemes)

                if token.text.last.map({ Self.puncts.contains(String($0)) }) == true {
                    let punct = String(token.text.last!)
                    result.append(punct)
                    punctuationAdded = true

                    if Self.sentenceEndingPunct.contains(punct) && index < processedTokens.count - 1 {
                        result.append(" ")
                    }
                }
            }
        }

        return result.joined()
    }

    private func addStressBeforeVowel(_ phoneme: String, stress: Character) -> String {
        for (index, char) in phoneme.enumerated() {
            if Self.vowels.contains(char) {
                if index == 0 {
                    return String(stress) + phoneme
                } else {
                    let insertIndex = phoneme.index(phoneme.startIndex, offsetBy: index)
                    return String(phoneme[..<insertIndex]) + String(stress) + String(phoneme[insertIndex...])
                }
            }
        }
        return phoneme
    }

    private func applyCustomStress(to phonemes: String, stressValue: Float) -> String {
        var result = phonemes

        // Remove existing stress markers first
        result = result.replacingOccurrences(of: String(Self.primaryStress), with: "")
        result = result.replacingOccurrences(of: String(Self.secondaryStress), with: "")

        if stressValue < -1 {
            // Remove all stress (already done above)
            return result
        } else if stressValue == -1 {
            // Add secondary stress
            return addStressBeforeVowel(result, stress: Self.secondaryStress)
        } else if stressValue >= 0 && stressValue < 1 {
            // Add secondary stress
            return addStressBeforeVowel(result, stress: Self.secondaryStress)
        } else if stressValue >= 1 {
            // Add primary stress
            return addStressBeforeVowel(result, stress: Self.primaryStress)
        }

        return result
    }

    private func isFunctionWord(_ word: String) -> Bool {
        let cleaned = word.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: String(Self.puncts.joined())))
        return Self.functionWords.contains(cleaned)
    }

    private func isContentWord(_ word: String) -> Bool {
        return !isFunctionWord(word) && word.count > 2 && isAlphabetic(word)
    }

    private func isAlphabetic(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        return Self.alphabeticRegex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }
}

// MARK: - Character Extension

private extension Character {
    var isWhitespace: Bool {
        return self == " " || self == "\t" || self == "\n" || self == "\r"
    }
}

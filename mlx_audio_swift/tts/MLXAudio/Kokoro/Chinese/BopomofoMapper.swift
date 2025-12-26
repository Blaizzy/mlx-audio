//
//  BopomofoMapper.swift
//  MLXAudio
//
//  Maps pinyin syllables to Bopomofo (注音符号) for Kokoro TTS.
//  Based on misaki's ZH_MAP from zh_frontend.py
//

import Foundation

/// Maps pinyin syllables to Bopomofo (注音符号) for Kokoro TTS.
public enum BopomofoMapper {

    // MARK: - Pinyin to Bopomofo Mapping

    /// Initials (声母): pinyin consonant -> bopomofo
    private static let initials: [String: String] = [
        "b": "ㄅ", "p": "ㄆ", "m": "ㄇ", "f": "ㄈ",
        "d": "ㄉ", "t": "ㄊ", "n": "ㄋ", "l": "ㄌ",
        "g": "ㄍ", "k": "ㄎ", "h": "ㄏ",
        "j": "ㄐ", "q": "ㄑ", "x": "ㄒ",
        "zh": "ㄓ", "ch": "ㄔ", "sh": "ㄕ", "r": "ㄖ",
        "z": "ㄗ", "c": "ㄘ", "s": "ㄙ",
    ]

    /// Finals (韵母): pinyin vowel -> bopomofo
    private static let finals: [String: String] = [
        // Simple vowels
        "a": "ㄚ", "o": "ㄛ", "e": "ㄜ", "i": "ㄧ", "u": "ㄨ", "v": "ㄩ",

        // Compound vowels
        "ai": "ㄞ", "ei": "ㄟ", "ao": "ㄠ", "ou": "ㄡ",
        "an": "ㄢ", "en": "ㄣ", "ang": "ㄤ", "eng": "ㄥ",
        "er": "ㄦ", "ie": "ㄝ",

        // Special finals (mapped to Chinese characters in Kokoro)
        "ii": "ㄭ",  // zi, ci, si
        "iii": "十",  // zhi, chi, shi, ri

        // i-medial compounds
        "ia": "压", "ian": "言", "iang": "阳", "iao": "要",
        "in": "阴", "ing": "应", "iong": "用", "iou": "又",

        // u-medial compounds
        "ua": "穵", "uai": "外", "uan": "万", "uang": "王",
        "uei": "为", "uen": "文", "ueng": "瓮", "uo": "我",
        "ong": "中",

        // ü-medial compounds
        "ve": "月", "van": "元", "vn": "云",
    ]

    /// Punctuation mapping
    private static let punctuation: [Character: String] = [
        ";": ";", ":": ":", ",": ",", ".": ".",
        "!": "!", "?": "?", "/": "/", "—": "—", "…": "…",
        "\"": "\"", "(": "(", ")": ")",
        "\u{201C}": "\u{201C}", "\u{201D}": "\u{201D}", " ": " ",
    ]

    /// Tone numbers (kept as-is)
    private static let tones: Set<Character> = ["1", "2", "3", "4", "5"]

    /// Erhua marker
    private static let erhua = "R"

    // MARK: - Public API

    /// Convert a pinyin syllable with tone to bopomofo
    /// e.g., "bei3" -> "ㄅㄟ3", "jing1" -> "ㄐ应1"
    public static func convert(_ pinyin: String) -> String {
        // Handle punctuation
        if pinyin.count == 1, let char = pinyin.first, let mapped = punctuation[char] {
            return mapped
        }

        // Extract tone number if present
        var syllable = pinyin.lowercased()
        var tone = ""
        if let last = syllable.last, tones.contains(last) {
            tone = String(last)
            syllable = String(syllable.dropLast())
        }

        // Handle erhua
        var hasErhua = false
        if syllable.hasSuffix("r") && syllable.count > 1 && syllable != "er" {
            // Check if it's erhua or just 'r' initial
            let withoutR = String(syllable.dropLast())
            if !withoutR.isEmpty && !"aeiouv".contains(withoutR.last!) {
                // Not erhua, it's 'r' initial like "ri"
            } else {
                hasErhua = true
                syllable = withoutR
            }
        }

        // Find initial
        var initial = ""
        var remaining = syllable

        // Try two-character initials first (zh, ch, sh)
        if syllable.count >= 2 {
            let twoChar = String(syllable.prefix(2))
            if let mapped = initials[twoChar] {
                initial = mapped
                remaining = String(syllable.dropFirst(2))
            }
        }

        // Try one-character initial
        if initial.isEmpty && !syllable.isEmpty {
            let oneChar = String(syllable.prefix(1))
            if let mapped = initials[oneChar] {
                initial = mapped
                remaining = String(syllable.dropFirst(1))
            }
        }

        // Handle special cases for zi, ci, si, zhi, chi, shi, ri
        if remaining == "i" {
            if ["z", "c", "s"].contains(String(syllable.prefix(1))) && syllable.count == 2 {
                remaining = "ii"  // zi, ci, si -> use ㄭ
            } else if ["zh", "ch", "sh", "r"].contains(String(syllable.prefix(2)))
                || (syllable.hasPrefix("r") && syllable.count == 2)
            {
                remaining = "iii"  // zhi, chi, shi, ri -> use 十
            }
        }

        // Handle ü -> v conversion (lü, nü, etc.)
        remaining = remaining.replacingOccurrences(of: "ü", with: "v")

        // Handle iou -> iu, uei -> ui, uen -> un (standard pinyin abbreviations)
        if remaining == "iu" { remaining = "iou" }
        if remaining == "ui" { remaining = "uei" }
        if remaining == "un" { remaining = "uen" }

        // Map final
        var final = ""
        if let mapped = finals[remaining] {
            final = mapped
        } else if !remaining.isEmpty {
            // Try to decompose compound final
            final = decomposeCompoundFinal(remaining)
        }

        // Build result
        var result = initial + final

        // Add erhua marker
        if hasErhua {
            result += erhua
        }

        // Add tone
        result += tone

        return result
    }

    /// Convert array of pinyin syllables to bopomofo
    public static func convert(_ pinyins: [String]) -> [String] {
        pinyins.map { convert($0) }
    }

    /// Convert pinyin string (space-separated) to bopomofo string
    public static func convertString(_ pinyin: String) -> String {
        let syllables = pinyin.split(separator: " ").map(String.init)
        return convert(syllables).joined()
    }

    // MARK: - Private Helpers

    /// Attempt to decompose a compound final that wasn't found in the dictionary
    private static func decomposeCompoundFinal(_ final: String) -> String {
        // Try matching longest substrings first
        var result = ""
        var remaining = final

        while !remaining.isEmpty {
            var matched = false

            // Try decreasing lengths
            for len in stride(from: min(4, remaining.count), through: 1, by: -1) {
                let prefix = String(remaining.prefix(len))
                if let mapped = finals[prefix] {
                    result += mapped
                    remaining = String(remaining.dropFirst(len))
                    matched = true
                    break
                }
            }

            if !matched {
                // Unknown character, keep as-is
                result += String(remaining.prefix(1))
                remaining = String(remaining.dropFirst())
            }
        }

        return result
    }

    // MARK: - Vocabulary

    /// Get all bopomofo tokens used by Kokoro
    public static var vocabulary: Set<String> {
        var tokens = Set<String>()

        // Add initials
        tokens.formUnion(initials.values)

        // Add finals
        tokens.formUnion(finals.values)

        // Add tones
        tokens.formUnion(tones.map(String.init))

        // Add punctuation
        tokens.formUnion(punctuation.values)

        // Add erhua
        tokens.insert(erhua)

        return tokens
    }
}

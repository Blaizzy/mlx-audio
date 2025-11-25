//
//  Voice.swift
//  MLXAudio
//
//  Unified voice model for all TTS engines.
//

import Foundation

/// Represents a voice option for TTS engines
public struct Voice: Identifiable, Hashable, Sendable {
    /// Unique identifier for the voice (e.g., "af_heart", "conversational_a")
    public let id: String

    /// Human-readable display name
    public let displayName: String

    /// Language/region code (e.g., "en-US", "en-GB", "ja-JP")
    public let languageCode: String

    public init(id: String, displayName: String, languageCode: String) {
        self.id = id
        self.displayName = displayName
        self.languageCode = languageCode
    }

    /// Flag emoji derived from language code
    public var languageFlag: String {
        // Handle special cases first
        guard languageCode.count >= 2 else { return "🏳️" }

        // Extract country code from language code (e.g., "en-US" -> "US")
        let countryCode: String
        if languageCode.contains("-") {
            countryCode = String(languageCode.suffix(2)).uppercased()
        } else {
            // Fallback mapping for simple language codes
            countryCode = Self.languageToCountry[languageCode.lowercased()] ?? "XX"
        }

        // Convert country code to flag emoji
        let base: UInt32 = 127397
        return countryCode.uppercased().unicodeScalars.compactMap {
            UnicodeScalar(base + $0.value)
        }.map { String($0) }.joined()
    }

    /// Mapping from single-letter voice prefixes to country codes
    /// Used for Kokoro-style voice names (e.g., "af_heart" -> American Female)
    private static let prefixToCountry: [Character: String] = [
        "a": "US",  // American
        "b": "GB",  // British
        "e": "ES",  // Spanish
        "f": "FR",  // French
        "h": "IN",  // Hindi
        "i": "IT",  // Italian
        "j": "JP",  // Japanese
        "p": "BR",  // Portuguese (Brazil)
        "z": "CN",  // Chinese
    ]

    /// Mapping from simple language codes to country codes
    private static let languageToCountry: [String: String] = [
        "en": "US",
        "es": "ES",
        "fr": "FR",
        "de": "DE",
        "it": "IT",
        "ja": "JP",
        "zh": "CN",
        "pt": "BR",
        "hi": "IN",
    ]

    /// Create a Voice from a Kokoro-style voice identifier (e.g., "af_heart")
    public static func fromKokoroID(_ id: String) -> Voice {
        let displayName = formatKokoroDisplayName(id)
        let languageCode = inferLanguageCode(from: id)
        return Voice(id: id, displayName: displayName, languageCode: languageCode)
    }

    /// Create a Voice for Marvis conversational voices
    public static func fromMarvisID(_ id: String) -> Voice {
        let displayName: String
        if id.hasPrefix("conversational_") {
            let voiceType = id.dropFirst("conversational_".count)
            displayName = "Conversational \(voiceType.uppercased())"
        } else {
            displayName = id.capitalized
        }
        return Voice(id: id, displayName: displayName, languageCode: "en-US")
    }

    // MARK: - Private Helpers

    private static func formatKokoroDisplayName(_ id: String) -> String {
        guard id.count >= 3 else { return id.capitalized }
        // Format: "af_heart" -> "Heart"
        let name = id.dropFirst(3)
        return name.capitalized
    }

    private static func inferLanguageCode(from id: String) -> String {
        guard let firstChar = id.first else { return "en-US" }
        let country = prefixToCountry[firstChar] ?? "US"
        let language = country == "US" || country == "GB" ? "en" :
                       country == "ES" ? "es" :
                       country == "FR" ? "fr" :
                       country == "IT" ? "it" :
                       country == "JP" ? "ja" :
                       country == "CN" ? "zh" :
                       country == "BR" ? "pt" :
                       country == "IN" ? "hi" : "en"
        return "\(language)-\(country)"
    }
}

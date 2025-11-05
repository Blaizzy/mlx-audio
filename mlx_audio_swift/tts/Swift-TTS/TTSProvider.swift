//
//  TTSProvider.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 13/04/2025.
//

import Foundation

// MARK: - TTS Provider Enum

enum TTSProvider: String, CaseIterable {
    case kokoro = "kokoro"
    case orpheus = "orpheus"
    case marvis = "marvis"

    var displayName: String {
        rawValue.capitalized
    }

    var defaultVoice: String {
        switch self {
        case .kokoro:
            return TTSVoice.bmGeorge.rawValue
        case .orpheus:
            return "dan"
        case .marvis:
            return MarvisSession.Voice.conversationalA.rawValue
        }
    }

    var availableVoices: [String] {
        switch self {
        case .kokoro:
            return TTSVoice.allCases.map { $0.rawValue }
        case .orpheus:
            return OrpheusVoice.allCases.map { $0.rawValue }
        case .marvis:
            return MarvisSession.Voice.allCases.map { $0.rawValue }
        }
    }

    var statusMessage: String {
        switch self {
        case .orpheus:
            return "Orpheus is currently quite slow (0.1x on M1). Working on it!\n\nBut it does support expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
        case .marvis:
            return "Marvis: Advanced conversational TTS with streaming support.\n\nNote: Downloads model weights on first use."
        case .kokoro:
            return ""
        }
    }

    func validateVoice(_ voice: String) -> Bool {
        switch self {
        case .kokoro:
            return TTSVoice.fromIdentifier(voice) != nil || TTSVoice(rawValue: voice) != nil
        case .orpheus:
            return OrpheusVoice(rawValue: voice) != nil
        case .marvis:
            return MarvisSession.Voice(rawValue: voice) != nil
        }
    }

    var errorMessage: String {
        switch self {
        case .kokoro:
            return "Invalid Kokoro voice selected"
        case .orpheus:
            return "Invalid Orpheus voice selected"
        case .marvis:
            return "Invalid Marvis voice selected: "
        }
    }
}

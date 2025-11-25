//
//  TTSProvider.swift
//  MLXAudio
//
//  Unified TTS provider enum for all platforms.
//  Single source of truth for provider selection and configuration.
//

import Foundation

/// Available TTS providers
public enum TTSProvider: String, CaseIterable, Identifiable, Sendable {
    case kokoro
    case orpheus
    case marvis
    case outetts

    public var id: String { rawValue }

    // MARK: - Display Properties

    /// Human-readable name for UI display
    public var displayName: String {
        switch self {
            case .outetts: "OuteTTS"
            default: rawValue.capitalized
        }
    }

    /// Description of the provider's capabilities
    public var description: String {
        switch self {
        case .kokoro:
            return "Fast, lightweight TTS with many voices"
        case .orpheus:
            return "High quality with emotional expressions"
        case .marvis:
            return "Advanced conversational TTS with streaming"
        case .outetts:
            return "TTS with speaker profiles"
        }
    }

    /// Status message shown in the UI (warnings, tips, etc.)
    public var statusMessage: String {
        switch self {
        case .kokoro:
            return ""
        case .orpheus:
            return "Supports expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
        case .marvis:
            return "Marvis: Advanced conversational TTS with streaming support.\n\nNote: Downloads model weights on first use."
        case .outetts:
            return "OuteTTS: Supports custom speaker profiles."
        }
    }

    // MARK: - Feature Flags

    /// Whether this provider supports streaming audio generation
    public var supportsStreaming: Bool {
        self == .marvis
    }

    /// Whether this provider supports speed adjustment
    public var supportsSpeed: Bool {
        self == .kokoro
    }

    /// Whether this provider supports emotional expressions
    public var supportsExpressions: Bool {
        self == .orpheus
    }

    /// Whether this provider supports quality level selection
    public var supportsQualityLevels: Bool {
        self == .marvis
    }

    public var supportsReferenceAudio: Bool {
        self == .outetts
    }

    // MARK: - Voice Management

    /// Default voice ID for this provider
    public var defaultVoiceID: String {
        switch self {
        case .kokoro:
            return "af_heart"
        case .orpheus:
            return "dan"
        case .marvis:
            return "conversational_a"
        case .outetts:
            return "default"
        }
    }

    /// All available voices for this provider
    public var availableVoices: [Voice] {
        switch self {
        case .kokoro:
            return Self.kokoroVoices
        case .orpheus:
            return Self.orpheusVoices
        case .marvis:
            return Self.marvisVoices
        case .outetts:
            return Self.outeTTSVoices
        }
    }

    /// Validate if a voice ID is valid for this provider
    public func validateVoice(_ voiceID: String) -> Bool {
        availableVoices.contains { $0.id == voiceID }
    }

    // MARK: - Voice Lists

    /// Kokoro voice definitions
    private static let kokoroVoices: [Voice] = [
        // American Female
        Voice.fromKokoroID("af_alloy"),
        Voice.fromKokoroID("af_aoede"),
        Voice.fromKokoroID("af_bella"),
        Voice.fromKokoroID("af_heart"),
        Voice.fromKokoroID("af_jessica"),
        Voice.fromKokoroID("af_kore"),
        Voice.fromKokoroID("af_nicole"),
        Voice.fromKokoroID("af_nova"),
        Voice.fromKokoroID("af_river"),
        Voice.fromKokoroID("af_sarah"),
        Voice.fromKokoroID("af_sky"),
        // American Male
        Voice.fromKokoroID("am_adam"),
        Voice.fromKokoroID("am_echo"),
        Voice.fromKokoroID("am_eric"),
        Voice.fromKokoroID("am_fenrir"),
        Voice.fromKokoroID("am_liam"),
        Voice.fromKokoroID("am_michael"),
        Voice.fromKokoroID("am_onyx"),
        Voice.fromKokoroID("am_puck"),
        Voice.fromKokoroID("am_santa"),
        // British Female
        Voice.fromKokoroID("bf_alice"),
        Voice.fromKokoroID("bf_emma"),
        Voice.fromKokoroID("bf_isabella"),
        Voice.fromKokoroID("bf_lily"),
        // British Male
        Voice.fromKokoroID("bm_daniel"),
        Voice.fromKokoroID("bm_fable"),
        Voice.fromKokoroID("bm_george"),
        Voice.fromKokoroID("bm_lewis"),
        // Spanish
        Voice.fromKokoroID("ef_dora"),
        Voice.fromKokoroID("em_alex"),
        // French
        Voice.fromKokoroID("ff_siwis"),
        // Hindi
        Voice.fromKokoroID("hf_alpha"),
        Voice.fromKokoroID("hf_beta"),
        Voice.fromKokoroID("hm_omega"),
        Voice.fromKokoroID("hm_psi"),
        // Italian
        Voice.fromKokoroID("if_sara"),
        Voice.fromKokoroID("im_nicola"),
        // Japanese
        Voice.fromKokoroID("jf_alpha"),
        Voice.fromKokoroID("jf_gongitsune"),
        Voice.fromKokoroID("jf_nezumi"),
        Voice.fromKokoroID("jf_tebukuro"),
        Voice.fromKokoroID("jm_kumo"),
        // Portuguese
        Voice.fromKokoroID("pf_dora"),
        Voice.fromKokoroID("pm_alex"),
        Voice.fromKokoroID("pm_santa"),
        // Chinese
        Voice.fromKokoroID("zf_xiaobei"),
        Voice.fromKokoroID("zf_xiaoni"),
        Voice.fromKokoroID("zf_xiaoxiao"),
        Voice.fromKokoroID("zf_xiaoyi"),
        Voice.fromKokoroID("zm_yunjian"),
        Voice.fromKokoroID("zm_yunxi"),
        Voice.fromKokoroID("zm_yunxia"),
        Voice.fromKokoroID("zm_yunyang"),
    ]

    /// Orpheus voice definitions
    private static let orpheusVoices: [Voice] = [
        Voice(id: "tara", displayName: "Tara", languageCode: "en-US"),
        Voice(id: "leah", displayName: "Leah", languageCode: "en-US"),
        Voice(id: "jess", displayName: "Jess", languageCode: "en-US"),
        Voice(id: "leo", displayName: "Leo", languageCode: "en-US"),
        Voice(id: "dan", displayName: "Dan", languageCode: "en-GB"),
        Voice(id: "mia", displayName: "Mia", languageCode: "en-US"),
        Voice(id: "zac", displayName: "Zac", languageCode: "en-US"),
        Voice(id: "zoe", displayName: "Zoe", languageCode: "en-US"),
    ]

    /// Marvis voice definitions
    private static let marvisVoices: [Voice] = [
        Voice.fromMarvisID("conversational_a"),
        Voice.fromMarvisID("conversational_b"),
    ]

    /// OuteTTS voice definitions (supports custom speaker profiles)
    private static let outeTTSVoices: [Voice] = [
        Voice(id: "default", displayName: "Default", languageCode: "en-US"),
        Voice(id: "custom", displayName: "Custom (Reference Audio)", languageCode: "en-US"),
    ]
}

// SesameVoiceManager.swift
// Voice prompt management for Sesame TTS
// Based on Python voice management system

import Foundation
import MLX

/// Voice configuration for Sesame TTS
/// Equivalent to Python's VoiceConfig
public struct VoiceConfig {
    public let name: String
    public let description: String
    public let language: String
    public let gender: String
    public let age: String
    public let accent: String

    public init(
        name: String,
        description: String,
        language: String = "en",
        gender: String = "neutral",
        age: String = "adult",
        accent: String = "neutral"
    ) {
        self.name = name
        self.description = description
        self.language = language
        self.gender = gender
        self.age = age
        self.accent = accent
    }
}

/// Voice prompt segment
/// Equivalent to Python's VoicePrompt
public struct VoicePrompt {
    public let speaker: Int
    public let text: String
    public let audioTokens: [Int]? // Optional pre-computed audio tokens
    public let isReference: Bool

    public init(
        speaker: Int,
        text: String,
        audioTokens: [Int]? = nil,
        isReference: Bool = false
    ) {
        self.speaker = speaker
        self.text = text
        self.audioTokens = audioTokens
        self.isReference = isReference
    }
}

/// Voice manager for Sesame TTS
/// Equivalent to Python's VoiceManager
public class SesameVoiceManager {
    private var voiceConfigs: [String: VoiceConfig] = [:]
    private var voicePrompts: [String: [VoicePrompt]] = [:]
    private let tokenizer: SesameTokenizer?

    public init(tokenizer: SesameTokenizer? = nil) {
        self.tokenizer = tokenizer
        setupDefaultVoices()
    }

    /// Set up default voice configurations
    /// Following Python's default voice setup
    private func setupDefaultVoices() {
        // Conversational voices
        voiceConfigs["conversational_a"] = VoiceConfig(
            name: "conversational_a",
            description: "Natural conversational voice A",
            language: "en",
            gender: "neutral",
            age: "adult",
            accent: "neutral"
        )

        voiceConfigs["conversational_b"] = VoiceConfig(
            name: "conversational_b",
            description: "Natural conversational voice B",
            language: "en",
            gender: "neutral",
            age: "adult",
            accent: "neutral"
        )

        // Professional voices
        voiceConfigs["professional_male"] = VoiceConfig(
            name: "professional_male",
            description: "Professional male voice",
            language: "en",
            gender: "male",
            age: "adult",
            accent: "neutral"
        )

        voiceConfigs["professional_female"] = VoiceConfig(
            name: "professional_female",
            description: "Professional female voice",
            language: "en",
            gender: "female",
            age: "adult",
            accent: "neutral"
        )

        // Casual voices
        voiceConfigs["casual_young"] = VoiceConfig(
            name: "casual_young",
            description: "Casual young adult voice",
            language: "en",
            gender: "neutral",
            age: "young",
            accent: "casual"
        )

        // Set up default prompts for each voice
        setupDefaultPrompts()
    }

    /// Set up default prompts for voices
    /// Following Python's prompt setup
    private func setupDefaultPrompts() {
        // Conversational A prompts
        voicePrompts["conversational_a"] = [
            VoicePrompt(
                speaker: 0,
                text: "Hello, I'm here to help you with anything you need.",
                isReference: true
            ),
            VoicePrompt(
                speaker: 0,
                text: "That's interesting! Can you tell me more about that?",
                isReference: false
            )
        ]

        // Conversational B prompts
        voicePrompts["conversational_b"] = [
            VoicePrompt(
                speaker: 1,
                text: "Hi there! What would you like to talk about today?",
                isReference: true
            ),
            VoicePrompt(
                speaker: 1,
                text: "I understand. Let me think about the best way to help you.",
                isReference: false
            )
        ]

        // Professional voices have more formal prompts
        voicePrompts["professional_male"] = [
            VoicePrompt(
                speaker: 2,
                text: "Good day. I'm pleased to assist you with your inquiry.",
                isReference: true
            ),
            VoicePrompt(
                speaker: 2,
                text: "I appreciate you bringing this to my attention.",
                isReference: false
            )
        ]

        voicePrompts["professional_female"] = [
            VoicePrompt(
                speaker: 3,
                text: "Hello, I'm here to provide you with professional assistance.",
                isReference: true
            ),
            VoicePrompt(
                speaker: 3,
                text: "Thank you for your patience. I'll address this matter promptly.",
                isReference: false
            )
        ]

        voicePrompts["casual_young"] = [
            VoicePrompt(
                speaker: 4,
                text: "Hey! What's up? I'm totally here to chat and help out!",
                isReference: true
            ),
            VoicePrompt(
                speaker: 4,
                text: "Oh cool! That sounds awesome. Tell me everything!",
                isReference: false
            )
        ]
    }

    /// Get voice configuration by name
    /// - Parameter name: Voice name
    /// - Returns: Voice configuration or nil if not found
    public func getVoiceConfig(name: String) -> VoiceConfig? {
        return voiceConfigs[name]
    }

    /// Get available voice names
    /// - Returns: Array of voice names
    public func getAvailableVoices() -> [String] {
        return Array(voiceConfigs.keys).sorted()
    }

    /// Get voice prompts for a specific voice
    /// - Parameter voiceName: Name of the voice
    /// - Returns: Array of voice prompts
    public func getVoicePrompts(voiceName: String) -> [VoicePrompt] {
        return voicePrompts[voiceName] ?? []
    }

    /// Add custom voice configuration
    /// - Parameters:
    ///   - config: Voice configuration
    ///   - prompts: Optional voice prompts
    public func addVoice(config: VoiceConfig, prompts: [VoicePrompt] = []) {
        voiceConfigs[config.name] = config
        if !prompts.isEmpty {
            voicePrompts[config.name] = prompts
        }
    }

    /// Convert voice prompts to Segment format for Model wrapper
    /// - Parameter voiceName: Name of the voice
    /// - Returns: Array of Segment objects
    public func getVoiceSegments(voiceName: String) -> [Segment] {
        let prompts = getVoicePrompts(voiceName: voiceName)
        return prompts.map { prompt in
            Segment(speaker: prompt.speaker, text: prompt.text, audio: nil)
        }
    }

    /// Get default voice segments (fallback)
    /// - Returns: Default conversational segments
    public func getDefaultSegments() -> [Segment] {
        return getVoiceSegments(voiceName: "conversational_a")
    }

    /// Validate voice exists
    /// - Parameter voiceName: Name of the voice to validate
    /// - Returns: True if voice exists
    public func validateVoice(voiceName: String) -> Bool {
        return voiceConfigs[voiceName] != nil
    }

    /// Get voice description
    /// - Parameter voiceName: Name of the voice
    /// - Returns: Voice description or empty string
    public func getVoiceDescription(voiceName: String) -> String {
        return voiceConfigs[voiceName]?.description ?? "Unknown voice"
    }

    /// Export voice configuration as dictionary
    /// - Parameter voiceName: Name of the voice
    /// - Returns: Dictionary representation
    public func exportVoice(voiceName: String) -> [String: Any]? {
        guard let config = voiceConfigs[voiceName] else { return nil }

        return [
            "name": config.name,
            "description": config.description,
            "language": config.language,
            "gender": config.gender,
            "age": config.age,
            "accent": config.accent,
            "prompts": voicePrompts[voiceName]?.map { prompt in
                [
                    "speaker": prompt.speaker,
                    "text": prompt.text,
                    "isReference": prompt.isReference
                ] as [String: Any]
            } ?? []
        ]
    }

    /// Import voice configuration from dictionary
    /// - Parameters:
    ///   - voiceData: Dictionary representation
    ///   - voiceName: Name for the voice
    public func importVoice(voiceData: [String: Any], voiceName: String) {
        guard let description = voiceData["description"] as? String,
              let language = voiceData["language"] as? String,
              let gender = voiceData["gender"] as? String,
              let age = voiceData["age"] as? String,
              let accent = voiceData["accent"] as? String else {
            return
        }

        let config = VoiceConfig(
            name: voiceName,
            description: description,
            language: language,
            gender: gender,
            age: age,
            accent: accent
        )

        var prompts: [VoicePrompt] = []
        if let promptsData = voiceData["prompts"] as? [[String: Any]] {
            for promptData in promptsData {
                if let speaker = promptData["speaker"] as? Int,
                   let text = promptData["text"] as? String,
                   let isReference = promptData["isReference"] as? Bool {
                    prompts.append(VoicePrompt(
                        speaker: speaker,
                        text: text,
                        isReference: isReference
                    ))
                }
            }
        }

        addVoice(config: config, prompts: prompts)
    }
}

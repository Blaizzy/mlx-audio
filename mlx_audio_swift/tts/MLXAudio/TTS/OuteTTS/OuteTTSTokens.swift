//
//  OuteTTSTokens.swift
//  MLXAudio
//
//  Special tokens for OuteTTS text and audio processing
//  Ported from: mlx_audio/tts/models/outetts/tokens.py
//

import Foundation

/// Special tokens used for OuteTTS text and audio processing
public struct OuteTTSSpecialTokens {
    // Sequence markers
    public let bos = "<|im_start|>"
    public let eos = "<|im_end|>"

    // Audio code tokens (format strings)
    public let c1 = "<|c1_%d|>"  // Codebook 1 token
    public let c2 = "<|c2_%d|>"  // Codebook 2 token

    // Text markers
    public let textStart = "<|text_start|>"
    public let textEnd = "<|text_end|>"

    // Voice characteristic markers
    public let voiceCharacteristicStart = "<|voice_characteristic_start|>"
    public let voiceCharacteristicEnd = "<|voice_characteristic_end|>"

    // Emotion markers
    public let emotionStart = "<|emotion_start|>"
    public let emotionEnd = "<|emotion_end|>"

    // Audio markers
    public let audioStart = "<|audio_start|>"
    public let audioEnd = "<|audio_end|>"

    // Time token (format string with 2 decimal places)
    public let time = "<|t_%.2f|>"

    // Code marker
    public let code = "<|code|>"

    // Audio feature tokens (format strings)
    public let energy = "<|energy_%d|>"
    public let spectralCentroid = "<|spectral_centroid_%d|>"
    public let pitch = "<|pitch_%d|>"

    // Word markers
    public let wordStart = "<|word_start|>"
    public let wordEnd = "<|word_end|>"

    // Feature markers
    public let features = "<|features|>"
    public let globalFeaturesStart = "<|global_features_start|>"
    public let globalFeaturesEnd = "<|global_features_end|>"

    public init() {}

    /// Format a codebook 1 token
    public func formatC1(_ value: Int) -> String {
        return String(format: c1, value)
    }

    /// Format a codebook 2 token
    public func formatC2(_ value: Int) -> String {
        return String(format: c2, value)
    }

    /// Format a time token
    public func formatTime(_ duration: Double) -> String {
        return String(format: time, duration)
    }

    /// Format an energy token
    public func formatEnergy(_ value: Int) -> String {
        return String(format: energy, value)
    }

    /// Format a spectral centroid token
    public func formatSpectralCentroid(_ value: Int) -> String {
        return String(format: spectralCentroid, value)
    }

    /// Format a pitch token
    public func formatPitch(_ value: Int) -> String {
        return String(format: pitch, value)
    }
}

/// Audio features for a word or segment
public struct OuteTTSAudioFeatures: Codable, Sendable {
    public var energy: Int
    public var spectralCentroid: Int
    public var pitch: Int

    enum CodingKeys: String, CodingKey {
        case energy
        case spectralCentroid = "spectral_centroid"
        case pitch
    }

    public init(energy: Int = 0, spectralCentroid: Int = 0, pitch: Int = 0) {
        self.energy = energy
        self.spectralCentroid = spectralCentroid
        self.pitch = pitch
    }
}

/// Word data with audio codes and features
public struct OuteTTSWordData: Codable, Sendable {
    public var word: String
    public var duration: Double
    public var c1: [Int]
    public var c2: [Int]
    public var features: OuteTTSAudioFeatures

    public init(word: String, duration: Double, c1: [Int], c2: [Int], features: OuteTTSAudioFeatures) {
        self.word = word
        self.duration = duration
        self.c1 = c1
        self.c2 = c2
        self.features = features
    }
}

/// Speaker profile
public struct OuteTTSSpeakerProfile: Codable, Sendable {
    public var text: String
    public var words: [OuteTTSWordData]
    public var globalFeatures: OuteTTSAudioFeatures

    enum CodingKeys: String, CodingKey {
        case text
        case words
        case globalFeatures = "global_features"
    }

    public init(text: String, words: [OuteTTSWordData], globalFeatures: OuteTTSAudioFeatures) {
        self.text = text
        self.words = words
        self.globalFeatures = globalFeatures
    }

    /// Load speaker profile from JSON file
    public static func load(from path: String) throws -> OuteTTSSpeakerProfile {
        let expandedPath = NSString(string: path).expandingTildeInPath
        let url = URL(fileURLWithPath: expandedPath)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(OuteTTSSpeakerProfile.self, from: data)
    }

    /// Save speaker profile to JSON file
    public func save(to path: String) throws {
        let expandedPath = NSString(string: path).expandingTildeInPath
        let url = URL(fileURLWithPath: expandedPath)

        // Create directory if needed
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
}

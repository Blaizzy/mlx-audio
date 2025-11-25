//
//  Constants.swift
//  MLXAudio
//
//  Centralized constants for magic numbers and configuration values.
//

import Foundation

/// TTS library constants
public enum TTSConstants {
    // MARK: - Audio

    public enum Audio {
        /// Default sample rate for Kokoro TTS (Hz)
        public static let kokoroSampleRate = 24000

        /// Default sample rate for Orpheus TTS (Hz)
        public static let orpheusSampleRate = 24000

        /// Default sample rate for Marvis TTS (Hz)
        public static let marvisSampleRate = 24000

        /// eSpeak-NG sample rate (Hz)
        public static let espeakSampleRate = 22050

        /// Size of audio buffer chunks for processing
        public static let bufferChunkSize = 32768

        /// Playback monitor timer interval (seconds)
        public static let playbackMonitorInterval: TimeInterval = 0.2

        /// Volume boost factor for playback
        public static let volumeBoostFactor: Float = 1.25

        /// Maximum audio sample value (for clipping prevention)
        public static let maxSampleValue: Float = 0.98
    }

    // MARK: - Memory

    public enum Memory {
        /// GPU cache limit for MLX (bytes) - 20 MB
        public static let gpuCacheLimit = 20 * 1024 * 1024
    }

    // MARK: - Timing

    public enum Timing {
        /// Maximum monitoring duration before timeout (seconds) - 60s
        public static let maxMonitoringDuration: TimeInterval = 60.0

        /// Default streaming interval for Marvis (seconds)
        public static let defaultStreamingInterval: Double = 0.5
    }

    // MARK: - Speed

    public enum Speed {
        /// Minimum speed multiplier
        public static let minimum: Float = 0.5

        /// Maximum speed multiplier
        public static let maximum: Float = 2.0

        /// Default speed multiplier
        public static let `default`: Float = 1.0

        /// Speed slider step size
        public static let step: Float = 0.1
    }

    // MARK: - Generation

    public enum Generation {
        /// Maximum sequence length for generation
        public static let maxSequenceLength = 2048

        /// Frames at which to perform periodic cleanup
        public static let cleanupInterval = 50

        /// Token interval for streaming updates (based on ~12.5 fps)
        public static func streamingIntervalTokens(for seconds: Double) -> Int {
            Int(seconds * 12.5)
        }
    }

    // MARK: - Model Configuration

    public enum Model {
        /// Number of codebooks for Marvis quality levels
        public enum MarvisCodebooks {
            public static let low = 8
            public static let medium = 16
            public static let high = 24
            public static let maximum = 32
        }
    }

    // MARK: - File Names

    public enum FileNames {
        /// Default output filename for Kokoro
        public static let kokoroOutput = "kokoro_output.wav"

        /// Default output filename for Orpheus
        public static let orpheusOutput = "orpheus_output.wav"

        /// Default output filename for Marvis
        public static let marvisOutput = "marvis_output.wav"
    }
}

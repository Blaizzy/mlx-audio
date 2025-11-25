//
//  Log.swift
//  MLXAudio
//
//  Structured logging utilities using os.log.
//

import Foundation
import os

/// Centralized logging for MLX Audio
///
/// Usage:
/// ```swift
/// Log.tts.debug("Starting generation for text: \(text, privacy: .private)")
/// Log.audio.error("Playback failed: \(error.localizedDescription)")
/// Log.model.info("Model loaded successfully")
/// ```
public enum Log {
    private static let subsystem = Bundle.main.bundleIdentifier ?? "com.mlx.audio"

    /// Logging for audio playback and processing
    public static let audio = Logger(subsystem: subsystem, category: "audio")

    /// Logging for TTS generation
    public static let tts = Logger(subsystem: subsystem, category: "tts")

    /// Logging for model loading and management
    public static let model = Logger(subsystem: subsystem, category: "model")

    /// Logging for UI events
    public static let ui = Logger(subsystem: subsystem, category: "ui")

    /// Logging for performance metrics
    public static let perf = Logger(subsystem: subsystem, category: "performance")
}

// MARK: - Convenience Extensions

extension Logger {
    /// Log a timing measurement
    func timing(_ label: String, duration: TimeInterval) {
        self.info("\(label): \(String(format: "%.3f", duration))s")
    }

    /// Log a real-time factor
    func rtf(_ label: String, rtf: Double) {
        self.info("\(label) RTF: \(String(format: "%.2f", rtf))x")
    }
}

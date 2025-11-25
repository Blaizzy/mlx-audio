//
//  AudioSessionManager.swift
//  MLXAudio
//
//  iOS-specific audio session management
//

#if os(iOS)
import AVFoundation

/// Configures AVAudioSession for iOS playback.
public enum AudioSessionManager {

    /// Configure the audio session for playback
    ///
    /// Sets up AVAudioSession with playback category and duck/mix options.
    /// Call once at app launch.
    public static func configure() {
        do {
            // Use .playback category to ensure audio plays even when device is in silent mode
            // .mixWithOthers allows audio to play alongside other apps
            // .duckOthers reduces volume of other audio when this app plays
            try AVAudioSession.sharedInstance().setCategory(
                .playback,
                mode: .default,
                options: [.duckOthers, .mixWithOthers]
            )
            try AVAudioSession.sharedInstance().setActive(true)

            let currentRoute = AVAudioSession.sharedInstance().currentRoute
            Log.audio.debug("Audio session configured, output: \(currentRoute.outputs.first?.portName ?? "unknown")")
        } catch {
            Log.audio.error("Audio session setup failed: \(error.localizedDescription)")
        }
    }
}
#endif

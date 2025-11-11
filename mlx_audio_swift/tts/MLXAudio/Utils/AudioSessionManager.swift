//
//  AudioSessionManager.swift
//   MLXAudio
//
//  Created by Sachin Desai on 5/17/25.
//

import Foundation
import AVFoundation
#if os(iOS)
import UIKit
#endif

/// A platform-agnostic audio session manager that handles platform differences between iOS and macOS
public class AudioSessionManager {

    /// Singleton instance
    public static let shared = AudioSessionManager()

    /// Private initializer for singleton pattern
    private init() {}

    /// Set up the audio session with appropriate categories
    public func setupAudioSession() {
        #if os(iOS)
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

            // Log the current audio route for debugging
            let currentRoute = AVAudioSession.sharedInstance().currentRoute
            print("Audio output route: \(currentRoute.outputs.first?.portName ?? "unknown")")
        } catch {
            print("Audio session setup failed: \(error)")
        }
        #endif
        // No equivalent action needed for macOS
    }

    /// Reset the audio session
    public func resetAudioSession() {
        #if os(iOS)
        do {
            try AVAudioSession.sharedInstance().setActive(false)
            try AVAudioSession.sharedInstance().setActive(true)
            try AVAudioSession.sharedInstance().setCategory(
                .playback,
                mode: .default,
                options: [.duckOthers, .mixWithOthers]
            )
        } catch {
            print("Failed to reset audio session: \(error)")
        }
        #endif
        // No equivalent action needed for macOS
    }

    /// Register for memory warnings
    public func registerForMemoryWarnings(target: Any, selector: Selector) {
        #if os(iOS)
        NotificationCenter.default.addObserver(
            target,
            selector: selector,
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
        #endif
    }

    /// Deactivate the audio session
    public func deactivateAudioSession() {
        #if os(iOS)
        do {
            try AVAudioSession.sharedInstance().setActive(false)
        } catch {
            print("Failed to deactivate audio session: \(error)")
        }
        #endif
        // No equivalent action needed for macOS
    }
}

//
//  AudioFilePlayer.swift
//  MLXAudio
//
//  Plays audio files using AVPlayer.
//  Used by views for replaying saved audio with progress tracking.
//

import AVFoundation
import Foundation

/// Plays audio files with progress tracking
@Observable
@MainActor
final class AudioFilePlayer {
    // MARK: - Public Properties

    /// Whether audio is currently playing
    var isPlaying: Bool = false

    /// Current playback position in seconds
    var currentTime: TimeInterval = 0

    /// Total duration of the loaded audio in seconds
    var duration: TimeInterval = 0

    /// URL of the currently loaded audio file
    var currentAudioURL: URL?

    // MARK: - Private Properties

    @ObservationIgnored private var player: AVPlayer?
    @ObservationIgnored private var timeObserver: Any?
    @ObservationIgnored private var statusObservation: NSKeyValueObservation?
    @ObservationIgnored private var durationObservation: NSKeyValueObservation?

    // MARK: - Initialization

    init() {}

    isolated deinit {
        stop()
    }

    // MARK: - Playback Control

    /// Load an audio file for playback
    /// - Parameter url: URL of the audio file to load
    func loadAudio(from url: URL) {
        // Stop any existing playback
        stop()

        // Create new player
        let item = AVPlayerItem(url: url)
        player = AVPlayer(playerItem: item)

        // Update state
        currentAudioURL = url
        currentTime = 0

        // Observe duration when ready
        durationObservation = item.observe(\.duration, options: [.new]) { [weak self] item, _ in
            Task { @MainActor [weak self] in
                let seconds = item.duration.seconds
                if seconds.isFinite {
                    self?.duration = seconds
                }
            }
        }

        // Observe playback status
        statusObservation = item.observe(\.status, options: [.new]) { [weak self] item, _ in
            Task { @MainActor [weak self] in
                if item.status == .readyToPlay {
                    let seconds = item.duration.seconds
                    if seconds.isFinite {
                        self?.duration = seconds
                    }
                    Log.audio.debug("Loaded audio: \(url.lastPathComponent), duration: \(self?.duration ?? 0)s")
                } else if item.status == .failed {
                    Log.audio.error("Failed to load audio: \(item.error?.localizedDescription ?? "unknown")")
                }
            }
        }

        // Add periodic time observer
        let interval = CMTime(seconds: 0.1, preferredTimescale: 600)
        timeObserver = player?.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            Task { @MainActor [weak self] in
                self?.currentTime = time.seconds
            }
        }

        // Observe when playback finishes
        NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: item,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.isPlaying = false
                self?.currentTime = 0
                self?.player?.seek(to: .zero)
                Log.audio.debug("Playback finished")
            }
        }
    }

    /// Start or resume playback
    func play() {
        guard player != nil else { return }

        player?.play()
        isPlaying = true

        Log.audio.debug("Playback started")
    }

    /// Pause playback
    func pause() {
        player?.pause()
        isPlaying = false

        Log.audio.debug("Playback paused")
    }

    /// Toggle between play and pause
    func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }

    /// Stop playback and reset to beginning
    func stop() {
        player?.pause()
        isPlaying = false
        currentTime = 0

        // Remove observers
        if let timeObserver {
            player?.removeTimeObserver(timeObserver)
        }
        timeObserver = nil
        statusObservation?.invalidate()
        statusObservation = nil
        durationObservation?.invalidate()
        durationObservation = nil

        NotificationCenter.default.removeObserver(self, name: .AVPlayerItemDidPlayToEndTime, object: player?.currentItem)

        player = nil

        Log.audio.debug("Playback stopped")
    }

    /// Seek to a specific time
    /// - Parameter time: Target time in seconds
    func seek(to time: TimeInterval) {
        guard let player else { return }
        let targetTime = CMTime(seconds: max(0, min(time, duration)), preferredTimescale: 600)
        player.seek(to: targetTime)
        currentTime = time
    }
}

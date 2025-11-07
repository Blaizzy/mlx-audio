//
//  AudioPlayerManager.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import Foundation
import AVFoundation
import Combine

class AudioPlayerManager: NSObject, ObservableObject {
    // Published properties for UI binding
    @Published var isPlaying: Bool = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var currentAudioURL: URL?

    private var player: AVAudioPlayer?
    private var timer: Timer?

    override init() {
        super.init()
    }

    deinit {
        stop()
    }

    // MARK: - Playback Control

    func loadAudio(from url: URL) {
        do {
            // Stop any existing playback
            stop()

            // Create new player
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.prepareToPlay()

            // Update state
            currentAudioURL = url
            duration = player?.duration ?? 0
            currentTime = 0

        } catch {
            print("Failed to load audio: \(error.localizedDescription)")
            currentAudioURL = nil
            duration = 0
            currentTime = 0
        }
    }

    func play() {
        guard let player = player else { return }

        player.play()
        isPlaying = true
        startTimer()
    }

    func pause() {
        player?.pause()
        isPlaying = false
        stopTimer()
    }

    func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }

    func stop() {
        player?.stop()
        isPlaying = false
        stopTimer()
        currentTime = 0
    }

    func seek(to time: TimeInterval) {
        guard let player = player else { return }
        player.currentTime = max(0, min(time, duration))
        currentTime = player.currentTime
    }

    // MARK: - Timer Management

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let player = self.player else { return }
            self.currentTime = player.currentTime
        }
        timer?.tolerance = 0.05
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

// MARK: - AVAudioPlayerDelegate

extension AudioPlayerManager: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        stopTimer()
        currentTime = 0
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("Audio decode error: \(error?.localizedDescription ?? "unknown")")
        isPlaying = false
        stopTimer()
    }
}

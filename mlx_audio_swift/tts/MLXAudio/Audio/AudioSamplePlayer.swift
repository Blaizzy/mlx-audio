//
//  AudioSamplePlayer.swift
//  MLXAudio
//
//  Plays raw audio samples using AVAudioEngine.
//  Used by TTS engines to play generated audio.
//

@preconcurrency import AVFoundation
import Foundation

/// Actor for thread-safe buffer tracking
private actor BufferTracker {
    private var scheduled: Int = 0
    private var completed: Int = 0

    func reset() {
        scheduled = 0
        completed = 0
    }

    func incrementScheduled() {
        scheduled += 1
    }

    func incrementCompleted() -> Bool {
        completed += 1
        return completed == scheduled && scheduled > 0
    }

    func areAllCompleted() -> Bool {
        completed == scheduled && scheduled > 0
    }

    var pendingCount: Int {
        max(0, scheduled - completed)
    }
}

/// Plays raw audio samples for TTS engines
@Observable
@MainActor
final class AudioSamplePlayer {
    // MARK: - Public State

    /// Whether audio is currently playing
    private(set) var isPlaying: Bool = false

    /// Number of samples currently queued for playback
    private(set) var queuedSampleCount: Int = 0

    // MARK: - Private Properties

    @ObservationIgnored private var engine: AVAudioEngine!
    @ObservationIgnored private var playerNode: AVAudioPlayerNode!
    @ObservationIgnored private var audioFormat: AVAudioFormat!
    @ObservationIgnored private let bufferTracker = BufferTracker()
    @ObservationIgnored private var playbackMonitorTimer: Timer?
    @ObservationIgnored private var hasStartedPlayback: Bool = false
    @ObservationIgnored private var playbackCompletionContinuation: CheckedContinuation<Void, Never>?

    private let sampleRate: Int

    // MARK: - Initialization

    /// Create an audio engine with the specified sample rate
    /// - Parameter sampleRate: Sample rate in Hz (default: 24000)
    init(sampleRate: Int = TTSConstants.Audio.kokoroSampleRate) {
        self.sampleRate = sampleRate
        setup()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Setup

    private func setup() {
        Log.audio.debug("Setting up audio engine at \(self.sampleRate)Hz")

        engine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        audioFormat = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: 1
        )
        guard audioFormat != nil else {
            Log.audio.error("Failed to create audio format")
            return
        }

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)

        do {
            try engine.start()
            Log.audio.debug("Audio engine started successfully")
        } catch {
            Log.audio.error("Failed to start audio engine: \(error.localizedDescription)")
        }
    }

    // MARK: - Public API

    /// Play audio samples and wait for playback to complete
    /// - Parameters:
    ///   - samples: Float audio samples
    ///   - volumeBoost: Optional volume multiplier (default: 1.25)
    func play(samples: [Float], volumeBoost: Float = TTSConstants.Audio.volumeBoostFactor) async {
        guard !samples.isEmpty else { return }

        await reset()

        guard let buffer = createBuffer(from: samples, volumeBoost: volumeBoost) else {
            Log.audio.error("Failed to create audio buffer")
            return
        }

        await scheduleBuffer(buffer)

        playerNode.play()
        isPlaying = true
        hasStartedPlayback = true

        // Await completion via continuation (resumed by checkIfPlaybackComplete)
        await withCheckedContinuation { continuation in
            self.playbackCompletionContinuation = continuation
            self.startPlaybackMonitoring()
        }
    }

    /// Wait for current playback to complete (used after enqueuing streaming audio)
    func awaitCompletion() async {
        guard isPlaying else { return }

        await withCheckedContinuation { continuation in
            self.playbackCompletionContinuation = continuation
            self.startPlaybackMonitoring()
        }
    }

    /// Enqueue audio samples for streaming playback
    /// - Parameters:
    ///   - samples: Float audio samples to enqueue
    ///   - prebufferSeconds: Seconds to buffer before starting playback (0 for immediate)
    func enqueue(samples: [Float], prebufferSeconds: Double = 0) {
        guard !samples.isEmpty else { return }

        resetEngineIfNeeded()

        // Schedule in small slices for smoother streaming
        let sliceSamples = max(1, Int(0.03 * Double(sampleRate))) // 30ms slices
        var offset = 0

        while offset < samples.count {
            let remaining = samples.count - offset
            let thisLength = min(sliceSamples, remaining)
            let slice = Array(samples[offset..<(offset + thisLength)])

            guard let buffer = createBuffer(from: slice, volumeBoost: 1.0) else {
                offset += thisLength
                continue
            }

            queuedSampleCount += thisLength
            let decrementAmount = thisLength

            Task { await bufferTracker.incrementScheduled() }

            playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.queuedSampleCount = max(0, self.queuedSampleCount - decrementAmount)
                    let allDone = await self.bufferTracker.incrementCompleted()
                    if allDone && !self.playerNode.isPlaying {
                        self.isPlaying = false
                    }
                }
            }

            // Start playback when prebuffer is reached
            if !hasStartedPlayback {
                let prebufferSamples = Int(prebufferSeconds * Double(sampleRate))
                if prebufferSamples == 0 || queuedSampleCount >= prebufferSamples {
                    playerNode.play()
                    hasStartedPlayback = true
                    isPlaying = true

                    // Retry if playback didn't start
                    if !playerNode.isPlaying {
                        Task {
                            try? await Task.sleep(for: .milliseconds(100))
                            self.playerNode.play()
                        }
                    }
                }
            } else if !playerNode.isPlaying {
                playerNode.play()
            }

            offset += thisLength
        }
    }

    /// Stop playback and reset state
    func stop() async {
        stopPlaybackMonitoring()

        // Resume any waiting continuation before stopping
        playbackCompletionContinuation?.resume()
        playbackCompletionContinuation = nil

        // Dispatch to background QoS to avoid priority inversion
        // (audio threads run at Default QoS, so calling from Background avoids inversion)
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .background).async { [playerNode] in
                playerNode?.stop()
                playerNode?.reset()
                continuation.resume()
            }
        }

        await resetBufferCounters()

        isPlaying = false
        hasStartedPlayback = false
        queuedSampleCount = 0

        Log.audio.debug("Audio playback stopped")
    }

    /// Reset the audio engine (useful after interruptions)
    func reset() async {
        await stop()

        // Reconnect components
        if playerNode.engine != nil {
            engine.detach(playerNode)
        }
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)

        // Restart engine
        do {
            try engine.start()
            Log.audio.debug("Audio engine reset successfully")
        } catch {
            Log.audio.error("Failed to restart audio engine: \(error.localizedDescription)")
        }
    }

    // MARK: - Buffer Creation

    private func createBuffer(from samples: [Float], volumeBoost: Float) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            return nil
        }

        buffer.frameLength = frameCount

        guard let channelData = buffer.floatChannelData else {
            return nil
        }

        // Copy samples with volume boost and clipping prevention
        let maxValue = TTSConstants.Audio.maxSampleValue
        for i in 0..<samples.count {
            let boosted = samples[i] * volumeBoost
            channelData[0][i] = min(max(boosted, -maxValue), maxValue)
        }

        return buffer
    }

    private func scheduleBuffer(_ buffer: AVAudioPCMBuffer) async {
        await bufferTracker.incrementScheduled()

        playerNode.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let allDone = await self.bufferTracker.incrementCompleted()
                if allDone {
                    self.checkIfPlaybackComplete()
                }
            }
        }
    }

    // MARK: - Playback Monitoring

    private func startPlaybackMonitoring() {
        stopPlaybackMonitoring()

        playbackMonitorTimer = Timer.scheduledTimer(withTimeInterval: TTSConstants.Audio.playbackMonitorInterval, repeats: true) { [weak self] timer in
            guard let self else {
                timer.invalidate()
                return
            }
            Task { @MainActor in
                self.checkIfPlaybackComplete()
            }
        }

        RunLoop.current.add(playbackMonitorTimer!, forMode: .common)

        // Fallback timeout
        Task {
            try? await Task.sleep(for: .seconds(TTSConstants.Timing.maxMonitoringDuration))
            await MainActor.run {
                if self.playbackMonitorTimer != nil {
                    self.stopPlaybackMonitoring()
                    self.isPlaying = false
                }
            }
        }
    }

    private func stopPlaybackMonitoring() {
        playbackMonitorTimer?.invalidate()
        playbackMonitorTimer = nil
    }

    private func checkIfPlaybackComplete() {
        Task {
            let allCompleted = await bufferTracker.areAllCompleted()
            let isActuallyPlaying = playerNode.isPlaying

            if !isActuallyPlaying && allCompleted {
                stopPlaybackMonitoring()
                isPlaying = false
                hasStartedPlayback = false

                // Resume any waiting continuation
                playbackCompletionContinuation?.resume()
                playbackCompletionContinuation = nil
            }
        }
    }

    // MARK: - Private Helpers

    private func resetBufferCounters() async {
        await bufferTracker.reset()
    }

    private func resetEngineIfNeeded() {
        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                Log.audio.error("Failed to restart audio engine: \(error.localizedDescription)")
            }
        }
    }
}

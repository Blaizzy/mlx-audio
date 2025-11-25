import Foundation
import AVFoundation
import Synchronization

/// Thread-safe counter for tracking queued audio samples
/// Wrapped in a class to allow capturing from actor-isolated context
final class QueuedSamplesCounter: Sendable {
    private let value = Atomic<Int>(0)

    func add(_ amount: Int) -> Int {
        value.add(amount, ordering: .relaxed).newValue
    }

    func subtract(_ amount: Int) {
        _ = value.subtract(amount, ordering: .relaxed)
    }

    func reset() {
        value.store(0, ordering: .relaxed)
    }
}

actor AudioPlayback {
    private let sampleRate: Double
    private let scheduleSliceSeconds: Double = 0.03 // 30ms slices

    private var audioEngine: AVAudioEngine!
    private var playerNode: AVAudioPlayerNode!
    private var audioFormat: AVAudioFormat!
    private let queuedSamples = QueuedSamplesCounter()
    private var hasStartedPlayback: Bool = false

    init(sampleRate: Double) {
        self.sampleRate = sampleRate

        // Set up audio engine inline (can't call actor-isolated methods from init)
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)

        guard let audioFormat else {
            return
        }

        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)

        do {
            try audioEngine.start()
        } catch {
            // Failed to start audio engine
        }
    }

    isolated deinit {
        stop()
    }

    func enqueue(_ samples: [Float], prebufferSeconds: Double) {
        guard let audioFormat else {
            return
        }
        let total = samples.count
        guard total > 0 else {
            return
        }

        let sliceSamples = max(1, Int(scheduleSliceSeconds * sampleRate))
        var offset = 0
        while offset < total {
            let remaining = total - offset
            let thisLen = min(sliceSamples, remaining)

            let frameLength = AVAudioFrameCount(thisLen)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameLength) else { break }
            buffer.frameLength = frameLength
            if let channelData = buffer.floatChannelData {
                for i in 0..<thisLen { channelData[0][i] = samples[offset + i] }
            }

            let currentQueued = queuedSamples.add(Int(frameLength))
            let decAmount = Int(frameLength)
            let counter = queuedSamples
            playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
                counter.subtract(decAmount)
            }

            // Start playback logic
            if !hasStartedPlayback {
                let prebufferSamples = Int(prebufferSeconds * sampleRate)
                // For non-streaming (prebufferSeconds = 0), start immediately
                // For streaming, wait for prebuffer
                if prebufferSamples == 0 || currentQueued >= prebufferSamples {
                    playerNode.play()
                    hasStartedPlayback = true

                    // Retry if playback didn't start
                    if !playerNode.isPlaying {
                        Task { [weak self] in
                            try? await Task.sleep(for: .milliseconds(100))
                            await self?.retryPlayback()
                        }
                    }
                }
            } else if !playerNode.isPlaying {
                playerNode.play()
            }

            offset += thisLen
        }
    }

    private func retryPlayback() {
        if let playerNode, !playerNode.isPlaying {
            playerNode.play()
        }
    }

    func stop() {
        if let playerNode {
            if playerNode.isPlaying {
                playerNode.stop()
            }
            playerNode.reset()
        }
        if let audioEngine, audioEngine.isRunning {
            audioEngine.stop()
        }
        hasStartedPlayback = false
        queuedSamples.reset()
    }

    func reset() {
        stop()

        // Reconnect components
        if let playerNode, playerNode.engine != nil {
            audioEngine.detach(playerNode)
        }
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)

        // Restart engine
        do {
            try audioEngine.start()
        } catch {
            // Failed to restart audio engine
        }
    }
}

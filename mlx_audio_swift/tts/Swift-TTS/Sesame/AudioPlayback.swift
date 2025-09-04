import Foundation
import AVFoundation

final class AudioPlayback {
    private let sampleRate: Double
    private let scheduleSliceSeconds: Double = 0.03 // 30ms slices

    private var audioEngine: AVAudioEngine!
    private var playerNode: AVAudioPlayerNode!
    private var audioFormat: AVAudioFormat!
    private var queuedSamples: Int = 0
    private var hasStartedPlayback: Bool = false

    init(sampleRate: Double) {
        self.sampleRate = sampleRate
        setup()
    }

    deinit {
        stop()
    }

    private func setup() {
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)
        do { try audioEngine.start() } catch { print("Failed to start audio engine: \(error)") }
    }

    func enqueue(_ samples: [Float], prebufferSeconds: Double) {
        guard let audioFormat else { return }
        let total = samples.count
        guard total > 0 else { return }

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

            queuedSamples += Int(frameLength)
            let decAmount = Int(frameLength)
            playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
                guard let self else { return }
                self.queuedSamples = max(0, self.queuedSamples - decAmount)
            }

            if !hasStartedPlayback {
                let prebufferSamples = Int(prebufferSeconds * sampleRate)
                if queuedSamples >= prebufferSamples {
                    playerNode.play()
                    hasStartedPlayback = true
                }
            }

            offset += thisLen
        }
    }

    func stop() {
        if let playerNode, playerNode.isPlaying { playerNode.stop() }
        if let audioEngine, audioEngine.isRunning { audioEngine.stop() }
    }
}


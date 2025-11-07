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
        #if os(iOS)
        AudioSessionManager.shared.setupAudioSession()
        #endif

        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)

        guard let audioFormat = audioFormat else {
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

            queuedSamples += Int(frameLength)
            let decAmount = Int(frameLength)
            playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
                guard let self else { return }
                self.queuedSamples = max(0, self.queuedSamples - decAmount)
            }

            // Start playback logic
            if !hasStartedPlayback {
                let prebufferSamples = Int(prebufferSeconds * sampleRate)
                // For non-streaming (prebufferSeconds = 0), start immediately
                // For streaming, wait for prebuffer
                if prebufferSamples == 0 || queuedSamples >= prebufferSamples {
                    playerNode.play()
                    hasStartedPlayback = true
                    
                    // Retry if playback didn't start (similar to Kokoro)
                    if !playerNode.isPlaying {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                            self?.playerNode.play()
                        }
                    }
                }
            } else if !playerNode.isPlaying {
                // If playback was stopped but hasStartedPlayback is still true, restart it
                playerNode.play()
            }

            offset += thisLen
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
        queuedSamples = 0
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

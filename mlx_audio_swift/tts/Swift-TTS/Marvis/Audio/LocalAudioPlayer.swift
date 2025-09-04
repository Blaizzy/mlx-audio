import AVFoundation
import Foundation

public final class LocalAudioPlayer {
    public let inputSampleRate: Double
    public let inputChannels: AVAudioChannelCount
    public var volume: Float {
        get { player.volume }
        set { player.volume = newValue }
    }

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let sourceFormat: AVAudioFormat
    private let queue = DispatchQueue(label: "LocalAudioPlayer.queue")

    private let drainGroup = DispatchGroup()
    private var pendingBuffers: Int = 0
    private var accepting: Bool = true

    private var started = false
    private var connected = false

    public init(sampleRate: Double, channels: Int = 1) {
        self.inputSampleRate = sampleRate
        self.inputChannels = AVAudioChannelCount(channels)
        guard let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(channels), interleaved: false)
        else { fatalError("Failed to create AVAudioFormat") }
        self.sourceFormat = fmt

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: sourceFormat)
        self.connected = true
    }

    deinit { stop() }

    public func start() throws {
        queue.async {
            do {
                #if os(iOS)
                try self.configureAudioSession()
                #endif

                if !self.engine.isRunning {
                    try self.engine.start()
                }
                if !self.player.isPlaying {
                    self.player.play()
                }
                self.started = true
            } catch {
                print("Error starting audio player: \(error)")
            }
        }
    }

    public func stop(waitForEnd: Bool = false, timeout: TimeInterval? = nil) {
        if waitForEnd {
            queue.sync { accepting = false }

            if let t = timeout {
                _ = drainGroup.wait(timeout: .now() + t)
            } else {
                drainGroup.wait()
            }
        }

        queue.sync {
            if player.isPlaying { player.stop() }
            if engine.isRunning { engine.stop() }
            started = false
            accepting = true
            pendingBuffers = 0
        }
    }

    public func reset() {
        queue.sync {
            player.stop()
            player.reset()
            engine.reset()
            started = false
        }
    }

    public var isRunning: Bool { queue.sync { started && engine.isRunning && player.isPlaying } }

    public func enqueue(samples: [Float]) {
        guard !samples.isEmpty else { return }

        queue.async {
            guard self.connected, self.accepting else { return }

            let frames = AVAudioFrameCount(samples.count / Int(self.inputChannels))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: self.sourceFormat, frameCapacity: frames) else { return }
            buffer.frameLength = frames

            samples.withUnsafeBufferPointer { src in
                guard let base = src.baseAddress else { return }
                let dst = buffer.floatChannelData!
                let n = Int(frames)
                if self.inputChannels == 1 {
                    dst[0].update(from: base, count: n)
                } else {
                    for ch in 0 ..< Int(self.inputChannels) {
                        dst[ch].update(from: base + ch * n, count: n)
                    }
                }
            }

            self.schedule(buffer)

            if !self.started {
                do { try self.start() } catch { print("LocalAudioPlayer start error: \(error)") }
            }
        }
    }

    public func enqueue(buffer: AVAudioPCMBuffer) {
        queue.async {
            guard self.connected, self.accepting else { return }

            self.schedule(buffer)
            if !self.started {
                do { try self.start() } catch { print("LocalAudioPlayer start error: \(error)") }
            }
        }
    }

    private func schedule(_ buffer: AVAudioPCMBuffer) {
        pendingBuffers += 1
        drainGroup.enter()

        player.scheduleBuffer(
            buffer,
            completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            guard let self else { return }
            queue.async {
                self.pendingBuffers -= 1
                self.drainGroup.leave()
            }
        }
    }

    #if os(iOS)
    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try session.setActive(true)
    }
    #endif
}

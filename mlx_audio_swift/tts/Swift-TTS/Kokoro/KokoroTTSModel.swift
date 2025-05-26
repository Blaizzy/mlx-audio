import AVFoundation
import MLX
import SwiftUI
#if os(iOS)
import UIKit
#endif

public class KokoroTTSModel: ObservableObject {
    private var kokoroTTSEngine: KokoroTTS!
    private var audioEngine: AVAudioEngine!
    private var playerNode: AVAudioPlayerNode!
    private var audioFormat: AVAudioFormat!

    // Buffer tracking for reliable playback completion detection
    private var scheduledBufferCount = 0
    private var completedBufferCount = 0
    private let bufferCountLock = NSLock() // Thread safety for buffer counters

    // State management
    private var isGenerating = false
    private var isPlayingAudio = false

    // Published property for UI updates - indicates generation OR playback is in progress
    @Published public var generationInProgress = false

    // A separate property to track if audio is currently playing
    @Published public var isAudioPlaying: Bool = false {
        didSet {
            // Avoid redundant operations for repeated identical values
            if oldValue != isAudioPlaying {
                // Whenever audio playing state changes, update generationInProgress
                // to ensure UI elements stay active during playback
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    if self.isAudioPlaying {
                        // Starting playback
                        self.generationInProgress = true
                        // Ensure internal state is consistent
                        self.isPlayingAudio = true
                        // Ensure monitoring is active if generation is complete
                        if !self.isGenerating && self.playbackMonitorTimer == nil {
                            self.startPlaybackMonitoring()
                        }
                    } else {
                        // Stopping playback
                        // Only set generationInProgress to false if both generation and playback are done
                        if !self.isGenerating {
                            self.generationInProgress = false
                            // Ensure internal state is consistent
                            self.isPlayingAudio = false
                            // Force UI refresh
                            self.objectWillChange.send()
                        }
                    }
                }
            }
        }
    }

    @Published public var audioGenerationTime: TimeInterval = 0

    public init() {
        kokoroTTSEngine = KokoroTTS()
        setupAudioSystem()
    }

    deinit {
         NotificationCenter.default.removeObserver(self)
         cleanupAudioSystem()
     }

    // MARK: - Audio System Setup

    private func setupAudioSystem() {
        print("Setting up audio system")

        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        audioFormat = AVAudioFormat(standardFormatWithSampleRate: Double(KokoroTTS.Constants.sampleRate), channels: 1)
        guard audioFormat != nil else {
            print("Failed to create audio format")
            return
        }

        // Use dedicated audio processing queue to avoid QoS inversions
        let audioQueue = DispatchQueue(label: "com.mlx.audio.processing", qos: .userInteractive)
        audioQueue.sync {
            // Use platform-agnostic AudioSessionManager
            AudioSessionManager.shared.setupAudioSession()

            audioEngine.attach(playerNode)
            audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)

            do {
                try audioEngine.start()
                print("Audio system started successfully")
            } catch {
                print("Failed to start audio engine: \(error)")
            }
        }
    }

    private func cleanupAudioSystem() {
        // Stop player node first, which is the likely source of QoS inversion
        if playerNode.isPlaying {
            playerNode.pause() // Use pause instead of stop to avoid blocking
        }

        // Then stop the audio engine
        if audioEngine.isRunning {
            audioEngine.pause() // Use pause instead of stop to avoid blocking
        }

        // Finally, deactivate the audio session
        AudioSessionManager.shared.deactivateAudioSession()
    }

    private func resetAudioSystem() {
        print("Resetting audio system")

        // Stop player node first to avoid QoS inversion
        if playerNode.isPlaying {
            playerNode.pause() // Use pause instead of stop to avoid blocking
        }

        // Then stop the audio engine
        if audioEngine.isRunning {
            audioEngine.pause() // Use pause instead of stop to avoid blocking
        }

        // Reset audio session using platform-agnostic manager
        AudioSessionManager.shared.resetAudioSession()

        // Reconnect components with proper error handling
        if playerNode.engine != nil {
            audioEngine.detach(playerNode)
        }
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)

        // Restart engine
        do {
            try audioEngine.start()
            print("Audio engine restarted")
        } catch {
            print("Failed to restart audio engine: \(error)")
        }
    }

    public func say(_ text: String, _ voice: TTSVoice, speed: Float = 1.0) {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
         guard !trimmedText.isEmpty else {
             return
         }

         // Reset timing metrics
         audioGenerationTime = 0.0

         // Update UI state immediately
         DispatchQueue.main.async {
             self.generationInProgress = true
             self.objectWillChange.send()
         }

         // Stop any ongoing playback
         if isGenerating || playerNode.isPlaying {
             stopPlayback()

             // We need to give the audio system time to reset
             // This is necessary for the AVAudioEngine to properly shut down
             Task {
                 // Wait briefly for audio system to fully reset
                 try? await Task.sleep(nanoseconds: 300_000_000) // 300ms

                 // Now start the new generation
                 self.startSpeechGeneration(text: trimmedText, voice: voice, speed: speed)
             }
             return
         }

         // No existing playback, start immediately
         startSpeechGeneration(text: trimmedText, voice: voice, speed: speed)
    }

    public func stopPlayback() {
        stopPlaybackMonitoring()
        resetBufferCounters()

        // Reset audio system with proper error handling
        do {
            playerNode.stop()
            playerNode.reset()

            // Additionally reset engine if running
            if audioEngine.isRunning {
                audioEngine.stop()
                try audioEngine.start()
            }
        } catch {
            print("Error resetting audio engine: \(error)")
        }

        // Reset all internal state flags
        isGenerating = false
        isPlayingAudio = false

        // Force UI update on main thread with proper sequencing
        DispatchQueue.main.async {
            // First notify observers of impending change
            self.objectWillChange.send()

            // Then update state properties in the correct order
            self.isAudioPlaying = false
            self.generationInProgress = false

            // Send another notification after state is updated
            self.objectWillChange.send()

            // Add a final verification check with slight delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
                guard let self = self else { return }

                // Check all state flags to ensure consistency
                let allFlagsCorrect = !self.isGenerating &&
                !self.isPlayingAudio &&
                !self.isAudioPlaying &&
                !self.generationInProgress

                // Force reset if any flag is still incorrect
                if !allFlagsCorrect {
                    print("Force resetting all UI states to false - inconsistency detected")

                    // Reset all flags again
                    self.isGenerating = false
                    self.isPlayingAudio = false
                    self.isAudioPlaying = false
                    self.generationInProgress = false

                    // Final notification
                    self.objectWillChange.send()

                    print("All UI states verified to be false")
                }
            }
        }

        // Reset TTS model in background with proper QoS
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self = self else { return }
            self.kokoroTTSEngine.resetModel()
        }
    }

    // MARK: - Buffer Tracking

     private func resetBufferCounters() {
         bufferCountLock.lock()
         defer { bufferCountLock.unlock() }

         scheduledBufferCount = 0
         completedBufferCount = 0
     }

     private func incrementScheduledBufferCount() {
         bufferCountLock.lock()
         defer { bufferCountLock.unlock() }

         scheduledBufferCount += 1
     }

     private func incrementCompletedBufferCount() {
         bufferCountLock.lock()
         defer { bufferCountLock.unlock() }

         completedBufferCount += 1

         // Check if all buffers completed
         if completedBufferCount == scheduledBufferCount && scheduledBufferCount > 0 {

             // Use main thread for UI updates
             DispatchQueue.main.async { [weak self] in
                 guard let self = self else { return }

                 // Only update if we're not generating new content
                 if !self.isGenerating {
                     // Update playback status
                     self.isPlayingAudio = false
                     self.isAudioPlaying = false

                     // Force UI update
                     self.objectWillChange.send()
                 }
             }
         }
     }

    // MARK: - Audio Generation and Playback

    private func startSpeechGeneration(text: String, voice: TTSVoice, speed: Float) {
        // Update internal state
        isGenerating = true

        // Reset buffer counters for the new generation
        resetBufferCounters()

        // Make sure the UI state is also set
        DispatchQueue.main.async {
            self.objectWillChange.send()
            self.generationInProgress = true
        }

        // Reset audio system to ensure clean state
        resetAudioSystem()

        // Start generation timer
        let generationStartTime = Date()

        // Use a local variable to track audio chunks that can be accessed in completion blocks
        var receivedAudioChunks = false

        do {
            // Use streaming by sentence approach
            try kokoroTTSEngine.generateAudio(
                voice: voice,
                text: text,
                speed: speed
            ) { [weak self] audioBuffer in
                guard let self = self else { return }

                // Mark that we've received at least one chunk
                receivedAudioChunks = true

                // Update generation time on first chunk
                if self.audioGenerationTime == 0.0 {
                    self.audioGenerationTime = Date().timeIntervalSince(generationStartTime)
                }

                // Play audio on main thread
                DispatchQueue.main.async {
                    self.playAudioChunk(audioBuffer)
                }
            }

            // After all sentences are processed, update the generation state
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }

                // Mark generation (but not playback) as complete
                // We do this regardless of whether chunks were received yet
                self.isGenerating = false

                // Keep generationInProgress true until playback completes

                // Check if player is actually playing something
                if self.playerNode.isPlaying {
                    // Start continuous monitoring of playback state
                    self.startPlaybackMonitoring()
                } else if receivedAudioChunks {
                    // We received chunks but player is not active
                    // This might happen with very short audio or fast processing
                    self.startPlaybackMonitoring()
                } else {
                    // No chunks received yet and player not active
                    // This is normal - chunks might still be coming

                    // We intentionally don't reset state here as chunks might still arrive
                    // The timer-based monitor or buffer callbacks will handle state cleanup
                }
            }
        } catch {
            // Stop any active monitoring
            stopPlaybackMonitoring()

            // Reset internal state
            isGenerating = false
            isPlayingAudio = false

            // Reset UI state with proper notification
            DispatchQueue.main.async {
                // First notify observers of impending change
                self.objectWillChange.send()

                // Reset all UI state flags
                self.isAudioPlaying = false
                self.generationInProgress = false

                // Final notification
                self.objectWillChange.send()
            }

            // Also reset the audio system to ensure clean state
            resetAudioSystem()
        }
    }

    private func playAudioChunk(_ audioBuffer: MLXArray) {
        // Skip empty chunks
        let audioShape = audioBuffer.shape
        guard !isAudioEmpty(shape: audioShape) else {
            print("Skipping empty audio chunk")
            return
        }

        // Extract audio data
        let (frameCount, audioData) = extractAudioData(from: audioBuffer)

        // Create PCM buffer
        guard let buffer = createAudioBuffer(frameCount: frameCount, audioData: audioData) else {
            print("Failed to create audio buffer")
            return
        }

        // Ensure audio engine is running
        if !audioEngine.isRunning {
            resetAudioSystem()
        }

        // Calculate buffer duration
        let bufferDuration = Double(frameCount) / Double(KokoroTTS.Constants.sampleRate)

        // Increment the scheduled buffer count before scheduling
        incrementScheduledBufferCount()

        // Schedule buffer playback with enhanced completion handling and buffer tracking
        playerNode.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self = self else { return }

            // Increment completed buffer count (thread-safe)
            self.incrementCompletedBufferCount()

            // Dispatch to main thread for UI updates
            DispatchQueue.main.async {
                // First verify if player is actually still playing anything
                let isActuallyPlaying = self.playerNode.isPlaying

                if !isActuallyPlaying {
                    // Check buffer counts for more accurate completion detection
                    self.bufferCountLock.lock()
                    let allBuffersCompleted = self.completedBufferCount == self.scheduledBufferCount
                    self.bufferCountLock.unlock()

                    if allBuffersCompleted {

                        // Directly update playback state
                        self.isPlayingAudio = false
                        self.isAudioPlaying = false

                        // If generation is also done, notify UI of change
                        if !self.isGenerating {
                            // Stop monitoring explicitly to avoid redundant state checks
                            self.stopPlaybackMonitoring()
                            self.objectWillChange.send()
                        }
                    }
                } else if !self.isGenerating {
                    // Generation is complete but audio is still playing
                    // Make sure our monitoring timer is active
                    if self.playbackMonitorTimer == nil {
                        self.startPlaybackMonitoring()
                    }
                }
            }
        }

        // Track audio playback state
        isPlayingAudio = true
        isAudioPlaying = true

        // Start playback if needed
        if !playerNode.isPlaying {
            playerNode.play()

            // Simple retry if player didn't start
            if !playerNode.isPlaying {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.playerNode.play()
                    self.isPlayingAudio = true
                    self.isAudioPlaying = true
                }
            }
        }
    }

    // MARK: - Helper Methods

    // Maintain a reference to the monitoring timer
    private var playbackMonitorTimer: Timer?

    private func startPlaybackMonitoring() {
        // Cancel any existing timer first
        stopPlaybackMonitoring()

        // Set audio playing state
        isAudioPlaying = true

        // Create a repeating timer that checks playback state every 0.2 seconds
        DispatchQueue.main.async {
            self.playbackMonitorTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] timer in
                guard let self = self else {
                    timer.invalidate()
                    return
                }

                self.checkIfPlaybackComplete()
            }

            // Make sure timer fires even during scrolling and user interaction
            RunLoop.current.add(self.playbackMonitorTimer!, forMode: .common)

            // Initial check for early detection
            self.checkIfPlaybackComplete()

            // Set a fallback timer to ensure monitoring stops eventually
            // This prevents monitoring from continuing indefinitely if playback state detection fails
            DispatchQueue.main.asyncAfter(deadline: .now() + 60.0) { [weak self] in
                guard let self = self, self.playbackMonitorTimer != nil else { return }

                self.stopPlaybackMonitoring()

                // Ensure playback state is reset
                if self.isAudioPlaying {
                    self.isPlayingAudio = false
                    self.isAudioPlaying = false
                    self.objectWillChange.send()
                }
            }
        }
    }

    private func stopPlaybackMonitoring() {
        // Check if timer exists before invalidating
        if playbackMonitorTimer != nil {
            playbackMonitorTimer?.invalidate()
            playbackMonitorTimer = nil
        }
    }

    private func checkIfPlaybackComplete() {
        // Double-check player state with a more reliable method
        let isActuallyPlaying = self.playerNode.isPlaying
        let hasScheduledBuffers = playerNode.engine?.isRunning ?? false

        // Check buffer counts for better completion detection
        bufferCountLock.lock()
        let allBuffersCompleted = completedBufferCount == scheduledBufferCount && scheduledBufferCount > 0
        bufferCountLock.unlock()

        // Use both playerNode.isPlaying AND buffer tracking for the most reliable detection
        if (!isActuallyPlaying && allBuffersCompleted) || (!isActuallyPlaying && !hasScheduledBuffers) {
            // Stop the monitoring timer immediately
            stopPlaybackMonitoring()

            // No more buffers are playing, mark playback as complete
            self.isPlayingAudio = false
            self.isAudioPlaying = false  // This will trigger generationInProgress update

            // Force UI update to refresh buttons state
            self.objectWillChange.send()
        }
    }

    private func isAudioEmpty(shape: [Int]) -> Bool {
        if shape.count == 1 {
            return shape[0] <= 1
        } else if shape.count == 2 {
            return shape[1] <= 1
        }
        return true
    }

    private func extractAudioData(from audioBuffer: MLXArray) -> (frameCount: Int, audioData: [Float]) {
        let audioShape = audioBuffer.shape

        // Handle different tensor shapes
        if audioShape.count == 1 {
            // 1D array [samples]
            let frameCount = audioShape[0]
            audioBuffer.eval()
            return (frameCount, audioBuffer.asArray(Float.self))
        } else if audioShape.count == 2 {
            // 2D array [1, samples]
            let frameCount = audioShape[1]
            let firstBatch = audioBuffer[0]
            firstBatch.eval()
            return (frameCount, firstBatch.asArray(Float.self))
        }

        // Fallback for unexpected shape
        return (0, [])
    }

    private func createAudioBuffer(frameCount: Int, audioData: [Float]) -> AVAudioPCMBuffer? {
        // Create buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(frameCount)) else {
            return nil
        }

        // Set frame length
        buffer.frameLength = buffer.frameCapacity

        // Copy data
        let channels = buffer.floatChannelData!
        let chunkSize = 32768 // 32K samples at a time

        for startIdx in stride(from: 0, to: min(frameCount, audioData.count), by: chunkSize) {
            autoreleasepool {
                let endIdx = min(startIdx + chunkSize, min(frameCount, audioData.count))

                // Copy with volume boost
                for i in startIdx..<endIdx {
                    if i < audioData.count && i < Int(buffer.frameCapacity) {
                        // Apply volume boost (25%) with clipping prevention
                        channels[0][i] = min(max(audioData[i] * 1.25, -0.98), 0.98)
                    }
                }
            }
        }
        return buffer
    }
}

extension AVAudioPCMBuffer {
    func saveToWavFile(at url: URL) throws {
        let audioFile = try AVAudioFile(forWriting: url,
                                      settings: format.settings,
                                      commonFormat: .pcmFormatFloat32,
                                      interleaved: false)
        try audioFile.write(from: self)
    }
}

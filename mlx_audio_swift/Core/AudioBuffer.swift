//
//  AudioBuffer.swift
//  MLXAudioCore
//
//  Created by MLX Audio Team on 2025-12-06.
//

import Foundation
import AVFoundation

/// Unified audio buffer type for passing audio data between components
public struct AudioBuffer: Sendable {
    /// Raw audio samples (mono, normalized to -1.0...1.0)
    public let samples: [Float]

    /// Sample rate in Hz
    public let sampleRate: Int

    /// Number of samples
    public var sampleCount: Int { samples.count }

    /// Duration in seconds
    public var duration: TimeInterval {
        Double(sampleCount) / Double(sampleRate)
    }

    /// Create an audio buffer from raw samples
    public init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    /// Create an audio buffer from Int16 samples (e.g., from microphone)
    public init(int16Samples: [Int16], sampleRate: Int) {
        self.samples = int16Samples.map { Float($0) / 32768.0 }
        self.sampleRate = sampleRate
    }

    /// Create an audio buffer from Data (Int16 format)
    public init(data: Data, sampleRate: Int) {
        let int16Array = data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Int16.self))
        }
        self.init(int16Samples: int16Array, sampleRate: sampleRate)
    }

    /// Load audio from file URL
    public static func load(from url: URL, targetSampleRate: Int = 16000) throws -> AudioBuffer {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioBufferError.bufferCreationFailed
        }

        try file.read(into: buffer)

        guard let floatData = buffer.floatChannelData else {
            throw AudioBufferError.noFloatData
        }

        // Convert to mono if stereo
        let channelCount = Int(format.channelCount)
        var samples = [Float](repeating: 0, count: Int(buffer.frameLength))

        if channelCount == 1 {
            samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))
        } else {
            // Mix down to mono
            for i in 0..<Int(buffer.frameLength) {
                var sum: Float = 0
                for ch in 0..<channelCount {
                    sum += floatData[ch][i]
                }
                samples[i] = sum / Float(channelCount)
            }
        }

        let sourceSampleRate = Int(format.sampleRate)

        // Resample if needed
        if sourceSampleRate != targetSampleRate {
            samples = resample(samples, from: sourceSampleRate, to: targetSampleRate)
        }

        return AudioBuffer(samples: samples, sampleRate: targetSampleRate)
    }

    /// Resample audio to a different sample rate
    public func resampled(to targetSampleRate: Int) -> AudioBuffer {
        if sampleRate == targetSampleRate {
            return self
        }
        let newSamples = Self.resample(samples, from: sampleRate, to: targetSampleRate)
        return AudioBuffer(samples: newSamples, sampleRate: targetSampleRate)
    }

    /// Convert to Int16 data (for recording/playback)
    public func toInt16Data() -> Data {
        let int16Samples = samples.map { Int16(max(-1, min(1, $0)) * 32767) }
        return int16Samples.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
    }

    // MARK: - Private

    private static func resample(_ samples: [Float], from sourceSampleRate: Int, to targetSampleRate: Int) -> [Float] {
        let ratio = Double(targetSampleRate) / Double(sourceSampleRate)
        let newCount = Int(Double(samples.count) * ratio)
        var resampled = [Float](repeating: 0, count: newCount)

        for i in 0..<newCount {
            let srcIndex = Double(i) / ratio
            let srcIndexInt = Int(srcIndex)
            let frac = Float(srcIndex - Double(srcIndexInt))

            if srcIndexInt + 1 < samples.count {
                // Linear interpolation
                resampled[i] = samples[srcIndexInt] * (1 - frac) + samples[srcIndexInt + 1] * frac
            } else if srcIndexInt < samples.count {
                resampled[i] = samples[srcIndexInt]
            }
        }

        return resampled
    }
}

public enum AudioBufferError: Error, LocalizedError {
    case bufferCreationFailed
    case noFloatData
    case fileNotFound(URL)
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .bufferCreationFailed:
            return "Failed to create audio buffer"
        case .noFloatData:
            return "Audio buffer has no float data"
        case .fileNotFound(let url):
            return "Audio file not found: \(url.path)"
        case .invalidFormat(let reason):
            return "Invalid audio format: \(reason)"
        }
    }
}

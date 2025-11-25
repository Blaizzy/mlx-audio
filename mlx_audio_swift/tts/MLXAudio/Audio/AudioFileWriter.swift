//
//  AudioFileWriter.swift
//  MLXAudio
//
//  Consolidated audio file saving utilities.
//  Replaces duplicate implementations across TTS engines.
//

import AVFoundation
import Foundation

/// Audio file format options
enum AudioFileFormat {
    case wav
    case caf

    var fileExtension: String {
        switch self {
        case .wav: return "wav"
        case .caf: return "caf"
        }
    }

    var commonFormat: AVAudioCommonFormat {
        switch self {
        case .wav, .caf: return .pcmFormatFloat32
        }
    }
}

/// Utility for saving audio samples to files
enum AudioFileWriter {
    /// Save audio samples to a file
    /// - Parameters:
    ///   - samples: Audio samples as Float array
    ///   - sampleRate: Sample rate (e.g., 24000 for Kokoro)
    ///   - directory: Target directory URL (defaults to documents directory)
    ///   - filename: Base filename without extension
    ///   - format: Output format (default: .wav for compatibility)
    /// - Returns: URL of the saved file
    /// - Throws: TTSError.fileIOError on failure
    static func save(
        samples: [Float],
        sampleRate: Int,
        to directory: URL? = nil,
        filename: String,
        format: AudioFileFormat = .wav
    ) throws -> URL {
        guard !samples.isEmpty else {
            throw TTSError.invalidArgument("Cannot save empty audio samples")
        }

        // Determine output directory
        let outputDirectory = directory ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]

        // Create full file URL
        let fileURL = outputDirectory.appendingPathComponent("\(filename).\(format.fileExtension)")

        // Create audio format
        guard let audioFormat = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: 1
        ) else {
            throw TTSError.fileIOError(underlying: NSError(
                domain: "AudioFileWriter",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"]
            ))
        }

        // Create buffer
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: audioFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw TTSError.fileIOError(underlying: NSError(
                domain: "AudioFileWriter",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"]
            ))
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)

        // Copy samples to buffer
        guard let channelData = buffer.floatChannelData else {
            throw TTSError.fileIOError(underlying: NSError(
                domain: "AudioFileWriter",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get channel data"]
            ))
        }

        for i in 0..<samples.count {
            channelData[0][i] = samples[i]
        }

        // Write to file
        do {
            let audioFile = try AVAudioFile(
                forWriting: fileURL,
                settings: audioFormat.settings,
                commonFormat: format.commonFormat,
                interleaved: false
            )
            try audioFile.write(from: buffer)
            Log.audio.debug("Audio saved to: \(fileURL.path)")
            return fileURL
        } catch {
            throw TTSError.fileIOError(underlying: error)
        }
    }

    /// Save multiple audio buffers to a single file
    /// - Parameters:
    ///   - buffers: Array of AVAudioPCMBuffer
    ///   - directory: Target directory URL
    ///   - filename: Base filename without extension
    ///   - format: Output format
    /// - Returns: URL of the saved file
    static func save(
        buffers: [AVAudioPCMBuffer],
        to directory: URL? = nil,
        filename: String,
        format: AudioFileFormat = .wav
    ) throws -> URL {
        guard !buffers.isEmpty else {
            throw TTSError.invalidArgument("Cannot save empty buffer array")
        }

        guard let firstBuffer = buffers.first else {
            throw TTSError.invalidArgument("No buffers provided")
        }

        let audioFormat = firstBuffer.format

        // Calculate total frames
        let totalFrames = buffers.reduce(0) { $0 + Int($1.frameLength) }

        // Create combined buffer
        guard let combinedBuffer = AVAudioPCMBuffer(
            pcmFormat: audioFormat,
            frameCapacity: AVAudioFrameCount(totalFrames)
        ) else {
            throw TTSError.fileIOError(underlying: NSError(
                domain: "AudioFileWriter",
                code: -4,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create combined buffer"]
            ))
        }

        combinedBuffer.frameLength = AVAudioFrameCount(totalFrames)

        // Copy all buffers into combined buffer
        var offset = 0
        for buffer in buffers {
            let frameCount = Int(buffer.frameLength)
            guard let srcData = buffer.floatChannelData?[0],
                  let dstData = combinedBuffer.floatChannelData?[0] else {
                continue
            }

            for i in 0..<frameCount {
                dstData[offset + i] = srcData[i]
            }
            offset += frameCount
        }

        // Determine output directory
        let outputDirectory = directory ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let fileURL = outputDirectory.appendingPathComponent("\(filename).\(format.fileExtension)")

        // Write to file
        do {
            let audioFile = try AVAudioFile(
                forWriting: fileURL,
                settings: audioFormat.settings,
                commonFormat: format.commonFormat,
                interleaved: false
            )
            try audioFile.write(from: combinedBuffer)
            Log.audio.debug("Combined audio saved to: \(fileURL.path)")
            return fileURL
        } catch {
            throw TTSError.fileIOError(underlying: error)
        }
    }
}

// MARK: - Stream Writer

/// Handles incremental writes for streaming audio
final class AudioStreamWriter {
    private var samples: [Float] = []
    private let sampleRate: Int
    private let directory: URL
    private let filename: String
    private let format: AudioFileFormat

    init(
        sampleRate: Int,
        to directory: URL? = nil,
        filename: String,
        format: AudioFileFormat = .wav
    ) {
        self.sampleRate = sampleRate
        self.directory = directory ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        self.filename = filename
        self.format = format
    }

    /// Append samples to the stream
    func append(samples newSamples: [Float]) {
        samples.append(contentsOf: newSamples)
    }

    /// Finalize and save the accumulated audio
    /// - Returns: URL of the saved file
    func finalize() throws -> URL {
        try AudioFileWriter.save(
            samples: samples,
            sampleRate: sampleRate,
            to: directory,
            filename: filename,
            format: format
        )
    }

    /// Current sample count
    var sampleCount: Int {
        samples.count
    }

    /// Current duration in seconds
    var duration: TimeInterval {
        Double(samples.count) / Double(sampleRate)
    }

    /// Reset the stream writer
    func reset() {
        samples.removeAll(keepingCapacity: true)
    }
}

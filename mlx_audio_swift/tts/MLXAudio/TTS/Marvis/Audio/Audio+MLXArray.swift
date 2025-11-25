@preconcurrency import AVFoundation
import Foundation
import MLX
import Synchronization

/// Returns (sampleRate, audio).
public func loadAudioArray(from url: URL) throws -> (Double, MLXArray) {
    let file = try AVAudioFile(forReading: url)

    let inFormat = file.processingFormat
    let totalFrames = AVAudioFrameCount(file.length)
    guard let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: totalFrames) else {
        throw NSError(domain: "WAVLoader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Buffer alloc failed"])
    }
    try file.read(into: inBuffer)

    if inFormat.commonFormat == .pcmFormatFloat32, let chans = inBuffer.floatChannelData {
        let frames = Int(inBuffer.frameLength)
        let channels: [[Float]] = (0..<Int(inFormat.channelCount)).map { c in
            let ptr = chans[c]
            return Array(UnsafeBufferPointer(start: ptr, count: frames))
        }
        return (inFormat.sampleRate, MLXArray(channels[0]))
    }

    let floatFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                    sampleRate: inFormat.sampleRate,
                                    channels: inFormat.channelCount,
                                    interleaved: false)
    guard let floatFormat = floatFormat else {
        throw NSError(domain: "WAVLoader", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to create float format"])
    }
    guard let converter = AVAudioConverter(from: inFormat, to: floatFormat) else {
        throw NSError(domain: "WAVLoader", code: -4, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"])
    }
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: floatFormat, frameCapacity: totalFrames) else {
        throw NSError(domain: "WAVLoader", code: -2, userInfo: [NSLocalizedDescriptionKey: "Out buffer alloc failed"])
    }

    let consumed = Atomic<Bool>(false)
    var convError: NSError?
    let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
        if consumed.exchange(true, ordering: .relaxed) {
            outStatus.pointee = .endOfStream
            return nil
        } else {
            outStatus.pointee = .haveData
            return inBuffer
        }
    }
    converter.convert(to: outBuffer, error: &convError, withInputFrom: inputBlock)
    if let e = convError { throw e }

    let frames = Int(outBuffer.frameLength)
    guard let floatChannelData = outBuffer.floatChannelData else {
        throw NSError(domain: "WAVLoader", code: -5, userInfo: [NSLocalizedDescriptionKey: "Failed to get float channel data"])
    }
    let channels: [[Float]] = (0..<Int(floatFormat.channelCount)).map { c in
        let ptr = floatChannelData[c]
        return Array(UnsafeBufferPointer(start: ptr, count: frames))
    }
    return (floatFormat.sampleRate, MLXArray(channels[0]))
}

public enum WAVWriterError: LocalizedError {
    case noFrames
    case bufferAllocFailed

    public var errorDescription: String? {
        switch self {
        case .noFrames:
            return "Audio has no frames to write"
        case .bufferAllocFailed:
            return "Failed to allocate audio buffer"
        }
    }
}

public func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let frames = audio.shape[0]
    guard frames > 0 else { throw WAVWriterError.noFrames }

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(1), interleaved: false) else {
        throw NSError(domain: "WAVWriter", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
    }
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
        throw WAVWriterError.bufferAllocFailed
    }
    buffer.frameLength = AVAudioFrameCount(frames)

    let channels = [audio.asArray(Float32.self)]
    guard let dst = buffer.floatChannelData else {
        throw WAVWriterError.bufferAllocFailed
    }
    for (c, channel) in channels.enumerated() {
        channel.withUnsafeBufferPointer { src in
            guard let baseAddress = src.baseAddress else { return }
            dst[c].update(from: baseAddress, count: frames)
        }
    }

    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
}

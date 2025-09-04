import AVFoundation
import Foundation
import MLX

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
                                    interleaved: false)!
    let converter = AVAudioConverter(from: inFormat, to: floatFormat)!
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: floatFormat, frameCapacity: totalFrames) else {
        throw NSError(domain: "WAVLoader", code: -2, userInfo: [NSLocalizedDescriptionKey: "Out buffer alloc failed"])
    }

    var consumed = false
    var convError: NSError?
    let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
        if consumed {
            outStatus.pointee = .endOfStream
            return nil
        } else {
            consumed = true
            outStatus.pointee = .haveData
            return inBuffer
        }
    }
    converter.convert(to: outBuffer, error: &convError, withInputFrom: inputBlock)
    if let e = convError { throw e }

    let frames = Int(outBuffer.frameLength)
    let channels: [[Float]] = (0..<Int(floatFormat.channelCount)).map { c in
        let ptr = outBuffer.floatChannelData![c]
        return Array(UnsafeBufferPointer(start: ptr, count: frames))
    }
    return (floatFormat.sampleRate, MLXArray(channels[0]))
}

public enum WAVWriterError: Error {
    case noFrames
    case bufferAllocFailed
}

public func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let frames = audio.shape[0]
    guard frames > 0 else { throw WAVWriterError.noFrames }

    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(1), interleaved: false)!
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
        throw WAVWriterError.bufferAllocFailed
    }
    buffer.frameLength = AVAudioFrameCount(frames)

    let channels = [audio.asArray(Float32.self)]
    if let dst = buffer.floatChannelData {
        for (c, channel) in channels.enumerated() {
            channel.withUnsafeBufferPointer { src in
                dst[c].update(from: src.baseAddress!, count: frames)
            }
        }
    }

    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
}

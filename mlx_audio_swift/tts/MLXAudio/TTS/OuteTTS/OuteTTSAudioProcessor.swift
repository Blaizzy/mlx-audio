//
//  OuteTTSAudioProcessor.swift
//  MLXAudio
//
//  Audio processing and feature extraction for OuteTTS speaker profiles
//  Ported from: mlx_audio/tts/models/outetts/audio_processor.py
//

import Foundation
import MLX
import Accelerate

// MARK: - Audio Feature Extraction

/// Calculates pitch using autocorrelation method
public func calculatePitch(
    audioArray: MLXArray,
    sampleRate: Int,
    minFreq: Float = 75.0,
    maxFreq: Float = 600.0,
    frameLength: Int = 400,
    hopLength: Int = 160,
    threshold: Float = 0.3
) -> [Float] {
    // Convert to numpy-like array
    var audioData = audioArray.asArray(Float.self)

    // Convert to mono if needed
    let numSamples = audioData.count

    // Pad audio
    let padLen = (frameLength - (numSamples % hopLength)) % hopLength
    audioData.append(contentsOf: [Float](repeating: 0, count: padLen))

    let numFrames = (audioData.count - frameLength) / hopLength + 1

    // Create frames
    var frames = [[Float]](repeating: [Float](repeating: 0, count: frameLength), count: numFrames)
    for i in 0..<numFrames {
        let start = i * hopLength
        frames[i] = Array(audioData[start..<(start + frameLength)])
    }

    // Apply Hanning window
    var window = [Float](repeating: 0, count: frameLength)
    vDSP_hann_window(&window, vDSP_Length(frameLength), Int32(vDSP_HANN_NORM))

    for i in 0..<numFrames {
        vDSP_vmul(frames[i], 1, window, 1, &frames[i], 1, vDSP_Length(frameLength))
    }

    // Compute autocorrelation using FFT
    let fftLength = frameLength * 2
    let log2n = vDSP_Length(log2(Float(fftLength)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        return [Float](repeating: 0, count: numFrames)
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    var pitches = [Float](repeating: 0, count: numFrames)

    for i in 0..<numFrames {
        // Zero-pad frame for FFT
        var paddedFrame = frames[i] + [Float](repeating: 0, count: frameLength)

        // Split into real and imaginary
        var realPart = [Float](repeating: 0, count: fftLength / 2)
        var imagPart = [Float](repeating: 0, count: fftLength / 2)

        // Perform FFT operations with proper pointer scoping
        realPart.withUnsafeMutableBufferPointer { realPtr in
            imagPart.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                // Convert to split complex
                paddedFrame.withUnsafeMutableBufferPointer { paddedPtr in
                    paddedPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftLength / 2) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(fftLength / 2))
                    }
                }

                // Forward FFT
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                // Inverse FFT for autocorrelation
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))
            }
        }

        // Get autocorrelation values
        var autocorr = [Float](repeating: 0, count: frameLength)
        for j in 0..<frameLength {
            if j < fftLength / 2 {
                autocorr[j] = realPart[j]
            }
        }

        // Find peak in valid frequency range
        let minIdx = max(1, Int(Float(sampleRate) / maxFreq))
        let maxIdx = min(frameLength, Int(Float(sampleRate) / minFreq))

        if minIdx < maxIdx {
            var peakIdx = minIdx
            var peakVal = autocorr[minIdx]
            for j in minIdx..<maxIdx {
                if autocorr[j] > peakVal {
                    peakVal = autocorr[j]
                    peakIdx = j
                }
            }

            // Check voicing threshold
            let autocorr0 = autocorr[0] + 1e-8
            if peakVal / autocorr0 > threshold && peakIdx > 0 {
                let pitch = Float(sampleRate) / Float(peakIdx)
                pitches[i] = min(max(pitch, minFreq), maxFreq)
            }
        }
    }

    return pitches
}

/// Extract single normalized pitch value from audio
public func extractSinglePitchValue(
    audioArray: MLXArray,
    sampleRate: Int,
    minFreq: Float = 75.0,
    maxFreq: Float = 600.0
) -> Float {
    let pitches = calculatePitch(
        audioArray: audioArray,
        sampleRate: sampleRate,
        minFreq: minFreq,
        maxFreq: maxFreq
    )

    // Calculate average pitch
    let nonZeroPitches = pitches.filter { $0 > 0 }
    let averagePitch = nonZeroPitches.isEmpty ? 0 : nonZeroPitches.reduce(0, +) / Float(nonZeroPitches.count)

    // Normalize to 0-1 range
    let normalized = (averagePitch - minFreq) / (maxFreq - minFreq)
    return min(max(normalized, 0.0), 1.0)
}

// MARK: - Feature Extractor

/// Extracts audio features (energy, spectral centroid, pitch)
public class OuteTTSFeatures {
    private let eps: Float = 1e-10

    public init() {}

    /// Scale value from 0-1 to 0-100
    public func scaleValue(_ value: Float) -> Int {
        return Int(round(value * 100))
    }

    /// Validate audio array
    public func validateAudio(_ audio: MLXArray) -> Bool {
        if audio.size == 0 {
            return false
        }
        let data = audio.asArray(Float.self)
        return !data.contains(where: { $0.isNaN || $0.isInfinite })
    }

    /// Get default features when audio is invalid
    public func getDefaultFeatures() -> OuteTTSAudioFeatures {
        return OuteTTSAudioFeatures(energy: 0, spectralCentroid: 0, pitch: 0)
    }

    /// Extract audio features from a segment
    public func extractAudioFeatures(audio: MLXArray, sampleRate: Int) -> OuteTTSAudioFeatures {
        guard validateAudio(audio) else {
            return getDefaultFeatures()
        }

        let audioData = audio.asArray(Float.self)

        // RMS Energy (normalized to 0-1)
        var squaredSum: Float = 0
        vDSP_svesq(audioData, 1, &squaredSum, vDSP_Length(audioData.count))
        let rmsEnergy = sqrt(squaredSum / Float(audioData.count))
        let normalizedEnergy = min(rmsEnergy, 1.0)

        // Spectral Centroid (normalized to 0-1)
        let spectralCentroid = computeSpectralCentroid(audioData, sampleRate: sampleRate)
        let normalizedCentroid = spectralCentroid / Float(sampleRate / 2)

        // Pitch (already normalized in extractSinglePitchValue)
        let pitch = extractSinglePitchValue(audioArray: audio, sampleRate: sampleRate)

        return OuteTTSAudioFeatures(
            energy: scaleValue(normalizedEnergy),
            spectralCentroid: scaleValue(min(normalizedCentroid, 1.0)),
            pitch: scaleValue(pitch)
        )
    }

    /// Compute spectral centroid using FFT
    private func computeSpectralCentroid(_ audio: [Float], sampleRate: Int) -> Float {
        let n = audio.count
        guard n > 0 else { return 0 }

        // Find next power of 2
        let fftSize = Int(pow(2, ceil(log2(Double(n)))))
        let log2n = vDSP_Length(log2(Float(fftSize)))

        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return 0
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Zero-pad audio
        var paddedAudio = audio + [Float](repeating: 0, count: fftSize - n)

        // Prepare split complex
        var realPart = [Float](repeating: 0, count: fftSize / 2)
        var imagPart = [Float](repeating: 0, count: fftSize / 2)
        var magnitudes = [Float](repeating: 0, count: fftSize / 2)

        // Perform FFT operations with proper pointer scoping
        realPart.withUnsafeMutableBufferPointer { realPtr in
            imagPart.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                paddedAudio.withUnsafeMutableBufferPointer { paddedPtr in
                    paddedPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftSize / 2) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(fftSize / 2))
                    }
                }

                // FFT
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                // Compute magnitude spectrum
                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(fftSize / 2))
            }
        }

        // Square root of magnitudes
        var count = Int32(fftSize / 2)
        vvsqrtf(&magnitudes, magnitudes, &count)

        // Compute frequencies
        let freqResolution = Float(sampleRate) / Float(fftSize)
        var frequencies = [Float](repeating: 0, count: fftSize / 2)
        for i in 0..<(fftSize / 2) {
            frequencies[i] = Float(i) * freqResolution
        }

        // Compute weighted sum
        var weightedSum: Float = 0
        vDSP_dotpr(frequencies, 1, magnitudes, 1, &weightedSum, vDSP_Length(fftSize / 2))

        var magnitudeSum: Float = 0
        vDSP_sve(magnitudes, 1, &magnitudeSum, vDSP_Length(fftSize / 2))

        return magnitudeSum > eps ? weightedSum / magnitudeSum : 0
    }
}

// MARK: - Audio Processor

public class OuteTTSAudioProcessor {
    public let features: OuteTTSFeatures
    public var audioCodec: DACCodec?
    public let sampleRate: Int

    public init(sampleRate: Int = 24000) {
        self.features = OuteTTSFeatures()
        self.sampleRate = sampleRate
    }

    /// Initialize with DAC codec
    public func loadCodec(repoId: String = DACCodec.defaultRepoId, progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }) async throws {
        self.audioCodec = try await DACCodec.fromPretrained(repoId: repoId, progressHandler: progressHandler)
    }

    /// Create speaker profile from transcribed audio data
    public func createSpeakerFromTranscription(
        audio: MLXArray,
        text: String,
        words: [(word: String, start: Double, end: Double)]
    ) async throws -> OuteTTSSpeakerProfile {
        guard let codec = audioCodec else {
            throw OuteTTSError.codecNotLoaded
        }

        // Encode audio to get codes
        let (_, codes) = codec.encode(audio.reshaped([1, 1, -1]))
        let codesArray = codes.asArray(Int32.self)

        // Parse codes into c1 and c2
        let numCodebooks = codes.shape[1]
        let timeSteps = codes.shape[2]

        var c1: [Int] = []
        var c2: [Int] = []

        for t in 0..<timeSteps {
            if numCodebooks > 0 {
                c1.append(Int(codesArray[t]))
            }
            if numCodebooks > 1 {
                c2.append(Int(codesArray[timeSteps + t]))
            }
        }

        // Extract global features
        let globalFeatures = features.extractAudioFeatures(audio: audio, sampleRate: sampleRate)

        // Tokens per second (approximately 75 for DAC at 24kHz)
        let tps: Double = 75.0
        let maxExtension = 20

        var wordCodes: [OuteTTSWordData] = []
        var start: Int? = nil

        let audioData = audio.asArray(Float.self)

        for (idx, wordInfo) in words.enumerated() {
            if start == nil {
                start = max(0, Int(wordInfo.start * tps) - maxExtension)
            }

            let word = wordInfo.word.trimmingCharacters(in: .whitespaces)

            let end: Int
            if idx == words.count - 1 {
                end = min(c1.count, Int(wordInfo.end * tps) + maxExtension)
            } else {
                end = Int(wordInfo.end * tps)
            }

            let wordC1 = Array(c1[start!..<min(end, c1.count)])
            let wordC2 = Array(c2[start!..<min(end, c2.count)])

            // Extract word audio segment
            let audioStart = Int(wordInfo.start * Double(sampleRate))
            let audioEnd = Int(wordInfo.end * Double(sampleRate))
            let wordAudio: [Float]
            if audioStart < audioEnd && audioEnd <= audioData.count {
                wordAudio = Array(audioData[audioStart..<audioEnd])
            } else {
                wordAudio = []
            }

            let wordFeatures = wordAudio.isEmpty
                ? features.getDefaultFeatures()
                : features.extractAudioFeatures(audio: MLXArray(wordAudio), sampleRate: sampleRate)

            start = end

            wordCodes.append(OuteTTSWordData(
                word: word,
                duration: round(Double(wordC1.count) / tps * 100) / 100,
                c1: wordC1,
                c2: wordC2,
                features: wordFeatures
            ))
        }

        return OuteTTSSpeakerProfile(
            text: OuteTTSPromptProcessor.normalizeText(text),
            words: wordCodes,
            globalFeatures: globalFeatures
        )
    }

    /// Save speaker profile to file
    public func saveSpeaker(_ speaker: OuteTTSSpeakerProfile, to path: String) throws {
        try speaker.save(to: path)
        print("Speaker saved to: \(path)")
    }

    /// Load speaker profile from file
    public func loadSpeaker(from path: String) throws -> OuteTTSSpeakerProfile {
        return try OuteTTSSpeakerProfile.load(from: path)
    }
}

// MARK: - Errors

public enum OuteTTSError: Error, LocalizedError {
    case codecNotLoaded
    case invalidAudio
    case speakerFileNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .codecNotLoaded:
            return "DAC codec not loaded. Call loadCodec() first."
        case .invalidAudio:
            return "Invalid or empty audio data"
        case .speakerFileNotFound(let path):
            return "Speaker file not found: \(path)"
        }
    }
}

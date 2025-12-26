//
//  ChineseTokenizer.swift
//  MLXAudio
//
//  Chinese word segmentation using jieba-style DAG-based algorithm.
//  Port of Python jieba library for Swift.
//

import Compression
import Foundation

/// Chinese word segmentation using jieba-style DAG-based algorithm.
final class ChineseTokenizer {

    /// Word frequency dictionary: word -> frequency
    private var wordFreq: [String: Int] = [:]

    /// Total frequency sum for probability calculation
    private var totalFreq: Double = 0

    /// Maximum word length in dictionary
    private var maxWordLength: Int = 0

    /// Minimum log probability for unknown words
    private var minLogProb: Double = -30.0

    /// Shared instance
    static let shared = ChineseTokenizer()

    private init() {}

    /// Load dictionary from gzipped binary data
    func loadDictionary(from url: URL) throws {
        let compressedData = try Data(contentsOf: url)
        let data = try decompressGzip(compressedData)
        parseBinaryDictionary(data)
    }

    /// Load dictionary from embedded data
    func loadDictionary(data: Data, isCompressed: Bool = true) throws {
        let uncompressed = isCompressed ? try decompressGzip(data) : data
        parseBinaryDictionary(uncompressed)
    }

    /// Parse binary dictionary format:
    /// [word_len:u16][word:utf8][freq:u32][pos_len:u8][pos:utf8]
    private func parseBinaryDictionary(_ data: Data) {
        wordFreq.removeAll()
        totalFreq = 0
        maxWordLength = 0

        var offset = 0
        while offset < data.count {
            // Read word length (u16)
            guard offset + 2 <= data.count else { break }
            let wordLen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2

            // Read word (utf8)
            guard offset + wordLen <= data.count else { break }
            let wordData = data[offset..<offset + wordLen]
            offset += wordLen
            guard let word = String(data: wordData, encoding: .utf8) else { continue }

            // Read frequency (u32)
            guard offset + 4 <= data.count else { break }
            let freq =
                Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16)
                | (Int(data[offset + 3]) << 24)
            offset += 4

            // Read POS length (u8)
            guard offset + 1 <= data.count else { break }
            let posLen = Int(data[offset])
            offset += 1

            // Skip POS (we don't need it for basic segmentation)
            offset += posLen

            wordFreq[word] = freq
            totalFreq += Double(freq)
            maxWordLength = max(maxWordLength, word.count)
        }
    }

    /// Decompress gzip data using Compression framework
    private func decompressGzip(_ data: Data) throws -> Data {
        guard data.count > 10 else {
            throw ChineseTokenizerError.invalidData
        }

        // Check gzip magic number
        guard data[0] == 0x1f && data[1] == 0x8b else {
            throw ChineseTokenizerError.invalidGzipHeader
        }

        // Parse gzip header properly
        var headerSize = 10  // Minimum header size
        let flags = data[3]

        // Skip optional fields
        if (flags & 0x04) != 0 {  // FEXTRA
            guard headerSize + 2 <= data.count else {
                throw ChineseTokenizerError.invalidGzipHeader
            }
            let extraLen = Int(data[headerSize]) | (Int(data[headerSize + 1]) << 8)
            headerSize += 2 + extraLen
        }
        if (flags & 0x08) != 0 {  // FNAME - null-terminated string
            while headerSize < data.count && data[headerSize] != 0 {
                headerSize += 1
            }
            headerSize += 1  // Skip null terminator
        }
        if (flags & 0x10) != 0 {  // FCOMMENT - null-terminated string
            while headerSize < data.count && data[headerSize] != 0 {
                headerSize += 1
            }
            headerSize += 1
        }
        if (flags & 0x02) != 0 {  // FHCRC
            headerSize += 2
        }

        guard headerSize + 8 < data.count else {
            throw ChineseTokenizerError.invalidGzipHeader
        }

        // Skip header and 8-byte trailer (CRC32 + ISIZE)
        let compressedPayload = Data(data.dropFirst(headerSize).dropLast(8))

        // Use Compression framework
        let bufferSize = 1024 * 1024  // 1MB buffer

        let result = compressedPayload.withUnsafeBytes { sourceBuffer -> Data? in
            guard let sourcePtr = sourceBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return nil
            }

            let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
            defer { destinationBuffer.deallocate() }

            var output = Data()
            let sourceSize = compressedPayload.count

            // Create decompression stream
            let stream = UnsafeMutablePointer<compression_stream>.allocate(capacity: 1)
            defer { stream.deallocate() }

            var status = compression_stream_init(stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB)
            guard status == COMPRESSION_STATUS_OK else { return nil }
            defer { compression_stream_destroy(stream) }

            stream.pointee.src_ptr = sourcePtr
            stream.pointee.src_size = sourceSize
            stream.pointee.dst_ptr = destinationBuffer
            stream.pointee.dst_size = bufferSize

            repeat {
                status = compression_stream_process(stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))

                if stream.pointee.dst_size < bufferSize {
                    output.append(destinationBuffer, count: bufferSize - stream.pointee.dst_size)
                    stream.pointee.dst_ptr = destinationBuffer
                    stream.pointee.dst_size = bufferSize
                }

                if status == COMPRESSION_STATUS_END {
                    break
                }

                guard status == COMPRESSION_STATUS_OK else {
                    return nil
                }
            } while true

            return output
        }

        guard let decompressedData = result else {
            throw ChineseTokenizerError.decompressionFailed
        }

        return decompressedData
    }

    /// Segment Chinese text into words
    func segment(_ text: String) -> [String] {
        guard !wordFreq.isEmpty else {
            // Fallback: character-by-character
            return text.map { String($0) }
        }

        let chars = Array(text)
        guard !chars.isEmpty else { return [] }

        // Build DAG (Directed Acyclic Graph)
        let dag = buildDAG(chars)

        // Find best path using dynamic programming
        let route = calculateRoute(chars, dag: dag)

        // Extract words from route
        var result: [String] = []
        var i = 0
        while i < chars.count {
            let (_, end) = route[i]!
            let word = String(chars[i...end])
            result.append(word)
            i = end + 1
        }

        return result
    }

    /// Build DAG: for each position, find all possible word endings
    private func buildDAG(_ chars: [Character]) -> [Int: [Int]] {
        var dag: [Int: [Int]] = [:]
        let n = chars.count

        for i in 0..<n {
            var endings: [Int] = []

            // Try all possible word lengths
            let maxLen = min(maxWordLength, n - i)
            for len in 1...maxLen {
                let word = String(chars[i..<i + len])
                if wordFreq[word] != nil {
                    endings.append(i + len - 1)
                }
            }

            // Always include single character as fallback
            if endings.isEmpty || !endings.contains(i) {
                endings.append(i)
            }

            dag[i] = endings.sorted()
        }

        return dag
    }

    /// Calculate optimal route using dynamic programming
    /// Returns: position -> (probability, end_position)
    private func calculateRoute(_ chars: [Character], dag: [Int: [Int]]) -> [Int: (Double, Int)] {
        let n = chars.count
        var route: [Int: (Double, Int)] = [:]

        // Base case: position after last character
        route[n] = (0.0, 0)

        // Work backwards
        for i in stride(from: n - 1, through: 0, by: -1) {
            var bestProb = -Double.infinity
            var bestEnd = i

            for end in dag[i] ?? [i] {
                let word = String(chars[i...end])
                let wordProb = logProbability(word)
                let futureProb = route[end + 1]?.0 ?? 0.0
                let totalProb = wordProb + futureProb

                if totalProb > bestProb {
                    bestProb = totalProb
                    bestEnd = end
                }
            }

            route[i] = (bestProb, bestEnd)
        }

        return route
    }

    /// Calculate log probability of a word
    private func logProbability(_ word: String) -> Double {
        if let freq = wordFreq[word] {
            return log(Double(freq) / totalFreq)
        }
        return minLogProb * Double(word.count)
    }

    /// Check if dictionary is loaded
    var isLoaded: Bool {
        !wordFreq.isEmpty
    }

    /// Get word frequency (for debugging)
    func frequency(of word: String) -> Int? {
        wordFreq[word]
    }
}

enum ChineseTokenizerError: Error {
    case invalidData
    case invalidGzipHeader
    case decompressionFailed
    case dictionaryNotLoaded
}

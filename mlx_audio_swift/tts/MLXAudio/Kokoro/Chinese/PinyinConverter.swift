//
//  PinyinConverter.swift
//  MLXAudio
//
//  Converts Chinese characters to pinyin.
//  Port of Python pypinyin library for Swift.
//

import Compression
import Foundation

/// Converts Chinese characters to pinyin.
final class PinyinConverter {

    /// Single character pinyin dictionary: Unicode codepoint -> [pinyin options]
    private var singlePinyin: [UInt32: [String]] = [:]

    /// Phrase pinyin dictionary: phrase -> [pinyin for each char]
    private var phrasePinyin: [String: [String]] = [:]

    /// Shared instance
    static let shared = PinyinConverter()

    private init() {}

    /// Load single character pinyin dictionary from URL
    func loadSinglePinyin(from url: URL) throws {
        let compressedData = try Data(contentsOf: url)
        try loadSinglePinyin(data: compressedData, isCompressed: true)
    }

    /// Load single character pinyin dictionary from data
    func loadSinglePinyin(data: Data, isCompressed: Bool = true) throws {
        let uncompressed = isCompressed ? try decompressGzip(data) : data
        parseSinglePinyinBinary(uncompressed)
    }

    /// Load phrase pinyin dictionary from URL
    func loadPhrasePinyin(from url: URL) throws {
        let compressedData = try Data(contentsOf: url)
        try loadPhrasePinyin(data: compressedData, isCompressed: true)
    }

    /// Load phrase pinyin dictionary from data
    func loadPhrasePinyin(data: Data, isCompressed: Bool = true) throws {
        let uncompressed = isCompressed ? try decompressGzip(data) : data
        parsePhrasePinyinBinary(uncompressed)
    }

    /// Parse single character pinyin binary format:
    /// [codepoint:u32][pinyin_count:u8][pinyin1_len:u8][pinyin1:utf8]...
    private func parseSinglePinyinBinary(_ data: Data) {
        singlePinyin.removeAll()

        var offset = 0
        while offset < data.count {
            // Read codepoint (u32)
            guard offset + 4 <= data.count else { break }
            let codepoint =
                UInt32(data[offset]) | (UInt32(data[offset + 1]) << 8) | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
            offset += 4

            // Read pinyin count (u8)
            guard offset + 1 <= data.count else { break }
            let count = Int(data[offset])
            offset += 1

            // Read each pinyin
            var pinyins: [String] = []
            for _ in 0..<count {
                guard offset + 1 <= data.count else { break }
                let pyLen = Int(data[offset])
                offset += 1

                guard offset + pyLen <= data.count else { break }
                let pyData = data[offset..<offset + pyLen]
                offset += pyLen

                if let py = String(data: pyData, encoding: .utf8) {
                    pinyins.append(py)
                }
            }

            singlePinyin[codepoint] = pinyins
        }
    }

    /// Parse phrase pinyin binary format:
    /// [phrase_len:u16][phrase:utf8][syllable_count:u8][[pinyin_len:u8][pinyin:utf8]]...
    private func parsePhrasePinyinBinary(_ data: Data) {
        phrasePinyin.removeAll()

        var offset = 0
        while offset < data.count {
            // Read phrase length (u16)
            guard offset + 2 <= data.count else { break }
            let phraseLen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2

            // Read phrase (utf8)
            guard offset + phraseLen <= data.count else { break }
            let phraseData = data[offset..<offset + phraseLen]
            offset += phraseLen
            guard let phrase = String(data: phraseData, encoding: .utf8) else { continue }

            // Read syllable count (u8)
            guard offset + 1 <= data.count else { break }
            let syllableCount = Int(data[offset])
            offset += 1

            // Read each syllable
            var syllables: [String] = []
            for _ in 0..<syllableCount {
                guard offset + 1 <= data.count else { break }
                let pyLen = Int(data[offset])
                offset += 1

                guard offset + pyLen <= data.count else { break }
                let pyData = data[offset..<offset + pyLen]
                offset += pyLen

                if let py = String(data: pyData, encoding: .utf8) {
                    syllables.append(py)
                }
            }

            phrasePinyin[phrase] = syllables
        }
    }

    /// Convert text to pinyin with tone numbers (Style.TONE3)
    /// Returns array of pinyin for each character/word
    func toPinyin(_ text: String, style: PinyinStyle = .tone3) -> [String] {
        let chars = Array(text)
        guard !chars.isEmpty else { return [] }

        var result: [String] = []
        var i = 0

        while i < chars.count {
            // Try to match phrases first (longer matches preferred)
            var matched = false

            for len in stride(from: min(8, chars.count - i), through: 2, by: -1) {
                let phrase = String(chars[i..<i + len])
                if let pinyins = phrasePinyin[phrase] {
                    // Convert diacritic pinyins to requested style (e.g., "háng" -> "hang2")
                    result.append(contentsOf: pinyins.map { convertStyle($0, to: style) })
                    i += len
                    matched = true
                    break
                }
            }

            if !matched {
                // Single character lookup
                let char = chars[i]
                let pinyin = getPinyin(for: char, style: style)
                result.append(pinyin)
                i += 1
            }
        }

        return result
    }

    /// Get pinyin for a single character
    func getPinyin(for char: Character, style: PinyinStyle = .tone3) -> String {
        guard let scalar = char.unicodeScalars.first else {
            return String(char)
        }

        let codepoint = scalar.value
        guard let pinyins = singlePinyin[codepoint], let firstPinyin = pinyins.first else {
            return String(char)
        }

        return convertStyle(firstPinyin, to: style)
    }

    /// Convert pinyin between styles
    private func convertStyle(_ pinyin: String, to style: PinyinStyle) -> String {
        switch style {
        case .tone3:
            // Convert from diacritic format to tone number format
            return convertDiacriticToTone3(pinyin)
        case .normal:
            // Remove tone marks and numbers
            return removeTones(pinyin)
        case .initials:
            return getInitial(pinyin)
        case .finals:
            return getFinal(pinyin)
        case .finalsTone3:
            let initial = getInitial(pinyin)
            return String(pinyin.dropFirst(initial.count))
        }
    }

    /// Convert diacritic pinyin to tone3 format (e.g., "ni" -> "ni3")
    private func convertDiacriticToTone3(_ pinyin: String) -> String {
        // If already has a tone number at the end, return as-is
        if let last = pinyin.last, last.isNumber {
            return pinyin
        }

        // Map diacritics to (base vowel, tone number)
        let diacriticToTone: [Character: (Character, String)] = [
            // First tone
            "ā": ("a", "1"), "ē": ("e", "1"), "ī": ("i", "1"),
            "ō": ("o", "1"), "ū": ("u", "1"), "ǖ": ("v", "1"),
            // Second tone
            "á": ("a", "2"), "é": ("e", "2"), "í": ("i", "2"),
            "ó": ("o", "2"), "ú": ("u", "2"), "ǘ": ("v", "2"),
            // Third tone
            "ǎ": ("a", "3"), "ě": ("e", "3"), "ǐ": ("i", "3"),
            "ǒ": ("o", "3"), "ǔ": ("u", "3"), "ǚ": ("v", "3"),
            // Fourth tone
            "à": ("a", "4"), "è": ("e", "4"), "ì": ("i", "4"),
            "ò": ("o", "4"), "ù": ("u", "4"), "ǜ": ("v", "4"),
        ]

        var result = ""
        var toneNumber = "5"  // Default neutral tone

        for char in pinyin {
            if let (baseVowel, tone) = diacriticToTone[char] {
                result.append(baseVowel)
                toneNumber = tone
            } else {
                result.append(char)
            }
        }

        // Append tone number at end
        return result + toneNumber
    }

    /// Remove tone marks from pinyin
    private func removeTones(_ pinyin: String) -> String {
        var result = pinyin
        // Remove tone numbers at end
        if let last = result.last, last.isNumber {
            result.removeLast()
        }
        // Convert accented vowels to plain
        let toneMap: [Character: Character] = [
            "ā": "a", "á": "a", "ǎ": "a", "à": "a",
            "ē": "e", "é": "e", "ě": "e", "è": "e",
            "ī": "i", "í": "i", "ǐ": "i", "ì": "i",
            "ō": "o", "ó": "o", "ǒ": "o", "ò": "o",
            "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
            "ǖ": "ü", "ǘ": "ü", "ǚ": "ü", "ǜ": "ü",
        ]
        return String(result.map { toneMap[$0] ?? $0 })
    }

    /// Get initial consonant
    private func getInitial(_ pinyin: String) -> String {
        let initials = [
            "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
        ]
        let lower = pinyin.lowercased()
        for initial in initials {
            if lower.hasPrefix(initial) {
                return initial
            }
        }
        return ""
    }

    /// Get final (vowel part)
    private func getFinal(_ pinyin: String) -> String {
        let initial = getInitial(pinyin)
        var final = String(pinyin.dropFirst(initial.count))
        // Remove tone number if present
        if let last = final.last, last.isNumber {
            final.removeLast()
        }
        return final
    }

    /// Decompress gzip data using Compression framework
    private func decompressGzip(_ data: Data) throws -> Data {
        guard data.count > 10 else {
            throw PinyinConverterError.invalidData
        }

        // Check gzip magic number
        guard data[0] == 0x1f && data[1] == 0x8b else {
            throw PinyinConverterError.invalidGzipHeader
        }

        // Parse gzip header properly
        var headerSize = 10  // Minimum header size
        let flags = data[3]

        // Skip optional fields
        if (flags & 0x04) != 0 {  // FEXTRA
            guard headerSize + 2 <= data.count else {
                throw PinyinConverterError.invalidGzipHeader
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
            throw PinyinConverterError.invalidGzipHeader
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
            throw PinyinConverterError.decompressionFailed
        }

        return decompressedData
    }

    var isLoaded: Bool {
        !singlePinyin.isEmpty
    }
}

enum PinyinStyle {
    case tone3  // e.g., "bei3" "jing1"
    case normal  // e.g., "bei" "jing"
    case initials  // e.g., "b" "j"
    case finals  // e.g., "ei" "ing"
    case finalsTone3  // e.g., "ei3" "ing1"
}

enum PinyinConverterError: Error {
    case invalidData
    case invalidGzipHeader
    case decompressionFailed
    case dictionaryNotLoaded
}

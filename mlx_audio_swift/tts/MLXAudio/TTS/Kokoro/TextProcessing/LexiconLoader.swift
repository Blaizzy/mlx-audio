//
//  LexiconLoader.swift
//  MLXAudio
//
//  Downloads and caches lexicon files from the misaki GitHub repository.
//

import Foundation

/// Utility class for loading lexicon files from GitHub.
/// Files are downloaded and cached on disk automatically.
class LexiconLoader {
  private init() {}

  // GitHub raw content base URL for misaki data files
  static let baseURL = "https://raw.githubusercontent.com/hexgrad/misaki/main/misaki/data"

  // Cache directory for lexicon files
  private static var cacheDirectory: URL {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    return cacheDir.appendingPathComponent("MLXAudio/kokoro/misaki-lexicons", isDirectory: true)
  }

  enum LexiconLoaderError: LocalizedError {
    case downloadFailed(String)
    case invalidJSON(String)

    var errorDescription: String? {
      switch self {
      case .downloadFailed(let message):
        return "Lexicon download failed: \(message)"
      case .invalidJSON(let message):
        return "Invalid lexicon JSON: \(message)"
      }
    }
  }

  /// Load a lexicon dictionary from GitHub (cached on disk).
  /// - Parameters:
  ///   - filename: The lexicon filename without extension (e.g., "us_gold")
  /// - Returns: A dictionary mapping words to phonemes
  static func loadLexicon(_ filename: String) async throws -> [String: String] {
    // Check if cached file exists
    let cachedURL = cacheDirectory.appendingPathComponent("\(filename).json")

    if FileManager.default.fileExists(atPath: cachedURL.path) {
      return try loadLexiconFromFile(cachedURL)
    }

    // Download from GitHub
    let remoteURL = URL(string: "\(baseURL)/\(filename).json")!

    let (data, response) = try await URLSession.shared.data(from: remoteURL)

    guard let httpResponse = response as? HTTPURLResponse,
          httpResponse.statusCode == 200 else {
      throw LexiconLoaderError.downloadFailed("Failed to download \(filename).json")
    }

    // Create cache directory if needed
    try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

    // Save to cache
    try data.write(to: cachedURL)

    // Parse and return
    return try parseLexiconData(data, filename: filename)
  }

  /// Load US English lexicons (gold + silver, merged)
  static func loadUSLexicon() async throws -> [String: String] {
    var lexicon: [String: String] = [:]

    // Load silver first (lower priority)
    if let silver = try? await loadLexicon("us_silver") {
      lexicon.merge(silver) { _, new in new }
    }

    // Load gold second (higher priority, overrides silver)
    let gold = try await loadLexicon("us_gold")
    lexicon.merge(gold) { _, new in new }

    return lexicon
  }

  /// Load British English lexicons (gold + silver, merged)
  static func loadGBLexicon() async throws -> [String: String] {
    var lexicon: [String: String] = [:]

    // Load silver first (lower priority)
    if let silver = try? await loadLexicon("gb_silver") {
      lexicon.merge(silver) { _, new in new }
    }

    // Load gold second (higher priority, overrides silver)
    let gold = try await loadLexicon("gb_gold")
    lexicon.merge(gold) { _, new in new }

    return lexicon
  }

  // MARK: - Private Helpers

  private static func loadLexiconFromFile(_ url: URL) throws -> [String: String] {
    let data = try Data(contentsOf: url)
    return try parseLexiconData(data, filename: url.lastPathComponent)
  }

  private static func parseLexiconData(_ data: Data, filename: String) throws -> [String: String] {
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw LexiconLoaderError.invalidJSON("Failed to parse \(filename)")
    }

    var processedLexicon: [String: String] = [:]

    for (key, value) in json {
      if let stringValue = value as? String {
        processedLexicon[key] = stringValue
      } else if let dictValue = value as? [String: Any],
                let defaultValue = dictValue["DEFAULT"] as? String {
        processedLexicon[key] = defaultValue
      }
    }

    return processedLexicon
  }

  /// Clear the lexicon cache
  static func clearCache() throws {
    if FileManager.default.fileExists(atPath: cacheDirectory.path) {
      try FileManager.default.removeItem(at: cacheDirectory)
    }
  }
}

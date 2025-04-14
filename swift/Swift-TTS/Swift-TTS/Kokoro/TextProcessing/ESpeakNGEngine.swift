//
import ESpeakNG

//  Kokoro-tts-lib
//
import Foundation

// ESpeakNG wrapper for phonemizing the text strings
final class ESpeakNGEngine {
  private var language: LanguageDialect = .none
  private var languageMapping: [String: String] = [:]

  enum ESpeakNGEngineError: Error {
    case dataBundleNotFound
    case couldNotInitialize
    case languageNotFound
    case internalError
    case languageNotSet
    case couldNotPhonemize
  }

  // Available languages
  enum LanguageDialect: String, CaseIterable {
    case none = ""
    case enUS = "en-us"
    case enGB = "en-gb"
  }

  // After constructing the wrapper, call setLanguage() before phonemizing any text
  init() throws {
    if let bundleURLStr = findDataBundlePath() {
      let initOK = espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, bundleURLStr, 0)

      if initOK != Constants.successAudioSampleRate {
        print("Internal espseak-ng error, could not initialize")
        throw ESpeakNGEngineError.couldNotInitialize
      }

      var languageList: Set<String> = []
      let voiceList = espeak_ListVoices(nil)
      var index = 0
      while let voicePointer = voiceList?.advanced(by: index).pointee {
        let voice = voicePointer.pointee
        if let cLang = voice.languages {
          let language = String(cString: cLang, encoding: .utf8)!
            .replacingOccurrences(of: "\u{05}", with: "")
            .replacingOccurrences(of: "\u{02}", with: "")
          languageList.insert(language)

          if let cName = voice.identifier {
            let name = String(cString: cName, encoding: .utf8)!
              .replacingOccurrences(of: "\u{05}", with: "")
              .replacingOccurrences(of: "\u{02}", with: "")
            languageMapping[language] = name
          }
        }

        index += 1
      }

      try LanguageDialect.allCases.forEach {
        if $0.rawValue.count > 0, !languageList.contains($0.rawValue) {
          print("Language dialect \($0) not found in espeak-ng voice list")
          throw ESpeakNGEngineError.languageNotFound
        }
      }
    } else {
      print("Couldn't find the espeak-ng data bundle, cannot initialize")
      throw ESpeakNGEngineError.dataBundleNotFound
    }
  }

  // Destructor
  deinit {
    let terminateOK = espeak_Terminate()
    print("ESpeakNGEngine termination OK: \(terminateOK == EE_OK)")
  }

  // Sets the language that will be used for phonemizing
  // If the function returns without throwing an exception then consider new language set!
  func setLanguage(for voice: TTSVoice) throws {
    guard let language = Constants.voice2Language[voice],
          let name = languageMapping[language.rawValue]
    else {
      throw ESpeakNGEngineError.languageNotFound
    }

    let result = espeak_SetVoiceByName((name as NSString).utf8String)

    if result == EE_NOT_FOUND {
      throw ESpeakNGEngineError.languageNotFound
    } else if result != EE_OK {
      throw ESpeakNGEngineError.internalError
    }

    self.language = language
  }

  // Phonemizes the text string that can then be passed to the next stage
  func phonemize(text: String) throws -> String {
    guard language != .none else {
      throw ESpeakNGEngineError.languageNotSet
    }

    guard !text.isEmpty else {
      return ""
    }

    // Split only on whitespace to preserve punctuation with words
    let words = text.split(whereSeparator: { $0.isWhitespace })
    var result: [String] = []
    
    for word in words {
      if word.isEmpty { continue }
      
      // Phonemize the word
      var textPtr = UnsafeRawPointer((String(word) as NSString).utf8String)
      let phonemes_mode = Int32((Int32(Character("_").asciiValue!) << 8) | 0x02)
      if let phonemes = ESpeakNG.espeak_TextToPhonemes(&textPtr, espeakCHARS_UTF8, phonemes_mode) {
        let processed = postProcessPhonemes(String(cString: phonemes))
        // If the original word ended with punctuation, add it back
        if let lastChar = word.last, lastChar.isPunctuation {
          result.append(processed + " " + String(lastChar))
        } else {
          result.append(processed)
        }
      }
    }

    return result.joined(separator: " ")
  }

  // Post processes manually phonemes before returning them
  // NOTE: This is currently only for English, handling other langauges requires different kind of postproccessing
  private func postProcessPhonemes(_ phonemes: String) -> String {
    var result = phonemes.trimmingCharacters(in: .whitespacesAndNewlines)
    for (old, new) in Constants.E2M {
      result = result.replacingOccurrences(of: old, with: new)
    }

    result = result.replacingOccurrences(of: "(\\S)\u{0329}", with: "ᵊ$1", options: .regularExpression)
    result = result.replacingOccurrences(of: "\u{0329}", with: "")

    if language == .enGB {
      result = result.replacingOccurrences(of: "e^ə", with: "ɛː")
      result = result.replacingOccurrences(of: "iə", with: "ɪə")
      result = result.replacingOccurrences(of: "ə^ʊ", with: "Q")
    } else {
      result = result.replacingOccurrences(of: "o^ʊ", with: "O")
      result = result.replacingOccurrences(of: "ɜːɹ", with: "ɜɹ")
      result = result.replacingOccurrences(of: "ɜː", with: "ɜɹ")
      result = result.replacingOccurrences(of: "ɪə", with: "iə")
      result = result.replacingOccurrences(of: "ː", with: "")
    }

    // For espeak < 1.52
    result = result.replacingOccurrences(of: "o", with: "ɔ")
    return result.replacingOccurrences(of: "^", with: "")
  }

  // Find the data bundle that is inside the framework
  private func findDataBundlePath() -> String? {
    if let frameworkBundle = Bundle(identifier: "com.kokoro.espeakng"),
       let dataBundleURL = frameworkBundle.url(forResource: "espeak-ng-data", withExtension: "bundle")
    {
      return dataBundleURL.path
    }
    return nil
  }

  private enum Constants {
    static let successAudioSampleRate = 22050
    static let E2M: [(String, String)] = [
      ("ʔˌn\u{0329}", "tn"), ("ʔn\u{0329}", "tn"), ("ʔn", "tn"), ("ʔ", "t"),
      ("a^ɪ", "I"), ("a^ʊ", "W"),
      ("d^ʒ", "ʤ"),
      ("e^ɪ", "A"), ("e", "A"),
      ("t^ʃ", "ʧ"),
      ("ɔ^ɪ", "Y"),
      ("ə^l", "ᵊl"),
      ("ʲo", "jo"), ("ʲə", "jə"), ("ʲ", ""),
      ("ɚ", "əɹ"),
      ("r", "ɹ"),
      ("x", "k"), ("ç", "k"),
      ("ɐ", "ə"),
      ("ɬ", "l"),
      ("\u{0303}", ""),
    ].sorted(by: { $0.0.count > $1.0.count })
    static let voice2Language: [TTSVoice: LanguageDialect] = [
        .afHeart: .enUS,
        .afBella: .enUS,
        .afNicole: .enUS,
        .afSarah: .enUS,
        .afSky: .enUS,
        .amAdam: .enUS,
        .amMichael: .enUS,
        .bfEmma: .enGB,
        .bfIsabella: .enGB,
        .bmGeorge: .enGB,
        .bmLewis: .enGB,
    ]
  }
}

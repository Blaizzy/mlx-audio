//
//  ESpeakNGEngine.swift
//  Kokoro-tts-lib
//

import Foundation
import libespeak_ng

final class ESpeakNGEngine {
  private var language: LanguageDialect = .none
  private var languageMapping: [String: String] = [:]

  enum ESpeakNGEngineError: LocalizedError {
    case dataBundleNotFound
    case couldNotInitialize
    case languageNotFound
    case internalError
    case languageNotSet
    case couldNotPhonemize
    case bundleInstallFailed(String)

    var errorDescription: String? {
      switch self {
      case .dataBundleNotFound:
        return "eSpeak-NG data bundle not found"
      case .couldNotInitialize:
        return "Could not initialize eSpeak-NG engine"
      case .languageNotFound:
        return "Language not found in eSpeak-NG"
      case .internalError:
        return "Internal eSpeak-NG error"
      case .languageNotSet:
        return "Language not set for eSpeak-NG engine"
      case .couldNotPhonemize:
        return "Could not phonemize text"
      case .bundleInstallFailed(let message):
        return "eSpeak-NG bundle installation failed: \(message)"
      }
    }
  }

  // Available languages
  public enum LanguageDialect: String, CaseIterable, Sendable {
    case none = ""
    case enUS = "en-us"
    case enGB = "en-gb"
    case jaJP = "ja"
    case znCN = "yue"
    case frFR = "fr-fr"
    case hiIN = "hi"
    case itIT = "it"
    case esES = "es"
    case ptBR = "pt-br"
  }

  // Get the directory for espeak-ng compiled data
  private static func getDataDirectory() -> URL {
    let fm = FileManager.default
    let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    let espeakDir = appSupport.appendingPathComponent("espeak-ng")
    try? fm.createDirectory(at: espeakDir, withIntermediateDirectories: true)
    return espeakDir
  }

  // After constructing the wrapper, call setLanguage() before phonemizing any text
  init() throws {
    #if !targetEnvironment(simulator)
    // Use EspeakLib to install and compile the data bundle
    let dataRoot = Self.getDataDirectory()

    Log.tts.info("ESpeakNG: Installing bundle to \(dataRoot.path)")
    do {
      try EspeakLib.ensureBundleInstalled(inRoot: dataRoot)
    } catch {
      Log.tts.error("ESpeakNG: Failed to install bundle: \(error.localizedDescription)")
      throw ESpeakNGEngineError.bundleInstallFailed(error.localizedDescription)
    }

    Log.tts.debug("ESpeakNG: Bundle installed successfully")

    // Initialize espeak-ng with the data path
    espeak_ng_InitializePath(dataRoot.path)
    let initResult = espeak_ng_Initialize(nil)

    if initResult != ENS_OK {
      Log.tts.error("ESpeakNG: espeak_ng_Initialize failed with status \(initResult.rawValue)")
      throw ESpeakNGEngineError.couldNotInitialize
    }

    // Set up output mode
    let outputResult = espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, 0, nil)
    if outputResult != ENS_OK {
      Log.tts.error("ESpeakNG: espeak_ng_InitializeOutput failed with status \(outputResult.rawValue)")
      throw ESpeakNGEngineError.couldNotInitialize
    }

    Log.tts.info("ESpeakNG: Initialized successfully")

    // Build language list
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

    Log.tts.debug("ESpeakNG: Found \(languageList.count) languages")

    try LanguageDialect.allCases.forEach {
      if $0.rawValue.count > 0, !languageList.contains($0.rawValue) {
        Log.tts.error("ESpeakNG: Language dialect \($0.rawValue) not found in espeak-ng voice list")
        throw ESpeakNGEngineError.languageNotFound
      }
    }
    #else
    throw ESpeakNGEngineError.couldNotInitialize
    #endif
  }

  // Destructor
  deinit {
    #if !targetEnvironment(simulator)
    let terminateOK = espeak_Terminate()
    Log.tts.debug("ESpeakNGEngine termination OK: \(terminateOK == EE_OK)")
    #endif
  }

  // Sets the language that will be used for phonemizing
  // If the function returns without throwing an exception then consider new language set!
  func setLanguage(for voice: TTSVoice) throws {
    #if !targetEnvironment(simulator)
    guard let language = PhonemeMapping.voice2Language[voice],
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
    #else
    throw ESpeakNGEngineError.languageNotFound
    #endif
  }

  public func languageForVoice(voice: TTSVoice) throws -> LanguageDialect {
    guard let language = PhonemeMapping.voice2Language[voice] else {
      throw ESpeakNGEngineError.languageNotFound
    }
    return language
  }

  // Phonemizes the text string that can then be passed to the next stage
  func phonemize(text: String) throws -> String {
    #if !targetEnvironment(simulator)
    guard language != .none else {
      throw ESpeakNGEngineError.languageNotSet
    }

    guard !text.isEmpty else {
      return ""
    }

    let textCopy = text as NSString
    var textPtr = UnsafeRawPointer(textCopy.utf8String)
    let phonemes_mode = Int32((Int32(Character("_").asciiValue!) << 8) | 0x02)

    // Use autoreleasepool to ensure memory is managed properly
    let result = autoreleasepool { () -> [String] in
      withUnsafeMutablePointer(to: &textPtr) { ptr in
        var resultWords: [String] = []
        while ptr.pointee != nil {
          if let result = espeak_TextToPhonemes(ptr, espeakCHARS_UTF8, phonemes_mode) {
            // Create a copy of the returned string to ensure we own the memory
            resultWords.append(String(cString: result, encoding: .utf8)!)
          }
        }
        return resultWords
      }
    }

    if !result.isEmpty {
      return postProcessPhonemes(result.joined(separator: " "))
    } else {
      throw ESpeakNGEngineError.couldNotPhonemize
    }
    #else
    throw ESpeakNGEngineError.couldNotPhonemize
    #endif
  }

  // Post processes manually phonemes before returning them
  // NOTE: This is currently only for English, handling other langauges requires different kind of postproccessing
  private func postProcessPhonemes(_ phonemes: String) -> String {
    var result = phonemes.trimmingCharacters(in: .whitespacesAndNewlines)
    for (old, new) in PhonemeMapping.E2M {
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

  private enum PhonemeMapping {
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
      .afAlloy: .enUS,
      .afAoede: .enUS,
      .afBella: .enUS,
      .afHeart: .enUS,
      .afJessica: .enUS,
      .afKore: .enUS,
      .afNicole: .enUS,
      .afNova: .enUS,
      .afRiver: .enUS,
      .afSarah: .enUS,
      .afSky: .enUS,
      .amAdam: .enUS,
      .amEcho: .enUS,
      .amEric: .enUS,
      .amFenrir: .enUS,
      .amLiam: .enUS,
      .amMichael: .enUS,
      .amOnyx: .enUS,
      .amPuck: .enUS,
      .amSanta: .enUS,
      .bfAlice: .enGB,
      .bfEmma: .enGB,
      .bfIsabella: .enGB,
      .bfLily: .enGB,
      .bmDaniel: .enGB,
      .bmFable: .enGB,
      .bmGeorge: .enGB,
      .bmLewis: .enGB,
      .efDora: .esES,
      .emAlex: .esES,
      .ffSiwis: .frFR,
      .hfAlpha: .hiIN,
      .hfBeta: .hiIN,
      .hfOmega: .hiIN,
      .hmPsi: .hiIN,
      .ifSara: .itIT,
      .imNicola: .itIT,
      .jfAlpha: .jaJP,
      .jfGongitsune: .jaJP,
      .jfNezumi: .jaJP,
      .jfTebukuro: .jaJP,
      .jmKumo: .jaJP,
      .pfDora: .ptBR,
      .pmSanta: .ptBR,
      .zfXiaobei: .znCN,
      .zfXiaoni: .znCN,
      .zfXiaoxiao: .znCN,
      .zfXiaoyi: .znCN,
      .zmYunjian: .znCN,
      .zmYunxi: .znCN,
      .zmYunxia: .znCN,
      .zmYunyang: .znCN
    ]
  }
}

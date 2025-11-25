//
//  Kokoro-tts-lib
//
import Foundation
import Hub
import MLX

// Utility class for loading voices from Hugging Face Hub.
// Voice files are downloaded as safetensors and cached on disk automatically.
class VoiceLoader {
  private init() {}

  // Hugging Face repo configuration
  static let defaultRepoId = "mlx-community/Kokoro-82M-bf16"

  static var availableVoices: [TTSVoice] {
    TTSVoice.allCases
  }

  /// Load a voice from Hugging Face Hub (safetensors).
  /// Files are cached locally by Hub.snapshot() to avoid re-downloading.
  static func loadVoice(
    _ voice: TTSVoice,
    repoId: String = defaultRepoId,
    progressHandler: @escaping (Progress) -> Void = { _ in }
  ) async throws -> MLXArray {
    let voiceId = voice.identifier
    let filename = "voices/\(voiceId).safetensors"

    let modelDirectoryURL = try await Hub.snapshot(
      from: repoId,
      matching: [filename],
      progressHandler: progressHandler
    )

    let voiceFileURL = modelDirectoryURL.appending(path: filename)
    return try loadVoiceFromFile(voiceFileURL)
  }

  /// Load voice array from a local safetensors file
  private static func loadVoiceFromFile(_ url: URL) throws -> MLXArray {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw VoiceLoaderError.voiceFileNotFound(url.lastPathComponent)
    }
    let weights = try MLX.loadArrays(url: url)
    guard let voiceArray = weights["voice"] else {
      throw VoiceLoaderError.invalidVoiceFile("Missing 'voice' key in safetensors file")
    }
    return voiceArray
  }

  enum VoiceLoaderError: LocalizedError {
    case voiceFileNotFound(String)
    case invalidVoiceFile(String)

    var errorDescription: String? {
      switch self {
      case .voiceFileNotFound(let filename):
        return "Voice file not found: \(filename). Check your internet connection and try again."
      case .invalidVoiceFile(let message):
        return "Invalid voice file: \(message)"
      }
    }
  }
}

// Extension to add utility methods to TTSVoice
extension TTSVoice {
  /// The voice identifier used for file names (e.g., "af_heart")
  var identifier: String {
    switch self {
    case .afAlloy: return "af_alloy"
    case .afAoede: return "af_aoede"
    case .afBella: return "af_bella"
    case .afHeart: return "af_heart"
    case .afJessica: return "af_jessica"
    case .afKore: return "af_kore"
    case .afNicole: return "af_nicole"
    case .afNova: return "af_nova"
    case .afRiver: return "af_river"
    case .afSarah: return "af_sarah"
    case .afSky: return "af_sky"
    case .amAdam: return "am_adam"
    case .amEcho: return "am_echo"
    case .amEric: return "am_eric"
    case .amFenrir: return "am_fenrir"
    case .amLiam: return "am_liam"
    case .amMichael: return "am_michael"
    case .amOnyx: return "am_onyx"
    case .amPuck: return "am_puck"
    case .amSanta: return "am_santa"
    case .bfAlice: return "bf_alice"
    case .bfEmma: return "bf_emma"
    case .bfIsabella: return "bf_isabella"
    case .bfLily: return "bf_lily"
    case .bmDaniel: return "bm_daniel"
    case .bmFable: return "bm_fable"
    case .bmGeorge: return "bm_george"
    case .bmLewis: return "bm_lewis"
    case .efDora: return "ef_dora"
    case .emAlex: return "em_alex"
    case .ffSiwis: return "ff_siwis"
    case .hfAlpha: return "hf_alpha"
    case .hfBeta: return "hf_beta"
    case .hfOmega: return "hm_omega"
    case .hmPsi: return "hm_psi"
    case .ifSara: return "if_sara"
    case .imNicola: return "im_nicola"
    case .jfAlpha: return "jf_alpha"
    case .jfGongitsune: return "jf_gongitsune"
    case .jfNezumi: return "jf_nezumi"
    case .jfTebukuro: return "jf_tebukuro"
    case .jmKumo: return "jm_kumo"
    case .pfDora: return "pf_dora"
    case .pmSanta: return "pm_santa"
    case .zfXiaobei: return "zf_xiaobei"
    case .zfXiaoni: return "zf_xiaoni"
    case .zfXiaoxiao: return "zf_xiaoxiao"
    case .zfXiaoyi: return "zf_xiaoyi"
    case .zmYunjian: return "zm_yunjian"
    case .zmYunxi: return "zm_yunxi"
    case .zmYunxia: return "zm_yunxia"
    case .zmYunyang: return "zm_yunyang"
    }
  }

  static func fromIdentifier(_ identifier: String) -> TTSVoice? {
    TTSVoice.allCases.first { $0.identifier == identifier }
  }
}

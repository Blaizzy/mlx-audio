//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Utility class for loading voices
class VoiceLoader {
  private init() {}

  static var availableVoices: [TTSVoice] {
    Array(Constants.voiceFiles.keys)
  }

  static func loadVoice(_ voice: TTSVoice) -> MLXArray {
    let (file, ext) = Constants.voiceFiles[voice]!
    let filePath = Bundle.main.path(forResource: file, ofType: ext)!
      print(filePath)
    return try! read3DArrayFromJson(file: filePath, shape: [510, 1, 256])!
  }

  private static func read3DArrayFromJson(file: String, shape: [Int]) throws -> MLXArray? {
    guard shape.count == 3 else { return nil }

    let data = try Data(contentsOf: URL(fileURLWithPath: file))
    let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])

    var aa = Array(repeating: Float(0.0), count: shape[0] * shape[1] * shape[2])
    var aaIndex = 0

    if let nestedArray = jsonObject as? [[[Any]]] {
      guard nestedArray.count == shape[0] else { return nil }
      for a in 0 ..< nestedArray.count {
        guard nestedArray[a].count == shape[1] else { return nil }
        for b in 0 ..< nestedArray[a].count {
          guard nestedArray[a][b].count == shape[2] else { return nil }
          for c in 0 ..< nestedArray[a][b].count {
            if let n = nestedArray[a][b][c] as? Double {
              aa[aaIndex] = Float(n)
              aaIndex += 1
            } else {
              fatalError("Cannot load value \(a), \(b), \(c) as double")
            }
          }
        }
      }
    } else {
      return nil
    }

    guard aaIndex == shape[0] * shape[1] * shape[2] else {
      fatalError("Mismatch in array size: \(aaIndex) vs \(shape[0] * shape[1] * shape[2])")
    }

    return MLXArray(aa).reshaped(shape)
  }

  public enum Constants {
    static let voiceFiles: [TTSVoice: (String, String)] = [
      .afAlloy: ("af_alloy", "json"),
      .afAoede: ("af_aoede", "json"),
      .afBella: ("af_bella", "json"),
      .afHeart: ("af_heart", "json"),
      .afJessica: ("af_jessica", "json"),
      .afKore: ("af_kore", "json"),
      .afNicole: ("af_nicole", "json"),
      .afNova: ("af_nova", "json"),
      .afRiver: ("af_river", "json"),
      .afSarah: ("af_sarah", "json"),
      .afSky: ("af_sky", "json"),
      .amAdam: ("am_adam", "json"),
      .amEcho: ("am_echo", "json"),
      .amEric: ("am_eric", "json"),
      .amFenrir: ("am_fenrir", "json"),
      .amLiam: ("am_liam", "json"),
      .amMichael: ("am_michael", "json"),
      .amOnyx: ("am_onyx", "json"),
      .amPuck: ("am_puck", "json"),
      .amSanta: ("am_santa", "json"),
      .bfAlice: ("bf_alice", "json"),
      .bfEmma: ("bf_emma", "json"),
      .bfIsabella: ("bf_isabella", "json"),
      .bfLily: ("bf_lily", "json"),
      .bmDaniel: ("bm_daniel", "json"),
      .bmFable: ("bm_fable", "json"),
      .bmGeorge: ("bm_george", "json"),
      .bmLewis: ("bm_lewis", "json"),
      .efDora: ("ef_dora", "json"),
      .emAlex: ("em_alex", "json"),
      .ffSiwis: ("ff_siwis", "json"),
      .hfAlpha: ("hf_alpha", "json"),
      .hfBeta: ("hf_beta", "json"),
      .hfOmega: ("hm_omega", "json"),
      .hmPsi: ("hm_psi", "json"),
      .ifSara: ("if_sara", "json"),
      .imNicola: ("im_nicola", "json"),
      .jfAlpha: ("jf_alpha", "json"),
      .jfGongitsune: ("jf_gongitsune", "json"),
      .jfNezumi: ("jf_nezumi", "json"),
      .jfTebukuro: ("jf_tebukuro", "json"),
      .jmKumo: ("jm_kumo", "json"),
      .pfDora: ("pf_dora", "json"),
      .pmSanta: ("pm_santa", "json"),
      .zfZiaobei: ("zf_xiaobei", "json"),
      .zfXiaoni: ("zf_xiaoni", "json"),
      .zfXiaoxiao: ("zf_xiaoxiao", "json"),
      .zfZiaoyi: ("zf_xiaoyi", "json"),
      .zmYunjian: ("zm_yunjian", "json"),
      .zmYunxi: ("zm_yunxi", "json"),
      .zmYunxia: ("zm_yunxia", "json"),
      .zmYunyang: ("zm_yunyang", "json")
    ]
  }
}

// Extension to add utility methods to TTSVoice
extension TTSVoice {
  static func fromIdentifier(_ identifier: String) -> TTSVoice? {
    let reverseMapping = Dictionary(
      VoiceLoader.Constants.voiceFiles.map { (voice, fileInfo) in
        (fileInfo.0, voice)
      },
      uniquingKeysWith: { first, _ in first } // In case of duplicates, keep the first one
    )
    return reverseMapping[identifier]
  }
}

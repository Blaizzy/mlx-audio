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
        for a in 0 ..< 510 {
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

  private enum Constants {
    static let voiceFiles: [TTSVoice: (String, String)] = [
      .afHeart: ("af_heart", "json"),
      .afBella: ("af_bella", "json"),
      .afNicole: ("af_nicole", "json"),
      .afSarah: ("af_sarah", "json"),
      .afSky: ("af_sky", "json"),
      .amAdam: ("am_adam", "json"),
      .amMichael: ("am_michael", "json"),
      .bfEmma: ("bf_emma", "json"),
      .bfIsabella: ("bf_isabella", "json"),
      .bmGeorge: ("bm_george", "json"),
      .bmLewis: ("bm_lewis", "json"),
    ]
  }
}

//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Utility class for loading and preprocessing the weights for the model
class OrpheusWeightLoader {
  private init() {}

  static func loadWeightsOrpheus() -> [String: MLXArray] {
    // Hey you - put tensor file in Orpheus/Resources folder
    let filePath = Bundle.main.path(forResource: "orpheus-3b-0.1-ft-4bit", ofType: "safetensors")!
    
    if !FileManager.default.fileExists(atPath: filePath) {
      fatalError("Orpheus: Weights not found at \(filePath)")
    }
    
    do {
      let weights = try MLX.loadArrays(url: URL(fileURLWithPath: filePath))

      var processedWeights: [String: MLXArray] = [:]
      
      let groupSize = 64
      for (key, value) in weights {
        if key.hasSuffix(".weight") {
          // Detect quantized weight by dtype uint32
          if value.dtype == .uint32 {
            // Look for associated scales and biases
            let scaleKey = key.replacingOccurrences(of: ".weight", with: ".scales")
            let biasKey = key.replacingOccurrences(of: ".weight", with: ".biases")
            if let scales = weights[scaleKey], let biases = weights[biasKey] {
                let deq = Dequantizer.dequantize(value, scales: scales, biases: biases, groupSize: groupSize, bits: 4)
              processedWeights[key] = deq

            } else {
              print("WARNING: Missing scales/biases for quantized weight \(key). Loading raw.")
              processedWeights[key] = value
            }
          } else {
            processedWeights[key] = value
          }
        } else {
          // Non-weight tensors keep original
          processedWeights[key] = value
        }
      }
      
      return processedWeights
    } catch {
      print("Orpheus: Error loading weights: \(error)")
      return [:]
    }
  }
}

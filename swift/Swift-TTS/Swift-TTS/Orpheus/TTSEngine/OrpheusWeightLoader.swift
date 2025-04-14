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
    
    // Check if the file exists
    if !FileManager.default.fileExists(atPath: filePath) {
      print("Orpheus: Weights not found at \(filePath)")
      return [:]
    }
    
    do {
      let weights = try MLX.loadArrays(url: URL(fileURLWithPath: filePath))
      var processedWeights: [String: MLXArray] = [:]
      
      for (key, value) in weights {
        // Process weights based on their type
        if key.contains("weight") {
          // Only transpose if necessary based on how weights are used downstream
          // Our linear function currently handles the transpose, so load as is.
          // if key.contains("q_proj") || key.contains("k_proj") || key.contains("v_proj") {
          //   // Transpose attention projection weights (2D arrays)
          //   processedWeights[key] = value.transposed(0, 1)
          // } else if key.contains("o_proj") {
          //   // Transpose output projection weights (2D arrays)
          //   processedWeights[key] = value.transposed(0, 1)
          // } else if key.contains("gate_proj") || key.contains("up_proj") || key.contains("down_proj") {
          //   // Transpose MLP weights (2D arrays)
          //   processedWeights[key] = value.transposed(0, 1)
          // } else {
          //   processedWeights[key] = value
          // }
          processedWeights[key] = value // Load all weights as they are
        } else {
          processedWeights[key] = value
        }
      }
      
      return processedWeights
    } catch {
      print("Orpheus: Error loading weights: \(error)")
      return [:]
    }
  }

  private static func checkArrayShape(arr: MLXArray) -> Bool {
    guard arr.shape.count != 3 else { return false }

    let outChannels = arr.shape[0]
    let kH = arr.shape[1]
    let kW = arr.shape[2]

    return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
  }
}

//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Utility class for loading and preprocessing the weights for the model
class KokoroWeightLoader {
    private init() {}
    
    static func loadWeights(url: URL? = nil) -> [String: MLXArray] {
        let modelURL = url ?? {
            let filePath = Bundle.main.path(forResource: "kokoro-v1_0", ofType: "safetensors")!
            return URL(fileURLWithPath: filePath)
        }()
        
        do {
            let weights = try MLX.loadArrays(url: modelURL)
            var sanitizedWeights: [String: MLXArray] = [:]
            
            for (key, value) in weights {
                if key.hasPrefix("bert") {
                    if key.contains("position_ids") {
                        continue
                    }
                    sanitizedWeights[key] = value
                } else if key.hasPrefix("predictor") {
                    if key.contains("F0_proj.weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("N_proj.weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                } else if key.hasPrefix("text_encoder") {
                    if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                } else if key.hasPrefix("decoder") {
                    if key.contains("noise_convs"), key.hasSuffix(".weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                }
            }
            
            return sanitizedWeights
        } catch {
            print("Kokoro: Error loading weights: \(error)")
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

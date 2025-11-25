//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import Hub

public class KokoroWeightLoader {
    private init() {}

    static public let defaultRepoId = "mlx-community/Kokoro-82M-bf16"
    static let defaultWeightsFilename = "kokoro-v1_0.safetensors"

    static func loadWeights(
        repoId: String = defaultRepoId,
        filename: String = defaultWeightsFilename,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> [String: MLXArray] {
        let modelDirectoryURL = try await Hub.snapshot(
            from: repoId,
            matching: [filename],
            progressHandler: progressHandler
        )
        let weightFileURL = modelDirectoryURL.appending(path: filename)
        return try loadWeights(from: weightFileURL)
    }

    static func loadWeights(from url: URL) throws -> [String: MLXArray] {
        let weights = try MLX.loadArrays(url: url)
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
    }

    private static func checkArrayShape(arr: MLXArray) -> Bool {
        guard arr.shape.count != 3 else { return false }

        let outChannels = arr.shape[0]
        let kH = arr.shape[1]
        let kW = arr.shape[2]

        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }
}

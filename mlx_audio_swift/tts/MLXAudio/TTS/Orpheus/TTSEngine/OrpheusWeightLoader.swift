//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import Hub

public class OrpheusWeightLoader {
    private init() {}

    static public let defaultRepoId = "mlx-community/orpheus-3b-0.1-ft-4bit"
    static let defaultWeightsFilename = "model.safetensors"

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
        // Load weights directly without dequantization
        // Quantized models have .weight (uint32 packed), .scales, and .biases
        // These will be loaded into QuantizedLinear layers by the Module system
        return try MLX.loadArrays(url: url)
    }
}

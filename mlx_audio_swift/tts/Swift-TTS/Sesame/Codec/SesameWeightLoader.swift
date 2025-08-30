//
// SesameWeightLoader for Sesame TTS Mimi Codec
// Handles loading and preprocessing PyTorch weights for MLX compatibility
// Based on Python Mimi's load_pytorch_weights method

import Foundation
import MLX
import MLXNN

/// Weight loader for Sesame TTS Mimi model
/// Handles PyTorch to MLX weight conversion and key mapping
class SesameWeightLoader {
    private init() {}

    /// Load weights from safetensors file and sanitize for MLX compatibility
    /// - Parameter url: URL to the safetensors file, nil uses bundled resource
    /// - Returns: Dictionary of sanitized weights
    static func loadWeights(url: URL? = nil) -> [String: MLXArray] {
        let modelURL = url ?? {
            // Default to bundled resource if available
            if let bundlePath = Bundle.main.path(forResource: "sesame-mimi", ofType: "safetensors", inDirectory: "Sesame/Resources") {
                return URL(fileURLWithPath: bundlePath)
            }
            // Fallback - this should be provided by caller
            fatalError("No Sesame weights URL provided and no bundled resource found")
        }()

        do {
            let rawWeights = try MLX.loadArrays(url: modelURL)
            return sanitizeWeights(rawWeights)
        } catch {
            print("Sesame: Error loading weights: \(error)")
            return [:]
        }
    }

    /// Sanitize PyTorch weights for MLX compatibility
    /// Follows the same logic as Python Mimi's load_pytorch_weights method
    private static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            var sanitizedKey = key
            var sanitizedValue = value

            // Apply key transformations based on Python's load_pytorch_weights logic
            sanitizedKey = transformKey(sanitizedKey)

            // Apply value transformations for different layer types
            sanitizedValue = transformValue(sanitizedKey, sanitizedValue)

            sanitizedWeights[sanitizedKey] = sanitizedValue
        }

        return sanitizedWeights
    }

    /// Transform weight key from PyTorch naming to MLX naming
    private static func transformKey(_ key: String) -> String {
        var transformedKey = key

        // DEBUG: Log all transformations
        let originalKey = transformedKey

        // Remove underscore prefixes (PyTorch naming convention)
        transformedKey = transformedKey.replacingOccurrences(of: "^_", with: "", options: .regularExpression)

        // Handle encoder model layers
        if transformedKey.hasPrefix("encoder.model.") {
            transformedKey = transformedKey.replacingOccurrences(of: "encoder.model.", with: "encoder.")
        }

        // Handle decoder model layers
        if transformedKey.hasPrefix("decoder.model.") {
            transformedKey = transformedKey.replacingOccurrences(of: "decoder.model.", with: "decoder.")
        }

        // Handle input projection weights
        if transformedKey.hasSuffix(".in_proj_weight") {
            transformedKey = transformedKey.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
        }

        // Handle gating linear layers
        if transformedKey.hasSuffix(".linear1.weight") {
            transformedKey = transformedKey.replacingOccurrences(of: ".linear1.weight", with: ".gating.linear1.weight")
        }
        if transformedKey.hasSuffix(".linear2.weight") {
            transformedKey = transformedKey.replacingOccurrences(of: ".linear2.weight", with: ".gating.linear2.weight")
        }

        // Handle upsample temporal conv layer key mapping
        if transformedKey.hasPrefix("upsample.convtr.convtr.convtr.") {
            transformedKey = transformedKey.replacingOccurrences(of: "upsample.convtr.convtr.convtr.", with: "upsample.convtr.convtr.convtr.")
        }

        // Handle encoder layer mappings (based on Python's hardcoded mapping)
        let encoderMappings = [
            1: 0,   // encoder.1. -> encoder.layers.0.residuals.0.
            4: 0,   // encoder.4. -> encoder.layers.0.residuals.0.
            7: 1,   // encoder.7. -> encoder.layers.1.residuals.0.
            10: 1,  // encoder.10. -> encoder.layers.1.residuals.0.
            13: 2,  // encoder.13. -> encoder.layers.2.residuals.0.
            16: 2   // encoder.16. -> encoder.layers.2.residuals.0.
        ]

        for (pytorchIdx, mlxIdx) in encoderMappings {
            if transformedKey.contains("encoder.\(pytorchIdx).") {
                transformedKey = transformedKey.replacingOccurrences(of: "encoder.\(pytorchIdx).", with: "encoder.layers.\(mlxIdx).residuals.0.")
            }
        }

        // Handle decoder layer mappings
        let decoderMappings = [
            2: 0,   // decoder.2. -> decoder.layers.0.upsample.
            5: 0,   // decoder.5. -> decoder.layers.0.residuals.0.
            8: 1,   // decoder.8. -> decoder.layers.1.upsample.
            11: 1,  // decoder.11. -> decoder.layers.1.residuals.0.
            14: 2,   // decoder.14. -> decoder.layers.2.upsample.
            17: 2   // decoder.17. -> decoder.layers.2.residuals.0.
        ]

        for (pytorchIdx, mlxIdx) in decoderMappings {
            if transformedKey.contains("decoder.\(pytorchIdx).") {
                if pytorchIdx % 3 == 2 { // upsample layers
                    transformedKey = transformedKey.replacingOccurrences(of: "decoder.\(pytorchIdx).", with: "decoder.layers.\(mlxIdx).upsample.")
                } else { // residual layers
                    transformedKey = transformedKey.replacingOccurrences(of: "decoder.\(pytorchIdx).", with: "decoder.layers.\(mlxIdx).residuals.0.")
                }
            }
        }

        // Handle initial/final conv layers
        transformedKey = transformedKey.replacingOccurrences(of: "decoder.0.", with: "decoder.init_conv1d.")
        transformedKey = transformedKey.replacingOccurrences(of: "decoder.20.", with: "decoder.final_conv1d.")
        transformedKey = transformedKey.replacingOccurrences(of: "encoder.0.", with: "encoder.init_conv1d.")
        transformedKey = transformedKey.replacingOccurrences(of: "encoder.19.", with: "encoder.final_conv1d.")

        // Handle conv block indices
        transformedKey = transformedKey.replacingOccurrences(of: ".block.1.", with: ".block.0.")
        transformedKey = transformedKey.replacingOccurrences(of: ".block.3.", with: ".block.1.")

        // DEBUG: Log transformations for temporal layers
        if originalKey.contains("downsample") || originalKey.contains("upsample") || transformedKey != originalKey {
            print("🔍 DEBUG transformKey: '\(originalKey)' -> '\(transformedKey)'")
        }

        return transformedKey
    }

    /// Transform weight values for MLX compatibility
    private static func transformValue(_ key: String, _ value: MLXArray) -> MLXArray {
        var transformedValue = value

        // Handle convolution weights (PyTorch: outC, inC, kSize → MLX: outC, kSize, inC)
        if key.contains(".conv.weight") ||
           key.contains(".output_proj.weight") ||
           key.contains(".input_proj.weight") {
            if value.shape.count == 3 {
                transformedValue = value.swappedAxes(-1, -2)
            }
        }

        // Handle transposed convolution weights (PyTorch: inC, outC, kSize → MLX: outC, kSize, inC)
        if key.contains(".convtr.weight") {
            if value.shape.count == 3 {
                // Special case for upsample layer: Input shape appears to be [512, 1, 4] from logs
                // but we need to check the actual input shape in logs
                if key.contains("upsample") {
                    // From logs: the raw PyTorch weight seems to be shape [512, 1, 4]
                    // MLX expects [512, 4, 1]
                    // So we need to swap the last two dimensions: [512, 1, 4] -> [512, 4, 1]
                    transformedValue = value.swappedAxes(-1, -2)
                } else {
                    // PyTorch transposed conv: [inC, outC, kSize] -> MLX: [outC, kSize, inC]
                    transformedValue = value.transposed(1, 2, 0)
                }
                
                print("🔧 DEBUG transformValue: convtr weight '\(key)' transformed from \(value.shape) to \(transformedValue.shape)")
            }
        }

        // Handle linear layer weights (PyTorch: outC, inC → MLX: inC, outC)
        if key.contains(".weight") &&
           !key.contains(".conv") &&
           !key.contains(".convtr") &&
           value.shape.count == 2 {
            transformedValue = value.transposed(1, 0)
        }

        return transformedValue
    }

    /// Load PyTorch weights with comprehensive key mapping (Python-compatible)
    /// - Parameters:
    ///   - url: URL to the safetensors file
    ///   - strict: Whether to fail on missing keys
    /// - Returns: Updated model with loaded weights
    static func loadPytorchWeights(model: Mimi, url: URL, strict: Bool = true) -> Mimi {
        let rawWeights = try? MLX.loadArrays(url: url)
        guard let weights = rawWeights else {
            if strict {
                fatalError("Failed to load weights from \(url)")
            }
            return model
        }

        print("Loading \(weights.count) weight tensors...")
        
        // DEBUG: Print some raw weight keys before transformation
        print("🔍 DEBUG: Sample raw weight keys:")
        for (i, key) in weights.keys.enumerated() {
            if i < 10 {
                print("  \(key)")
            }
        }
        
        var sanitizedWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            var sanitizedKey = key
            var sanitizedValue = value

            // Apply comprehensive key transformations
            let originalKey = sanitizedKey
            sanitizedKey = transformKey(sanitizedKey)
            sanitizedValue = transformValue(sanitizedKey, sanitizedValue)

            // DEBUG: Log key transformations for conv layers
            if originalKey.contains("conv") || originalKey.contains("upsample") || originalKey.contains("downsample") {
                print("🔍 DEBUG key transform: '\(originalKey)' -> '\(sanitizedKey)' (shape: \(sanitizedValue.shape))")
            }

            sanitizedWeights[sanitizedKey] = sanitizedValue
        }

        // Print some keys for debugging
        print("Sample sanitized weight keys:")
        for (i, key) in sanitizedWeights.keys.enumerated() {
            if i < 5 {
                print("  \(key)")
            }
        }
        
        // DEBUG: Look for upsample/downsample weights specifically
        print("🔍 DEBUG: Looking for temporal conv weights...")
        for key in sanitizedWeights.keys {
            if key.contains("upsample") || key.contains("downsample") {
                print("  Found temporal weight: \(key) (shape: \(sanitizedWeights[key]!.shape))")
            }
        }

        // Convert to nested dictionary format for ModuleParameters
        var nestedWeights: [String: NestedItem<String, MLXArray>] = [:]
        for (key, value) in sanitizedWeights {
            nestedWeights[key] = .value(value)
        }

        // Apply weights to model using MLX's update method
        do {
            try model.update(parameters: ModuleParameters(values: nestedWeights), verify: strict ? .all : .none)
            print("Successfully loaded and applied Mimi weights")
            
            // DEBUG: Check if our temporal layers got weights
            print("🔍 DEBUG: Checking if temporal layers received weights...")
            print("  Upsample weight shape: \(model.upsample.weight.shape)")
            print("  Upsample weight range: \(model.upsample.weight.min().item(Float.self)) to \(model.upsample.weight.max().item(Float.self))")
            print("  Downsample conv weight shape: \(model.downsample.conv.weight.shape)")
            print("  Downsample weight range: \(model.downsample.conv.weight.min().item(Float.self)) to \(model.downsample.conv.weight.max().item(Float.self))")
            
            return model
        } catch {
            print("🚨 ERROR: Failed to load weights into model: \(error)")
            if strict {
                fatalError("Failed to load weights into model: \(error)")
            } else {
                print("Warning: Failed to load some weights: \(error)")
                return model
            }
        }
    }

    /// Create Mimi model from pretrained weights
    /// - Parameters:
    ///   - config: Mimi configuration
    ///   - repoId: HuggingFace repository ID (future use)
    ///   - filename: Weight filename
    /// - Returns: Initialized Mimi model with loaded weights
    static func fromPretrained(
        config: MimiConfig,
        repoId: String? = nil,
        filename: String = "tokenizer-e351c8d8-checkpoint125.safetensors"
    ) -> Mimi {
        let model = Mimi(config)

        // For now, expect weights to be provided via URL
        // Future: Implement HuggingFace download
        print("Sesame: Please provide weight file URL for loading pretrained model")
        print("Usage: loadPytorchWeights(model: model, url: weightFileURL)")

        return model
    }

    /// Standard Mimi 2024.07 configuration factory
    static func mimi202407(numCodebooks: Int = 32) -> MimiConfig {
        let seanet = SeanetConfig(
            dimension: 512,
            channels: 1,
            causal: true,
            nfilters: 64,
            nresidualLayers: 1,
            ratios: [8, 6, 5, 4],
            ksize: 7,
            residualKsize: 3,
            lastKsize: 3,
            dilationBase: 2,
            padMode: "constant",
            trueSkip: true,
            compress: 2
        )

        let transformer = TransformerConfig(
            dModel: seanet.dimension,
            numHeads: 8,
            numLayers: 8,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: 0.01,
            positionalEmbedding: "rope",
            useConvBlock: false,
            crossAttention: false,
            convKernelSize: 3,
            useConvBias: true,
            gating: false,
            norm: "layer_norm",
            context: 250,
            maxPeriod: 10000,
            maxSeqLen: 8192,
            kvRepeat: 1,
            dimFeedforward: 2048,
            convLayout: true
        )

        return MimiConfig(
            channels: 1,
            sampleRate: 24000,
            frameRate: 12.5,
            renormalize: true,
            seanet: seanet,
            transformer: transformer,
            quantizerNq: numCodebooks,
            quantizerBins: 2048,
            quantizerDim: 256
        )
    }
}

// MARK: - Configuration Extensions

extension MimiConfig {
    /// Create standard Mimi 2024.07 configuration
    static func standard202407(numCodebooks: Int = 32) -> MimiConfig {
        return SesameWeightLoader.mimi202407(numCodebooks: numCodebooks)
    }
}

extension Mimi {
    /// Load PyTorch weights from file
    func loadPytorchWeights(url: URL, strict: Bool = true) -> Mimi {
        return SesameWeightLoader.loadPytorchWeights(model: self, url: url, strict: strict)
    }

    /// Create from pretrained weights
    static func fromPretrained(
        repoId: String? = nil,
        filename: String = "tokenizer-e351c8d8-checkpoint125.safetensors",
        numCodebooks: Int = 32
    ) -> Mimi {
        let config = MimiConfig.standard202407(numCodebooks: numCodebooks)
        return SesameWeightLoader.fromPretrained(
            config: config,
            repoId: repoId,
            filename: filename
        )
    }
}
//
//  DACModel.swift
//  MLXAudio
//
//  DAC (Descript Audio Codec) encoder and decoder
//  Ported from: mlx_audio/codec/models/descript/dac.py
//

import Foundation
import MLX
import MLXNN
import Hub

// MARK: - Encoder Block

/// Encoder block with residual units and strided convolution
/// Uses Sequential to match Python weight key structure
class DACEncoderBlock: Module, UnaryLayer {
    let block: Sequential

    init(dim: Int, stride: Int) {
        self.block = Sequential(layers: [
            DACResidualUnit(dim: dim / 2, dilation: 1),
            DACResidualUnit(dim: dim / 2, dilation: 3),
            DACResidualUnit(dim: dim / 2, dilation: 9),
            DACSnake1d(channels: dim / 2),
            DACWNConv1d(
                inChannels: dim / 2,
                outChannels: dim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            )
        ])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return block(x)
    }
}

// MARK: - Encoder

/// DAC Encoder: audio -> latent representation
/// Uses Sequential to match Python weight key structure
class DACEncoder: Module, UnaryLayer {
    let block: Sequential
    let encDim: Int

    init(dModel: Int = 64, strides: [Int] = [2, 4, 8, 8], dLatent: Int = 64) {
        var currentDim = dModel
        var layers: [UnaryLayer] = []

        // Initial convolution
        layers.append(DACWNConv1d(
            inChannels: 1,
            outChannels: dModel,
            kernelSize: 7,
            padding: 3
        ))

        // Encoder blocks
        for stride in strides {
            currentDim *= 2
            layers.append(DACEncoderBlock(dim: currentDim, stride: stride))
        }

        // Final layers
        layers.append(DACSnake1d(channels: currentDim))
        layers.append(DACWNConv1d(
            inChannels: currentDim,
            outChannels: dLatent,
            kernelSize: 3,
            padding: 1
        ))

        self.block = Sequential(layers: layers)
        self.encDim = currentDim

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] -> apply conv
        let y = block(x)
        // Output: [batch, time, latent_dim] (move axis for compatibility)
        return y.transposed(axes: [0, 2, 1])
    }
}

// MARK: - Decoder Block

/// Decoder block with upsampling and residual units
/// Uses Sequential to match Python weight key structure
class DACDecoderBlock: Module, UnaryLayer {
    let block: Sequential

    init(inputDim: Int, outputDim: Int, stride: Int) {
        self.block = Sequential(layers: [
            DACSnake1d(channels: inputDim),
            DACWNConvTranspose1d(
                inChannels: inputDim,
                outChannels: outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            ),
            DACResidualUnit(dim: outputDim, dilation: 1),
            DACResidualUnit(dim: outputDim, dilation: 3),
            DACResidualUnit(dim: outputDim, dilation: 9)
        ])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return block(x)
    }
}

// MARK: - Decoder

/// DAC Decoder: latent representation -> audio
/// Uses Sequential to match Python weight key structure
class DACDecoder: Module, UnaryLayer {
    let model: Sequential

    init(inputChannel: Int, channels: Int, rates: [Int], dOut: Int = 1) {
        var layers: [UnaryLayer] = []

        // Initial convolution (model.layers.0)
        layers.append(DACWNConv1d(
            inChannels: inputChannel,
            outChannels: channels,
            kernelSize: 7,
            padding: 3
        ))

        // Decoder blocks (model.layers.1 to model.layers.N)
        for (i, stride) in rates.enumerated() {
            let inputDim = channels / Int(pow(2.0, Double(i)))
            let outputDim = channels / Int(pow(2.0, Double(i + 1)))
            layers.append(DACDecoderBlock(
                inputDim: inputDim,
                outputDim: outputDim,
                stride: stride
            ))
        }

        // Final layers
        let finalDim = channels / Int(pow(2.0, Double(rates.count)))
        layers.append(DACSnake1d(channels: finalDim))
        layers.append(DACWNConv1d(
            inChannels: finalDim,
            outChannels: dOut,
            kernelSize: 7,
            padding: 3
        ))

        self.model = Sequential(layers: layers)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = model(x)
        // Apply tanh activation (no weights, not in Sequential)
        return MLX.tanh(y)
    }
}

// MARK: - DAC Configuration

/// Configuration for DAC model
struct DACConfig: Codable {
    let encoderDim: Int
    let encoderRates: [Int]
    let latentDim: Int?
    let decoderDim: Int
    let decoderRates: [Int]
    let nCodebooks: Int
    let codebookSize: Int
    let codebookDim: Int
    let sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_hidden_size"
        case encoderRates = "downsampling_ratios"
        case latentDim = "hidden_size"
        case decoderDim = "decoder_hidden_size"
        case decoderRates = "upsampling_ratios"
        case nCodebooks = "n_codebooks"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case sampleRate = "sampling_rate"
    }

    static let speechDefault = DACConfig(
        encoderDim: 64,
        encoderRates: [2, 4, 5, 8],
        latentDim: nil,
        decoderDim: 1536,
        decoderRates: [8, 5, 4, 2],
        nCodebooks: 2,
        codebookSize: 1024,
        codebookDim: 8,
        sampleRate: 24000
    )
}

// MARK: - DAC Model

/// DAC (Descript Audio Codec) - Full encoder-decoder model with quantization
public final class DACCodec {
    public static let defaultRepoId = "mlx-community/dac-speech-24khz-1.5kbps"

    private let encoder: DACEncoder
    private let decoder: DACDecoder
    private let quantizer: DACResidualVectorQuantize
    private let config: DACConfig
    public let hopLength: Int
    public let sampleRate: Int

    private init(config: DACConfig, encoder: DACEncoder, decoder: DACDecoder, quantizer: DACResidualVectorQuantize) {
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.hopLength = config.encoderRates.reduce(1, *)
        self.sampleRate = config.sampleRate
    }

    /// Load DAC model from Hugging Face Hub
    public static func fromPretrained(
        repoId: String = defaultRepoId,
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> DACCodec {
        // Download model files from Hub
        let modelDir = try await HubApi.shared.snapshot(
            from: repoId,
            matching: ["*.safetensors", "*.json"],
            progressHandler: progressHandler
        )

        // Load config
        let configURL = modelDir.appending(path: "config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(DACConfig.self, from: configData)

        // Calculate latent dim if not specified
        let latentDim = config.latentDim ?? (config.encoderDim * Int(pow(2.0, Double(config.encoderRates.count))))

        // Initialize model components
        let encoder = DACEncoder(
            dModel: config.encoderDim,
            strides: config.encoderRates,
            dLatent: latentDim
        )

        let decoder = DACDecoder(
            inputChannel: latentDim,
            channels: config.decoderDim,
            rates: config.decoderRates
        )

        let quantizer = DACResidualVectorQuantize(
            inputDim: latentDim,
            nCodebooks: config.nCodebooks,
            codebookSize: config.codebookSize,
            codebookDim: config.codebookDim
        )

        // Load weights
        let weightsURL = modelDir.appending(path: "model.safetensors")
        let weights = try MLX.loadArrays(url: weightsURL)

        // Apply weights to encoder
        try applyWeights(to: encoder, weights: weights, prefix: "encoder")

        // Apply weights to decoder
        try applyWeights(to: decoder, weights: weights, prefix: "decoder")

        // Apply weights to quantizer
        try applyWeights(to: quantizer, weights: weights, prefix: "quantizer")

        return DACCodec(config: config, encoder: encoder, decoder: decoder, quantizer: quantizer)
    }

    /// Encode audio to latent codes
    public func encode(_ audioData: MLXArray, nQuantizers: Int? = nil) -> (z: MLXArray, codes: MLXArray) {
        // Preprocess audio
        let audio = preprocess(audioData)

        // Encode
        let z = encoder(audio.transposed(axes: [0, 2, 1]))

        // Quantize
        let (zQ, codes, _, _, _) = quantizer(z, nQuantizers: nQuantizers)

        return (zQ, codes)
    }

    /// Decode latent representation to audio
    public func decode(_ z: MLXArray) -> MLXArray {
        return decoder(z.transposed(axes: [0, 2, 1]))
    }

    /// Decode from codes directly
    public func decodeFromCodes(_ codes: MLXArray) -> MLXArray {
        let (zQ, _, _) = quantizer.fromCodes(codes)
        return decode(zQ)
    }

    /// Preprocess audio data (pad to hop length)
    private func preprocess(_ audioData: MLXArray) -> MLXArray {
        let length = audioData.shape[audioData.ndim - 1]
        let rightPad = Int(ceil(Double(length) / Double(hopLength))) * hopLength - length

        if rightPad > 0 {
            // Pad the last dimension
            return MLX.padded(audioData, widths: [IntOrPair([0, 0]), IntOrPair([0, 0]), IntOrPair([0, rightPad])])
        }
        return audioData
    }

    /// Apply weights from dictionary to a module
    private static func applyWeights(to module: Module, weights: [String: MLXArray], prefix: String) throws {
        // Filter weights for this prefix and strip the prefix
        let filteredWeights = weights.filter { $0.key.hasPrefix(prefix + ".") }
            .mapKeys { String($0.dropFirst(prefix.count + 1)) }

        let parameters = ModuleParameters.unflattened(filteredWeights)
        try module.update(parameters: parameters, verify: [.noUnusedKeys])
    }
}

// MARK: - Dictionary Extension

extension Dictionary where Key == String {
    func mapKeys(_ transform: (Key) -> Key) -> [Key: Value] {
        Dictionary(uniqueKeysWithValues: map { (transform($0.key), $0.value) })
    }
}

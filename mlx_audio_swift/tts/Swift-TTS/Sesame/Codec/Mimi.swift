//
// Mimi Codec for Sesame TTS
// Using MLXNN directly (like Kokoro/Orpheus) for standard components

import Foundation
import MLX
import MLXNN
import MLXFast

/// Configuration for Mimi codec (matches Python MimiConfig)
struct MimiConfig {
    let channels: Int
    let sampleRate: Float
    let frameRate: Float
    let renormalize: Bool
    let seanet: SeanetConfig
    let transformer: TransformerConfig
    let quantizerNq: Int
    let quantizerBins: Int
    let quantizerDim: Int



    /// Create Mimi 2024.07 configuration (matches Python)
    static func mimi202407(numCodebooks: Int) -> MimiConfig {
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

/// Main Mimi codec class - matching Python implementation
class Mimi: Module {
    @ModuleInfo var encoder: SeanetEncoder
    @ModuleInfo var decoder: SeanetDecoder
    @ModuleInfo var quantizer: SplitResidualVectorQuantizer
    @ModuleInfo var downsample: ConvDownsample1d
    @ModuleInfo var upsample: ConvTrUpsample1d
    @ModuleInfo var encoderTransformer: ProjectedTransformer
    @ModuleInfo var decoderTransformer: ProjectedTransformer

    // Cache management
    private var encoderCache: [KVCacheProtocol]?
    private var decoderCache: [KVCacheProtocol]?

    private(set) var config: MimiConfig
    private let downsampleStride: Int

    /// Sample rate of the codec (read-only property)
    var sampleRate: Float {
        return config.sampleRate
    }

    /// Frame rate of the codec (read-only property)
    var frameRate: Float {
        return config.frameRate
    }

    init(_ config: MimiConfig) {
        self.config = config

        // Calculate downsample stride for temporal processing
        let encoderFrameRate = Float(config.sampleRate) / Float(config.seanet.ratios.reduce(1, *))
        self.downsampleStride = Int(encoderFrameRate / Float(config.frameRate))

        // Initialize encoder
        self._encoder.wrappedValue = SeanetEncoder(config.seanet)

        // Initialize decoder
        self._decoder.wrappedValue = SeanetDecoder(config.seanet)

        // Initialize quantizer - FIXED: Match Python configuration exactly
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: config.quantizerDim,           // 256 - matches Python cfg.quantizer_dim
            inputDim: config.seanet.dimension,  // 512 - matches Python dim (seanet dimension)
            outputDim: config.seanet.dimension, // 512 - matches Python dim (seanet dimension)
            nq: config.quantizerNq,             // 32 - matches Python cfg.quantizer_nq
            bins: config.quantizerBins          // 2048 - matches Python cfg.quantizer_bins
        )

        // Initialize temporal processing components
        self._downsample.wrappedValue = ConvDownsample1d(
            stride: downsampleStride,
            dim: config.seanet.dimension,
            causal: config.seanet.causal
        )

        self._upsample.wrappedValue = ConvTrUpsample1d(
            stride: downsampleStride,
            dim: config.seanet.dimension,
            causal: config.seanet.causal
        )

        // Initialize transformers
        let encoderTransformer = ProjectedTransformer.encoder(
            dModel: config.seanet.dimension,
            inputDim: config.seanet.dimension,
            outputDim: config.seanet.dimension
        )
        self._encoderTransformer.wrappedValue = encoderTransformer

        let decoderTransformer = ProjectedTransformer.decoder(
            dModel: config.seanet.dimension,
            inputDim: config.seanet.dimension,
            outputDim: config.seanet.dimension
        )
        self._decoderTransformer.wrappedValue = decoderTransformer

        // Initialize caches
        self.encoderCache = encoderTransformer.makeCache()
        self.decoderCache = decoderTransformer.makeCache()

        super.init()
    }

    /// Encode audio to discrete codes - matching Python architecture
    func encode(_ audio: MLXArray) -> MLXArray {
        // Ensure proper shape [batch, channels, time]
        var x = audio
        if x.ndim == 2 {
            x = x.expandedDimensions(axis: 0)  // Add batch dimension
        }
        if x.shape[1] != config.channels {
            // Transpose if needed
            x = x.swappedAxes(1, 2)
        }

        // Step 1: Seanet encoding
        x = encoder(x)

        // Step 2: Encoder transformer processing
        if let cache = encoderCache {
            x = encoderTransformer(x, cache: cache)[0]
        }

        // Step 3: Temporal downsampling for compression
        x = downsample(x)

        // Step 4: Quantization to discrete codes
        return quantizer.encode(x)
    }

    /// Decode discrete codes to audio - matching Python architecture
    func decode(_ codes: MLXArray) -> MLXArray {
        print("🔍 DEBUG Mimi decode: input codes shape = \(codes.shape)")
        
        // Validate input shape
        guard codes.ndim == 3 else {
            fatalError("Mimi decode expects 3D input [batch, codebooks, time], got \(codes.ndim)D: \(codes.shape)")
        }
        
        guard codes.shape[1] == numCodebooks else {
            fatalError("Mimi decode expects \(numCodebooks) codebooks, got \(codes.shape[1]) in shape \(codes.shape)")
        }
        
        // Step 1: Dequantize from discrete codes to latent representation
        print("🔍 DEBUG Mimi decode: calling quantizer.decode...")
        var x = quantizer.decode(codes)
        print("🔍 DEBUG Mimi decode: quantizer.decode returned shape = \(x.shape)")

        // Step 2: Temporal upsampling for reconstruction
        print("🔍 DEBUG Mimi decode: calling upsample with input shape = \(x.shape)")
        x = upsample(x)
        print("🔍 DEBUG Mimi decode: upsample returned shape = \(x.shape)")

        // Step 3: Decoder transformer processing
        if let cache = decoderCache {
            print("🔍 DEBUG Mimi decode: calling decoder transformer...")
            x = decoderTransformer(x, cache: cache)[0]
            print("🔍 DEBUG Mimi decode: decoder transformer returned shape = \(x.shape)")
        }

        // Step 4: Seanet decoding to audio
        print("🔍 DEBUG Mimi decode: calling seanet decoder...")
        let decoded = decoder(x)
        print("🔍 DEBUG Mimi decode: seanet decoder returned shape = \(decoded.shape)")

        // Return audio in original format
        return decoded
    }

    /// Get frame rate for temporal alignment
    var outputFrameRate: Float {
        return frameRate
    }

    /// Get sample rate
    var outputSampleRate: Float {
        return sampleRate
    }

    /// Get number of codebooks
    var numCodebooks: Int {
        return config.quantizerNq
    }

    // MARK: - Weight Loading

    // Weight loading is handled by SesameWeightLoader extension
    // See SesameWeightLoader.swift for loadPytorchWeights implementation

    /// Reset all caches (for new sequences)
    func resetCache() {
        encoderCache?.forEach { $0.reset() }
        decoderCache?.forEach { $0.reset() }
    }

    /// Get current cache state for inspection
    func getCacheState() -> (encoder: [(MLXArray?, MLXArray?)]?, decoder: [(MLXArray?, MLXArray?)]?) {
        let encoderState = encoderCache?.map {
            ($0.keys, $0.values) as (MLXArray?, MLXArray?)
        }
        let decoderState = decoderCache?.map {
            ($0.keys, $0.values) as (MLXArray?, MLXArray?)
        }
        return (encoderState, decoderState)
    }

    /// Get transformer configurations
    var encoderTransformerConfig: TransformerConfig {
        return encoderTransformer.transformerConfig
    }

    var decoderTransformerConfig: TransformerConfig {
        return decoderTransformer.transformerConfig
    }

    /// Get codebook size
    var codebookSize: Int {
        return config.quantizerBins
    }

    // MARK: - Streaming Methods (Matching Python Implementation)

    /// Reset all streaming state
    func resetState() {
        encoder.resetState()
        decoder.resetState()

        // Reset transformer caches
        encoderCache?.forEach { $0.reset() }
        decoderCache?.forEach { $0.reset() }
    }

    /// Encode audio step-by-step for streaming
    func encodeStep(_ xs: MLXArray) -> MLXArray {
        var x = xs

        // Ensure proper shape
        if x.ndim == 2 {
            x = x.expandedDimensions(axis: 0)
        }
        if x.shape[1] != config.channels {
            x = x.swappedAxes(1, 2)
        }

        // Step 1: Seanet encoding (streaming)
        x = encoder.step(x)

        // Step 2: Encoder transformer processing
        if let cache = encoderCache {
            x = encoderTransformer(x, cache: cache)[0]
        }

        // Step 3: Temporal downsampling (streaming)
        x = downsample.step(x)

        // Step 4: Quantization
        return quantizer.encode(x)
    }

    /// Decode discrete codes step-by-step for streaming
    func decodeStep(_ xs: MLXArray) -> MLXArray {
        // Step 1: Dequantize
        var x = quantizer.decode(xs)

        // Step 2: Temporal upsampling (streaming)
        x = upsample.step(x)

        // Step 3: Decoder transformer processing
        if let cache = decoderCache {
            x = decoderTransformer(x, cache: cache)[0]
        }

        // Step 4: Seanet decoding (streaming)
        return decoder.step(x)
    }

    /// Warmup the model for initialization
    func warmup() {
        let pcm = MLXArray.zeros([1, 1, 1920 * 4])
        let codes = encode(pcm)
        let pcmOut = decode(codes)
        MLX.eval(pcmOut)
    }
}

/// Streaming version of Mimi decoder (for real-time generation)
/// Equivalent to Python's MimiStreamingDecoder
public class MimiStreamingDecoder {
    private let mimi: Mimi

    init(_ mimi: Mimi) {
        self.mimi = mimi
        reset()
    }

    /// Reset the underlying codec state
    func reset() {
        mimi.resetState()
    }

    /// Decode a sequence of audio tokens incrementally
    public func decodeFrames(_ tokens: MLXArray) -> MLXArray {
        var inputTokens = tokens

        // Ensure proper shape [batch, codebooks, time]
        if inputTokens.ndim == 2 {
            inputTokens = inputTokens.expandedDimensions(axis: 0)
        }

        var pcm: [MLXArray] = []

        // Process each frame incrementally
        for t in 0..<inputTokens.shape[2] {
            let stepTokens = inputTokens[0..., 0..., t..<(t + 1)]
            let framePcm = mimi.decodeStep(stepTokens)
            pcm.append(framePcm)
        }

        // Concatenate all frames along the time axis (last axis)
        return MLX.concatenated(pcm, axis: -1)
    }

    /// Get the underlying Mimi model
    var model: Mimi {
        return mimi
    }
}
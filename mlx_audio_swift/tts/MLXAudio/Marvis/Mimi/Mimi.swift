import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

// MARK: - Configs

public struct MimiConfig {
    public let channels: Int
    public let sampleRate: Double
    public let frameRate: Double
    public let renormalize: Bool
    public let seanet: SeanetConfig
    public let transformer: TransformerConfig
    public let quantizerNQ: Int
    public let quantizerBins: Int
    public let quantizerDim: Int

    public init(
        channels: Int,
        sampleRate: Double,
        frameRate: Double,
        renormalize: Bool,
        seanet: SeanetConfig,
        transformer: TransformerConfig,
        quantizerNQ: Int,
        quantizerBins: Int,
        quantizerDim: Int
    ) {
        self.channels = channels
        self.sampleRate = sampleRate
        self.frameRate = frameRate
        self.renormalize = renormalize
        self.seanet = seanet
        self.transformer = transformer
        self.quantizerNQ = quantizerNQ
        self.quantizerBins = quantizerBins
        self.quantizerDim = quantizerDim
    }
}

@inline(__always) private func product(_ xs: [Int]) -> Int { xs.reduce(1, *) }

public func mimi_202407(numCodebooks: Int) -> MimiConfig {
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
        padMode: .constant,
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
        maxPeriod: 10_000,
        maxSeqLen: 8_192,
        kvRepeat: 1,
        dimFeedforward: 2_048,
        convLayout: true // transformer expects [B,C,T] at API boundary
    )
    return MimiConfig(
        channels: 1,
        sampleRate: 24_000,
        frameRate: 12.5,
        renormalize: true,
        seanet: seanet,
        transformer: transformer,
        quantizerNQ: numCodebooks,
        quantizerBins: 2_048,
        quantizerDim: 256
    )
}

// MARK: - Mimi

public final class Mimi: Module {
    public let cfg: MimiConfig

    @ModuleInfo public var encoder: SeanetEncoder
    @ModuleInfo public var decoder: SeanetDecoder
    @ModuleInfo public var quantizer: SplitResidualVectorQuantizer

    @ModuleInfo public var encoder_transformer: ProjectedTransformer
    @ModuleInfo public var decoder_transformer: ProjectedTransformer

    @ModuleInfo public var downsample: ConvDownsample1d
    @ModuleInfo public var upsample: ConvTrUpsample1d

    public private(set) var encoderCache: [KVCache]
    public private(set) var decoderCache: [KVCache]

    private let downsampleStride: Int

    public init(cfg: MimiConfig) {
        self.cfg = cfg

        let encFPS = cfg.sampleRate / Double(product(cfg.seanet.ratios))
        self.downsampleStride = Int(encFPS / cfg.frameRate)

        self._encoder = ModuleInfo(wrappedValue: SeanetEncoder(cfg: cfg.seanet))
        self._decoder = ModuleInfo(wrappedValue: SeanetDecoder(cfg: cfg.seanet))

        self._quantizer = ModuleInfo(wrappedValue: SplitResidualVectorQuantizer(
            dim: cfg.quantizerDim,
            inputDim: cfg.seanet.dimension,
            outputDim: cfg.seanet.dimension,
            nq: cfg.quantizerNQ,
            bins: cfg.quantizerBins
        ))

        self._encoder_transformer = ModuleInfo(wrappedValue: ProjectedTransformer(
            cfg: cfg.transformer,
            inputDim: cfg.seanet.dimension,
            outputDims: [cfg.seanet.dimension]
        ))
        self._decoder_transformer = ModuleInfo(wrappedValue: ProjectedTransformer(
            cfg: cfg.transformer,
            inputDim: cfg.seanet.dimension,
            outputDims: [cfg.seanet.dimension]
        ))

        self._downsample = ModuleInfo(wrappedValue: ConvDownsample1d(
            stride: downsampleStride, dim: cfg.seanet.dimension, causal: true
        ))
        self._upsample = ModuleInfo(wrappedValue: ConvTrUpsample1d(
            stride: downsampleStride, dim: cfg.seanet.dimension, causal: true
        ))

        self.encoderCache = _encoder_transformer.wrappedValue.makeCache()
        self.decoderCache = _decoder_transformer.wrappedValue.makeCache()
    }

    public func resetState() {
        encoder.resetState()
        decoder.resetState()
        for c in decoderCache { c.reset() }
        for c in encoderCache { c.reset() }
    }

    public var frameRate: Double { cfg.frameRate }
    public var sampleRate: Double { cfg.sampleRate }

    public func encode(_ xs: MLXArray) -> MLXArray {
        encoder.resetState()
        for c in encoderCache { c.reset() }

        var z = encoder(xs)
        z = encoder_transformer(z, cache: encoderCache)[0]
        z = downsample(z)
        return quantizer.encode(z) // [B, nq, Tq]
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        decoder.resetState()
        for c in decoderCache { c.reset() }

        var z = quantizer.decode(codes) // [B, Cdim, Tq]
        z = upsample(z)
        z = decoder_transformer(z, cache: decoderCache)[0]
        return decoder(z) // [B, 1, T]
    }

    public func encodeStep(_ xs: MLXArray) -> MLXArray {
        var z = encoder.step(xs)
        z = encoder_transformer(z, cache: encoderCache)[0]
        z = downsample.step(z)
        z = quantizer.encode(z)
        return z
    }

    public func decodeStep(_ codes: MLXArray) -> MLXArray {
        var z = quantizer.decode(codes)
        z = upsample.step(z)
        z = decoder_transformer(z, cache: decoderCache)[0]
        z = decoder.step(z)
        return z
    }
}

// MARK: - Streaming

public final class MimiStreamingDecoder {
    private let mimi: Mimi

    public init(_ mimi: Mimi) {
        self.mimi = mimi
        reset()
    }

    public func reset() {
        mimi.decoder.resetState()
        mimi.upsample.resetState()
        for c in mimi.decoderCache { c.reset() }
    }

    public func decodeFrames(_ tokens: MLXArray) -> MLXArray {
        let tok = (tokens.ndim == 2) ? tokens.expandedDimensions(axes: [0]) : tokens // ensure [B,C,T]
        let T = tok.shape[2]

        var pcs: [MLXArray] = []
        for t in 0 ..< T {
            let left = split(tok, indices: [t], axis: 2)
            let mid = split(left[1], indices: [1], axis: 2)[0]
            pcs.append(mimi.decodeStep(mid))
        }
        return concatenated(pcs, axis: 2) // [B, 1, samples]
    }
}

public extension Mimi {
    static func fromPretrained(repoId: String = "kyutai/moshiko-pytorch-bf16", filename: String = "tokenizer-e351c8d8-checkpoint125.safetensors", progressHandler: @escaping (Progress) -> Void) async throws -> Mimi {
        print("[Mimi] Starting Mimi model loading from \(repoId)")

        print("[Mimi] Creating configuration...")
        let cfg = mimi_202407(numCodebooks: 32)

        print("[Mimi] Initializing Mimi model with config...")
        let modelInitStart = CFAbsoluteTimeGetCurrent()
        let model = Mimi(cfg: cfg)
        let modelInitTime = CFAbsoluteTimeGetCurrent() - modelInitStart
        print(String(format: "[Mimi] Model initialization completed in %.2f seconds", modelInitTime))

        print("[Mimi] Downloading/snapshotting weights file...")
        let snapshotStart = CFAbsoluteTimeGetCurrent()
        let weightFileURL = try await Hub.snapshot(from: repoId, matching: filename, progressHandler: progressHandler).appending(path: filename)
        let snapshotTime = CFAbsoluteTimeGetCurrent() - snapshotStart
        print(String(format: "[Mimi] Weights file snapshot completed in %.2f seconds", snapshotTime))

        print("[Mimi] Loading weight arrays from safetensors file...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        var weights = [String: MLXArray]()
        let w = try loadArrays(url: weightFileURL)
        for (key, value) in w {
            weights[key] = value
        }
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "[Mimi] Weight arrays loaded in %.2f seconds. Total weights: %d", loadTime, weights.count))

        print("[Mimi] Sanitizing weights...")
        let sanitizeStart = CFAbsoluteTimeGetCurrent()
        weights = model.sanitize(weights: weights)
        let sanitizeTime = CFAbsoluteTimeGetCurrent() - sanitizeStart
        print(String(format: "[Mimi] Weights sanitized in %.2f seconds. Final weight count: %d", sanitizeTime, weights.count))

        print("[Mimi] Processing codebook updates...")
        let filterStart = CFAbsoluteTimeGetCurrent()
        func filterFn(_ module: Module, _ name: String, _ item: ModuleItem) -> Bool {
            if let codebook = module as? EuclideanCodebook, name == "initialized" {
                codebook.updateInPlace()
            }
            return true
        }
        _ = model.filterMap(filter: filterFn)
        let filterTime = CFAbsoluteTimeGetCurrent() - filterStart
        print(String(format: "[Mimi] Codebook processing completed in %.2f seconds", filterTime))

        print("[Mimi] Updating model parameters...")
        let updateStart = CFAbsoluteTimeGetCurrent()
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        let updateTime = CFAbsoluteTimeGetCurrent() - updateStart
        print(String(format: "[Mimi] Model parameters updated in %.2f seconds", updateTime))

        print("[Mimi] Evaluating model...")
        let evalStart = CFAbsoluteTimeGetCurrent()
        eval(model)
        let evalTime = CFAbsoluteTimeGetCurrent() - evalStart
        print(String(format: "[Mimi] Model evaluation completed in %.2f seconds", evalTime))

        print("[Mimi] Mimi model loading completed successfully")
        return model
    }

    private func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (rawKey, rawVal) in weights {
            var k = rawKey
                .split(separator: ".")
                .map { seg -> String in
                    if seg.hasPrefix("_") { return String(seg.dropFirst()) }
                    return String(seg)
                }
                .joined(separator: ".")

            if k.hasPrefix("encoder.model.") {
                k = k.replacingOccurrences(of: "encoder.model.", with: "encoder.")
            }
            if k.hasPrefix("decoder.model.") {
                k = k.replacingOccurrences(of: "decoder.model.", with: "decoder.")
            }

            if k.hasSuffix(".in_proj_weight") {
                k = k.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
            }
            if k.hasSuffix(".linear1.weight") {
                k = k.replacingOccurrences(of: ".linear1.weight", with: ".gating.linear1.weight")
            }
            if k.hasSuffix(".linear2.weight") {
                k = k.replacingOccurrences(of: ".linear2.weight", with: ".gating.linear2.weight")
            }

            let decIdx = [2, 5, 8, 11]
            for (layerIdx, decoderIdx) in decIdx.enumerated() {
                k = k.replacingOccurrences(of: "decoder.\(decoderIdx).",
                                           with: "decoder.layers.\(layerIdx).upsample.")
                k = k.replacingOccurrences(of: "decoder.\(decoderIdx + 1).",
                                           with: "decoder.layers.\(layerIdx).residuals.0.")
            }
            let encIdx = [1, 4, 7, 10]
            for (layerIdx, encoderIdx) in encIdx.enumerated() {
                k = k.replacingOccurrences(of: "encoder.\(encoderIdx).",
                                           with: "encoder.layers.\(layerIdx).residuals.0.")
                k = k.replacingOccurrences(of: "encoder.\(encoderIdx + 2).",
                                           with: "encoder.layers.\(layerIdx).downsample.")
            }

            k = k.replacingOccurrences(of: "decoder.0.", with: "decoder.init_conv1d.")
            k = k.replacingOccurrences(of: "decoder.14.", with: "decoder.final_conv1d.")
            k = k.replacingOccurrences(of: "encoder.0.", with: "encoder.init_conv1d.")
            k = k.replacingOccurrences(of: "encoder.14.", with: "encoder.final_conv1d.")
            k = k.replacingOccurrences(of: ".block.1.", with: ".block.0.")
            k = k.replacingOccurrences(of: ".block.3.", with: ".block.1.")

            var v = rawVal
            if k.hasSuffix(".conv.weight")
                || k.hasSuffix(".output_proj.weight")
                || k.hasSuffix(".input_proj.weight") {
                if v.ndim >= 2 {
                    v = swappedAxes(v, v.ndim - 1, v.ndim - 2)
                }
            }
            if k.hasSuffix(".convtr.weight") {
                if v.ndim == 3 {
                    var w = swappedAxes(v, 0, 1) // [1,0,2]
                    w = swappedAxes(w, 1, 2) // [1,2,0]
                    v = w
                }
            }

            out[k] = v
        }

        return out
    }
}

// MARK: -

public final class MimiTokenizer {
    let codec: Mimi
    public init(_ codec: Mimi) {
        codec.train(false)
        self.codec = codec
    }
}

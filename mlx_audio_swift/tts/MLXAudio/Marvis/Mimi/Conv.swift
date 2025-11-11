import Foundation
import MLX
import MLXNN

// MARK: - Conv1d (NCL wrapper over MLX's NLC)

public final class Conv1d: Module {

    public var weight: MLXArray
    public var bias: MLXArray?

    public let padding: Int
    public let groups: Int
    public let stride: Int
    public let dilation: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        // Uniform init in [-scale, scale]
        let scale: Float = 1.0 / Float(inChannels * ksize)
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outChannels, ksize, inChannels / groups]
        )
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
    }

    // NCL -> NLC -> conv1d -> NCL
    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let xsNLC = swappedAxes(xsNCL, 1, 2)
        var y = conv1d(
            xsNLC, weight,
            stride: stride, padding: padding,
            dilation: dilation, groups: groups
        )
        if let b = bias { y = y + b }
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - ConvTranspose1d (NCL wrapper)

public final class ConvTranspose1d: Module {

    public var weight: MLXArray
    public var bias: MLXArray?

    public let padding: Int
    public let groups: Int
    public let stride: Int
    public let ksize: Int
    public let inChannels: Int
    public let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        // Weight shape mirrors your Python: (out_channels // groups, ksize, in_channels)
        let scale: Float = 1.0 / Float(inChannels * ksize)
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outChannels / groups, ksize, inChannels]
        )
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.ksize = ksize
        self.inChannels = inChannels
        self.outChannels = outChannels
    }

    // Expand weight as needed to emulate grouped depthwise transposed conv like the Python version
    private func expandedWeightAndGroups() -> (MLXArray, Int) {
        if groups == inChannels && groups == outChannels {
            var eyeW = eye(outChannels)
                .asType(weight.dtype)
                .reshaped([outChannels, 1, outChannels])
            eyeW = repeated(eyeW, count: ksize, axis: 1) // repeat along kernel dim
            let wRep = repeated(weight, count: groups, axis: 0)
            return (wRep * eyeW, 1)
        } else if groups > 1 {
            fatalError("groups > 1 (non-depthwise) not supported in ConvTranspose1d")
        } else {
            return (weight, groups)
        }
    }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let xsNLC = swappedAxes(xsNCL, 1, 2)
        let (wEff, gEff) = expandedWeightAndGroups()
        var y = convTransposed1d(xsNLC, wEff, stride: stride, padding: padding, groups: gEff)
        if let b = bias { y = y + b }
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - Normalized wrappers (kept as simple pass-through like Python)

public final class NormConv1d: Module {
    @ModuleInfo public var conv: Conv1d

    public init(
        inChannels: Int, outChannels: Int, ksize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, dilation: Int = 1, bias: Bool = true
    ) {
        self._conv = ModuleInfo(wrappedValue: Conv1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: padding, groups: groups, dilation: dilation, bias: bias
        ))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
}

public final class NormConvTranspose1d: Module {
    @ModuleInfo public var convtr: ConvTranspose1d

    public init(
        inChannels: Int, outChannels: Int, ksize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, bias: Bool = true
    ) {
        self._convtr = ModuleInfo(wrappedValue: ConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: padding, groups: groups, bias: bias
        ))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray { convtr(xs) }
}

// MARK: - Helpers

@inline(__always)
func getExtraPaddingForConv1d(xs: MLXArray, ksize: Int, stride: Int, paddingTotal: Int) -> Int {
    let len = xs.shape[2]
    let nframes = max(len + paddingTotal - ksize, 0)
    let nf = Double(nframes) / Double(stride) + 1.0
    let idealLen = (Int(ceil(nf)) - 1) * stride + ksize - paddingTotal
    return max(0, idealLen - len)
}

// Unpad along last axis using split (avoids relying on slicing syntax)
@inline(__always)
func unpad1d(_ xs: MLXArray, unpadL: Int, unpadR: Int) -> MLXArray {
    let L = xs.shape[2]
    let parts = split(xs, indices: [unpadL, L - unpadR], axis: 2)
    return parts[1] // middle segment
}

// MARK: - StreamableConv1d

public final class StreamableConv1d: Module {
    private let causal: Bool
    private let padMode: PadMode
    private let ksizeBase: Int
    @ModuleInfo public var conv: NormConv1d

    private var prevXs: MLXArray? = nil
    private var leftPadApplied = false
    private let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        dilation: Int,
        groups: Int,
        bias: Bool,
        causal: Bool,
        padMode: PadMode
    ) {
        self.causal = causal
        self.padMode = padMode
        self.ksizeBase = ksize
        self._conv = ModuleInfo(wrappedValue: NormConv1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: 0, groups: groups, dilation: dilation, bias: bias
        ))
        self.outChannels = outChannels
    }

    public func resetState() {
        prevXs = nil
        leftPadApplied = false
    }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        // Effective kernel size with dilation
        let dil = conv.conv.dilation
        let kEff = (ksizeBase - 1) * dil + 1
        let paddingTotal = kEff - conv.conv.stride
        let extra = getExtraPaddingForConv1d(
            xs: xsNCL, ksize: kEff, stride: conv.conv.stride, paddingTotal: paddingTotal
        )
        let z = IntOrPair(0)
        let pad: (Int, Int) = {
            if causal { return (paddingTotal, 0) }
            let pr = paddingTotal / 2
            return (paddingTotal - pr, pr)
        }()
        let (padL, padR) = pad
        let widths: [IntOrPair] = [z, z, IntOrPair((padL, padR + extra))]
        let xPad = padded(xsNCL, widths: widths, mode: padMode)
        return conv(xPad)
    }

    // Streaming step; input/output are NCL
    public func step(_ xsNCL: MLXArray) -> MLXArray {
        let b = xsNCL.shape[0]
        let len = xsNCL.shape[2]
        if len == 0 { return MLXArray.zeros([b, outChannels, 0]) }

        let stride = conv.conv.stride
        let dilation = conv.conv.dilation
        let kEff = (ksizeBase - 1) * dilation + 1

        var x = xsNCL
        if !leftPadApplied {
            leftPadApplied = true
            let padTotal = kEff - stride
            x = padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((padTotal, 0))], mode: padMode)
        }

        if let prev = prevXs {
            x = concatenated([prev, x], axis: 2)
        }

        let L = x.shape[2]
        let nframes = max(L + stride - kEff, 0) / stride
        if nframes > 0 {
            let offset = nframes * stride
            // stash tail for next call: x[..., offset:]
            let tailSplit = split(x, indices: [offset], axis: 2)
            prevXs = tailSplit.count > 1 ? tailSplit[1] : nil
            
            let inLen = (nframes - 1) * stride + kEff
            let keep = split(x, indices: [inLen], axis: 2)[0]
            return conv(keep)
        } else {
            prevXs = x
            return MLXArray.zeros([b, outChannels, 0])
        }
    }
}

// MARK: - StreamableConvTranspose1d

public final class StreamableConvTranspose1d: Module {
    private let causal: Bool
    private let ksize: Int
    @ModuleInfo public var convtr: NormConvTranspose1d

    private var prevYs: MLXArray? = nil
    private let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        groups: Int,
        bias: Bool,
        causal: Bool
    ) {
        self.causal = causal
        self.ksize = ksize
        self._convtr = ModuleInfo(wrappedValue: NormConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: 0, groups: groups, bias: bias
        ))
        self.outChannels = outChannels
    }

    public func resetState() { prevYs = nil }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let stride = convtr.convtr.stride
        let paddingTotal = max(ksize - stride, 0)
        let y = convtr(xsNCL)
        let (unL, unR): (Int, Int) = {
            if causal { return (0, paddingTotal) }
            let r = paddingTotal / 2
            return (paddingTotal - r, r)
        }()
        return unpad1d(y, unpadL: unL, unpadR: unR)
    }

    public func step(_ xsNCL: MLXArray) -> MLXArray {
        let b = xsNCL.shape[0]
        let len = xsNCL.shape[2]
        if len == 0 { return MLXArray.zeros([b, outChannels, 0]) }

        let stride = convtr.convtr.stride
        var y = convtr(xsNCL)
        let ot = y.shape[2]

        if var prev = prevYs {
            let pt = prev.shape[2]
            if let b = convtr.convtr.bias { prev = prev - b.reshaped([1, b.shape[0], 1]) }
            // overlap-add
            let head = split(y, indices: [pt], axis: 2)
            let combined = head[0] + prev
            y = concatenated([combined, head[1]], axis: 2)
        }

        let invalid = ksize - stride
        let parts = split(y, indices: [max(ot - invalid, 0)], axis: 2)
        let valid = parts[0]
        prevYs = parts.count > 1 ? parts[1] : nil
        return valid
    }
}

// MARK: - Upsample/Downsample wrappers

public final class ConvDownsample1d: Module {
    @ModuleInfo public var conv: StreamableConv1d

    public init(stride: Int, dim: Int, causal: Bool) {
        self._conv = ModuleInfo(wrappedValue: StreamableConv1d(
            inChannels: dim, outChannels: dim, ksize: 2*stride,
            stride: stride, dilation: 1, groups: 1, bias: false,
            causal: causal, padMode: .edge
        ))
    }

    public func resetState() { conv.resetState() }
    public func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
    public func step(_ xs: MLXArray) -> MLXArray { conv.step(xs) }
}

public final class ConvTrUpsample1d: Module {
    @ModuleInfo public var convtr: StreamableConvTranspose1d

    public init(stride: Int, dim: Int, causal: Bool) {
        self._convtr = ModuleInfo(wrappedValue: StreamableConvTranspose1d(
            inChannels: dim, outChannels: dim, ksize: 2*stride,
            stride: stride, groups: dim, bias: false, causal: causal
        ))
    }

    public func resetState() { convtr.resetState() }
    public func callAsFunction(_ xs: MLXArray) -> MLXArray { convtr(xs) }
    public func step(_ xs: MLXArray) -> MLXArray { convtr.step(xs) }
}

import MLX
import MLXNN

public final class KVCache: AttentionCache {

    public let nKVHeads: Int
    public let kHeadDim: Int
    public let vHeadDim: Int

    public private(set) var keys: MLXArray? = nil   // [B, H, Tk, Dh_k]
    public private(set) var values: MLXArray? = nil // [B, H, Tk, Dh_v]
    public private(set) var offset: Int = 0
    public var step: Int = 256

    public init(headDim: Int, nKVHeads: Int, step: Int = 256) {
        self.nKVHeads = nKVHeads
        self.kHeadDim = headDim
        self.vHeadDim = headDim
        self.step = step
    }

    public init(kHeadDim: Int, vHeadDim: Int, nKVHeads: Int, step: Int = 256) {
        self.nKVHeads = nKVHeads
        self.kHeadDim = kHeadDim
        self.vHeadDim = vHeadDim
        self.step = step
    }

    public func reset() {
        offset = 0
        keys = nil
        values = nil
    }

    public var state: (MLXArray?, MLXArray?) { (keys, values) }

    public func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        // Basic shape checks
        let B = k.shape[0]
        precondition(k.shape[1] == nKVHeads && v.shape[1] == nKVHeads, "nKVHeads mismatch")
        let t = k.shape[2]
        precondition(k.shape[3] == kHeadDim, "k head dim mismatch")
        precondition(v.shape[3] == vHeadDim, "v head dim mismatch")
        if let kk = keys { precondition(kk.shape[0] == B, "batch size changed in KV cache") }

        ensureCapacity(timeToAppend: t, batch: B, kDType: k.dtype, vDType: v.dtype)

        let prev = offset
        offset += t

        if let kBase = keys, let vBase = values {
            keys = replaceSlice(base: kBase, axis: 2, start: prev, length: t, with: k)
            values = replaceSlice(base: vBase, axis: 2, start: prev, length: t, with: v)
        }

        let kUsed = split(keys!, indices: [offset], axis: 2)[0]
        let vUsed = split(values!, indices: [offset], axis: 2)[0]
        return (kUsed, vUsed)
    }

    // MARK: - Internal helpers

    private func ensureCapacity(timeToAppend t: Int, batch B: Int, kDType: DType, vDType: DType) {
        let prev = offset
        if keys == nil || (prev + t) > keys!.shape[2] {
            let nSteps = (t + step - 1) / step
            let allocT = nSteps * step

            let newK = MLXArray.zeros([B, nKVHeads, allocT, kHeadDim]).asType(kDType)
            let newV = MLXArray.zeros([B, nKVHeads, allocT, vHeadDim]).asType(vDType)

            if var kExisting = keys, var vExisting = values {
                if prev % step != 0 {
                    kExisting = split(kExisting, indices: [prev], axis: 2)[0]
                    vExisting = split(vExisting, indices: [prev], axis: 2)[0]
                }
                keys = concatenated([kExisting, newK], axis: 2)
                values = concatenated([vExisting, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }
    }

    private func replaceSlice(base: MLXArray, axis: Int, start: Int, length: Int, with repl: MLXArray) -> MLXArray {
        let split1 = split(base, indices: [start], axis: axis)
        let left = split1[0]
        let right = split1[1]
        let split2 = split(right, indices: [length], axis: axis)
        let rightRest = split2.count > 1 ? split2[1] : concatenated([], axis: axis) // empty along axis
        return concatenated([left, repl, rightRest], axis: axis)
    }
}
//
// KVCache for Sesame TTS ProjectedTransformer
// Efficient key-value caching for transformer attention layers
// Based on MLX Python implementation

import Foundation
import MLX
import MLXFast

/// Base KVCache protocol
protocol KVCacheProtocol {
    var offset: Int { get set }
    var keys: MLXArray? { get set }
    var values: MLXArray? { get set }

    func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func reset()
}

/// Standard KVCache implementation
class KVCache: KVCacheProtocol {
    let nKvHeads: Int
    let kHeadDim: Int
    let vHeadDim: Int
    var offset: Int = 0
    var keys: MLXArray?
    var values: MLXArray?
    private let step: Int = 256

    init(headDim: Int, nKvHeads: Int) {
        self.nKvHeads = nKvHeads
        self.kHeadDim = headDim
        self.vHeadDim = headDim
    }

    init(headDim: (Int, Int), nKvHeads: Int) {
        self.nKvHeads = nKvHeads
        self.kHeadDim = headDim.0
        self.vHeadDim = headDim.1
    }

    func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset

        // Grow cache if needed
        if self.keys == nil || (prev + keys.shape[2]) > self.keys!.shape[2] {
            let B = keys.shape[0]
            let nSteps = (step + keys.shape[2] - 1) / step
            let kShape = [B, nKvHeads, nSteps * step, kHeadDim]
            let vShape = [B, nKvHeads, nSteps * step, vHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let existingKeys = self.keys, let existingValues = self.values {
                // Handle partial step case
                if prev % step != 0 {
                    self.keys = existingKeys[0..<prev, axis: 2]
                    self.values = existingValues[0..<prev, axis: 2]
                }
                self.keys = MLX.concatenated([existingKeys, newK], axis: 2)
                self.values = MLX.concatenated([existingValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        // Update cache with new keys and values
        self.offset += keys.shape[2]
        self.keys![prev..<offset, axis: 2] = keys
        self.values![prev..<offset, axis: 2] = values

        return (
            self.keys![0..<offset, axis: 2],
            self.values![0..<offset, axis: 2]
        )
    }

    func reset() {
        offset = 0
        keys = nil
        values = nil
    }

    var state: (MLXArray?, MLXArray?) {
        return (keys, values)
    }
}

/// Rotating KVCache with fixed maximum size
class RotatingKVCache: KVCacheProtocol {
    let nKvHeads: Int
    let kHeadDim: Int
    let vHeadDim: Int
    let maxSize: Int
    let keep: Int
    var offset: Int = 0
    var keys: MLXArray?
    var values: MLXArray?
    private let step: Int = 256
    private var idx: Int = 0

    init(headDim: Int, nKvHeads: Int, maxSize: Int, keep: Int = 0) {
        self.nKvHeads = nKvHeads
        self.kHeadDim = headDim
        self.vHeadDim = headDim
        self.maxSize = maxSize
        self.keep = keep
    }

    init(headDim: (Int, Int), nKvHeads: Int, maxSize: Int, keep: Int = 0) {
        self.nKvHeads = nKvHeads
        self.kHeadDim = headDim.0
        self.vHeadDim = headDim.1
        self.maxSize = maxSize
        self.keep = keep
    }

    private func trim(_ trimSize: Int, _ v: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toConcat: [MLXArray] = []

        if trimSize > 0 {
            toConcat.append(v[0..<keep, axis: 2])
            toConcat.append(v[(trimSize + keep)..<v.shape[2], axis: 2])
        } else {
            toConcat.append(v)
        }

        if let append = append {
            toConcat.append(append)
        }

        return MLX.concatenated(toConcat, axis: 2)
    }

    func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset
        let B = keys.shape[0]
        let S = keys.shape[2]

        // Prefill mode (S > 1)
        if S > 1 {
            if self.keys == nil {
                self.keys = keys
                self.values = values
            } else {
                let trimSize = self.keys!.shape[2] - maxSize + 1
                self.keys = trim(trimSize, self.keys!, append: keys)
                self.values = trim(trimSize, self.values!, append: values)
            }
            offset += S
            idx = self.keys!.shape[2]
            return (self.keys!, self.values!)
        }

        // Generation mode
        if self.keys == nil || (prev >= self.keys!.shape[2] && self.keys!.shape[2] < maxSize) {
            let newSize = min(step, maxSize - prev)
            let kShape = [B, nKvHeads, newSize, kHeadDim]
            let vShape = [B, nKvHeads, newSize, vHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let existingKeys = self.keys, let existingValues = self.values {
                self.keys = MLX.concatenated([existingKeys, newK], axis: 2)
                self.values = MLX.concatenated([existingValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            idx = prev
        }

        // Trim if needed
        let trimSize = self.keys!.shape[2] - maxSize
        if trimSize > 0 {
            self.keys = trim(trimSize, self.keys!)
            self.values = trim(trimSize, self.values!)
            idx = maxSize
        }

        // Rotate
        if idx == maxSize {
            idx = keep
        }

        // Assign new values
        self.keys![idx..<(idx + 1), axis: 2] = keys
        self.values![idx..<(idx + 1), axis: 2] = values
        offset += 1
        idx += 1

        // Return appropriate slice
        if offset < maxSize {
            return (
                self.keys![0..<offset, axis: 2],
                self.values![0..<offset, axis: 2]
            )
        }
        return (self.keys!, self.values!)
    }

    func reset() {
        offset = 0
        idx = 0
        keys = nil
        values = nil
    }

    var state: (MLXArray?, MLXArray?) {
        return (keys, values)
    }
}

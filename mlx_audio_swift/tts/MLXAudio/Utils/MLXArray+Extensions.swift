import MLX

// MARK: - arange extension
// Temporary workaround until (if) `arange` is added to MLX Swift.
// Provides Python-compatible arange syntax: MLXArray.arange(stop) or MLXArray.arange(start, stop, step: step)

extension MLXArray {

    /// Generate values in the half-open interval `[0, stop)`.
    ///
    /// Example:
    /// ```swift
    /// let r = MLXArray.arange(10) // [0, 1, 2, ..., 9]
    /// ```
    ///
    /// - Parameter stop: End of the sequence (exclusive).
    /// - Returns: An array containing the generated range as int32.
    static func arange(_ stop: Int) -> MLXArray {
        guard stop > 0 else { return MLXArray([Int32]()) }
        return MLXArray(Array(0..<stop).map { Int32($0) })
    }

    /// Generate values in the half-open interval `[start, stop)` spaced by `step`.
    ///
    /// Example:
    /// ```swift
    /// let r = MLXArray.arange(10)              // [0, 1, 2, ..., 9]
    /// let r2 = MLXArray.arange(2, 10, step: 2) // [2, 4, 6, 8]
    /// ```
    ///
    /// - Parameters:
    ///   - start: Starting value of the sequence. Defaults to 0.
    ///   - stop: End of the sequence (exclusive).
    ///   - step: Spacing between values. Defaults to 1.
    /// - Returns: An array containing the generated range as int32.
    static func arange(_ start: Int = 0, _ stop: Int, step: Int = 1) -> MLXArray {
        guard step != 0 else { fatalError("Step cannot be zero.") }
        guard (step > 0 && start < stop) || (step < 0 && start > stop) else {
            return MLXArray([Int32]())
        }
        return MLXArray(Swift.stride(from: start, to: stop, by: step).map { Int32($0) })
    }

    /// Generate values in the half-open interval `[start, stop)` with a given dtype.
    static func arange(_ start: Int = 0, _ stop: Int, step: Int = 1, dtype: DType) -> MLXArray {
        guard step != 0 else { fatalError("Step cannot be zero.") }
        guard (step > 0 && start < stop) || (step < 0 && start > stop) else {
            return MLXArray([Int32]())
        }
        let array = MLXArray(Swift.stride(from: start, to: stop, by: step).map { Int32($0) })
        return dtype == .int32 ? array : array.asType(dtype)
    }

    /// Generate floating point values in the half-open interval `[start, stop)`.
    static func arange(_ start: Double, _ stop: Double, step: Double = 1.0, dtype: DType = .float32) -> MLXArray {
        guard step != 0.0 else { fatalError("Step cannot be zero.") }
        guard (step > 0 && start < stop) || (step < 0 && start > stop) else {
            return MLXArray([Float]())
        }
        let array = MLXArray(Swift.stride(from: start, to: stop, by: step).map { Float($0) })
        return dtype == .float32 ? array : array.asType(dtype)
    }
}

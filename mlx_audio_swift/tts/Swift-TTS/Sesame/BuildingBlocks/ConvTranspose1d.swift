//
// Sesame BuildingBlocks - Using MLXNN directly (like Kokoro/Orpheus)
// This file provides any custom building blocks needed for Sesame TTS

import Foundation
import MLX
import MLXNN

// MARK: - Custom Building Blocks (only when needed)

// Example: Custom ConvTranspose1d if needed for Sesame
// For now, we use MLXNN.ConvTransposed1d directly in our modules
class SesameConvTranspose1d: MLXNN.ConvTransposed1d {
    /// Initialize with Sesame-specific parameters if needed
    convenience init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.init(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }
}

//
// Sesame BuildingBlocks - Using MLXNN directly (like Kokoro/Orpheus)
// This file provides any custom building blocks needed for Sesame TTS
// All basic MLXNN components are used directly without wrappers

import Foundation
import MLX
import MLXNN

// MARK: - Custom Building Blocks (only when needed)

// Example: Custom Conv1d with specific initialization for Sesame
// Only create custom classes when we need behavior different from MLXNN
class SesameConv1d: MLXNN.Conv1d {
    /// Initialize with Sesame-specific Xavier initialization
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

        // Custom initialization if needed
        // MLXNN.Conv1d already provides good defaults
    }
}

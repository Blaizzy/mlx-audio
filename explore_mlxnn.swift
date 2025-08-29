import MLX
import MLXNN

// Print available MLXNN types
print("=== MLXNN Types ===")
print("Convolution related:")
if let convType = NSClassFromString("MLXNN.Conv1d") {
    print("Conv1d class found: \(convType)")
} else {
    print("Conv1d class not found")
}

if let convTransposeType = NSClassFromString("MLXNN.ConvTranspose1d") {
    print("ConvTranspose1d class found: \(convTransposeType)")
} else {
    print("ConvTranspose1d class not found")
}

// Try to access MLXNN namespace
print("\n=== MLXNN Module Contents ===")
let mirror = Mirror(reflecting: MLXNN.self)
print("MLXNN children count: \(mirror.children.count)")
for child in mirror.children {
    if let label = child.label {
        print("- \(label): \(type(of: child.value))")
    }
}

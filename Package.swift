// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "mlx-audio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(
            name: "MLXAudio",
            targets: ["MLXAudio", "ESpeakNG"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.2"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
    ],
    targets: [
        .binaryTarget(
            name: "ESpeakNG",
            path: "mlx_audio_swift/tts/MLXAudio/Kokoro/Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "MLXAudio",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
                "ESpeakNG"
            ],
            path: "mlx_audio_swift/tts/MLXAudio",
            exclude: ["Preview Content", "Assets.xcassets", "MLXAudioApp.swift", "MLXAudio.entitlements"],
            resources: [
                .process("Kokoro/Resources") // Kokoro voices
            ]
        ),
        .testTarget(
            name: "MLXAudioTests",
            dependencies: ["MLXAudio"],
            path: "mlx_audio_swift/tts/Tests"
        ),
    ]
)

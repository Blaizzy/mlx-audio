// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "mlx-audio",
    platforms: [.macOS("15.4"), .iOS("18.4")],
    products: [
        .library(
            name: "MLXAudio",
            targets: ["MLXAudio"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
        .package(url: "https://github.com/espeak-ng/espeak-ng-spm.git", branch: "master"),
    ],
    targets: [
        .target(
            name: "MLXAudio",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "libespeak-ng", package: "espeak-ng-spm"),
                .product(name: "espeak-ng-data", package: "espeak-ng-spm"),
            ],
            path: "mlx_audio_swift/tts/MLXAudio"
        ),
        .testTarget(
            name: "MLXAudioTests",
            dependencies: ["MLXAudio"],
            path: "mlx_audio_swift/tts/Tests"
        ),
    ]
)

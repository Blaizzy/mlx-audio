// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "mlx-audio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(
            name: "MLXAudio",
            targets: ["MLXAudio", "MLXESpeakNG"]
        ),
        .library(
            name: "MLXAudioSTT",
            targets: ["MLXAudioSTT"]
        ),
        .library(
            name: "MLXAudioSTS",
            targets: ["MLXAudioSTS"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "2.29.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
        .package(url: "https://github.com/pvieito/PythonKit.git", branch: "master"),
    ],
    targets: [
        // MARK: - TTS
        .binaryTarget(
            name: "MLXESpeakNG",
            path: "mlx_audio_swift/tts/MLXAudio/Kokoro/Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "MLXAudio",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                "MLXESpeakNG"
            ],
            path: "mlx_audio_swift/tts/MLXAudio",
            exclude: ["Preview Content", "Assets.xcassets", "MLXAudioApp.swift", "MLXAudio.entitlements"],
            resources: [
                .process("Kokoro/Resources")
            ]
        ),
        .testTarget(
            name: "MLXAudioTests",
            dependencies: ["MLXAudio"],
            path: "mlx_audio_swift/tts/Tests"
        ),

        // MARK: - Core (shared utilities)
        .target(
            name: "MLXAudioCore",
            dependencies: [
                .product(name: "PythonKit", package: "PythonKit"),
            ],
            path: "mlx_audio_swift/Core"
        ),

        // MARK: - STT (Speech-to-Text via PythonKit)
        .target(
            name: "MLXAudioSTT",
            dependencies: [
                "MLXAudioCore",
                .product(name: "PythonKit", package: "PythonKit"),
            ],
            path: "mlx_audio_swift/stt"
        ),

        // MARK: - STS (Speech-to-Speech pipeline)
        .target(
            name: "MLXAudioSTS",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioSTT",
            ],
            path: "mlx_audio_swift/sts"
        ),

        // MARK: - STT/STS Tests
        .testTarget(
            name: "MLXAudioSTTTests",
            dependencies: ["MLXAudioSTT", "MLXAudioSTS", "MLXAudioCore"],
            path: "mlx_audio_swift/Tests"
        ),
    ]
)

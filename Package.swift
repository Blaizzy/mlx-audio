// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MLXSwiftTTS",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "MLXSwiftTTS",
            targets: ["MLXSwiftTTS","ESpeakNG"]),
    ],
    dependencies: [
         .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.2")
    ],
    targets: [
        .binaryTarget(
            name: "ESpeakNG",
            path: "mlx_audio_swift/tts/Swift-TTS/Kokoro/Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "MLXSwiftTTS",
            dependencies: [.product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
             "ESpeakNG"
           ],
            path: "mlx_audio_swift/tts/Swift-TTS",
            exclude: ["Preview Content","Assets.xcassets","Swift_TTSApp.swift","Swift_TTS.entitlements"]),
        .testTarget(
            name: "MLXSwiftTTSTests",
            dependencies: ["MLXSwiftTTS"]),
    ]
)
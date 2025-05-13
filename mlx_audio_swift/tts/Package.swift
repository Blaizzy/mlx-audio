// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MLXSwiftPackage",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "MLXSwiftPackage",
            targets: ["MLXSwiftPackage"]),
    ],
    dependencies: [
         .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.2")
    ],
    targets: [
        .binaryTarget(
            name: "ESpeakNG",
            path: "Swift-TTS/Kokoro/Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "MLXSwiftPackage",
            dependencies: [.product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
             "ESpeakNG"
           ],
            path: "Swift-TTS",
            exclude: ["Preview Content","Assets.xcassets"]),
        .testTarget(
            name: "MLXSwiftPackageTests",
            dependencies: ["MLXSwiftPackage"]),
    ]
)
# Xcode Build Troubleshooting for MLX-Audio Swift Framework

## Metal Toolchain Error with Xcode Beta

### Problem
When building the MLXAudio framework with Xcode Beta versions, you may encounter this error:

```
error: cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain
```

This error occurs because Xcode Beta versions often ship without the complete Metal Toolchain component, which is required to compile Metal shaders (.metal files) used by the mlx-swift dependency.

### Root Cause
- Xcode Beta installations may have incomplete toolchain components
- The mlx-swift framework dependency requires Metal shader compilation
- Metal Toolchain component is needed but not installed by default in some Beta versions

### Solution

**Step 1: Install Metal Toolchain**
```bash
# For Xcode Beta (adjust path as needed)
env DEVELOPER_DIR=/Applications/Xcode-beta.app xcodebuild -downloadComponent MetalToolchain

# For Xcode Beta with specific version number
env DEVELOPER_DIR=/Applications/Xcode-26.0.0-Beta.app xcodebuild -downloadComponent MetalToolchain
```

**Step 2: Verify Installation**
After the download completes (approximately 688MB), you should see:
```
Done downloading: Metal Toolchain 17A5295f.
```

**Step 3: Retry Build**
```bash
# Command line build
env DEVELOPER_DIR=/Applications/Xcode-beta.app xcrun xcodebuild -IDEClonedSourcePackagesDirPathOverride=$PWD/.dependencies -skipMacroValidation -skipPackagePluginValidation -derivedDataPath $PWD/.derivedData build -scheme MLXAudio -destination platform=macOS,arch=arm64

# Or simply rebuild in Xcode GUI
```

## Alternative Solutions

### Use Stable Xcode
If you have a stable Xcode installation, you can switch to it:

```bash
# Check available Xcode installations
ls /Applications/ | grep -i xcode

# Use stable Xcode instead
env DEVELOPER_DIR=/Applications/Xcode.app xcrun xcodebuild -scheme MLXAudio -destination platform=macOS,arch=arm64 build
```

### Fix xcode-select Path
Ensure xcode-select points to the correct installation:

```bash
# Check current setting
xcode-select -p

# Switch to stable Xcode if needed
sudo xcode-select --switch /Applications/Xcode.app

# Or switch to Beta if Metal Toolchain is installed
sudo xcode-select --switch /Applications/Xcode-beta.app
```

## Build Commands Reference

### Standard Build Commands

```bash
# Basic build
swift build

# Xcode project build
xcodebuild -scheme MLXAudio -destination platform=macOS,arch=arm64 build

# With custom derived data path
xcodebuild -derivedDataPath $PWD/.derivedData build -scheme MLXAudio -destination platform=macOS,arch=arm64

# Full command with all flags (as used in CI/testing)
env DEVELOPER_DIR=/Applications/Xcode-beta.app xcrun xcodebuild -IDEClonedSourcePackagesDirPathOverride=$PWD/.dependencies -skipMacroValidation -skipPackagePluginValidation -derivedDataPath $PWD/.derivedData build -scheme MLXAudio -destination platform=macOS,arch=arm64
```

### Debug Information Commands

```bash
# Check Xcode version
xcodebuild -version

# List available SDKs
xcodebuild -showsdks

# List schemes
xcodebuild -list

# Check Metal toolchain availability
which metal
```

## Requirements

- **Hardware**: Apple Silicon Mac (M1 or newer)
- **OS**: macOS 14.0+
- **Xcode**: 15.0+ or compatible Beta with Metal Toolchain
- **Dependencies**: MLX Swift framework will be automatically resolved

## Related Issues

This fix resolves build failures in:
- Swift Package Manager builds
- Xcode GUI builds  
- CI/CD pipeline builds using Xcode Beta
- Command line builds with `xcodebuild`

The issue specifically affects compilation of Metal shaders in the mlx-swift dependency, which provides GPU-accelerated operations for the TTS models.
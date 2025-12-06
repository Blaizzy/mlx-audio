//
//  PythonSetup.swift
//  MLXAudioCore
//
//  Created by MLX Audio Team on 2025-12-06.
//

import Foundation

#if canImport(PythonKit)
import PythonKit
#endif

public enum PythonSetupError: Error, LocalizedError {
    case resourceNotFound(String)
    case initializationFailed(String)
    case moduleNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .resourceNotFound(let resource):
            return "Python resource not found: \(resource)"
        case .initializationFailed(let reason):
            return "Python initialization failed: \(reason)"
        case .moduleNotFound(let module):
            return "Python module not found: \(module)"
        }
    }
}

public enum PythonSetup {
    private static var isInitialized = false

    /// Initialize Python environment from bundled Python.xcframework
    ///
    /// Call this once at app startup before using any Python functionality.
    ///
    /// - Parameters:
    ///   - pythonResourceName: Name of the bundled python resource (default: "python")
    ///   - appResourceName: Name of the app scripts resource (default: "app")
    /// - Throws: PythonSetupError if initialization fails
    public static func initialize(
        pythonResourceName: String = "python",
        appResourceName: String = "app"
    ) throws {
        guard !isInitialized else { return }

        #if canImport(PythonKit)
        // Find bundled Python resources
        guard let pythonHome = Bundle.main.path(forResource: pythonResourceName, ofType: nil) else {
            throw PythonSetupError.resourceNotFound("PYTHONHOME (\(pythonResourceName))")
        }

        // App scripts path is optional
        let appPath = Bundle.main.path(forResource: appResourceName, ofType: nil)

        // Set environment variables
        setenv("PYTHONHOME", pythonHome, 1)

        // Build PYTHONPATH
        var pythonPath = "\(pythonHome)/lib/python3.12/site-packages"
        if let appPath = appPath {
            pythonPath = "\(appPath):\(pythonPath)"
        }
        setenv("PYTHONPATH", pythonPath, 1)

        // Disable Python's signal handlers (important for embedding)
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1)

        // Initialize Python interpreter via PythonKit
        // PythonKit auto-initializes the interpreter on first import
        do {
            _ = try Python.attemptImport("sys")
        } catch {
            throw PythonSetupError.initializationFailed("Failed to initialize Python: \(error)")
        }

        isInitialized = true
        #else
        throw PythonSetupError.initializationFailed("PythonKit not available")
        #endif
    }

    /// Check if Python is initialized
    ///
    /// Returns true if `initialize()` completed successfully.
    /// Does not re-verify Python state for performance reasons.
    public static var isPythonReady: Bool {
        #if canImport(PythonKit)
        return isInitialized
        #else
        return false
        #endif
    }

    /// Get Python version string
    public static var pythonVersion: String? {
        #if canImport(PythonKit)
        guard isPythonReady else { return nil }
        let sys = Python.import("sys")
        return String(sys.version)
        #else
        return nil
        #endif
    }

    /// Verify that mlx_audio is importable
    public static func verifyMLXAudio() throws -> String {
        #if canImport(PythonKit)
        guard isPythonReady else {
            throw PythonSetupError.initializationFailed("Python not initialized")
        }

        do {
            let mlxAudio = try Python.attemptImport("mlx_audio")
            if let version = String(mlxAudio.__version__) {
                return version
            }
            return "unknown"
        } catch {
            throw PythonSetupError.moduleNotFound("mlx_audio")
        }
        #else
        throw PythonSetupError.initializationFailed("PythonKit not available")
        #endif
    }

    /// Clean up Python interpreter (call at app termination)
    ///
    /// Note: PythonKit does not expose Py_FinalizeEx(), and finalizing an embedded
    /// Python interpreter is generally problematic (modules may not clean up properly).
    /// This method resets the internal state but does not actually finalize Python.
    public static func finalize() {
        #if canImport(PythonKit)
        guard isInitialized else { return }
        // PythonKit does not expose Py_FinalizeEx - reset internal state only
        isInitialized = false
        #endif
    }
}

//
//  TTSError.swift
//  MLXAudio
//
//  Unified error handling across all TTS engines.
//

import Foundation

/// Unified error type for all TTS operations
public enum TTSError: LocalizedError {
    /// The model hasn't been loaded yet
    case modelNotLoaded

    /// Audio generation failed
    case generationFailed(underlying: Error)

    /// Audio playback failed
    case audioPlaybackFailed(underlying: Error)

    /// The requested voice is not valid for this engine
    case invalidVoice(String)

    /// Not enough memory to load or run the model
    case insufficientMemory

    /// The operation was cancelled by the user
    case cancelled

    /// Model download or loading failed
    case modelLoadFailed(underlying: Error)

    /// Invalid reference audio provided
    case invalidReferenceAudio(String)

    /// Voice not found in available voices
    case voiceNotFound(String)

    /// File I/O error
    case fileIOError(underlying: Error)

    /// Invalid configuration or arguments
    case invalidArgument(String)

    // MARK: - LocalizedError

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded. Call load() first."
        case .generationFailed(let error):
            return "Generation failed: \(error.localizedDescription)"
        case .audioPlaybackFailed(let error):
            return "Playback failed: \(error.localizedDescription)"
        case .invalidVoice(let id):
            return "Invalid voice: \(id)"
        case .insufficientMemory:
            return "Insufficient memory for model."
        case .cancelled:
            return "Operation was cancelled."
        case .modelLoadFailed(let error):
            return "Failed to load model: \(error.localizedDescription)"
        case .invalidReferenceAudio(let message):
            return "Invalid reference audio: \(message)"
        case .voiceNotFound(let name):
            return "Voice not found: \(name)"
        case .fileIOError(let error):
            return "File I/O error: \(error.localizedDescription)"
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        }
    }

    public var failureReason: String? {
        switch self {
        case .modelNotLoaded:
            return "The TTS model must be loaded before generating audio."
        case .generationFailed:
            return "An error occurred during audio synthesis."
        case .audioPlaybackFailed:
            return "The audio system encountered an error during playback."
        case .invalidVoice:
            return "The specified voice is not available for this TTS engine."
        case .insufficientMemory:
            return "The device does not have enough memory to run this model."
        case .cancelled:
            return "The user cancelled the operation."
        case .modelLoadFailed:
            return "The model weights could not be downloaded or loaded."
        case .invalidReferenceAudio:
            return "The reference audio file is invalid or in an unsupported format."
        case .voiceNotFound:
            return "The requested voice preset could not be found."
        case .fileIOError:
            return "A file system operation failed."
        case .invalidArgument:
            return "An invalid argument was provided."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .modelNotLoaded:
            return "Call the load() method before attempting to generate audio."
        case .generationFailed:
            return "Try again with different text or check the error details."
        case .audioPlaybackFailed:
            return "Check that the audio session is configured correctly."
        case .invalidVoice:
            return "Use availableVoices to see the list of valid voices."
        case .insufficientMemory:
            return "Close other applications to free up memory, or use a smaller model."
        case .cancelled:
            return nil
        case .modelLoadFailed:
            return "Check your internet connection and try again."
        case .invalidReferenceAudio:
            return "Provide a mono WAV file at 24kHz sample rate."
        case .voiceNotFound:
            return "Check that the voice preset file exists in the model directory."
        case .fileIOError:
            return "Check file permissions and available disk space."
        case .invalidArgument:
            return "Review the method documentation for valid argument values."
        }
    }
}

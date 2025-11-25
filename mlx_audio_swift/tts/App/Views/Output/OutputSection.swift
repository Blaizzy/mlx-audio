//
//  OutputSection.swift
//  MLXAudio
//
//  Audio output and playback section.
//

import SwiftUI

struct OutputSection: View {
    @Environment(AppState.self) private var appState

    @State private var playerManager = AudioFilePlayer()

    var body: some View {
        VStack(spacing: 16) {
            // Status Banner
            if !appState.statusMessage.isEmpty {
                StatusBanner(
                    message: appState.statusMessage,
                    progress: appState.isModelLoading ? appState.loadingProgress : nil,
                    isError: appState.error != nil
                )
            }

            // Audio Player
            if let audioURL = appState.lastGeneratedAudioURL {
                AudioPlayerView(audioURL: audioURL)
                    .environment(playerManager)
            } else {
                AudioPlayerPlaceholder()
            }

            // Generation Stats
            if let result = appState.lastResult {
                GenerationStatsView(result: result)
            }

            // Share Button
            if let audioURL = appState.lastGeneratedAudioURL {
                ShareLink(item: audioURL) {
                    Label("Share Audio", systemImage: "square.and.arrow.up")
                        .padding(.vertical, 4)
                }
                .buttonStyle(.bordered)
            }
        }
        .onChange(of: appState.isGenerating) { wasGenerating, isGenerating in
            if isGenerating {
                // Stop playback when new generation starts
                playerManager.stop()
            } else if !isGenerating && wasGenerating {
                // Reload audio when generation completes
                if let url = appState.lastGeneratedAudioURL {
                    playerManager.loadAudio(from: url)
                }
            }
        }
    }
}

/// Display generation statistics
private struct GenerationStatsView: View {
    let result: AudioResult

    var body: some View {
        HStack(spacing: 16) {
            if let duration = result.duration {
                StatItem(
                    label: "Duration",
                    value: String(format: "%.2fs", duration)
                )
            }

            StatItem(
                label: "Time",
                value: String(format: "%.2fs", result.processingTime)
            )

            if let rtf = result.realTimeFactor {
                StatItem(
                    label: "RTF",
                    value: String(format: "%.2fx", rtf)
                )
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

private struct StatItem: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .monospacedDigit()
            Text(label)
                .font(.caption2)
        }
    }
}

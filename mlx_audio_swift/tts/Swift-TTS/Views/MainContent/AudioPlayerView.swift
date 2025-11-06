//
//  AudioPlayerView.swift
//  Swift-TTS
//
//  Created by Claude Code
//

import SwiftUI
import AVFoundation

struct AudioPlayerView: View {
    let audioURL: URL?
    let isPlaying: Bool
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onPlayPause: () -> Void
    let onSeek: (TimeInterval) -> Void

    private var progress: Double {
        guard duration > 0 else { return 0 }
        return currentTime / duration
    }

    var body: some View {
        VStack(spacing: 12) {
            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    Rectangle()
                        .fill(Color(nsColor: .separatorColor))
                        .frame(height: 4)
                        .cornerRadius(2)

                    // Progress
                    Rectangle()
                        .fill(Color.accentColor)
                        .frame(width: geometry.size.width * progress, height: 4)
                        .cornerRadius(2)
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let newProgress = value.location.x / geometry.size.width
                            let newTime = max(0, min(duration, newProgress * duration))
                            onSeek(newTime)
                        }
                )
            }
            .frame(height: 4)

            // Controls
            HStack {
                // Play/Pause button
                Button(action: onPlayPause) {
                    Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.title)
                        .foregroundColor(audioURL != nil ? .primary : .secondary)
                }
                .buttonStyle(.plain)
                .disabled(audioURL == nil)

                Spacer()

                // Time display
                Text(formatTime(currentTime))
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()

                Text("/")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(formatTime(duration))
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

// Empty state placeholder for when no audio is available
struct AudioPlayerPlaceholder: View {
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "play.circle.fill")
                    .font(.title)
                    .foregroundColor(.secondary)
                Text("Audio Player")
                    .foregroundColor(.secondary)
                Spacer()
                Text("0:00")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }

            ProgressView(value: 0.0)
                .progressViewStyle(.linear)
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }
}

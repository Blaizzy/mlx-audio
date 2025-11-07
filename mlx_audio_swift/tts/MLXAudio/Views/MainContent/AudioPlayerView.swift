//
//  AudioPlayerView.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI
import AVFoundation
#if canImport(UIKit)
import UIKit
#endif

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

    private var separatorColor: Color {
        #if os(macOS)
        Color(nsColor: .separatorColor)
        #else
        Color(UIColor.separator)
        #endif
    }

    private var controlBackgroundColor: Color {
        #if os(macOS)
        Color(nsColor: .controlBackgroundColor)
        #else
        Color(UIColor.secondarySystemBackground)
        #endif
    }

    var body: some View {
        HStack(spacing: 16) {
            // Play/Pause button
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 48))
                    .foregroundColor(audioURL != nil ? .primary : .secondary)
            }
            .buttonStyle(.plain)
            .disabled(audioURL == nil)
            .padding(.leading, 16)

            // Progress bar and time
            VStack(spacing: 8) {
                // Progress bar
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // Background
                        Rectangle()
                            .fill(separatorColor)
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

                // Time display
                HStack {
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

                    Spacer()
                }
            }
            .padding(.trailing, 16)
        }
        .padding(.vertical, 12)
        .background(controlBackgroundColor)
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
    private var controlBackgroundColor: Color {
        #if os(macOS)
        Color(nsColor: .controlBackgroundColor)
        #else
        Color(UIColor.secondarySystemBackground)
        #endif
    }

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: "play.circle.fill")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
                .padding(.leading, 16)

            VStack(spacing: 8) {
                ProgressView(value: 0.0)
                    .progressViewStyle(.linear)
                    .frame(height: 4)

                HStack {
                    Text("0:00")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .monospacedDigit()
                    Text("/")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("0:00")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .monospacedDigit()
                    Spacer()
                }
            }
            .padding(.trailing, 16)
        }
        .padding(.vertical, 12)
        .background(controlBackgroundColor)
        .cornerRadius(8)
    }
}

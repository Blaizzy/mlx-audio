//
//  AudioPlayerView.swift
//  MLXAudio
//
//  Audio playback controls with progress bar.
//

import SwiftUI

struct AudioPlayerView: View {
    @Environment(AudioFilePlayer.self) private var playerManager

    let audioURL: URL?

    var body: some View {
        VStack(spacing: 12) {
            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.secondary.opacity(0.2))
                        .frame(height: 4)

                    Rectangle()
                        .fill(Color.accentColor)
                        .frame(width: geometry.size.width * progress, height: 4)
                }
                .clipShape(RoundedRectangle(cornerRadius: 2))
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let newProgress = value.location.x / geometry.size.width
                            let newTime = max(0, min(playerManager.duration, newProgress * playerManager.duration))
                            playerManager.seek(to: newTime)
                        }
                )
            }
            .frame(height: 4)

            // Controls
            HStack(spacing: 16) {
                // Time display
                Text(formatTime(playerManager.currentTime))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(width: 40, alignment: .leading)

                Spacer()

                // Play/Pause
                Button {
                    if audioURL != nil {
                        if playerManager.currentAudioURL != audioURL {
                            playerManager.loadAudio(from: audioURL!)
                        }
                        playerManager.togglePlayPause()
                    }
                } label: {
                    Image(systemName: playerManager.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44))
                }
                .buttonStyle(.plain)
                .disabled(audioURL == nil)

                Spacer()

                // Duration
                Text(formatTime(playerManager.duration))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(width: 40, alignment: .trailing)
            }
        }
        .padding()
        .background(.background.secondary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .onChange(of: audioURL) { _, newURL in
            if let url = newURL {
                playerManager.loadAudio(from: url)
            }
        }
    }

    private var progress: Double {
        guard playerManager.duration > 0 else { return 0 }
        return playerManager.currentTime / playerManager.duration
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

/// Placeholder when no audio is available
struct AudioPlayerPlaceholder: View {
    var body: some View {
        VStack(spacing: 12) {
            Rectangle()
                .fill(Color.secondary.opacity(0.2))
                .frame(height: 4)
                .clipShape(RoundedRectangle(cornerRadius: 2))

            HStack(spacing: 16) {
                Text("0:00")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(width: 40, alignment: .leading)

                Spacer()

                Image(systemName: "play.circle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(.secondary)

                Spacer()

                Text("0:00")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(width: 40, alignment: .trailing)
            }
        }
        .padding()
        .background(.background.secondary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

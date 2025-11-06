//
//  TTSMainView.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct TTSMainView: View {
    @Binding var text: String
    @Binding var status: String
    let selectedProvider: TTSProvider
    let marvisSession: MarvisSession?
    @ObservedObject var audioPlayerManager: AudioPlayerManager

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Title with Marvis Status
            HStack {
                Text("Text to Speech")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                Spacer()

                // Marvis Status (conditional)
                if selectedProvider == .marvis {
                    MarvisStatusIndicator(session: marvisSession)
                }
            }

            // Text Input Section
            TextInputSection(text: $text)

            // Provider Info
            if !selectedProvider.statusMessage.isEmpty {
                InfoBox(message: selectedProvider.statusMessage)
            }

            Spacer()

            // Audio Player
            if audioPlayerManager.currentAudioURL != nil {
                AudioPlayerView(
                    audioURL: audioPlayerManager.currentAudioURL,
                    isPlaying: audioPlayerManager.isPlaying,
                    currentTime: audioPlayerManager.currentTime,
                    duration: audioPlayerManager.duration,
                    onPlayPause: { audioPlayerManager.togglePlayPause() },
                    onSeek: { time in audioPlayerManager.seek(to: time) }
                )
            } else {
                AudioPlayerPlaceholder()
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

struct InfoBox: View {
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "info.circle.fill")
                .foregroundColor(.blue)
            Text(message)
                .font(.caption)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(Color.blue.opacity(0.1))
        .cornerRadius(8)
    }
}

struct MarvisStatusIndicator: View {
    let session: MarvisSession?

    private var controlBackgroundColor: Color {
        #if os(macOS)
        Color(nsColor: .controlBackgroundColor)
        #else
        Color(UIColor.secondarySystemBackground)
        #endif
    }

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(session != nil ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(session != nil ? "Connected" : "Not Connected")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(controlBackgroundColor)
        .cornerRadius(12)
    }
}


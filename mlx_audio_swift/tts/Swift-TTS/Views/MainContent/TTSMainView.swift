//
//  TTSMainView.swift
//  Swift-TTS
//
//  Created by Claude Code
//

import SwiftUI

struct TTSMainView: View {
    @Binding var text: String
    @Binding var status: String
    let selectedProvider: TTSProvider

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Title
            HStack {
                Text("Text to Speech")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                Spacer()
            }

            // Text Input Section
            TextInputSection(text: $text)

            // Provider Info
            if !selectedProvider.statusMessage.isEmpty {
                InfoBox(message: selectedProvider.statusMessage)
            }

            Spacer()

            // Audio Player Placeholder (for future)
            AudioPlayerPlaceholder()
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
                Text("00:00")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            ProgressView(value: 0.0)
                .progressViewStyle(.linear)
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }
}

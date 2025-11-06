//
//  TTSInspectorView.swift
//  Swift-TTS
//
//  Created by Claude Code
//

import SwiftUI

struct TTSInspectorView: View {
    @Binding var selectedProvider: TTSProvider
    @Binding var selectedVoice: String
    @Binding var status: String

    let isGenerating: Bool
    let canGenerate: Bool
    let marvisSession: MarvisSession?
    let onGenerate: () -> Void
    let onStop: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("Settings")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
            }
            .padding()
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            // Content
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // Model Section
                    ModelPickerSection(
                        selectedProvider: $selectedProvider,
                        selectedVoice: $selectedVoice
                    )

                    Divider()

                    // Voice Section
                    VoicePickerSection(
                        provider: selectedProvider,
                        selectedVoice: $selectedVoice
                    )

                    Divider()

                    // Marvis Status (conditional)
                    if selectedProvider == .marvis {
                        MarvisStatusSection(session: marvisSession)
                        Divider()
                    }

                    // Controls
                    ControlsSection(
                        isGenerating: isGenerating,
                        canGenerate: canGenerate,
                        onGenerate: onGenerate,
                        onStop: onStop
                    )

                    // Status Display
                    if !status.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Status")
                                .font(.headline)
                                .foregroundColor(.secondary)
                            Text(status)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(nsColor: .controlBackgroundColor))
                                .cornerRadius(6)
                        }
                    }
                }
                .padding()
            }
        }
        .frame(width: 300) // Fixed inspector width
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct MarvisStatusSection: View {
    let session: MarvisSession?

    var body: some View {
        HStack {
            Circle()
                .fill(session != nil ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(session != nil ? "Marvis Ready" : "Not Initialized")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

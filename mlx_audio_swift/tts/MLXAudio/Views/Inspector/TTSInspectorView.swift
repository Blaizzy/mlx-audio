//
//  TTSInspectorView.swift
//  MLXAudio
//
//  Created by Claude Code
//

import SwiftUI

struct TTSInspectorView: View {
    @Binding var selectedProvider: TTSProvider
    @Binding var selectedVoice: String
    @Binding var selectedQuality: MarvisSession.QualityLevel
    @Binding var status: String
    @Binding var autoPlay: Bool

    let isGenerating: Bool
    let canGenerate: Bool
    let marvisSession: MarvisSession?
    let onGenerate: () -> Void
    let onStop: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
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

                    // Quality Section (Marvis only)
                    if selectedProvider == .marvis {
                        QualityPickerSection(selectedQuality: $selectedQuality)
                        Divider()
                    }

                    // Auto-play toggle
                    AutoPlaySection(autoPlay: $autoPlay)

                    Divider()

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
    }
}

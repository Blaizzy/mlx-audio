//
//  TTSInspectorView.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct TTSInspectorView: View {
    @Binding var selectedProvider: TTSProvider
    @Binding var selectedVoice: String
    @Binding var selectedQuality: MarvisSession.QualityLevel
    @Binding var status: String
    @Binding var autoPlay: Bool
    @Binding var useStreaming: Bool
    @Binding var streamingInterval: Double

    let isGenerating: Bool
    let canGenerate: Bool
    let marvisSession: MarvisSession?
    let onGenerate: () -> Void
    let onStop: () -> Void

    private var controlBackgroundColor: Color {
        #if os(macOS)
        Color(nsColor: .controlBackgroundColor)
        #else
        Color(UIColor.secondarySystemBackground)
        #endif
    }

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

                    // Streaming toggle (Marvis only)
                    if selectedProvider == .marvis {
                        StreamingSection(useStreaming: $useStreaming)
                        Divider()

                        // Streaming interval (Marvis only, when streaming enabled)
                        if useStreaming {
                            StreamingIntervalSection(streamingInterval: $streamingInterval)
                            Divider()
                        }
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
                                .background(controlBackgroundColor)
                                .cornerRadius(6)
                        }
                    }
                }
                .padding()
            }
        }
    }
}

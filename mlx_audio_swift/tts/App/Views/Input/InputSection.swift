//
//  InputSection.swift
//  MLXAudio
//
//  Text input and generation controls section.
//

import SwiftUI

struct InputSection: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        @Bindable var appState = appState
        VStack(spacing: 16) {
            // Text Input
            TextInputView(
                text: $appState.inputText,
                isDisabled: appState.isGenerating
            )

            // Auto-play toggle
            Toggle(isOn: $appState.autoPlay) {
                Label("Auto-play", systemImage: "play.circle")
            }
            .toggleStyle(.switch)

            // Generate Button
            GenerateButton(
                isLoading: appState.isModelLoading,
                isGenerating: appState.isGenerating,
                canGenerate: !appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                onGenerate: {
                    Task {
                        if !appState.isLoaded {
                            try? await appState.loadEngine()
                        }
                        await appState.generate()
                    }
                },
                onStop: {
                    Task { await appState.stop() }
                }
            )

            // Streaming option for Marvis
            if appState.selectedProvider.supportsStreaming {
                Button {
                    Task {
                        if !appState.isLoaded {
                            try? await appState.loadEngine()
                        }
                        await appState.generateStreaming()
                    }
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "waveform.path")
                        Text("Stream")
                    }
                    .padding(.vertical, 4)
                }
                .buttonStyle(.bordered)
                .disabled(appState.isModelLoading || appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }
}

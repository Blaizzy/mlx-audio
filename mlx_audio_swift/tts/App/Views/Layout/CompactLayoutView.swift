//
//  CompactLayoutView.swift
//  MLXAudio
//
//  Layout for iPhone and iPad portrait (NavigationStack with inline controls).
//

import SwiftUI

struct CompactLayoutView: View {
    @Environment(AppState.self) private var appState
    
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    ProviderAndVoiceSection()
                    InputSection()
                    OutputSection()
                }
                .padding()
            }
            .navigationTitle("MLX Audio")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsSheet()
            }
        }
    }
}

// Only used in compact size class (otherwise these selectors are shown in the toolbar)
private struct ProviderAndVoiceSection: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        @Bindable var appState = appState
        HStack(spacing: 12) {
            ProviderPickerView(
                selectedProvider: appState.selectedProvider,
                onSelect: { provider in
                    Task { await appState.selectProvider(provider) }
                }
            )

            Spacer()

            VoicePickerView(
                voices: appState.availableVoices,
                selectedVoiceID: $appState.selectedVoiceID
            )
        }
    }
}

/// Full settings sheet for compact layout
private struct SettingsSheet: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                SettingsSection()
                    .padding()
            }
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                }
            }
        }
    }
}

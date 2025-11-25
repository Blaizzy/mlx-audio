//
//  ProviderPickerView.swift
//  MLXAudio
//
//  TTS provider selection component.
//

import SwiftUI

struct ProviderPickerView: View {
    let selectedProvider: TTSProvider
    let onSelect: (TTSProvider) -> Void

    var body: some View {
        Menu {
            ForEach(TTSProvider.allCases) { provider in
                Button {
                    onSelect(provider)
                } label: {
                    HStack {
                        Text(provider.displayName)
                        if provider == selectedProvider {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "waveform")
                Text(selectedProvider.displayName)
            }
        }
    }
}

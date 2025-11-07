//
//  ModelPickerSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct ModelPickerSection: View {
    @Binding var selectedProvider: TTSProvider
    @Binding var selectedVoice: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model")
                .font(.headline)
                .foregroundColor(.secondary)

            Picker("", selection: $selectedProvider) {
                ForEach(TTSProvider.allCases, id: \.self) { provider in
                    Text(provider.displayName).tag(provider)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
        }
        .onChange(of: selectedProvider) { _, newProvider in
            selectedVoice = newProvider.defaultVoice
        }
    }
}
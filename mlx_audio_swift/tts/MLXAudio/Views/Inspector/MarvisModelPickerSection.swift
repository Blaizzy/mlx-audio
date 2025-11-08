//
//  MarvisModelPickerSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 11/08/25.
//

import SwiftUI

struct MarvisModelPickerSection: View {
    @Binding var selectedModel: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Marvis Model")
                .font(.headline)
                .foregroundColor(.secondary)

            Picker("", selection: $selectedModel) {
                ForEach(MarvisSession.ModelVariant.allCases, id: \.self) { model in
                    Text(model.displayName).tag(model.repoId)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
        }
    }
}

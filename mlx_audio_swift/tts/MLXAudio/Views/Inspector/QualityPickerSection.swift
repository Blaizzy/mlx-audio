//
//  QualityPickerSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct QualityPickerSection: View {
    @Binding var selectedQuality: MarvisSession.QualityLevel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Quality")
                .font(.headline)
                .foregroundColor(.secondary)

            // Segmented Picker for Quality
            Picker("Quality", selection: $selectedQuality) {
                ForEach(MarvisSession.QualityLevel.allCases, id: \.self) { quality in
                    Text(quality.rawValue.capitalized).tag(quality)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            // Quality description
            Text(qualityDescription)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var qualityDescription: String {
        switch selectedQuality {
        case .low:
            return "8 codebooks - Fastest generation, lower quality"
        case .medium:
            return "16 codebooks - Balanced speed and quality"
        case .high:
            return "24 codebooks - Slower, better quality"
        case .maximum:
            return "32 codebooks - Slowest, best quality"
        }
    }
}

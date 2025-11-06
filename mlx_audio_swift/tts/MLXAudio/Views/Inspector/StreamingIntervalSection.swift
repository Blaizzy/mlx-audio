//
//  StreamingIntervalSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct StreamingIntervalSection: View {
    @Binding var streamingInterval: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Streaming Interval")
                    .font(.headline)
                    .foregroundColor(.secondary)

                Spacer()

                Text(String(format: "%.1fs", streamingInterval))
                    .font(.subheadline)
                    .bold()
            }

            Slider(value: $streamingInterval, in: 0.1...1.0, step: 0.1)

            Text("Time between audio chunks (lower = faster response, higher = more efficient)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

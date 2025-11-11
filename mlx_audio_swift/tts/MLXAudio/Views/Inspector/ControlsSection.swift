//
//  ControlsSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct ControlsSection: View {
    let isGenerating: Bool
    let canGenerate: Bool
    let onGenerate: () -> Void
    let onStop: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            // Generate Button
            Button(action: onGenerate) {
                HStack {
                    if isGenerating {
                        ProgressView()
                            .controlSize(.small)
                            .scaleEffect(0.8)
                        Text("Generating...")
                    } else {
                        Image(systemName: "waveform")
                        Text("Generate")
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 36)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canGenerate || isGenerating)

            // Stop Button
            Button(action: onStop) {
                HStack {
                    Image(systemName: "stop.fill")
                    Text("Stop")
                }
                .frame(maxWidth: .infinity)
                .frame(height: 32)
            }
            .buttonStyle(.bordered)
            .disabled(!isGenerating)
        }
    }
}

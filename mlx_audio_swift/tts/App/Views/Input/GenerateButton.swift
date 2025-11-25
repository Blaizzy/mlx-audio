//
//  GenerateButton.swift
//  MLXAudio
//
//  Primary action button for audio generation.
//

import SwiftUI

struct GenerateButton: View {
    let isLoading: Bool
    let isGenerating: Bool
    let canGenerate: Bool
    let onGenerate: () -> Void
    let onStop: () -> Void

    var body: some View {
        Button {
            if isGenerating {
                onStop()
            } else {
                onGenerate()
            }
        } label: {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .controlSize(.small)
                    Text("Loading Model...")
                } else if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                    Text("Stop")
                } else {
                    Image(systemName: "play.fill")
                    Text("Generate")
                }
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.borderedProminent)
        .disabled(isLoading || (!isGenerating && !canGenerate))
    }
}

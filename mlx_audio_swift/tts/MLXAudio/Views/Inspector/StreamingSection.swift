//
//  StreamingSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct StreamingSection: View {
    @Binding var useStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Generation Mode")
                .font(.headline)
                .foregroundColor(.secondary)

            Toggle("Use Streaming", isOn: $useStreaming)
                .toggleStyle(.switch)

            Text(useStreaming ? "Real-time audio streaming with progress feedback" : "Generate complete audio before playback")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

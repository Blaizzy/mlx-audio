//
//  AutoPlaySection.swift
//  Swift-TTS
//
//  Created by Claude Code
//

import SwiftUI

struct AutoPlaySection: View {
    @Binding var autoPlay: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Playback")
                .font(.headline)
                .foregroundColor(.secondary)

            Toggle("Auto-play", isOn: $autoPlay)
                .toggleStyle(.switch)
        }
    }
}

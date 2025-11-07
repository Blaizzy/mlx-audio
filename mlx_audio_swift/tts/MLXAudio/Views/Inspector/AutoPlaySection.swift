//
//  AutoPlaySection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
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

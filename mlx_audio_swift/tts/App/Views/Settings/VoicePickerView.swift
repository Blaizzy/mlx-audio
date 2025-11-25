//
//  VoicePickerView.swift
//  MLXAudio
//
//  Voice selection component using the unified Voice model.
//

import SwiftUI

struct VoicePickerView: View {
    let voices: [Voice]
    @Binding var selectedVoiceID: String

    private var selectedVoice: Voice? {
        voices.first { $0.id == selectedVoiceID }
    }

    var body: some View {
        Menu {
            ForEach(voices) { voice in
                Button {
                    selectedVoiceID = voice.id
                } label: {
                    if voice.id == selectedVoiceID {
                        Label("\(voice.languageFlag) \(voice.displayName)", systemImage: "checkmark")
                    } else {
                        Text("\(voice.languageFlag) \(voice.displayName)")
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                if let voice = selectedVoice {
                    Text("\(voice.languageFlag) \(voice.displayName)")
                        .lineLimit(1)
                } else {
                    Text("Select Voice")
                }
                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.background.secondary)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
    }
}

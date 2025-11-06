//
//  VoicePickerSection.swift
//  Swift-TTS
//
//  Created by Claude Code
//

import SwiftUI

struct VoicePickerSection: View {
    let provider: TTSProvider
    @Binding var selectedVoice: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Voice")
                .font(.headline)
                .foregroundColor(.secondary)

            // Voice Card
            HStack(spacing: 12) {
                Circle()
                    .fill(Color.blue)
                    .frame(width: 44, height: 44)
                    .overlay(
                        Image(systemName: "person.wave.2")
                            .foregroundColor(.white)
                    )

                VStack(alignment: .leading, spacing: 2) {
                    Text(selectedVoice.capitalized)
                        .font(.body)
                        .fontWeight(.medium)
                    Text(voiceLanguage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Button(action: {}) {
                    Image(systemName: "gearshape")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)

            // Voice Dropdown
            Picker("", selection: $selectedVoice) {
                ForEach(provider.availableVoices, id: \.self) { voice in
                    Text(voice.capitalized).tag(voice)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
        }
    }

    private var voiceLanguage: String {
        // Extract language from voice name prefix
        // af = American Female, bm = British Male, etc.
        let prefix = String(selectedVoice.prefix(2))
        switch prefix {
        case "af", "am": return "English (American)"
        case "bf", "bm": return "English (British)"
        case "jf", "jm": return "Japanese"
        case "zf", "zm": return "Chinese (Mandarin)"
        case "ff": return "French"
        case "ef", "em": return "Spanish"
        case "hf", "hm": return "Hindi"
        case "if", "im": return "Italian"
        case "pf", "pm": return "Portuguese"
        default: return "English"
        }
    }
}

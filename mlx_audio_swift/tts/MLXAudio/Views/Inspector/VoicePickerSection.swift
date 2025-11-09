//
//  VoicePickerSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct VoicePickerSection: View {
    let provider: TTSProvider
    @Binding var selectedVoice: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Voice")
                .font(.headline)
                .foregroundColor(.secondary)

            // Icon and Voice Picker
            HStack(spacing: 12) {
                Circle()
                    .fill(Color.blue)
                    .frame(width: 36, height: 36)
                    .overlay(
                        Image(systemName: "person.wave.2")
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                    )

                VStack(alignment: .leading, spacing: 0) {
                    Picker("", selection: $selectedVoice) {
                        ForEach(provider.availableVoices, id: \.self) { voice in
                            Text(voice.capitalized).tag(voice)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()

                    Text(voiceLanguage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
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

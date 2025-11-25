//
//  TextInputView.swift
//  MLXAudio
//
//  Reusable text input component for TTS.
//

import SwiftUI

struct TextInputView: View {
    @Binding var text: String
    var isDisabled: Bool = false
    var characterLimit: Int = 5000

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Text("Text Input")
                    .font(.headline)

                Spacer()

                Text("\(text.count) / \(characterLimit)")
                    .font(.caption)
                    .foregroundStyle(text.count > characterLimit ? .red : .secondary)

                if !text.isEmpty {
                    Button {
                        text = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }

            // Text Editor
            ZStack(alignment: .topLeading) {
                if text.isEmpty {
                    Text("Enter text to synthesize...")
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 12)
                }

                TextEditor(text: $text)
                    .font(.body)
                    .focused($isFocused)
                    .scrollContentBackground(.hidden)
                    .disabled(isDisabled)
            }
            .frame(minHeight: 120, maxHeight: 300)
            .padding(8)
            .background(.background.secondary)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(.separator, lineWidth: 1)
            )
        }
    }
}

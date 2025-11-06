//
//  TextInputSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct TextInputSection: View {
    @Binding var text: String
    @FocusState private var isFocused: Bool

    private let characterLimit = 5000

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with character count
            HStack {
                Text("Text Input")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                Text("\(text.count) / \(characterLimit) characters")
                    .font(.caption)
                    .foregroundColor(text.count > characterLimit ? .red : .secondary)
            }

            // Text Editor
            ZStack(alignment: .topLeading) {
                if text.isEmpty {
                    Text("Enter text to synthesize...")
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 12)
                }

                TextEditor(text: $text)
                    .font(.body)
                    .focused($isFocused)
                    .frame(minHeight: 150, maxHeight: 300)
                    .scrollContentBackground(.hidden)
                    .background(Color(nsColor: .textBackgroundColor))
            }
            .background(Color(nsColor: .textBackgroundColor))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
            )

            // Clear button
            if !text.isEmpty {
                Button("Clear") {
                    text = ""
                }
                .buttonStyle(.link)
            }
        }
    }
}

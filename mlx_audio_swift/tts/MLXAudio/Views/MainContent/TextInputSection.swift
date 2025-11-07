//
//  TextInputSection.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct TextInputSection: View {
    @Binding var text: String
    @FocusState private var isFocused: Bool

    private let characterLimit = 5000

    private var textBackgroundColor: Color {
        #if os(macOS)
        Color(nsColor: .textBackgroundColor)
        #else
        Color(UIColor.systemBackground)
        #endif
    }

    private var separatorColor: Color {
        #if os(macOS)
        Color(nsColor: .separatorColor)
        #else
        Color(UIColor.separator)
        #endif
    }

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
                    .background(textBackgroundColor)
            }
            .background(textBackgroundColor)
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(separatorColor, lineWidth: 1)
            )

            // Clear button
            if !text.isEmpty {
                Button("Clear") {
                    text = ""
                }
                #if os(macOS)
                .buttonStyle(.link)
                #else
                .buttonStyle(.plain)
                #endif
            }
        }
    }
}

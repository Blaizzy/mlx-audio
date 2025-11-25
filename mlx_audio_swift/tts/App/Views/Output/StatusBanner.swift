//
//  StatusBanner.swift
//  MLXAudio
//
//  Status message display with loading progress.
//

import SwiftUI

struct StatusBanner: View {
    let message: String
    var progress: Double? = nil
    var isError: Bool = false

    var body: some View {
        HStack(spacing: 12) {
            if let progress = progress, progress > 0 && progress < 1 {
                ProgressView(value: progress)
                    .progressViewStyle(.circular)
                    .controlSize(.small)
            }

            Text(message)
                .font(.callout)
                .foregroundStyle(isError ? .red : .secondary)
                .lineLimit(2)

            Spacer()
        }
    }
}

/// Inline status for compact layouts
struct InlineStatus: View {
    let message: String
    var isError: Bool = false

    var body: some View {
        Text(message)
            .font(.caption)
            .foregroundStyle(isError ? .red : .secondary)
            .lineLimit(1)
    }
}

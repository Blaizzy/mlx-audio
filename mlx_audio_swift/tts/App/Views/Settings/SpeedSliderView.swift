//
//  SpeedSliderView.swift
//  MLXAudio
//
//  Speed adjustment slider (Kokoro only).
//

import SwiftUI

struct SpeedSliderView: View {
    @Binding var speed: Float
    var isDisabled: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Speed", systemImage: "gauge.with.needle")
                    .font(.headline)
                    .foregroundStyle(.secondary)

                Spacer()

                Text(String(format: "%.1fx", speed))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }

            HStack(spacing: 12) {
                Image(systemName: "tortoise")
                    .foregroundStyle(.secondary)
                    .font(.caption)

                Slider(
                    value: $speed,
                    in: TTSConstants.Speed.minimum...TTSConstants.Speed.maximum,
                    step: TTSConstants.Speed.step
                )
                .disabled(isDisabled)

                Image(systemName: "hare")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            }

            // Quick presets
            HStack(spacing: 8) {
                ForEach([0.5, 1.0, 1.5, 2.0], id: \.self) { preset in
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            speed = Float(preset)
                        }
                    } label: {
                        Text(String(format: "%.1fx", preset))
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                speed == Float(preset)
                                    ? Color.accentColor.opacity(0.2)
                                    : Color.clear
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    .buttonStyle(.plain)
                    .disabled(isDisabled)
                }
            }
        }
    }
}

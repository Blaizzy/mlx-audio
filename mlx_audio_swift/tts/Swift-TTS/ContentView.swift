//
//  ContentView.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 13/04/2025.
//

import SwiftUI
import MLX

struct ContentView: View {

    @State private var kokoroTTSModel: KokoroTTSModel? = nil
    @State private var orpheusTTSModel: OrpheusTTSModel? = nil
    @State private var sesameTTS: SesameTTS? = nil

    @State private var sayThis : String = "Hello Everybody"
    @State private var status : String = ""

    private var availableProviders = ["kokoro", "orpheus", "sesame"]
    @State private var chosenProvider : String = "sesame"  // Default to Sesame
    @State private var availableVoices: [String] = SesameVoice.allCases.map { $0.rawValue }
    @State private var chosenVoice: String = SesameVoice.conversational_a.rawValue

    var body: some View {
        VStack {
            Image(systemName: "mouth")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("TTS Examples")
                .font(.headline)
                .padding()

            Picker("Choose a provider", selection: $chosenProvider) {
                ForEach(availableProviders, id: \.self) { provider in
                    Text(provider.capitalized)
                }
            }
            .onChange(of: chosenProvider) { _, newProvider in
                if newProvider == "orpheus" {
                    availableVoices = OrpheusVoice.allCases.map { $0.rawValue }
                    chosenVoice = availableVoices.first ?? "dan"

                    status = "Orpheus is currently quite slow (0.1x on M1).  Working on it!\n\nBut it does support expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
                } else if newProvider == "sesame" {
                    availableVoices = SesameVoice.allCases.map { $0.rawValue }
                    chosenVoice = availableVoices.first ?? SesameVoice.conversational_a.rawValue

                    status = "Sesame CSM-1B: Advanced conversational TTS with streaming support.\n\nNote: Requires model weights to be downloaded from HuggingFace (sesame/csm-1b)"
                } else {
                    // kokoro
                    availableVoices = TTSVoice.allCases.map { $0.rawValue }
                    chosenVoice = availableVoices.first ?? TTSVoice.bmGeorge.rawValue

                    status = ""
                }
            }
            .padding()
            .padding(.bottom, 0)

            // Voice picker
            Picker("Choose a voice", selection: $chosenVoice) {
                ForEach(availableVoices, id: \.self) { voice in
                    Text(voice.capitalized)
                }
            }
            .padding()
            .padding(.top, 0)

            TextField("Enter text", text: $sayThis).padding()

            // Show model status for Sesame
            if chosenProvider == "sesame" {
                HStack {
                    Circle()
                        .fill(sesameTTS?.isLoaded ?? false ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(sesameTTS?.isLoaded ?? false ? "Model Loaded" : "Model Not Loaded")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)

                // Show model info if loaded
                if let modelInfo = sesameTTS?.modelInfo {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Model: \(modelInfo["name"] as? String ?? "Unknown")")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Architecture: \(modelInfo["architecture"] as? String ?? "Unknown")")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Sample Rate: \(modelInfo["sample_rate"] as? Int ?? 0) Hz")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 4)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(4)
                }
            }

            Button(action: {
                Task {
                    status = "Generating..."
                    if chosenProvider == "kokoro" {
                        if kokoroTTSModel == nil {
                            kokoroTTSModel = KokoroTTSModel()
                        }

                        if let kokoroVoice = TTSVoice.fromIdentifier(chosenVoice) ?? TTSVoice(rawValue: chosenVoice) {
                            kokoroTTSModel!.say(sayThis, kokoroVoice)
                        } else {
                            status = "Invalid Kokoro voice selected"
                        }

                    } else if chosenProvider == "orpheus" {
                        if orpheusTTSModel == nil {
                            orpheusTTSModel = OrpheusTTSModel()
                        }

                        if let orpheusVoice = OrpheusVoice(rawValue: chosenVoice) {
                            await orpheusTTSModel!.say(sayThis, orpheusVoice)
                        } else {
                            status = "Invalid Orpheus voice selected"
                        }

                    } else if chosenProvider == "sesame" {
                        // Initialize Sesame TTS if needed
                        if sesameTTS == nil {
                            sesameTTS = SesameTTS()
                        }

                        // Check if model is loaded
                        if !sesameTTS!.isLoaded {
                            do {
                                try sesameTTS!.loadModel()
                                status = "Sesame model loaded successfully!"
                                try await Task.sleep(nanoseconds: 2_000_000_000) // 2 second delay
                                status = "Generating with Sesame..."
                            } catch {
                                status = "Failed to load Sesame model: \(error.localizedDescription)"
                                return
                            }
                        }

                        // Generate audio with Sesame
                        if let sesameVoice = SesameVoice(rawValue: chosenVoice) {
                            do {
                                let audio = try sesameTTS!.generateAudio(text: sayThis, voice: sesameVoice)
                                status = "Sesame generation complete! Audio shape: \(audio.shape)"
                            } catch {
                                status = "Sesame generation failed: \(error.localizedDescription)"
                            }
                        } else {
                            status = "Invalid Sesame voice selected"
                        }
                    }

                    if chosenProvider != "sesame" {
                        status = "Done"
                    }
                }
            }, label: {
                Text("Generate")
                    .font(.title2)
                    .padding()
            })
            .buttonStyle(.borderedProminent)

            // Debug button for Sesame
            if chosenProvider == "sesame" {
                Button(action: {
                    if let sesame = sesameTTS {
                        sesame.printStatus()
                        print("\n--- Available Voices ---")
                        sesame.printAvailableVoices()
                        status = "Debug info printed to console"
                    } else {
                        status = "Sesame TTS not initialized"
                    }
                }, label: {
                    Text("Debug Info")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                })
                .buttonStyle(.bordered)
                .padding(.top, 4)
            }

            ScrollView {
                Text(status)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding()
            }
            .frame(height: 100)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

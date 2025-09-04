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
    @State private var marvisTTS: MarvisTTS? = nil

    @State private var sayThis : String = "Hello Everybody"
    @State private var status : String = ""

    private var availableProviders = ["kokoro", "orpheus", "Marvis"]
    @State private var chosenProvider : String = "Marvis"  // Default to Marvis
    @State private var availableVoices: [String] = MarvisTTS.Voice.allCases.map { $0.rawValue }
    @State private var chosenVoice: String = MarvisTTS.Voice.conversationalA.rawValue

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
                } else if newProvider == "Marvis" {
                    availableVoices = MarvisTTS.Voice.allCases.map { $0.rawValue }
                    chosenVoice = availableVoices.first ?? MarvisTTS.Voice.conversationalA.rawValue

                    status = "Marvis TTS: Advanced conversational TTS with streaming support.\n\nNote: Requires model weights to be downloaded from HuggingFace (sesame/csm-1b)"
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

            // Show model status for Marvis
            if chosenProvider == "Marvis" {
                HStack {
                    Circle()
                        .fill(marvisTTS != nil ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(marvisTTS != nil ? "Marvis TTS Ready" : "Marvis TTS Not Initialized")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)

                // Show model info if loaded
                if marvisTTS != nil {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Model: Marvis TTS")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Architecture: Sesame + Mimi")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Sample Rate: \(Int(marvisTTS!.sampleRate)) Hz")
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

                    } else if chosenProvider == "Marvis" {
                        // Initialize Marvis TTS if needed
                        if marvisTTS == nil {
                            do {
                                status = "Loading Marvis TTS model..."
                                marvisTTS = try await MarvisTTS.fromPretrained(progressHandler: { progress in
                                    status = "Loading Marvis TTS: \(Int(progress.fractionCompleted * 100))%"
                                })
                                status = "Marvis TTS model loaded successfully!"
                            } catch {
                                status = "Failed to load Marvis TTS model: \(error.localizedDescription)"
                                return
                            }
                        }

                        // Generate audio with Marvis TTS
                        if let marvisVoice = MarvisTTS.Voice(rawValue: chosenVoice) {
                            do {
                                status = "Generating with Marvis TTS..."
                                let results = try marvisTTS!.generate(text: sayThis, voice: marvisVoice)
                                if let result = results.first {
                                    status = "Marvis TTS generation complete! Audio shape: \(result.audio.count) samples, Sample rate: \(result.sampleRate)Hz"
                                } else {
                                    status = "No audio generated"
                                }
                            } catch {
                                status = "Marvis TTS generation failed: \(error.localizedDescription)"
                            }
                        } else {
                            status = "Invalid Marvis voice selected: \(chosenVoice)"
                        }
                    }

                    if chosenProvider != "Marvis" {
                        status = "Done"
                    }
                }
            }, label: {
                Text("Generate")
                    .font(.title2)
                    .padding()
            })
            .buttonStyle(.borderedProminent)


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

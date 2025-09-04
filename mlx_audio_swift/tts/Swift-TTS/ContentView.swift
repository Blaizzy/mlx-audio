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
    @State private var sesameTTSModel: SesameTTS? = nil

    @State private var sayThis : String = "Hello Everybody"
    @State private var status : String = ""

    @State private var chosenProvider: TTSProvider = .sesame  // Default to Sesame
    @State private var chosenVoice: String = SesameTTS.Voice.conversationalA.rawValue

    var body: some View {
        VStack {
            Image(systemName: "mouth")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("TTS Examples")
                .font(.headline)
                .padding()

            Picker("Choose a provider", selection: $chosenProvider) {
                ForEach(TTSProvider.allCases, id: \.self) { provider in
                    Text(provider.displayName)
                }
            }
            .onChange(of: chosenProvider) { _, newProvider in
                chosenVoice = newProvider.defaultVoice
                status = newProvider.statusMessage
            }
            .padding()
            .padding(.bottom, 0)

            // Voice picker
            Picker("Choose a voice", selection: $chosenVoice) {
                ForEach(chosenProvider.availableVoices, id: \.self) { voice in
                    Text(voice.capitalized)
                }
            }
            .padding()
            .padding(.top, 0)

            TextField("Enter text", text: $sayThis).padding()

            // Show model status for Sesame
            if chosenProvider == .sesame {
                HStack {
                    Circle()
                        .fill(sesameTTSModel != nil ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(sesameTTSModel != nil ? "Sesame TTS Ready" : "Sesame TTS Not Initialized")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)

                // Show model info if loaded
                if sesameTTSModel != nil {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Model: Sesame TTS")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Architecture: Sesame + Mimi")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Sample Rate: \(Int(sesameTTSModel!.sampleRate)) Hz")
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
                    switch chosenProvider {
                    case .kokoro:
                        generateWithKokoro()
                    case .orpheus:
                        await generateWithOrpheus()
                    case .sesame:
                        await generateWithSesame()
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

    // MARK: - TTS Generation Methods

    private func generateWithKokoro() {
        if kokoroTTSModel == nil {
            kokoroTTSModel = KokoroTTSModel()
        }

        if chosenProvider.validateVoice(chosenVoice),
           let kokoroVoice = TTSVoice.fromIdentifier(chosenVoice) ?? TTSVoice(rawValue: chosenVoice) {
            kokoroTTSModel!.say(sayThis, kokoroVoice)
            status = "Done"
        } else {
            status = chosenProvider.errorMessage
        }
    }

    private func generateWithOrpheus() async {
        if orpheusTTSModel == nil {
            orpheusTTSModel = OrpheusTTSModel()
        }

        if chosenProvider.validateVoice(chosenVoice),
           let orpheusVoice = OrpheusVoice(rawValue: chosenVoice) {
            await orpheusTTSModel!.say(sayThis, orpheusVoice)
            status = "Done"
        } else {
            status = chosenProvider.errorMessage
        }
    }

    private func generateWithSesame() async {
        // Initialize Sesame TTS if needed
        if sesameTTSModel == nil {
            do {
                status = "Loading Sesame TTS model..."
                sesameTTSModel = try await SesameTTS.fromPretrained(progressHandler: { progress in
                    status = "Loading Sesame TTS: \(Int(progress.fractionCompleted * 100))%"
                })
                status = "Sesame TTS model loaded successfully!"
            } catch {
                status = "Failed to load Sesame TTS model: \(error.localizedDescription)"
                return
            }
        }

        // Generate audio with Sesame TTS
        if chosenProvider.validateVoice(chosenVoice),
           let sesameVoice = SesameTTS.Voice(rawValue: chosenVoice) {
            do {
                status = "Generating with Sesame TTS..."
                let results = try sesameTTSModel!.generate(text: sayThis, voice: sesameVoice)
                if let result = results.first {
                    status = "Sesame TTS generation complete! Audio shape: \(result.audio.count) samples, Sample rate: \(result.sampleRate)Hz"
                } else {
                    status = "No audio generated"
                }
            } catch {
                status = "Sesame TTS generation failed: \(error.localizedDescription)"
            }
        } else {
            status = chosenProvider.errorMessage + chosenVoice
        }
    }
}

#Preview {
    ContentView()
}

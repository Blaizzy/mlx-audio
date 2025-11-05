//
//  ContentView.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 13/04/2025.
//

import SwiftUI
import MLX

struct ContentView: View {

    @StateObject private var kokoroTTSModel = KokoroTTSModel()
    @State private var orpheusTTSModel: OrpheusTTSModel? = nil
    @State private var marvisSession: MarvisSession? = nil

    @State private var text: String = "Hello Everybody"
    @State private var status: String = ""

    @State private var chosenProvider: TTSProvider = .marvis  // Default to Marvis
    @State private var chosenVoice: String = MarvisSession.Voice.conversationalA.rawValue

    // Loading and playing states
    @State private var isMarvisLoading = false
    @State private var isOrpheusGenerating = false

    // Computed property to check if any generation is in progress
    private var isCurrentlyGenerating: Bool {
        kokoroTTSModel.generationInProgress || isOrpheusGenerating || isMarvisLoading
    }


    var body: some View {
        VStack {
            Text("MLX Audio Eval")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding()

            Picker("Choose a provider", selection: $chosenProvider) {
                ForEach(TTSProvider.allCases, id: \.self) { provider in
                    Text(provider.displayName)
                }
            }
            .onChange(of: chosenProvider) { newProvider in
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
            
            HStack {
                TextField("Enter text", text: $text)
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
            .padding()
            
            // Show model status for Marvis
            if chosenProvider == .marvis {
                HStack {
                    Circle()
                        .fill(marvisSession != nil ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(marvisSession != nil ? "Marvis Ready" : "Marvis Not Initialized")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)

                // Show model info if loaded
                if marvisSession != nil {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Model: Marvis")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Architecture: Marvis + Mimi")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Sample Rate: \(Int(marvisSession!.sampleRate)) Hz")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 4)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(4)
                }
            }
            
            // Generate and Stop buttons
            HStack(spacing: 12) {
                // Generate button
                Button(action: {
                    // Validate text is not empty
                    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmedText.isEmpty else {
                        status = "Please enter some text before generating audio."
                        return
                    }

                    Task {
                        status = "Generating..."
                        switch chosenProvider {
                        case .kokoro:
                            generateWithKokoro()
                        case .orpheus:
                            isOrpheusGenerating = true
                            await generateWithOrpheus()
                            isOrpheusGenerating = false
                        case .marvis:
                            await generateWithMarvis()
                        }
                    }
                }, label: {
                    HStack {
                        if isCurrentlyGenerating {
                            ProgressView()
                                .controlSize(.small)
                                .scaleEffect(0.8)
                            Text(isMarvisLoading ? "Loading..." : "Generating...")
                        } else {
                            Text("Generate")
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 32)
                })
                .buttonStyle(.borderedProminent)
                .disabled(isCurrentlyGenerating || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                // Stop button
                Button(action: {
                    // Stop generation based on provider
                    switch chosenProvider {
                    case .kokoro:
                        kokoroTTSModel.stopPlayback()
                    case .orpheus:
                        // Orpheus doesn't have stop functionality yet
                        status = "Orpheus generation cannot be stopped"
                    case .marvis:
                        marvisSession?.cleanupMemory()
                        isMarvisLoading = false
                    }
                    status = "Generation stopped"
                }, label: {
                    HStack {
                        Image(systemName: "stop.fill")
                        Text("Stop")
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 32)
                })
                .buttonStyle(.bordered)
                .disabled(!isCurrentlyGenerating)
            }
            .padding(.horizontal)
            
            // Streaming toggle removed for now
            
            
            ScrollView {
                Text(status)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding()
            }
            .frame(height: 150)
        }
        .padding()
    }
    
    // MARK: - TTS Generation Methods
    
    private func generateWithKokoro() {
        if chosenProvider.validateVoice(chosenVoice),
           let kokoroVoice = TTSVoice.fromIdentifier(chosenVoice) ?? TTSVoice(rawValue: chosenVoice) {
            kokoroTTSModel.say(text, kokoroVoice)
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
            await orpheusTTSModel!.say(text, orpheusVoice)
            status = "Done"
        } else {
            status = chosenProvider.errorMessage
        }
    }
    
    private func generateWithMarvis() async {
        // Initialize Marvis if needed with bound voice
        if marvisSession == nil {
            do {
                isMarvisLoading = true
                status = "Loading Marvis..."
                guard let voice = MarvisSession.Voice(rawValue: chosenVoice) else {
                    status = "\(chosenProvider.errorMessage)\(chosenVoice)"
                    isMarvisLoading = false
                    return
                }
                marvisSession = try await MarvisSession(voice: voice, progressHandler: { progress in
                    status = "Loading Marvis: \(Int(progress.fractionCompleted * 100))%"
                })
                status = "Marvis loaded successfully!"
                isMarvisLoading = false
            } catch {
                status = "Failed to load Marvis: \(error.localizedDescription)"
                isMarvisLoading = false
                return
            }
        }

        // Generate audio using bound configuration
        do {
            status = "Generating with Marvis..."
            let result = try await marvisSession!.generate(for: text)
            status = "Marvis generation complete! Audio: \(result.audio.count) samples @ \(result.sampleRate)Hz"
        } catch {
            status = "Marvis generation failed: \(error.localizedDescription)"
        }
    }
}

#Preview {
    ContentView()
}

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
    @State private var sesameSession: SesameSession? = nil
    
    @State private var sayThis : String = "Hello Everybody"
    @State private var status : String = ""
    
    @State private var chosenProvider: TTSProvider = .sesame  // Default to Sesame
    @State private var chosenVoice: String = SesameSession.Voice.conversationalA.rawValue
    
    
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
                        .fill(sesameSession != nil ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(sesameSession != nil ? "Sesame Ready" : "Sesame Not Initialized")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)
                
                // Show model info if loaded
                if sesameSession != nil {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Model: Sesame")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Architecture: Sesame + Mimi")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Sample Rate: \(Int(sesameSession!.sampleRate)) Hz")
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
            
            // Streaming toggle removed for now
            
            
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
        // Initialize Sesame if needed with bound voice
        if sesameSession == nil {
            do {
                status = "Loading Sesame..."
                guard let voice = SesameSession.Voice(rawValue: chosenVoice) else {
                    status = chosenProvider.errorMessage + chosenVoice
                    return
                }
                sesameSession = try await SesameSession(voice: voice, progressHandler: { progress in
                    status = "Loading Sesame: \(Int(progress.fractionCompleted * 100))%"
                })
                status = "Sesame loaded successfully!"
            } catch {
                status = "Failed to load Sesame: \(error.localizedDescription)"
                return
            }
        }
        
        // Generate audio using bound configuration
        do {
            status = "Generating with Sesame..."
            let result = try await sesameSession!.generate(for: sayThis)
            status = "Sesame generation complete! Audio: \(result.audio.count) samples @ \(result.sampleRate)Hz"
        } catch {
            status = "Sesame generation failed: \(error.localizedDescription)"
        }
    }
}

#Preview {
    ContentView()
}

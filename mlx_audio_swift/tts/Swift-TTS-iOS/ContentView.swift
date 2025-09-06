//
//  ContentView.swift
//   Swift-TTS-iOS
//
//  Created by Sachin Desai on 5/17/25.
//

import SwiftUI
import MLX

// MARK: - TTS Provider Enum

enum TTSProvider: String, CaseIterable {
    case kokoro = "kokoro"
    case sesame = "sesame"
    
    var displayName: String {
        rawValue.capitalized
    }
    
    var statusMessage: String {
        switch self {
        case .kokoro:
            return ""
        case .sesame:
            return "Sesame TTS: Advanced conversational TTS with streaming support."
        }
    }
}

struct ContentView: View {
    @State private var speed = 1.0
    @State public var text = ""
    @State private var showAlert = false
    
    @FocusState private var isTextEditorFocused: Bool
    @State private var chosenProvider: TTSProvider = .kokoro
    
    // TTS Models
    @ObservedObject var kokoroViewModel: KokoroTTSModel
    
    // Alias for backward compatibility
    var viewModel: KokoroTTSModel { kokoroViewModel }
    @State private var sesameTTSModel: SesameSession? = nil
    @State private var isSesameLoading = false
    @State private var isSesamePlaying = false
    @State private var status = ""
    @State private var chosenVoice = "conversational_a"
    
    @StateObject private var speakerModel = SpeakerViewModel()
    
    var body: some View {
        NavigationStack {
            ZStack {
                backgroundView
                
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 16) {
                        // Provider Selection
                        VStack(alignment: .leading, spacing: 8) {
                            Text("TTS Provider")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            
                            Picker("Choose a provider", selection: $chosenProvider) {
                                ForEach(TTSProvider.allCases, id: \.self) { provider in
                                    Text(provider.displayName)
                                }
                            }
                            .pickerStyle(.segmented)
                            .disabled(isSesameLoading || kokoroViewModel.generationInProgress)
                            .onChange(of: chosenProvider) { _, newProvider in
                                // Reset speaker selection when switching providers
                                speakerModel.selectedSpeakerId = 0
                                status = newProvider.statusMessage
                            }
                        }
                        
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 12) {
                                compactSpeakerView(
                                    selectedSpeakerId: $speakerModel.selectedSpeakerId,
                                    title: chosenProvider == .kokoro ? "Speaker" : "Voice",
                                    speakers: chosenProvider == .kokoro ? speakerModel.kokoroSpeakers : speakerModel.sesameSpeakers
                                )
                                .frame(maxWidth: .infinity)
                            }
                        }
                        
                        speedControlView
                        textInputView
                        
                        actionButtonsView
                    }
                    .padding([.horizontal, .bottom])
                }
                .toolbar {
                    ToolbarItem(placement: .principal) {
                        VStack(spacing: 0) {
                            HStack {
                                Text(chosenProvider.displayName)
                                    .font(.title)
                                if isSesameLoading {
                                    ProgressView()
                                        .controlSize(.small)
                                        .padding(.leading, 8)
                                }
                            }
                            if chosenProvider == .kokoro {
                                Text("Time to first audio sample: \(kokoroViewModel.audioGenerationTime > 0 ? String(format: "%.2f", kokoroViewModel.audioGenerationTime) : "--")s")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else {
                                Text(chosenProvider.statusMessage)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .scrollContentBackground(.hidden)
                .alert("Empty Text", isPresented: $showAlert) {
                    Button("OK", role: .cancel) { }
                } message: {
                    Text("Please enter some text before generating audio.")
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    if isTextEditorFocused {
                        dismissKeyboard()
                        isTextEditorFocused = false
                    }
                }
            }
        }
        // Sync generation progress to speakerModel.isGenerating
        .onChange(of: kokoroViewModel.generationInProgress) { _, newValue in
            if chosenProvider == .kokoro {
                speakerModel.isGenerating = newValue
            }
        }
        .onChange(of: isSesameLoading) { _, newValue in
            if chosenProvider == .sesame {
                speakerModel.isGenerating = newValue
            }
        }
    }
    
    private var backgroundView: some View {
        Color(.systemBackground)
            .ignoresSafeArea()
    }
    
    private func compactSpeakerView(selectedSpeakerId: Binding<Int>, title: String, speakers: [Speaker]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            Menu {
                ForEach(speakers) { speaker in
                    Button(action: {
                        selectedSpeakerId.wrappedValue = speaker.id
                    }) {
                        HStack {
                            Text("\(speaker.flag) \(speaker.displayName)")
                            if selectedSpeakerId.wrappedValue == speaker.id {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                HStack {
                    if let speaker = speakers.first(where: { $0.id == selectedSpeakerId.wrappedValue }) {
                        Text(speaker.flag)
                        Text(speaker.displayName)
                            .lineLimit(1)
                            .foregroundStyle(.primary)
                    }
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(.tertiarySystemBackground))
                )
            }
            .disabled((chosenProvider == .kokoro && kokoroViewModel.generationInProgress) ||
                      (chosenProvider == .sesame && isSesameLoading))
        }
    }
    
    private var speedControlView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Speed")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1fx", speed))
                    .font(.subheadline)
                    .bold()
            }
            
            Slider(value: $speed, in: 0.5...2.0, step: 0.1)
                .tint(.accentColor)
                .disabled(viewModel.generationInProgress)
        }
    }
    
    private var textInputView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Text Input")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            ZStack(alignment: .topLeading) {
                TextEditor(text: $text)
                    .font(.body)
                    .frame(minHeight: 120)
                    .scrollContentBackground(.hidden)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.tertiarySystemBackground))
                    )
                    .focused($isTextEditorFocused)
                    .disabled(viewModel.generationInProgress)
                    .onTapGesture {
                        // Explicitly focus the text editor when tapped
                        if !isTextEditorFocused && !viewModel.generationInProgress {
                            isTextEditorFocused = true
                        }
                    }
                
                if text.isEmpty {
                    Text("Enter your text here...")
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 20)
                        .padding(.top, 25)
                        .allowsHitTesting(false)
                }
            }
        }
    }
    
    private var actionButtonsView: some View {
        HStack(spacing: 12) {
            // generatge button
            Button {
                if isTextEditorFocused {
                    dismissKeyboard()
                    isTextEditorFocused = false
                }
                
                let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
                
                Task {
                    if chosenProvider == .kokoro {
                        // Prepare text and speaker for Kokoro
                        let speaker = speakerModel.getPrimarySpeaker().first!
                        
                        // Set memory constraints for MLX and start generation
                        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
                        kokoroViewModel.say(t, TTSVoice.fromIdentifier(speaker.name) ?? .afHeart, speed: Float(speed))
                    } else if chosenProvider == .sesame {
                        // Initialize Sesame TTS if needed
                        if sesameTTSModel == nil {
                            isSesameLoading = true
                            do {
                                sesameTTSModel = try await SesameSession.fromPretrained(progressHandler: { progress in
                                    // Update loading status if needed
                                })
                                isSesameLoading = false
                            } catch {
                                isSesameLoading = false
                                print("Failed to load Sesame TTS: \(error)")
                                return
                            }
                        }
                        
                        // Generate with Sesame TTS
                        let selectedSesameVoice: SesameSession.Voice
                        if chosenVoice == "conversational_a" {
                            selectedSesameVoice = .conversationalA
                        } else if chosenVoice == "conversational_b" {
                            selectedSesameVoice = .conversationalB
                        } else {
                            selectedSesameVoice = .conversationalA // Default fallback
                        }
                        
                        do {
                            status = "Generating with Sesame TTS (streaming)..."
                            isSesamePlaying = true
                            let stream = sesameTTSModel!.stream(text: t, voice: selectedSesameVoice)
                            var totalSamples = 0
                            for try await chunk in stream {
                                totalSamples += chunk.sampleCount
                                status = "Streaming... \(totalSamples) samples (RTF ~\(chunk.realTimeFactor))"
                            }
                            status = "Sesame TTS generation complete!"
                        } catch {
                            isSesamePlaying = false
                            status = "Sesame TTS generation failed: \(error.localizedDescription)"
                        }
                    }
                }
            } label: {
                HStack {
                    if (chosenProvider == .kokoro && kokoroViewModel.generationInProgress) ||
                        (chosenProvider == .sesame && isSesameLoading) {
                        ProgressView()
                            .controlSize(.small)
                        Text(chosenProvider == .sesame && isSesameLoading ? "Loading..." : "Generating...")
                    } else {
                        Text("Generate")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .frame(maxWidth: .infinity, minHeight: 44)
            .disabled((chosenProvider == .kokoro && kokoroViewModel.generationInProgress) ||
                      (chosenProvider == .sesame && isSesameLoading) ||
                      text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            
            // Stop button
            Button {
                if chosenProvider == .kokoro {
                    kokoroViewModel.stopPlayback()
                } else if chosenProvider == .sesame {
                    // Stop Sesame TTS playback and reset session
                    sesameTTSModel?.cleanupMemory()
                    isSesamePlaying = false
                    status = "Sesame TTS playback stopped"
                }
            } label: {
                HStack {
                    Image(systemName: "stop.fill")
                    Text("Stop")
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .frame(maxWidth: .infinity, minHeight: 44)
            .tint(.red)
            .disabled((chosenProvider == .kokoro && !kokoroViewModel.isAudioPlaying) ||
                      (chosenProvider == .sesame && !isSesamePlaying))
        }
    }
}

struct Speaker: Identifiable {
    let id: Int
    let name: String
    
    var flag: String {
        if name.lowercased() == "none" {
            return "⚪️" // Empty/None speaker icon
        }
        
        guard name.count >= 2 else { return "🏳️" }
        let country = name.prefix(1)
        
        // Determine country flag
        let countryFlag: String
        switch country {
        case "a": countryFlag = "🇺🇸" // USA
        case "b": countryFlag = "🇬🇧" // British
        case "e": countryFlag = "🇪🇸" // Spain
        case "f": countryFlag = "🇫🇷" // French
        case "h": countryFlag = "🇮🇳" // Hindi
        case "i": countryFlag = "🇮🇹" // Italian
        case "j": countryFlag = "🇯🇵" // Japanese
        case "p": countryFlag = "🇧🇷" // Brazil
        case "z": countryFlag = "🇨🇳" // Chinese
        default: countryFlag = "🏳️"
        }
        
        return countryFlag
    }
    
    var displayName: String {
        if name.lowercased() == "none" {
            return "None" // Special case for None option
        }
        
        // Handle Sesame conversational voices
        if name.hasPrefix("conversational_") {
            let voiceType = name.dropFirst("conversational_".count)
            return "Conversational \(voiceType.uppercased())"
        }
        
        // Handle Kokoro speakers (original logic)
        guard name.count >= 2 else { return name }
        let cleanName = name.dropFirst(3).capitalized
        return "\(cleanName)"
    }
}

class SpeakerViewModel: ObservableObject {
    @Published var selectedSpeakerId: Int = 0
    @Published var selectedSpeakerId2: Int = -1
    @Published var isGenerating: Bool = false
    
    // All Kokoro speakers
    private let _kokoroSpeakers: [Speaker] = [
        Speaker(id: -1, name: "none"),
        Speaker(id: 0, name: "af_alloy"),
        Speaker(id: 1, name: "af_aoede"),
        Speaker(id: 2, name: "af_bella"),
        Speaker(id: 3, name: "af_heart"),
        Speaker(id: 4, name: "af_jessica"),
        Speaker(id: 5, name: "af_kore"),
        Speaker(id: 6, name: "af_nicole"),
        Speaker(id: 7, name: "af_nova"),
        Speaker(id: 8, name: "af_river"),
        Speaker(id: 9, name: "af_sarah"),
        Speaker(id: 10, name: "af_sky"),
        Speaker(id: 11, name: "am_adam"),
        Speaker(id: 12, name: "am_echo"),
        Speaker(id: 13, name: "am_eric"),
        Speaker(id: 14, name: "am_fenrir"),
        Speaker(id: 15, name: "am_liam"),
        Speaker(id: 16, name: "am_michael"),
        Speaker(id: 17, name: "am_onyx"),
        Speaker(id: 18, name: "am_puck"),
        Speaker(id: 19, name: "am_santa"),
        Speaker(id: 20, name: "bf_alice"),
        Speaker(id: 21, name: "bf_emma"),
        Speaker(id: 22, name: "bf_isabella"),
        Speaker(id: 23, name: "bf_lily"),
        Speaker(id: 24, name: "bm_daniel"),
        Speaker(id: 25, name: "bm_fable"),
        Speaker(id: 26, name: "bm_george"),
        Speaker(id: 27, name: "bm_lewis"),
        Speaker(id: 28, name: "ef_dora"),
        Speaker(id: 29, name: "em_alex"),
        Speaker(id: 30, name: "ff_siwis"),
        Speaker(id: 31, name: "hf_alpha"),
        Speaker(id: 32, name: "hf_beta"),
        Speaker(id: 33, name: "hm_omega"),
        Speaker(id: 34, name: "hm_psi"),
        Speaker(id: 35, name: "if_sara"),
        Speaker(id: 36, name: "im_nicola"),
        Speaker(id: 37, name: "jf_alpha"),
        Speaker(id: 38, name: "jf_gongitsune"),
        Speaker(id: 39, name: "jf_nezumi"),
        Speaker(id: 40, name: "jf_tebukuro"),
        Speaker(id: 41, name: "jm_kumo"),
        Speaker(id: 42, name: "pf_dora"),
        Speaker(id: 43, name: "pm_alex"),
        Speaker(id: 44, name: "pm_santa"),
        Speaker(id: 45, name: "zf_xiaobei"),
        Speaker(id: 46, name: "zf_xiaoni"),
        Speaker(id: 47, name: "zf_xiaoxiao"),
        Speaker(id: 48, name: "zf_xiaoyi"),
        Speaker(id: 49, name: "zm_yunjian"),
        Speaker(id: 50, name: "zm_yunxi"),
        Speaker(id: 51, name: "zm_yunxia"),
        Speaker(id: 52, name: "zm_yunyang"),
    ]
    
    // Sesame voices (simplified for iOS version)
    private let _sesameSpeakers: [Speaker] = [
        Speaker(id: 0, name: "conversational_a"),
        Speaker(id: 1, name: "conversational_b"),
    ]
    
    // Public accessors
    var kokoroSpeakers: [Speaker] { _kokoroSpeakers }
    var sesameSpeakers: [Speaker] { _sesameSpeakers }
    
    // Dynamic speakers based on selected provider
    var speakers: [Speaker] {
        // This will be set from outside based on chosenProvider
        _kokoroSpeakers // Default to Kokoro
    }
    
    func updateSpeakers(for provider: TTSProvider) {
        // Update the speakers list based on provider
        // We'll handle this in the ContentView by using the appropriate speaker list
    }
    
    func getPrimarySpeaker() -> [Speaker] {
        speakers.filter { $0.id == selectedSpeakerId }
    }
    
    func getSecondarySpeaker() -> [Speaker] {
        speakers.filter { $0.id == selectedSpeakerId2 }
    }
    
    func getSpeaker(id: Int) -> Speaker? {
        speakers.first { $0.id == id }
    }
}

extension View {
    func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                        to: nil,
                                        from: nil,
                                        for: nil)
    }
}

#Preview {
    ContentView(kokoroViewModel: KokoroTTSModel())
}

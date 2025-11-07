//
//  ContentView.swift
//   MLXAudio-iOS
//
//  Created by Sachin Desai on 5/17/25.
//

import SwiftUI
import MLX

// MARK: - TTS Provider Enum

enum TTSProvider: String, CaseIterable {
    case marvis = "marvis"
    case kokoro = "kokoro"
    
    var displayName: String {
        rawValue.capitalized
    }
    
    var statusMessage: String {
        switch self {
        case .kokoro:
            return ""
        case .marvis:
            return "Marvis TTS: Advanced conversational TTS with streaming support."
        }
    }
}

struct ContentView: View {
    // MARK: - Constants
    private let mlxGPUCacheLimit = 20 * 1024 * 1024  // 20MB cache limit
    
    // MARK: - State Properties
    @State private var speed = 1.0
    @State public var text = "How are you doing today?"
    @State private var showAlert = false
    
    @FocusState private var isTextEditorFocused: Bool
    @State private var chosenProvider: TTSProvider = .marvis
    
    // MARK: - TTS Models
    @ObservedObject var kokoroViewModel: KokoroTTSModel

    // Alias for backward compatibility
    var viewModel: KokoroTTSModel { kokoroViewModel }
    @State private var marvisSession: MarvisSession? = nil
    @State private var isMarvisLoading = false
    @State private var isMarvisPlaying = false
    @State private var status = ""
    @State private var chosenVoice = "conversational_a"
    @State private var chosenQuality: MarvisSession.QualityLevel = .maximum
    @State private var marvisAudioGenerationTime: TimeInterval = 0
    @State private var useStreaming: Bool = false
    @State private var streamingInterval: Double = 0.5

    @StateObject private var speakerModel = SpeakerViewModel()
    
    var body: some View {
        NavigationStack {
            ZStack {
                backgroundView
                
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 16) {
                        // Provider Status Header
                        VStack(spacing: 4) {
                            HStack {
                                Text(chosenProvider.displayName)
                                    .font(.title)
                                if isMarvisLoading {
                                    ProgressView()
                                        .controlSize(.small)
                                        .padding(.leading, 8)
                                }
                            }
                            if chosenProvider == .kokoro {
                                Text("Time to first audio sample: \(kokoroViewModel.audioGenerationTime > 0 ? String(format: "%.2f", kokoroViewModel.audioGenerationTime) : "--")s")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else if chosenProvider == .marvis {
                                Text("Time to first audio sample: \(marvisAudioGenerationTime > 0 ? String(format: "%.2f", marvisAudioGenerationTime) : "--")s")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.top, 8)
                        
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
                            .disabled(isMarvisLoading || kokoroViewModel.generationInProgress)
                            .onChange(of: chosenProvider) { newProvider in
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
                                    speakers: chosenProvider == .kokoro ? speakerModel.kokoroSpeakers : speakerModel.marvisSpeakers
                                )
                                .frame(maxWidth: .infinity)
                            }
                        }

                        // Quality picker for Marvis
                        if chosenProvider == .marvis {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Quality")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)

                                Picker("Quality", selection: $chosenQuality) {
                                    ForEach(MarvisSession.QualityLevel.allCases, id: \.self) { quality in
                                        Text(quality.rawValue.capitalized).tag(quality)
                                    }
                                }
                                .pickerStyle(.segmented)
                                .disabled(isMarvisLoading)

                                Text(qualityDescription)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            // Streaming toggle
                            VStack(alignment: .leading, spacing: 8) {
                                Toggle("Use Streaming", isOn: $useStreaming)
                                    .disabled(isMarvisLoading)

                                Text(useStreaming ? "Real-time audio streaming with progress feedback" : "Generate complete audio before playback")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            // Streaming interval (when streaming enabled)
                            if useStreaming {
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        Text("Streaming Interval")
                                            .font(.subheadline)
                                            .foregroundStyle(.secondary)

                                        Spacer()

                                        Text(String(format: "%.1fs", streamingInterval))
                                            .font(.subheadline)
                                            .bold()
                                    }

                                    Slider(value: $streamingInterval, in: 0.1...1.0, step: 0.1)
                                        .tint(.accentColor)
                                        .disabled(isMarvisLoading)

                                    Text("Time between audio chunks (lower = faster response)")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }

                        speedControlView
                        textInputView
                        
                        // Status display
                        if !status.isEmpty {
                            Text(status)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                                .frame(maxWidth: .infinity)
                        }
                        
                        actionButtonsView
                    }
                    .padding([.horizontal, .bottom])
                }
                .navigationTitle("MLX Audio Eval")
                .navigationBarTitleDisplayMode(.large)
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
        .onChange(of: kokoroViewModel.generationInProgress) { newValue in
            if chosenProvider == .kokoro {
                speakerModel.isGenerating = newValue
            }
        }
        .onChange(of: isMarvisLoading) { newValue in
            if chosenProvider == .marvis {
                speakerModel.isGenerating = newValue
            }
        }
    }
    
    // MARK: - View Components
    
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
                      (chosenProvider == .marvis && isMarvisLoading))
        }
    }
    
    private var speedControlView: some View {
        VStack(alignment: .leading, spacing: 8) {
            if chosenProvider == .kokoro {
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
    }
    
    private var textInputView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Text Input")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                if !text.isEmpty {
                    Button {
                        text = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                            .font(.subheadline)
                    }
                }
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
                    Text("Enter the text here...")
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
            // Generate button
            Button {
                if isTextEditorFocused {
                    dismissKeyboard()
                    isTextEditorFocused = false
                }
                
                let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
                
                Task {
                    if chosenProvider == .kokoro {
                        // Prepare text and speaker for Kokoro
                        let speaker = speakerModel.getPrimarySpeaker().first!
                        
                        // Set memory constraints for MLX and start generation
                        MLX.GPU.set(cacheLimit: mlxGPUCacheLimit)
                        kokoroViewModel.say(text, TTSVoice.fromIdentifier(speaker.name) ?? .afHeart, speed: Float(speed))
                    } else if chosenProvider == .marvis {
                        // Initialize Marvis TTS if needed
                        if marvisSession == nil {
                            isMarvisLoading = true
                            do {
                                marvisSession = try await MarvisSession.fromPretrained(progressHandler: { progress in
                                    // Update loading status if needed
                                })
                                isMarvisLoading = false
                            } catch {
                                isMarvisLoading = false
                                print("Failed to load Marvis TTS: \(error)")
                                return
                            }
                        }

                        // Generate with Marvis TTS
                        let selectedMarvisVoice: MarvisSession.Voice
                        if chosenVoice == "conversational_a" {
                            selectedMarvisVoice = .conversationalA
                        } else if chosenVoice == "conversational_b" {
                            selectedMarvisVoice = .conversationalB
                        } else {
                            selectedMarvisVoice = .conversationalA // Default fallback
                        }

                        do {
                            isMarvisPlaying = true
                            marvisAudioGenerationTime = 0

                            if useStreaming {
                                // Use streaming API
                                status = "Streaming with Marvis TTS..."
                                let stream = marvisSession!.stream(
                                    text: text,
                                    voice: selectedMarvisVoice,
                                    qualityLevel: chosenQuality,
                                    streamingInterval: streamingInterval
                                )
                                var totalSamples = 0
                                var isFirstChunk = true
                                for try await chunk in stream {
                                    if isFirstChunk {
                                        marvisAudioGenerationTime = chunk.processingTime
                                        isFirstChunk = false
                                    }
                                    totalSamples += chunk.sampleCount
                                    status = "Streaming... \(totalSamples) samples (RTF ~\(String(format: "%.2f", chunk.realTimeFactor)))"
                                }
                                status = "Marvis TTS streaming complete!"
                            } else {
                                // Use non-streaming API
                                status = "Generating with Marvis TTS..."
                                let result = try await marvisSession!.generateRaw(
                                    for: text,
                                    quality: chosenQuality
                                )
                                marvisAudioGenerationTime = result.processingTime
                                status = "Marvis TTS generation complete! \(result.sampleCount) samples"
                            }

                            isMarvisPlaying = false
                        } catch {
                            isMarvisPlaying = false
                            isMarvisLoading = false
                            status = "Marvis TTS generation failed: \(error.localizedDescription)"
                        }
                    }
                }
            } label: {
                HStack {
                    if (chosenProvider == .kokoro && kokoroViewModel.generationInProgress) ||
                        (chosenProvider == .marvis && isMarvisLoading) {
                        ProgressView()
                            .controlSize(.small)
                        Text(chosenProvider == .marvis && isMarvisLoading ? "Loading..." : "Generating...")
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
                      (chosenProvider == .marvis && isMarvisLoading) ||
                      text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            
            // Stop button
            Button {
                if chosenProvider == .kokoro {
                    kokoroViewModel.stopPlayback()
                } else if chosenProvider == .marvis {
                    // Stop Marvis TTS playback and reset session
                    do {
                        try marvisSession?.cleanupMemory()
                    } catch {
                        print("Failed to cleanup Marvis memory: \(error)")
                    }
                    isMarvisPlaying = false
                    status = "Marvis TTS playback stopped"
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
                      (chosenProvider == .marvis && !isMarvisPlaying))
        }
    }

    // MARK: - Helper Functions

    private var qualityDescription: String {
        switch chosenQuality {
        case .low:
            return "8 codebooks - Fastest, lower quality"
        case .medium:
            return "16 codebooks - Balanced"
        case .high:
            return "24 codebooks - Slower, better quality"
        case .maximum:
            return "32 codebooks - Slowest, best quality"
        }
    }
}

// MARK: - Speaker Model

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
        
        // Handle Marvis conversational voices
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

// MARK: - Speaker ViewModel

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
    
    // Marvis voices (simplified for iOS version)
    private let _marvisSpeakers: [Speaker] = [
        Speaker(id: 0, name: "conversational_a"),
        Speaker(id: 1, name: "conversational_b"),
    ]

    // Public accessors
    var kokoroSpeakers: [Speaker] { _kokoroSpeakers }
    var marvisSpeakers: [Speaker] { _marvisSpeakers }
    
    // Dynamic speakers based on selected provider
    var speakers: [Speaker] {
        // This will be set from outside based on chosenProvider
        _kokoroSpeakers // Default to Kokoro
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

// MARK: - View Extension

extension View {
    func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                        to: nil,
                                        from: nil,
                                        for: nil)
    }
}

// MARK: - Preview

#Preview {
    ContentView(kokoroViewModel: KokoroTTSModel())
}

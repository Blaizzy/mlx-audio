//
//  ContentView.swift
//   Swift-TTS-iOS
//
//  Created by Sachin Desai on 5/17/25.
//

import SwiftUI
import MLX

struct ContentView: View {
    @State private var speed = 1.0
    @State public var text = ""
    @State private var showAlert = false
  
    @FocusState private var isTextEditorFocused: Bool
    @ObservedObject var viewModel: KokoroTTSModel
    @StateObject private var speakerModel = SpeakerViewModel()
    
    var body: some View {
        NavigationStack {
            ZStack {
                backgroundView
                
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 12) {
                                compactSpeakerView(
                                    selectedSpeakerId: $speakerModel.selectedSpeakerId,
                                    title: "Speaker"
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
                                Text("Kokoro")
                                    .font(.title)
                            }
                            Text("Time to first audio sample: \(viewModel.audioGenerationTime > 0 ? String(format: "%.2f", viewModel.audioGenerationTime) : "--")s")
                            .font(.caption)
                            .foregroundStyle(.secondary)
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
        // Sync viewModel.generationInProgress to speakerModel.isGenerating
        .onChange(of: viewModel.generationInProgress) { _, newValue in
            speakerModel.isGenerating = newValue
        }
    }
    
    private var backgroundView: some View {
        Color(.systemBackground)
            .ignoresSafeArea()
    }
    
    private func compactSpeakerView(selectedSpeakerId: Binding<Int>, title: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            Menu {
                ForEach(speakerModel.speakers) { speaker in
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
                    if let speaker = speakerModel.getSpeaker(id: selectedSpeakerId.wrappedValue) {
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
            .disabled(viewModel.generationInProgress)
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
                
                // Prepare text and speaker
                let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
                let speaker = speakerModel.getPrimarySpeaker().first!
                
                // Set memory constraints for MLX and start generation
                MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
                viewModel.say(t, TTSVoice.fromIdentifier(speaker.name) ?? .afHeart, speed: Float(speed))
            } label: {
                HStack {
                    if viewModel.generationInProgress {
                        ProgressView()
                            .controlSize(.small)
                        Text("Generating...")
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
            .disabled(viewModel.generationInProgress || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            
            // Stop button
            Button {
                viewModel.stopPlayback()
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
            .disabled(!viewModel.isAudioPlaying)
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

        guard name.count >= 2 else { return name }
        let cleanName = name.dropFirst(3).capitalized
        return "\(cleanName)"
    }
}

class SpeakerViewModel: ObservableObject {
    @Published var selectedSpeakerId: Int = 0
    @Published var selectedSpeakerId2: Int = -1
    @Published var isGenerating: Bool = false

    let speakers: [Speaker] = [
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
  ContentView(viewModel: KokoroTTSModel())
}

//
//  ContentView.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 13/04/2025.
//

import SwiftUI
import MLX

struct ContentView: View {

    // MARK: - State Management
    @StateObject private var kokoroTTSModel = KokoroTTSModel()
    @State private var orpheusTTSModel: OrpheusTTSModel? = nil
    @State private var marvisSession: MarvisSession? = nil

    @State private var text: String = "Hello Everybody"
    @State private var status: String = ""

    @State private var chosenProvider: TTSProvider = .kokoro
    @State private var chosenVoice: String = TTSVoice.bmGeorge.rawValue

    // Sidebar selection
    @State private var selectedSidebarItem: SidebarItem = .textToSpeech

    // Loading and playing states
    @State private var isMarvisLoading = false
    @State private var isOrpheusGenerating = false

    // MARK: - Computed Properties

    // Computed property to check if any generation is in progress
    private var isCurrentlyGenerating: Bool {
        kokoroTTSModel.generationInProgress || isOrpheusGenerating || isMarvisLoading
    }

    private var canGenerate: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // MARK: - Body

    var body: some View {
        NavigationSplitView {
            // Left: Sidebar
            SidebarView(selection: $selectedSidebarItem)
                .frame(width: 250)
        } detail: {
            // Center + Right: Main Content + Inspector
            HSplitView {
                // Center: Main Content Area
                TTSMainView(
                    text: $text,
                    status: $status,
                    selectedProvider: chosenProvider
                )
                .frame(minWidth: 400)

                // Right: Inspector Panel
                TTSInspectorView(
                    selectedProvider: $chosenProvider,
                    selectedVoice: $chosenVoice,
                    status: $status,
                    isGenerating: isCurrentlyGenerating,
                    canGenerate: canGenerate,
                    marvisSession: marvisSession,
                    onGenerate: handleGenerate,
                    onStop: handleStop
                )
            }
        }
        .frame(minWidth: 1200, minHeight: 700)
        .navigationTitle("MLX Audio")
        .onChange(of: chosenProvider) { _, newProvider in
            status = newProvider.statusMessage
        }
    }

    // MARK: - Actions

    private func handleGenerate() {
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
    }

    private func handleStop() {
        switch chosenProvider {
        case .kokoro:
            kokoroTTSModel.stopPlayback()
        case .orpheus:
            status = "Orpheus generation cannot be stopped"
        case .marvis:
            marvisSession?.cleanupMemory()
            isMarvisLoading = false
        }
        status = "Generation stopped"
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

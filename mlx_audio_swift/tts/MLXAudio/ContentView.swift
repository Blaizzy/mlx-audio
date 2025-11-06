//
//  ContentView.swift
//  MLXAudio
//
//  Created by Ben Harraway on 13/04/2025.
//

import SwiftUI
import MLX
import AVFoundation

struct ContentView: View {

    // MARK: - State Management
    @StateObject private var kokoroTTSModel = KokoroTTSModel()
    @StateObject private var audioPlayerManager = AudioPlayerManager()
    @State private var orpheusTTSModel: OrpheusTTSModel? = nil
    @State private var marvisSession: MarvisSession? = nil
    @State private var marvisLastAudioURL: URL?

    @State private var text: String = "Hello Everybody"
    @State private var status: String = ""

    @State private var chosenProvider: TTSProvider = .marvis
    @State private var chosenVoice: String = MarvisSession.Voice.conversationalA.rawValue
    @State private var chosenQuality: MarvisSession.QualityLevel = .maximum

    // Sidebar selection
    @State private var selectedSidebarItem: SidebarItem = .textToSpeech

    // Loading and playing states
    @State private var isMarvisLoading = false
    @State private var isOrpheusGenerating = false

    // Auto-play setting
    @State private var autoPlay: Bool = true

    // Streaming setting
    @State private var useStreaming: Bool = false
    @State private var streamingInterval: Double = 0.5

    // Inspector visibility
    @State private var isInspectorPresent: Bool = true

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
            // Main Content Area
            TTSMainView(
                text: $text,
                status: $status,
                selectedProvider: chosenProvider,
                marvisSession: marvisSession,
                audioPlayerManager: audioPlayerManager
            )
            .frame(minWidth: 400)
            .inspector(isPresented: $isInspectorPresent) {
                // Right: Inspector Panel
                TTSInspectorView(
                    selectedProvider: $chosenProvider,
                    selectedVoice: $chosenVoice,
                    selectedQuality: $chosenQuality,
                    status: $status,
                    autoPlay: $autoPlay,
                    useStreaming: $useStreaming,
                    streamingInterval: $streamingInterval,
                    isGenerating: isCurrentlyGenerating,
                    canGenerate: canGenerate,
                    marvisSession: marvisSession,
                    onGenerate: handleGenerate,
                    onStop: handleStop
                )
                .inspectorColumnWidth(min: 250, ideal: 300, max: 400)
            }
        }
        .frame(minWidth: 1200, minHeight: 700)
        .navigationTitle("MLX Audio")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: { isInspectorPresent.toggle() }) {
                    Label("Toggle Inspector", systemImage: "sidebar.right")
                }
            }
        }
        .onChange(of: chosenProvider) { _, newProvider in
            status = newProvider.statusMessage
        }
        .onChange(of: kokoroTTSModel.lastGeneratedAudioURL) { _, newURL in
            if let url = newURL {
                audioPlayerManager.loadAudio(from: url)
            }
        }
        .onChange(of: orpheusTTSModel?.lastGeneratedAudioURL) { _, newURL in
            if let url = newURL {
                audioPlayerManager.loadAudio(from: url)
            }
        }
        .onChange(of: marvisLastAudioURL) { _, newURL in
            if let url = newURL {
                audioPlayerManager.loadAudio(from: url)
            }
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
            do {
                try marvisSession?.cleanupMemory()
            } catch {
                print("Failed to cleanup Marvis memory: \(error)")
            }
            isMarvisLoading = false
        }
        status = "Generation stopped"
    }

    // MARK: - TTS Generation Methods

    private func generateWithKokoro() {
        if chosenProvider.validateVoice(chosenVoice),
           let kokoroVoice = TTSVoice.fromIdentifier(chosenVoice) ?? TTSVoice(rawValue: chosenVoice) {
            kokoroTTSModel.say(text, kokoroVoice, autoPlay: autoPlay)
            status = "Done"
        } else {
            status = chosenProvider.errorMessage
        }
    }

    private func generateWithOrpheus() async {
        if orpheusTTSModel == nil {
            orpheusTTSModel = OrpheusTTSModel()
        }

        guard let orpheusTTSModel = orpheusTTSModel else {
            status = "Failed to initialize Orpheus model"
            return
        }

        if chosenProvider.validateVoice(chosenVoice),
           let orpheusVoice = OrpheusVoice(rawValue: chosenVoice) {
            await orpheusTTSModel.say(text, orpheusVoice, autoPlay: autoPlay)
            status = "Done"
        } else {
            status = chosenProvider.errorMessage
        }
    }

    private func generateWithMarvis() async {
        guard await initializeMarvisSessionIfNeeded() else {
            return
        }

        do {
            isMarvisLoading = true
            defer { isMarvisLoading = false }

            if useStreaming {
                try await generateWithMarvisStreaming()
            } else {
                try await generateWithMarvisNonStreaming()
            }
        } catch {
            status = "Marvis generation failed: \(error.localizedDescription)"
            isMarvisLoading = false
        }
    }

    private func initializeMarvisSessionIfNeeded() async -> Bool {
        guard marvisSession == nil else { return true }

        do {
            isMarvisLoading = true
            status = "Loading Marvis..."
            defer { isMarvisLoading = false }

            guard let voice = MarvisSession.Voice(rawValue: chosenVoice) else {
                status = "\(chosenProvider.errorMessage)\(chosenVoice)"
                return false
            }

            marvisSession = try await MarvisSession(voice: voice, progressHandler: { progress in
                status = "Loading Marvis: \(Int(progress.fractionCompleted * 100))%"
            }, playbackEnabled: autoPlay)
            status = "Marvis loaded successfully!"
            return true
        } catch {
            status = "Failed to load Marvis: \(error.localizedDescription)"
            return false
        }
    }

    private func generateWithMarvisStreaming() async throws {
        status = "Streaming with Marvis..."

        guard let voice = MarvisSession.Voice(rawValue: chosenVoice) else {
            status = "Invalid voice selection"
            return
        }

        guard let marvisSession = marvisSession else {
            status = "Marvis session not initialized"
            return
        }

        let stream = marvisSession.stream(text: text, voice: voice, qualityLevel: chosenQuality, streamingInterval: streamingInterval)
        var totalSamples = 0
        var firstChunk = true

        for try await chunk in stream {
            if firstChunk {
                status = "Streaming: First chunk received (\(chunk.sampleCount) samples)"
                firstChunk = false
            }
            totalSamples += chunk.sampleCount
            status = "Streaming: \(totalSamples) samples, RTF: \(String(format: "%.2f", chunk.realTimeFactor))"
        }

        status = "Marvis streaming complete! Total: \(totalSamples) samples"
    }

    private func generateWithMarvisNonStreaming() async throws {
        guard let marvisSession = marvisSession else {
            status = "Marvis session not initialized"
            return
        }

        status = "Generating with Marvis..."
        let result = autoPlay
            ? try await marvisSession.generate(for: text, quality: chosenQuality)
            : try await marvisSession.generateRaw(for: text, quality: chosenQuality)

        // Save Marvis audio to file
        saveMarvisAudio(result: result)

        status = "Marvis generation complete! Audio: \(result.audio.count) samples @ \(result.sampleRate)Hz"
    }

    private func saveMarvisAudio(result: MarvisSession.GenerationResult) {
        // Create audio buffer from result
        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(result.sampleRate), channels: 1) else {
            print("Failed to create audio format for Marvis audio")
            return
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(result.sampleCount)) else {
            print("Failed to create buffer for Marvis audio")
            return
        }

        buffer.frameLength = buffer.frameCapacity
        guard let channels = buffer.floatChannelData else {
            print("Failed to get channel data for Marvis audio")
            return
        }
        for i in 0..<result.audio.count {
            channels[0][i] = result.audio[i]
        }

        // Save to file
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let audioFileURL = documentsPath.appendingPathComponent("marvis_output.wav")

        do {
            let audioFile = try AVAudioFile(forWriting: audioFileURL,
                                          settings: format.settings,
                                          commonFormat: .pcmFormatFloat32,
                                          interleaved: false)
            try audioFile.write(from: buffer)
            print("Marvis audio saved to: \(audioFileURL.path)")
            marvisLastAudioURL = audioFileURL
        } catch {
            print("Failed to save Marvis audio: \(error)")
        }
    }
}

#Preview {
    ContentView()
}

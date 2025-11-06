//
//  OrpheusTTSModel.swift
//  MLXAudio
//
//  Created by Ben Harraway on 13/04/2025.
//

import AVFoundation
import MLX
import SwiftUI

class OrpheusTTSModel: ObservableObject {
    let orpheusTTSEngine: OrpheusTTS?
    let audioEngine: AVAudioEngine!
    let playerNode: AVAudioPlayerNode!

    // Published audio file URL
    @Published var lastGeneratedAudioURL: URL?
    
    init() {
        do {
            try orpheusTTSEngine = OrpheusTTS()
        } catch {
            orpheusTTSEngine = nil
            print("Orpheus error.  Could not init. \(error.localizedDescription)")
        }
        
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        audioEngine.attach(playerNode)
    }
       
    func say(_ text: String, _ voice: OrpheusVoice, autoPlay: Bool = true) async {
        if let orpheusTTSEngine = orpheusTTSEngine {
            let mainTimer = BenchmarkTimer.shared.create(id: "TTSGeneration")

            let audioBuffer: MLXArray
            do {
                audioBuffer = try await orpheusTTSEngine.generateAudio(voice: voice, text: text)
            } catch {
                print("Error generating audio: \(error)")
                BenchmarkTimer.shared.stop(id: "TTSGeneration")
                return
            }

            BenchmarkTimer.shared.stop(id: "TTSGeneration")
            BenchmarkTimer.shared.printLog(id: "TTSGeneration")

            BenchmarkTimer.shared.reset()

            let audio = audioBuffer[0].asArray(Float.self)

            let sampleRate = 24000.0
            let audioLength = Double(audio.count) / sampleRate
            print("Audio length: " + String(format: "%.4f", audioLength))

            print("\(mainTimer!.deltaTime)")
            print("Speed: " + String(format: "%.2f", audioLength / mainTimer!.deltaTime))

            guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1) else {
              print("Failed to create audio format")
              return
            }

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)) else {
              print("Couldn't create buffer")
              return
            }

            buffer.frameLength = buffer.frameCapacity
            guard let channels = buffer.floatChannelData else {
                print("Failed to get channel data")
                return
            }
            for i in 0 ..< audio.count {
              channels[0][i] = audio[i]
            }

            // Save audio file
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioFileURL = documentsPath.appendingPathComponent("output.wav")

            do {
                try buffer.saveToWavFile(at: audioFileURL)
                print("Audio saved to: \(audioFileURL.path)")
                lastGeneratedAudioURL = audioFileURL
            } catch {
                print("Failed to save audio: \(error)")
            }

            // Only play if autoPlay is enabled
            if autoPlay {
                audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
                do {
                  try audioEngine.start()
                } catch {
                  print("Audio engine failed to start: \(error.localizedDescription)")
                  return
                }

                await playerNode.scheduleBuffer(buffer, at: nil, options: .interrupts)
                playerNode.play()
            }
        }
    }

    func saveAudioFile(to destinationUrl: URL) async {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let sourceUrl = documentsPath.appendingPathComponent("output.wav")
        
        do {
            if FileManager.default.fileExists(atPath: destinationUrl.path) {
                try FileManager.default.removeItem(at: destinationUrl)
            }
            try FileManager.default.copyItem(at: sourceUrl, to: destinationUrl)
        } catch {
            print("Failed to save audio: \(error)")
        }
    }
}

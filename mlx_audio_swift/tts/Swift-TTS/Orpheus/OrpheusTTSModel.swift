//
//  OrpheusTTSModel.swift
//  Swift-TTS
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
       
    func say(_ text: String, _ voice: OrpheusVoice) async {
        if let orpheusTTSEngine = orpheusTTSEngine {
            let mainTimer = BenchmarkTimer.shared.create(id: "TTSGeneration")
            let audioBuffer = try! await orpheusTTSEngine.generateAudio(voice: voice, text: text)
            BenchmarkTimer.shared.stop(id: "TTSGeneration")
            BenchmarkTimer.shared.printLog(id: "TTSGeneration")

            BenchmarkTimer.shared.reset()

            let audio = audioBuffer[0].asArray(Float.self)

            let sampleRate = 24000.0
            let audioLength = Double(audio.count) / sampleRate
            print("Audio length: " + String(format: "%.4f", audioLength))

            print("\(mainTimer!.deltaTime)")
            print("Speed: " + String(format: "%.2f", audioLength / mainTimer!.deltaTime))

            let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)) else {
              print("Couldn't create buffer")
              return
            }

            buffer.frameLength = buffer.frameCapacity
            let channels = buffer.floatChannelData!
            for i in 0 ..< audio.count {
              channels[0][i] = audio[i]
            }

            // Save audio file
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioFileURL = documentsPath.appendingPathComponent("output.wav")
            
            do {
                try buffer.saveToWavFile(at: audioFileURL)
                print("Audio saved to: \(audioFileURL.path)")
            } catch {
                print("Failed to save audio: \(error)")
            }

            audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
            do {
              try audioEngine.start()
            } catch {
              print("Audio engine failed to start: \(error.localizedDescription)")
              return
            }

            playerNode.scheduleBuffer(buffer, at: nil, options: .interrupts, completionHandler: nil)
            playerNode.play()
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

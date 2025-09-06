//
//  Swift_TTS_iOSApp.swift
//  Swift-TTS-iOS
//
//  Created by Sachin Desai on 5/20/25.
//

import SwiftUI
import Swift_TTS

@main
struct Swift_TTS_iOSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView(kokoroViewModel: KokoroTTSModel())
                .onAppear {
                    AudioSessionManager.shared.setupAudioSession()
                }
        }
    }
}

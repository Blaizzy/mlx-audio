//
//  MLXAudio_iOSApp.swift
//  MLXAudio-iOS
//
//  Created by Sachin Desai on 5/20/25.
//

import SwiftUI

@main
struct MLXAudio_iOSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView(kokoroViewModel: KokoroTTSModel())
                .onAppear {
                    AudioSessionManager.shared.setupAudioSession()
                }
        }
    }
}

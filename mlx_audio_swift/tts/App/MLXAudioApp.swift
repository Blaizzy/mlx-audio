//
//  MLXAudioApp.swift
//  MLX Audio Demo
//
//  Main app entry point.
//

import SwiftUI

@main
struct MLXAudioApp: App {
    init() {
        #if os(iOS)
        AudioSessionManager.configure()
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .windowStyle(.automatic)
        .defaultSize(width: 900, height: 600)
        #endif
    }
}

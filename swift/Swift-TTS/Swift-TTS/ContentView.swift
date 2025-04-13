//
//  ContentView.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 13/04/2025.
//

import SwiftUI

struct ContentView: View {
    
    private let kokoroTTSModel = KokoroTTSModel()
    @State private var sayThis : String = "Hello Everybody"
    @State private var status : String = ""
    
    var body: some View {
        VStack {
            Image(systemName: "mouth")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("TTS Examples")
                .font(.headline)
                .padding()
            
            TextField("Enter text", text: $sayThis)
            
            Button("Kokoro") {
                Task {
                    status = "Generating..."
                    await kokoroTTSModel.say(sayThis, .bmGeorge)
                    status = "Done"
                }
            }
            
            Button("Orpheus") {
                
            }
            
            Text(status)
                .font(.caption)
                .padding()
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

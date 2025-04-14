//
//  ContentView.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 13/04/2025.
//

import SwiftUI

struct ContentView: View {
    
    @State private var kokoroTTSModel: KokoroTTSModel? = nil
    @State private var orpheusTTSModel: OrpheusTTSModel? = nil
    
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
                    if kokoroTTSModel == nil {
                        kokoroTTSModel = KokoroTTSModel()
                    }
                    await kokoroTTSModel!.say(sayThis, .bmGeorge)
                    status = "Done"
                }
            }
            
            Button("Orpheus") {
                Task {
                    status = "Generating..."
                    if orpheusTTSModel == nil {
                        orpheusTTSModel = OrpheusTTSModel()
                    }
                    await orpheusTTSModel!.say(sayThis, .tara)
                    status = "Done"
                }
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

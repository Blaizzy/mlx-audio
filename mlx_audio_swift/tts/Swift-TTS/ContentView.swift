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
    
    private var availableProviders = ["kokoro", "orpheus"]
    @State private var chosenProvider : String = "kokoro"

    var body: some View {
        VStack {
            Image(systemName: "mouth")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("TTS Examples")
                .font(.headline)
                .padding()
                        
            Picker("Choose a provider", selection: $chosenProvider) {
                ForEach(availableProviders, id: \.self) { provider in
                    Text(provider.capitalized)
                }
            }
            .onChange(of: chosenProvider) { newProvider in
                if newProvider == "orpheus" {
                    status = "Orpheus is currently quite slow (0.05x on M1).  Working on it!\n\nBut it does support expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
                } else {
                    status = ""
                }
            }
            .padding()

            TextField("Enter text", text: $sayThis).padding()
            
            Button(action: {
                Task {
                    status = "Generating..."
                    if chosenProvider == "kokoro" {
                        if kokoroTTSModel == nil {
                            kokoroTTSModel = KokoroTTSModel()
                        }
                        await kokoroTTSModel!.say(sayThis, .bmGeorge)
                        
                    } else if chosenProvider == "orpheus" {
                        if orpheusTTSModel == nil {
                            orpheusTTSModel = OrpheusTTSModel()
                        }
                        await orpheusTTSModel!.say(sayThis, .tara)
                    }
                    
                    status = "Done"
                }
            }, label: {
                Text("Generate")
                    .font(.title2)
                    .padding()
            })
            .buttonStyle(.borderedProminent)

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

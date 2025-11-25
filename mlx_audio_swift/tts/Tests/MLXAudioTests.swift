//
//  MLXAudioTests.swift
//  MLXAudioTests
//
//  Created by Ben Harraway on 14/04/2025.
//

import Testing

struct MLXAudioTests {

    @Test func example() async throws {
        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
    }

    @Test @MainActor func testKokoroEngineInitializes() async {
        let engine = KokoroEngine()
        #expect(engine.isLoaded == false)
        #expect(engine.isGenerating == false)
        #expect(engine.availableVoices.count > 0)
    }

    @Test @MainActor func testOrpheusEngineInitializes() async {
        let engine = OrpheusEngine()
        #expect(engine.isLoaded == false)
        #expect(engine.isGenerating == false)
        #expect(engine.availableVoices.count > 0)
    }

    @Test @MainActor func testMarvisEngineInitializes() async {
        let engine = MarvisEngine()
        #expect(engine.isLoaded == false)
        #expect(engine.isGenerating == false)
        #expect(engine.availableVoices.count > 0)
    }

    @Test func testTTSProviderHasVoices() async {
        for provider in TTSProvider.allCases {
            #expect(provider.availableVoices.count > 0)
            #expect(provider.defaultVoiceID.isEmpty == false)
        }
    }
}

//
//  MLXAudioTests.swift
//  MLXAudioTests
//
//  Created by Ben Harraway on 14/04/2025.
//

import Testing

@testable import MLXAudio
@testable import MLXESpeakNG

struct MLXAudioTests {

    func example() async throws {
        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
    }
    
    func testViewBodyDoesNotCrash() {
        _ = ContentView().body
    }
    
    func testKokoro() async {
        let kokoroTTSModel = KokoroTTSModel()
        kokoroTTSModel.say("test", .bmGeorge)
    }

    func testOrpheus() async {
        let orpheusTTSModel = OrpheusTTSModel()
        await orpheusTTSModel.say("test", .tara)
    }
}

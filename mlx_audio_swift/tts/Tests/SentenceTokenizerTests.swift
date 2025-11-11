//
//  SentenceTokenizerTests.swift
//   MLXAudio
//

import XCTest
import Foundation
import NaturalLanguage

@testable import MLXAudio

final class SentenceTokenizerTests: XCTestCase {

    // MARK: - English Tests

    func testEnglishBasicSentences() {
        let text = "Hello world. This is a test. How are you?"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Should produce at least one chunk")
        XCTAssertEqual(result.count, 1, "Short text should merge into one chunk")
        XCTAssertEqual(result[0], "Hello world. This is a test. How are you?")

        // Validate no empty chunks
        XCTAssertFalse(result.contains(""), "Should not contain empty chunks")
    }

    func testEnglishWithAbbreviations() {
        let text = "Dr. Smith went to the U.S.A. yesterday. He met Prof. Johnson at 3:30 p.m. They discussed the project."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Should produce at least one chunk")
        XCTAssertLessThanOrEqual(result.count, 2, "Should merge into 1-2 chunks for optimal TTS")

        // Verify abbreviations are kept together (not split incorrectly)
        let fullText = result.joined(separator: " ")
        XCTAssertTrue(fullText.contains("Dr. Smith"), "Dr. abbreviation should be preserved")
        XCTAssertTrue(fullText.contains("U.S.A."), "Country abbreviation should be preserved")
        XCTAssertTrue(fullText.contains("Prof. Johnson"), "Professor abbreviation should be preserved")
        XCTAssertTrue(fullText.contains("p.m."), "Time abbreviation should be preserved")

        // Validate chunk constraints
        for chunk in result {
            XCTAssertLessThanOrEqual(chunk.count, 300, "Latin chunks should be ≤300 chars")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "No empty chunks")
        }
    }

    func testEnglishLongText() {
        let text = "This is a very long sentence that should test the chunking mechanism. " +
                   "It contains multiple clauses and should be split appropriately for TTS processing. " +
                   "The tokenizer should optimize chunk sizes for better speech synthesis. " +
                   "Each chunk should be within the optimal length range for natural speech flow."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Long text should produce chunks")
        XCTAssertGreaterThanOrEqual(result.count, 1, "Should produce at least one chunk")

        // Test chunking behavior
        let totalLength = text.count
        if totalLength > 300 {
            XCTAssertGreaterThan(result.count, 1, "Long text (>\(totalLength) chars) should split into multiple chunks")
        }

        // Validate each chunk
        for (index, chunk) in result.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 300, "Chunk \(index) should be ≤300 chars (was \(chunk.count))")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk \(index) should not be empty")
        }

        // Verify text preservation (total content should match)
        let reconstructed = result.joined(separator: " ")
        XCTAssertEqual(reconstructed.count, text.count, "Total character count should be preserved")
    }

    func testEnglishQuestionAndExclamation() {
        let text = "What's your name? I'm excited! This is amazing."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Should produce at least one chunk")
        XCTAssertEqual(result.count, 1, "Short text with strong punctuation should merge into one chunk")

        // Verify punctuation preservation
        let chunk = result[0]
        XCTAssertTrue(chunk.contains("What's your name?"), "Question should be preserved")
        XCTAssertTrue(chunk.contains("I'm excited!"), "Exclamation should be preserved")
        XCTAssertTrue(chunk.contains("This is amazing."), "Statement should be preserved")

        // Verify strong sentence endings are detected
        XCTAssertTrue(chunk.contains("?"), "Question mark should be preserved")
        XCTAssertTrue(chunk.contains("!"), "Exclamation mark should be preserved")
    }

    // MARK: - French Tests

    func testFrenchBasicSentences() {
        let text = "Bonjour le monde. Comment allez-vous? J'espère que vous allez bien."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "French text should produce chunks")
        XCTAssertEqual(result.count, 1, "Short French text should merge into one chunk")

        let chunk = result[0]
        XCTAssertTrue(chunk.contains("Bonjour le monde"), "French greeting should be preserved")
        XCTAssertTrue(chunk.contains("Comment allez-vous?"), "French question should be preserved")
        XCTAssertTrue(chunk.contains("J'espère"), "French apostrophe should be preserved")

        // Validate Latin script chunking rules
        XCTAssertLessThanOrEqual(chunk.count, 300, "French should follow Latin chunking rules (≤300 chars)")
    }

    func testFrenchWithAccents() {
        let text = "C'est très intéressant. Les élèves étudient français. Où êtes-vous né?"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "French text with accents should produce chunks")
        XCTAssertEqual(result.count, 2, "French text with strong endings should split appropriately")

        // Verify content preservation across chunks
        let fullText = result.joined(separator: " ")
        XCTAssertTrue(fullText.contains("très intéressant"), "French accents should be preserved")
        XCTAssertTrue(fullText.contains("élèves étudient"), "Multiple accents should be preserved")
        XCTAssertTrue(fullText.contains("Où êtes-vous"), "French circumflex should be preserved")

        // Validate each chunk
        for (index, chunk) in result.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 300, "Chunk \(index) should follow Latin rules (≤300 chars)")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk \(index) should not be empty")
        }
    }

    // MARK: - Spanish Tests

    func testSpanishBasicSentences() {
        let text = "Hola mundo. ¿Cómo estás? ¡Esto es increíble!"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Spanish text should produce chunks")
        XCTAssertEqual(result.count, 1, "Short Spanish text should merge into one chunk")

        let chunk = result[0]
        XCTAssertTrue(chunk.contains("Hola mundo"), "Spanish greeting should be preserved")
        XCTAssertTrue(chunk.contains("¿Cómo estás?"), "Spanish inverted question should be preserved")
        XCTAssertTrue(chunk.contains("¡Esto es increíble!"), "Spanish inverted exclamation should be preserved")

        // Validate Spanish punctuation handling
        XCTAssertTrue(chunk.contains("¿") && chunk.contains("?"), "Both question marks should be preserved")
        XCTAssertTrue(chunk.contains("¡") && chunk.contains("!"), "Both exclamation marks should be preserved")
    }

    func testSpanishInvertedPunctuation() {
        let text = "¿Hablas español? ¡Qué maravilloso! Me gusta mucho este idioma."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Spanish text should produce chunks")
        XCTAssertEqual(result.count, 1, "Short Spanish text should merge into one chunk")

        let chunk = result[0]
        XCTAssertTrue(chunk.contains("¿Hablas español?"), "Inverted question should be preserved")
        XCTAssertTrue(chunk.contains("¡Qué maravilloso!"), "Inverted exclamation should be preserved")

        // Test inverted punctuation detection
        XCTAssertTrue(chunk.hasPrefix("¿"), "Should start with inverted question mark")
        XCTAssertTrue(chunk.contains("¡Qué"), "Should contain inverted exclamation")
    }

    // MARK: - Italian Tests

    func testItalianBasicSentences() {
        let text = "Ciao mondo. Come stai? Spero che tu stia bene."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Italian text should produce chunks")

        // Verify complete text preservation
        let fullText = result.joined(separator: " ")
        XCTAssertTrue(fullText.contains("Ciao mondo"), "Italian greeting should be preserved")
        XCTAssertTrue(fullText.contains("Come stai?"), "Complete Italian question should be preserved")
        XCTAssertTrue(fullText.contains("Spero che tu stia bene"), "Complete Italian statement should be preserved")

        // Validate Latin script chunking rules
        for (index, chunk) in result.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 300, "Chunk \(index) should follow Latin rules (≤300 chars)")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk \(index) should not be empty")
        }
    }

    func testItalianWithApostrophes() {
        let text = "L'Italia è bella. Non c'è problema. Quest'anno andrò in vacanza."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Italian text with apostrophes should produce chunks")

        // Verify complete text preservation including apostrophes
        let fullText = result.joined(separator: " ")
        XCTAssertTrue(fullText.contains("L'Italia è bella"), "Italian apostrophe construction should be preserved")
        XCTAssertTrue(fullText.contains("Non c'è problema"), "Complete sentence with apostrophe should be preserved")
        XCTAssertTrue(fullText.contains("Quest'anno andrò in vacanza"), "Complete sentence with apostrophe should be preserved")

        // Validate apostrophe preservation
        XCTAssertTrue(fullText.contains("L'"), "L' apostrophe should be preserved")
        XCTAssertTrue(fullText.contains("c'è"), "c' apostrophe should be preserved")
        XCTAssertTrue(fullText.contains("Quest'"), "Quest' apostrophe should be preserved")
    }

    // MARK: - Portuguese Tests

    func testPortugueseBasicSentences() {
        let text = "Olá mundo. Como você está? Eu espero que você esteja bem."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Portuguese text should produce chunks")

        // Verify complete text preservation
        let fullText = result.joined(separator: " ")
        XCTAssertTrue(fullText.contains("Olá mundo"), "Portuguese greeting should be preserved")
        XCTAssertTrue(fullText.contains("Como você está?"), "Complete Portuguese question should be preserved")
        XCTAssertTrue(fullText.contains("Eu espero que você esteja bem"), "Complete Portuguese statement should be preserved")

        // Validate Latin script chunking rules
        for (index, chunk) in result.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 300, "Chunk \(index) should follow Latin rules (≤300 chars)")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk \(index) should not be empty")
        }
    }

    func testPortugueseWithTildes() {
        let text = "Esta é uma lição. Não tenho irmãos. A educação é importante."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Portuguese text with tildes should produce chunks")

        // Verify complete text preservation including tildes
        let fullText = result.joined(separator: " ")
        XCTAssertTrue(fullText.contains("Esta é uma lição"), "Complete sentence with tilde should be preserved")
        XCTAssertTrue(fullText.contains("Não tenho irmãos"), "Complete sentence with tilde should be preserved")
        XCTAssertTrue(fullText.contains("A educação é importante"), "Complete sentence with tilde should be preserved")

        // Validate tilde preservation
        XCTAssertTrue(fullText.contains("lição"), "Portuguese tilde (ã) should be preserved")
        XCTAssertTrue(fullText.contains("Não"), "Portuguese tilde (ã) should be preserved")
        XCTAssertTrue(fullText.contains("irmãos"), "Portuguese tilde (ã) should be preserved")
        XCTAssertTrue(fullText.contains("educação"), "Portuguese tilde (ã) should be preserved")
    }

    // MARK: - Japanese Tests

    func testJapaneseBasicSentences() {
        let text = "こんにちは世界。元気ですか？今日はいい天気ですね。"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Japanese text should produce chunks")
        XCTAssertEqual(result.count, 1, "Short Japanese text should merge into one chunk")

        let chunk = result[0]
        XCTAssertTrue(chunk.contains("こんにちは世界"), "Japanese greeting should be preserved")
        XCTAssertTrue(chunk.contains("元気ですか"), "Japanese question should be preserved")
        XCTAssertTrue(chunk.contains("いい天気"), "Japanese phrase should be preserved")

        // Validate CJK punctuation
        XCTAssertTrue(chunk.contains("。"), "Japanese period should be preserved")
        XCTAssertTrue(chunk.contains("？"), "Japanese question mark should be preserved")

        // Validate CJK chunking rules
        XCTAssertLessThanOrEqual(chunk.count, 200, "Japanese should follow CJK chunking rules (≤200 chars)")
    }

    func testJapaneseMixedScripts() {
        let text = "私の名前は田中です。Appleが好きです。今日は2024年です。"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Japanese mixed script text should produce chunks")
        XCTAssertEqual(result.count, 1, "Short mixed script text should merge into one chunk")

        let chunk = result[0]
        XCTAssertTrue(chunk.contains("田中です"), "Kanji name should be preserved")
        XCTAssertTrue(chunk.contains("Appleが好き"), "Mixed script (Latin+Japanese) should be preserved")
        XCTAssertTrue(chunk.contains("2024年"), "Numbers with Japanese should be preserved")

        // Test mixed script handling
        XCTAssertTrue(chunk.contains("Apple"), "Latin script in Japanese should be preserved")
        XCTAssertTrue(chunk.contains("2024"), "Arabic numerals should be preserved")
    }

    func testJapaneseNoSpaces() {
        let text = "日本語は難しいです。でも面白いと思います。頑張って勉強します。"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Japanese text without spaces should produce chunks")
        XCTAssertEqual(result.count, 1, "Short Japanese text should merge into one chunk")

        let chunk = result[0]
        XCTAssertTrue(chunk.contains("難しいです"), "Japanese adjective should be preserved")
        XCTAssertTrue(chunk.contains("面白い"), "Japanese expression should be preserved")
        XCTAssertTrue(chunk.contains("勉強します"), "Japanese verb should be preserved")

        // Verify no spaces added incorrectly
        XCTAssertFalse(chunk.contains(" でも"), "Should not add spaces in Japanese text")
        XCTAssertFalse(chunk.contains(" 頑張って"), "Should not add spaces in Japanese text")
    }

    // MARK: - Chinese Tests

    func testChineseSimplifiedBasicSentences() {
        let text = "你好世界。你好吗？今天天气很好。"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Chinese simplified text should produce chunks")

        // Verify complete text preservation
        let fullText = result.joined(separator: "")  // No separator for CJK
        XCTAssertTrue(fullText.contains("你好世界"), "Chinese greeting should be preserved")
        XCTAssertTrue(fullText.contains("你好吗"), "Complete Chinese question should be preserved")
        XCTAssertTrue(fullText.contains("今天天气很好"), "Complete Chinese statement should be preserved")

        // Validate CJK punctuation
        XCTAssertTrue(fullText.contains("。"), "Chinese period should be preserved")
        XCTAssertTrue(fullText.contains("？"), "Chinese question mark should be preserved")

        // Validate CJK chunking rules
        for (index, chunk) in result.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 200, "Chunk \(index) should follow CJK rules (≤200 chars)")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk \(index) should not be empty")
        }
    }

    func testChineseTraditionalBasicSentences() {
        let text = "你好世界。你好嗎？今天天氣很好。"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Chinese traditional text should produce chunks")

        // Verify complete text preservation
        let fullText = result.joined(separator: "")  // No separator for CJK
        XCTAssertTrue(fullText.contains("你好世界"), "Chinese greeting should be preserved")
        XCTAssertTrue(fullText.contains("你好嗎"), "Complete Chinese traditional question should be preserved")
        XCTAssertTrue(fullText.contains("今天天氣很好"), "Complete Chinese traditional statement should be preserved")

        // Validate CJK punctuation and traditional characters
        XCTAssertTrue(fullText.contains("。"), "Chinese period should be preserved")
        XCTAssertTrue(fullText.contains("？"), "Chinese question mark should be preserved")
        XCTAssertTrue(fullText.contains("嗎"), "Traditional character 嗎 should be preserved")
        XCTAssertTrue(fullText.contains("氣"), "Traditional character 氣 should be preserved")

        // Validate CJK chunking rules
        for (index, chunk) in result.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 200, "Chunk \(index) should follow CJK rules (≤200 chars)")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk \(index) should not be empty")
        }
    }

    func testChineseLongText() {
        let text = "中国是一个历史悠久的国家。它有着丰富的文化传统和美丽的自然风景。人们非常友善和热情。我很喜欢中国菜和中国文化。"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty)
        // Should split appropriately for CJK chunking (max 200 chars)
        for chunk in result {
            XCTAssertLessThanOrEqual(chunk.count, 200)
        }
    }

    // MARK: - Hindi Tests

    func testHindiBasicSentences() {
        let text = "नमस्ते दुनिया। आप कैसे हैं? मुझे हिंदी बहुत पसंद है।"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty, "Hindi text should produce chunks")

        // Verify complete text preservation
        let fullText = result.joined(separator: " ")
        XCTAssertTrue(fullText.contains("नमस्ते दुनिया"), "Hindi greeting should be preserved")
        XCTAssertTrue(fullText.contains("आप कैसे हैं"), "Complete Hindi question should be preserved")
        XCTAssertTrue(fullText.contains("मुझे हिंदी बहुत पसंद है"), "Complete Hindi statement should be preserved")

        // Validate Devanagari punctuation
        XCTAssertTrue(fullText.contains("।"), "Devanagari danda should be preserved")
        XCTAssertTrue(fullText.contains("?"), "Question mark should be preserved")

        // Validate Indic chunking rules
        for (index, chunk) in result.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 250, "Chunk \(index) should follow Indic rules (≤250 chars)")
            XCTAssertGreaterThan(chunk.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk \(index) should not be empty")
        }
    }

    func testHindiDevanagariPunctuation() {
        let text = "यह एक परीक्षा है। क्या आप हिंदी बोलते हैं॥ भारत एक सुंदर देश है।"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(result.joined().contains("परीक्षा है"))
        XCTAssertTrue(result.joined().contains("बोलते हैं"))
        XCTAssertTrue(result.joined().contains("सुंदर देश"))
    }

    func testHindiMixedPunctuation() {
        let text = "मेरा नाम राज है. आप का नाम क्या है? हिंदी सीखना आसान है।"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty)
        // Should handle both Western (.) and Devanagari (।) punctuation
        XCTAssertTrue(result.joined().contains("राज है"))
        XCTAssertTrue(result.joined().contains("क्या है"))
        XCTAssertTrue(result.joined().contains("आसान है"))
    }

    func testHindiLongText() {
        let text = "भारत एक विविधताओं से भरा देश है। यहाँ कई भाषाएँ बोली जाती हैं। हिंदी सबसे ज्यादा बोली जाने वाली भाषा है। भारतीय संस्कृति बहुत समृद्ध है।"
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty)
        // Should split appropriately for Indic chunking (max 250 chars)
        for chunk in result {
            XCTAssertLessThanOrEqual(chunk.count, 250)
        }
    }

    // MARK: - Edge Cases

    func testEmptyString() {
        let result = SentenceTokenizer.splitIntoSentences(text: "")
        XCTAssertTrue(result.isEmpty)
    }

    func testSingleCharacter() {
        let result = SentenceTokenizer.splitIntoSentences(text: "A")
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], "A")
    }

    func testOnlyPunctuation() {
        let result = SentenceTokenizer.splitIntoSentences(text: "!?.")
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], "!?.")
    }

    func testWhitespaceOnly() {
        let result = SentenceTokenizer.splitIntoSentences(text: "   \n\t  ")
        XCTAssertTrue(result.isEmpty)
    }

    func testMixedLanguages() {
        let text = "Hello world. こんにちは。Bonjour monde."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(result.joined().contains("Hello world"))
        XCTAssertTrue(result.joined().contains("こんにちは"))
        XCTAssertTrue(result.joined().contains("Bonjour monde"))
    }

    // MARK: - Chunking Optimization Tests

    func testLatinChunkingLimits() {
        let longSentence = String(repeating: "This is a test sentence. ", count: 20) // ~500 chars
        let result = SentenceTokenizer.splitIntoSentences(text: longSentence)

        // Should split into multiple chunks, each ≤300 chars
        for chunk in result {
            XCTAssertLessThanOrEqual(chunk.count, 300, "Latin chunks should be ≤300 characters")
        }
    }

    func testCJKChunkingLimits() {
        let longSentence = String(repeating: "これはテストです。", count: 30) // ~300 chars
        let result = SentenceTokenizer.splitIntoSentences(text: longSentence)

        // Should split into multiple chunks, each ≤200 chars
        for chunk in result {
            XCTAssertLessThanOrEqual(chunk.count, 200, "CJK chunks should be ≤200 characters")
        }
    }

    func testIndicChunkingLimits() {
        let longSentence = String(repeating: "यह एक परीक्षा है। ", count: 20) // ~400 chars
        let result = SentenceTokenizer.splitIntoSentences(text: longSentence)

        // Should split into multiple chunks, each ≤250 chars
        for chunk in result {
            XCTAssertLessThanOrEqual(chunk.count, 250, "Indic chunks should be ≤250 characters")
        }
    }

    func testShortSentenceMerging() {
        let text = "Hi. Bye. OK. Yes. No."
        let result = SentenceTokenizer.splitIntoSentences(text: text)

        // Short sentences should be merged for optimal TTS chunks
        XCTAssertLessThan(result.count, 5, "Short sentences should be merged")
        XCTAssertTrue(result.joined().contains("Hi"))
        XCTAssertTrue(result.joined().contains("Bye"))
        XCTAssertTrue(result.joined().contains("OK"))
    }

    // MARK: - Performance Tests

    func testPerformanceWithLargeText() {
        let largeText = String(repeating: "This is a performance test sentence. ", count: 1000)

        // Configure performance test
        let options = XCTMeasureOptions()
        options.iterationCount = 5

        measure(options: options) {
            _ = SentenceTokenizer.splitIntoSentences(text: largeText)
        }
    }

    func testPerformanceWithMultipleLanguages() {
        let mixedText = """
        Hello world. This is English.
        こんにちは世界。これは日本語です。
        你好世界。这是中文。
        Bonjour monde. C'est du français.
        Hola mundo. Esto es español.
        नमस्ते दुनिया। यह हिंदी है।
        """

        // Configure performance test
        let options = XCTMeasureOptions()
        options.iterationCount = 5

        measure(options: options) {
            _ = SentenceTokenizer.splitIntoSentences(text: mixedText)
        }
    }
}

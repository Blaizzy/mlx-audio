//
//  KokoroTokenizerTests.swift
//   MLXAudio
//

import XCTest
import Foundation

@testable import MLXAudio

final class KokoroTokenizerTests: XCTestCase {
    
    private var tokenizer: KokoroTokenizer!
    private var mockEngine: ESpeakNGEngine!
    
    override func setUp() {
        super.setUp()
        // Create a real ESpeakNGEngine for testing
        do {
            mockEngine = try ESpeakNGEngine()
            tokenizer = KokoroTokenizer(engine: mockEngine)
            try tokenizer.setLanguage(for: .afAlloy) // Set to English US for consistent testing
        } catch {
            XCTFail("Failed to set up test environment: \(error)")
        }
    }
    
    override func tearDown() {
        tokenizer = nil
        mockEngine = nil
        super.tearDown()
    }
    
    // MARK: - Basic Phonemization Tests
    
    func testBasicEnglishPhonemization() throws {
        let result = try tokenizer.phonemize("hello world")
        
        XCTAssertFalse(result.phonemes.isEmpty, "Should produce phonemes")
        XCTAssertGreaterThan(result.tokens.count, 0, "Should produce tokens")
        
        // Test that it contains recognizable phonetic content
        let phonemes = result.phonemes
        XCTAssertTrue(phonemes.contains("h") || phonemes.contains("ˈ") || phonemes.contains("l"), 
                     "Should contain phonetic characters for 'hello'")
    }
    
    func testEmptyStringPhonemization() throws {
        let result = try tokenizer.phonemize("")
        XCTAssertTrue(result.phonemes.isEmpty, "Empty string should produce empty phonemes")
        XCTAssertTrue(result.tokens.isEmpty, "Empty string should produce no tokens")
    }
    
    // MARK: - Direct Phoneme Specification Tests
    
    func testDirectPhonemeSpecification() throws {
        let result = try tokenizer.phonemize("[hello](/həˈloʊ/)")
        
        XCTAssertEqual(result.phonemes, "həˈloʊ", "Should use exact phonemes specified")
        XCTAssertEqual(result.tokens.count, 1, "Should produce 1 token")
        XCTAssertEqual(result.tokens[0].text, "hello", "Token text should be original word")
        XCTAssertEqual(result.tokens[0].phonemes, "həˈloʊ", "Token phonemes should be specified phonemes")
    }
    
    func testMultipleDirectPhonemeSpecifications() throws {
        let result = try tokenizer.phonemize("[hello](/həˈloʊ/) [world](/wɜɹld/)")
        
        XCTAssertEqual(result.tokens[0].phonemes, "həˈloʊ", "Should contain first specified phonemes")
        XCTAssertEqual(result.tokens[1].phonemes, "wɜɹld", "Should contain second specified phonemes")
        XCTAssertEqual(result.tokens.count, 2, "Should produce 2 tokens")
    }
    
    // MARK: - Alias/Replacement Text Tests
    
    func testAliasReplacement() throws {
        let result = try tokenizer.phonemize("[Dr.](Doctor)")
        
        XCTAssertFalse(result.phonemes.isEmpty, "Should phonemize replacement text")
        XCTAssertEqual(result.tokens.count, 1, "Should produce 1 token")
        XCTAssertEqual(result.tokens[0].text, "Doctor", "Token text should be replacement text")
        
        // Should contain phonetic representation of "Doctor", not "Dr."
        XCTAssertTrue(result.phonemes.contains("d") || result.phonemes.contains("ɑ") || result.phonemes.contains("k"), 
                     "Should contain phonetic elements of 'Doctor'")
    }
    
    func testAcronymExpansion() throws {
        let result = try tokenizer.phonemize("[NASA](N A S A)")
        
        XCTAssertFalse(result.phonemes.isEmpty, "Should phonemize expanded acronym")
        XCTAssertEqual(result.tokens.count, 1, "Should produce 1 token")
        XCTAssertEqual(result.tokens[0].text, "N A S A", "Token text should be replacement text")
        
        // Should phonemize the spelled-out version "N A S A"
        XCTAssertTrue(result.phonemes.contains("n") || result.phonemes.contains("ɛ") || result.phonemes.contains("s"), 
                     "Should contain phonetic elements of spelled out letters")
    }
    
    func testMultipleAliases() throws {
        let result = try tokenizer.phonemize("[Dr.](Doctor) [Mr.](Mister)")
        
        XCTAssertFalse(result.phonemes.isEmpty, "Should contain phonemized replacements")
        XCTAssertEqual(result.tokens.count, 2, "Should produce 2 tokens")
        
        // Verify token structure
        XCTAssertEqual(result.tokens[0].text, "Doctor", "First token should be replacement text")
        XCTAssertEqual(result.tokens[1].text, "Mister", "Second token should be replacement text")
    }
    
    // MARK: - Stress Control Tests
    
    func testPrimaryStressControl() throws {
        let result = try tokenizer.phonemize("[important](1.5)")
        
        XCTAssertTrue(result.phonemes.contains("ˈ"), "Should contain primary stress marker")
        XCTAssertFalse(result.phonemes.contains("1.5"), "Should not contain stress value as text")
        XCTAssertFalse(result.phonemes.contains("point"), "Should not convert stress value to words")
        XCTAssertEqual(result.tokens[0].text, "important", "Token text should be original word")
        XCTAssertEqual(result.tokens[0].stress, 1.5, "Token should store stress value")
    }
    
    func testSecondaryStressControl() throws {
        let result = try tokenizer.phonemize("[the](-1)")
        
        XCTAssertTrue(result.phonemes.contains("ˌ"), "Should contain secondary stress marker")
        XCTAssertFalse(result.phonemes.contains("-1"), "Should not contain stress value as text")
        XCTAssertEqual(result.tokens[0].text, "the", "Token text should be original word")
        XCTAssertEqual(result.tokens[0].stress, -1.0, "Token should store stress value")
    }
    
    func testStressRemoval() throws {
        let result = try tokenizer.phonemize("[world](-2)")
        
        // Should not contain stress markers after removal
        let hasStress = result.phonemes.contains("ˈ") || result.phonemes.contains("ˌ")
        XCTAssertFalse(hasStress, "Should not contain stress markers after removal")
        XCTAssertFalse(result.phonemes.contains("-2"), "Should not contain stress value as text")
        XCTAssertEqual(result.tokens[0].text, "world", "Token text should be original word")
        XCTAssertEqual(result.tokens[0].stress, -2.0, "Token should store stress value")
    }
    
    func testFloatingPointStress() throws {
        let result = try tokenizer.phonemize("[hello](0.5)")
        
        XCTAssertTrue(result.phonemes.contains("ˌ"), "Fractional positive should add secondary stress")
        XCTAssertFalse(result.phonemes.contains("0.5"), "Should not contain stress value as text")
        XCTAssertFalse(result.phonemes.contains("point"), "Should not convert stress value to words")
        XCTAssertEqual(result.tokens[0].text, "hello", "Token text should be original word")
        XCTAssertEqual(result.tokens[0].stress, 0.5, "Token should store exact stress value")
    }
    
    // MARK: - Currency/Time/Decimal Processing Tests
    
    func testCurrencyProcessing() throws {
        let result = try tokenizer.phonemize("$5.50")
        
        XCTAssertEqual(result.tokens[0].text, "5", "Should contain 5")
        XCTAssertEqual(result.tokens[1].text, "dollars", "Should convert currency symbol")
        XCTAssertEqual(result.tokens[2].text, "and", "")
        XCTAssertEqual(result.tokens[3].text, "50", "Should contain cents")
        XCTAssertEqual(result.tokens[4].text, "cents", "Should include cents unit")
    }
    
    func testTimeProcessing() throws {
        let result = try tokenizer.phonemize("5:30")
        
        XCTAssertEqual(result.tokens[0].text, "5", "Should contain hour")
        XCTAssertEqual(result.tokens[1].text, "30", "Should contain minutes")
    }
    
    func testDecimalProcessing() throws {
        let result = try tokenizer.phonemize("3.14")
        
        XCTAssertEqual(result.tokens[0].text, "3", "Should contain integer part")
        XCTAssertEqual(result.tokens[1].text, "point", "Should include decimal point word")
        XCTAssertEqual(result.tokens[2].text, "1", "Should contain first decimal digit")
        XCTAssertEqual(result.tokens[3].text, "4", "Should contain second decimal digit")
    }
    
    func testDecimalInStressControlNotProcessed() throws {
        let result = try tokenizer.phonemize("[important](1.5)")
        
        // This is the key test - ensures our fix works
        XCTAssertFalse(result.phonemes.contains("point"), "Decimal in stress control should not be processed")
        XCTAssertFalse(result.phonemes.contains("1 point 5"), "Should not convert stress value to words")
        XCTAssertTrue(result.phonemes.contains("ˈ"), "Should apply stress instead")
        XCTAssertEqual(result.tokens[0].stress, 1.5, "Should store stress value correctly")
    }
    
    func testTimeInReplacementText() throws {
        let result = try tokenizer.phonemize("[5:30](half past five)")
        
        XCTAssertFalse(result.phonemes.contains("5 thirty"), "Should not auto-process time in replacement")
        XCTAssertEqual(result.tokens[0].text, "half past five", "Should preserve original text")
        
        // Should phonemize "half past five" instead of processing "5:30"
        let phonemes = result.phonemes.lowercased()
        XCTAssertTrue(phonemes.contains("h") || phonemes.contains("f"), "Should contain phonetic elements of 'half'")
    }
    
    // MARK: - Combined Features Tests
    
    func testMixedFeatures() throws {
        let result = try tokenizer.phonemize("[Dr.](Doctor) said [hello](/həˈloʊ/) at [5:30](five)")
        
        // Should handle alias, direct phoneme, and replacement in one text
        XCTAssertEqual(result.tokens.count, 5, "Should produce 5 tokens (Doctor, said, hello, at, five)")
        
        // Verify token types
        XCTAssertEqual(result.tokens[0].text, "Doctor", "First token should be alias")
        XCTAssertEqual(result.tokens[1].text, "said", "Second token should be regular word")
        XCTAssertEqual(result.tokens[2].text, "hello", "Third token should be direct phoneme")
        XCTAssertEqual(result.tokens[3].text, "at", "Fourth token should be replacement")
        XCTAssertEqual(result.tokens[4].text, "five", "Fifth token should be alias")
        
        // Verify direct phoneme is preserved
        XCTAssertEqual(result.tokens[2].phonemes, "həˈloʊ", "Direct phonemes should be preserved")
    }
    
    func testStressWithDirectPhonemes() throws {
        let result = try tokenizer.phonemize("[hello](/həloʊ/) [world](1.5)")
        
        XCTAssertEqual(result.tokens.count, 2, "Should produce 2 tokens")
        
        // First token should use direct phonemes as-is
        XCTAssertEqual(result.tokens[0].phonemes, "həloʊ", "Direct phonemes should be preserved")
        XCTAssertNil(result.tokens[0].stress, "Direct phonemes should not have stress applied")
        
        // Second token should have stress applied
        XCTAssertEqual(result.tokens[1].text, "world", "Second token should be world")
        XCTAssertEqual(result.tokens[1].stress, 1.5, "Stress control should be applied")
    }
    
    // MARK: - Punctuation Handling Tests
    
    func testPunctuationPreservation() throws {
        let result = try tokenizer.phonemize("Hello!")
        
        XCTAssertTrue(result.phonemes.contains("!"), "Should preserve exclamation mark")
        XCTAssertFalse(result.phonemes.isEmpty, "Should contain phonemized content")
    }
    
    func testQuestionMarks() throws {
        let result = try tokenizer.phonemize("How?")
        
        XCTAssertTrue(result.phonemes.contains("?"), "Should preserve question mark")
        XCTAssertFalse(result.phonemes.isEmpty, "Should contain phonemized content")
    }
    
    // MARK: - Language Setting Tests
    
    func testLanguageSettingEnUS() throws {
        // Test setting US English voice
        try tokenizer.setLanguage(for: .afAlloy) // US voice
        
        // Should not throw error for valid voice
        XCTAssertNoThrow(try tokenizer.setLanguage(for: .afAlloy))
    }
    
    func testLanguageSettingNonUS() throws {
        // Test setting British English voice
        try tokenizer.setLanguage(for: .bfAlice) // GB voice
        
        // Should not throw error for valid voice
        XCTAssertNoThrow(try tokenizer.setLanguage(for: .bfAlice))
    }
    
    // MARK: - Stress Application Tests
    
    func testDefaultStressApplication() throws {
        let result = try tokenizer.phonemize("important")
        
        // Should apply default stress rules
        XCTAssertTrue(result.phonemes.contains("ˈ") || result.phonemes.contains("ˌ"), "Should contain some stress marker")
        XCTAssertFalse(result.phonemes.isEmpty, "Should contain phonemized content")
    }
    
    func testFunctionWordStressHandling() throws {
        let result = try tokenizer.phonemize("the important book")
        
        // Should handle function words and content words
        XCTAssertGreaterThan(result.tokens.count, 1, "Should produce multiple tokens")
        XCTAssertFalse(result.phonemes.isEmpty, "Should contain phonemized content")
        
        // Check that we have the expected words
        let tokenTexts = result.tokens.map { $0.text }
        XCTAssertTrue(tokenTexts.contains("the"), "Should contain 'the'")
        XCTAssertTrue(tokenTexts.contains("important"), "Should contain 'important'")
        XCTAssertTrue(tokenTexts.contains("book"), "Should contain 'book'")
    }
    
    // MARK: - Error Handling Tests
    
    func testInvalidPhonemeSpecification() throws {
        // Test malformed direct phoneme syntax
        let result = try tokenizer.phonemize("[hello](/invalid")
        
        // Should treat as regular text since phoneme spec is malformed
        XCTAssertFalse(result.phonemes.isEmpty, "Should fallback to normal phonemization")
        XCTAssertGreaterThan(result.tokens.count, 0, "Should produce tokens")
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceWithLongText() throws {
        let longText = String(repeating: "Today is a very long sentence. ", count: 50)
        
        let options = XCTMeasureOptions()
        options.iterationCount = 3
        
        measure(options: options) {
            do {
                _ = try tokenizer.phonemize(longText)
            } catch {
                XCTFail("Performance test should not throw error: \(error)")
            }
        }
    }
    
    func testPerformanceWithManyFeatures() throws {
        let complexText = "[hello](/həˈloʊ/) [world](1.5) costs $5.50 at [5:30](five thirty)"
        
        let options = XCTMeasureOptions()
        options.iterationCount = 5
        
        measure(options: options) {
            do {
                _ = try tokenizer.phonemize(complexText)
            } catch {
                XCTFail("Performance test should not throw error: \(error)")
            }
        }
    }
    
    // MARK: - Edge Cases
    
    func testVeryLongPhonemeSpecification() throws {
        let longPhonemes = String(repeating: "əˈ", count: 100)
        let result = try tokenizer.phonemize("[test](/\(longPhonemes)/)")
        
        XCTAssertEqual(result.phonemes, longPhonemes, "Should handle very long phoneme specifications")
    }
    
    func testNestedBrackets() throws {
        let result = try tokenizer.phonemize("[[test]]")
        
        // Should treat as normal text since it's not valid syntax
        XCTAssertFalse(result.phonemes.isEmpty, "Should phonemize nested brackets as text")
        XCTAssertGreaterThan(result.tokens.count, 0, "Should produce tokens")
    }
    
    func testSpecialCharactersInText() throws {
        let result = try tokenizer.phonemize("test@#$%")
        
        XCTAssertFalse(result.phonemes.isEmpty, "Should handle special characters")
        XCTAssertGreaterThan(result.tokens.count, 0, "Should produce tokens")
    }
    
    func testEmptyFeatureSpecification() throws {
        let result = try tokenizer.phonemize("[test]()")
        
        // Empty replacement should be handled gracefully
        XCTAssertTrue(result.phonemes.isEmpty, "Should handle empty feature specification")
        XCTAssertEqual(result.tokens.count, 0, "Shouldn't produce a token")
    }
}

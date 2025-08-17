import Foundation
import MLX

public class OrpheusTokenizer {
    private let tokenizerConfig: [String: Any]
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let continuingSubwordPrefix: String?
    private let endOfWordSuffix: String?
    private let unkToken: String?
    
    // Add vocabSize property
    public var vocabSize: Int {
        return vocab.count
    }
    
    // Hashable struct for BPE pairs
    private struct StringPair: Hashable {
        let first: String
        let second: String
    }
    
    // Build a merge rank dictionary for fast lookup (once)
    private lazy var mergeRanks: [StringPair: Int] = {
        var dict = [StringPair: Int]()
        for (i, pair) in merges.enumerated() {
            dict[StringPair(first: pair.0, second: pair.1)] = i
        }
        return dict
    }()
    
    private func getPairs(_ symbols: [String]) -> Set<StringPair> {
        var pairs = Set<StringPair>()
        for i in 0..<(symbols.count - 1) {
            pairs.insert(StringPair(first: symbols[i], second: symbols[i + 1]))
        }
        return pairs
    }
    
    public init() throws {
        // Hey you - put tokenizer config file in Orpheus/Resources folder
        let filePath = Bundle.main.path(forResource: "tokenizer_config", ofType: "json")!        
        if !FileManager.default.fileExists(atPath: filePath) {
            fatalError("Orpheus: Tokenizer at \(filePath)")
        }
        
        // Load tokenizer configuration
        guard let configPath = Bundle.main.path(forResource: "tokenizer_config", ofType: "json"),
              let configData = try? Data(contentsOf: URL(fileURLWithPath: configPath)),
              let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw TokenizerError.configNotFound
        }
        tokenizerConfig = config
        
        // Extract BPE-specific config
        continuingSubwordPrefix = config["continuing_subword_prefix"] as? String
        endOfWordSuffix = config["end_of_word_suffix"] as? String
        unkToken = config["unk_token"] as? String
        
        // Load vocabulary and merges from tokenizer.json
        guard let tokenizerPath = Bundle.main.path(forResource: "tokenizer", ofType: "json"),
              let tokenizerData = try? Data(contentsOf: URL(fileURLWithPath: tokenizerPath)),
              let tokenizerDict = try? JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any],
              let model = tokenizerDict["model"] as? [String: Any],
              let vocabDict = model["vocab"] as? [String: Int],
              let mergesArray = model["merges"] as? [[String]] else {            
            throw TokenizerError.tokenizerNotFound
        }
        
        vocab = vocabDict
        
        // Convert merges to tuples and print first few for debugging
        merges = mergesArray.map { pair in
            (pair[0], pair[1])
        }
    }
    
    // MARK: - Prepare Input IDs (matches llama.py)
    public func prepareInputIds(prompts: [String], voice: String? = nil, refAudio: MLXArray? = nil, refText: String? = nil) -> (MLXArray, MLXArray) {
        // TESTING
//        let fake_token_values: [Int32] = [128259, 128000, 83, 5169, 25, 24748, 128009, 128260]
//        let token_ids_for_batch = MLXArray(fake_token_values).reshaped([1, -1])
//
//        // Create a dummy attention mask (all true, indicating all tokens are attended)
//        let attention_mask_for_batch = MLXArray.ones(token_ids_for_batch.shape, dtype: .bool)
//
//        return (token_ids_for_batch, attention_mask_for_batch)
        // END TESTING
        
        // Special token IDs
        let startToken: Int = 128259
        let endTokens: [Int] = [128009, 128260]
        let padToken: Int = 128263
        let audioStartTokens: [Int] = [128261, 128257]
        let audioEndTokens: [Int] = [128258, 128262]
        
        var promptsToUse = prompts
        let audioInputIds: MLXArray? = nil
        var audioTranscriptIds: MLXArray? = nil
        
        if let refAudio = refAudio, let refText = refText {
            // TODO: Implement encodeAudioToCodes in Swift to match Python
            // audioInputIds = encodeAudioToCodes(refAudio) + audioCodeOffset
            // audioTranscriptIds = tokenize(text: refText)
        } else if let voice = voice {
            
        }
        
        // Tokenize prompts
        var promptInputIds: [MLXArray] = []
        for prompt in promptsToUse {
            let tokens = tokenizeText(prompt)
            promptInputIds.append(MLXArray(tokens))
        }
        
        // Find max length for padding
        let maxLen = promptInputIds.map { $0.count }.max() ?? 0
        
        var batchInputIds: [MLXArray] = []
        for inputIds in promptInputIds {
            var modifiedInputIds: [MLXArray] = []
            let paddingLen = maxLen - inputIds.count
            if paddingLen > 0 {
                let padArray = MLX.full([paddingLen], values: padToken)
                modifiedInputIds.append(padArray)
            }
            
            // Reference audio and transcript (not implemented yet)
            if let audioInputIds = audioInputIds, let audioTranscriptIds = audioTranscriptIds {
                let start = MLXArray([startToken])
                let end = MLXArray(endTokens)
                let audioStart = MLXArray(audioStartTokens)
                let audioEnd = MLXArray(audioEndTokens)
                let refInputIds = MLX.concatenated([start, audioTranscriptIds, end, audioStart, audioInputIds, audioEnd], axis: 0)
                modifiedInputIds.append(refInputIds)
            }
            
            // Prompt
            let start = MLXArray([startToken])
            let end = MLXArray(endTokens)
            let beginOfText = MLXArray([128000])
            let onePromptInputIds = MLX.concatenated([start, beginOfText, inputIds, end], axis: 0)
            modifiedInputIds.append(onePromptInputIds)
            
            let batch = MLX.concatenated(modifiedInputIds, axis: 0)
            batchInputIds.append(batch)
        }
        
        let batchInput = stack(batchInputIds, axis: 0)
        let padArray = MLXArray([padToken])
        let batchMask = MLX.notEqual(batchInput, padArray)
        return (batchInput, batchMask)
    }
    
    // Utility: stack arrays along a new axis (like MLX stack)
    public func stack(_ arrays: [MLXArray], axis: Int = 0) -> MLXArray {
        let expanded = arrays.map { $0.expandDims(at: axis) }
        return MLX.concatenated(expanded, axis: axis)
    }
    
    private func preprocessText(_ text: String) -> String {
        return text.precomposedStringWithCanonicalMapping
    }
    
    private func getBestPair(_ symbols: [String]) -> (pair: StringPair, rank: Int, index: Int)? {
        var bestPair: (pair: StringPair, rank: Int, index: Int)? = nil

        for i in 0..<(symbols.count - 1) {
            let pair = StringPair(first: symbols[i], second: symbols[i + 1])
            if let rank = mergeRanks[pair] {
                if bestPair == nil || rank < bestPair!.rank {
                    bestPair = (pair: pair, rank: rank, index: i)
                }
            }
        }
        return bestPair
    }

    private func applyBPE(_ symbols: [String]) -> [String] {
        var currentSymbols = symbols
        
        // Handle prefix space (using "Ġ" representation)
        var initialSymbolsForBPE: [String] = []
        for symbol in currentSymbols { // currentSymbols here are the raw byte strings from preprocessed text
            if symbol == " " {
                initialSymbolsForBPE.append("Ġ")
            } else {
                initialSymbolsForBPE.append(symbol)
            }
        }
        currentSymbols = initialSymbolsForBPE
        // print("Initial symbols with space handling: \(currentSymbols)") // Already confirmed by user

        // Apply BPE merges based on rank priority - one merge at a time
        while true {
            // Find the highest-priority merge rule that applies to any adjacent pair in currentSymbols.
            // getBestPair returns the pair with the lowest rank, and its first index of occurrence.
            guard let (pairToMerge, _ /* rank */, indexToMergeAt) = getBestPair(currentSymbols) else {
                // No pair in currentSymbols is found in mergeRanks, or currentSymbols has < 2 elements.
                break
            }

            // Perform the single merge at indexToMergeAt
            var newSymbols = [String]()
            
            // Add elements before the merge point
            if indexToMergeAt > 0 {
                newSymbols.append(contentsOf: currentSymbols[0..<indexToMergeAt])
            }
            
            // Add the merged element
            newSymbols.append(pairToMerge.first + pairToMerge.second)
            
            // Add elements after the merged pair
            let nextElementIndexAfterMergedPair = indexToMergeAt + 2
            if nextElementIndexAfterMergedPair < currentSymbols.count {
                newSymbols.append(contentsOf: currentSymbols[nextElementIndexAfterMergedPair..<currentSymbols.count])
            }
            
            currentSymbols = newSymbols
        }
        
        return currentSymbols
    }
    
    private func tokenizeText(_ text: String) -> [Int] {
        let preprocessed = preprocessText(text)
        
        // Convert to UTF-8 bytes, then to string symbols
        let byteSymbols = [UInt8](preprocessed.utf8).map { String(UnicodeScalar($0)) }
        
        // Apply BPE
        let mergedSymbols = applyBPE(byteSymbols)
        
        // Map to vocab IDs
        var tokens: [Int] = []
        for symbol in mergedSymbols {
            if let id = vocab[symbol] {
                tokens.append(id)
            } else if let unk = unkToken, let unkId = vocab[unk] {
                tokens.append(unkId)
            } else {
                // If no UNK token, split into bytes
                for byte in symbol.utf8 {
                    if let id = vocab[String(UnicodeScalar(byte))] {
                        tokens.append(id)
                    }
                }
            }
        }
        
        return tokens
    }
    
    enum TokenizerError: Error {
        case configNotFound
        case specialTokensNotFound
        case tokenizerNotFound
    }
}

// MLXArray extension for expandDims
extension MLXArray {
    func expandDims(at axis: Int) -> MLXArray {
        var newShape = self.shape
        newShape.insert(1, at: axis)
        return self.reshaped(newShape)
    }
} 

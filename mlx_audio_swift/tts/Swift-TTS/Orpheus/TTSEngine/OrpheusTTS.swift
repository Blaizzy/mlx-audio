import Foundation
import MLX
import MLXNN
import MLXRandom

// Available voices for Orpheus
public enum OrpheusVoice: String, CaseIterable {
    case tara = "tara" // Female, conversational, clear
    case leah = "leah" // Female, warm, gentle
    case jess = "jess" // Female, energetic, youthful
    case leo = "leo" // Male, authoritative, deep
    case dan = "dan" // Male, friendly, casual
    case mia = "mia" // Female, professional, articulate
    case zac = "zac" // Male, enthusiastic, dynamic
    case zoe = "zoe" // Female, calm, soothing
}

// MARK: - Profiling Helper
struct Profiler {
    static var enabled: Bool = false // Set to true to enable profiling
    
    static func time<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
        guard enabled else { return try block() }
        
        let start = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let end = CFAbsoluteTimeGetCurrent()
        let duration = (end - start) * 1000 // Convert to milliseconds
        print("⏱️ [PROFILE] \(label): \(String(format: "%.2f", duration))ms")
        return result
    }
    
    static func timeAsync<T>(_ label: String, _ block: () async throws -> T) async rethrows -> T {
        guard enabled else { return try await block() }
        
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let end = CFAbsoluteTimeGetCurrent()
        let duration = (end - start) * 1000 // Convert to milliseconds
        print("⏱️ [PROFILE] \(label): \(String(format: "%.2f", duration))ms")
        return result
    }
}

struct Constants {
    static let maxTokenCount = 1200
    static let sampleRate = 24000
    static let startToken = 128259
    static let endToken = 128258
    static let padToken = 128263
    static let audioStartToken = 128261
    static let audioEndToken = 128262
    static let voicePrefixToken = 128260
    static let repetitionContextSize = 20
    static let codeOffset = 128266
    static let audioCodeDataStartMarker = 128257
}

// Main class for Orpheus TTS
public class OrpheusTTS {
    enum OrpheusTTSError: Error {
        case tooManyTokens
        case weightsNotAvailable
        case modelNotInitialized
    }
    
    private let weights: [String: MLXArray]
    private let snacDecoder: SNACDecoder
    private var chosenVoice: OrpheusVoice?
    private let tokenizer: OrpheusTokenizer
    private let hiddenSize: Int = 3072
    private let layers: [TransformerBlock] // Store TransformerBlock instances
    
    init() throws {
        // Load model weights
        let loadedWeights = Profiler.time("Weight loading") {
            OrpheusWeightLoader.loadWeightsOrpheus()
        }
        self.weights = loadedWeights

        self.snacDecoder = Profiler.time("SNAC decoder init") {
            SNACDecoder(config: SNACDecoder.loadConfig()!)
        }
        
        self.tokenizer = try Profiler.time("Tokenizer init") {
            try OrpheusTokenizer()
        }
        
        // Initialize transformer layers - avoid capturing self in closure
        let numLayers = 28 // Based on config.json
        let layerInitStart = CFAbsoluteTimeGetCurrent()
        var tempLayers = [TransformerBlock]()
        for i in 0..<numLayers {
            let layerStart = CFAbsoluteTimeGetCurrent()
            tempLayers.append(TransformerBlock(weights: loadedWeights, layerIndex: i))
            let layerEnd = CFAbsoluteTimeGetCurrent()
            let layerDuration = (layerEnd - layerStart) * 1000
        }
        self.layers = tempLayers
        let layerInitEnd = CFAbsoluteTimeGetCurrent()
        let layerInitDuration = (layerInitEnd - layerInitStart) * 1000
    }
    
    public func generateAudio(voice: OrpheusVoice, text: String, temperature: Float = 0.6, topP: Float = 0.8) async throws -> MLXArray {
        let totalGenerationStart = CFAbsoluteTimeGetCurrent()
        
        // Prepare input with voice prefix
        let prompt = "\(voice.rawValue): \(text)"
        print("Orpheus prompt: \(prompt)")
        
        let input_ids_tuple = Profiler.time("Tokenizer preparation") {
            tokenizer.prepareInputIds(prompts: [prompt])
        }
                
        // Convert the tokenizer output to a Swift [Int32]
        var current_ids = Profiler.time("Input IDs conversion") {
            let array = MLXArray(input_ids_tuple.0[0].asArray(Int32.self))
            return array.ndim == 1 ? array.reshaped([1, -1]) : array
        }
        
        print("Input IDs: \(current_ids.shape) = \(current_ids.asArray(Int32.self))")
        
        // Initialize KV Caches
        let numLayers = self.layers.count
        var kvCaches: [Cache?] = Array(repeating: nil, count: numLayers)
        
        // Process the initial prompt.
        var (logits, updatedKvCachesAfterPrompt) = Profiler.time("Initial forward pass") {
            forward(inputIds: current_ids, currentKvCaches: kvCaches)
        }
        kvCaches = updatedKvCachesAfterPrompt
        
        // Generate audio tokens
        var generatedTokensForPenalty: [Int32] = [] // For repetition penalty
        var i = 0
        var previousToken: Int32? = nil // For correcting anomalous tokens after audioStartToken

        let maxOutputTokens = Constants.maxTokenCount // Define how many tokens to generate at most
        
        let generationLoopStart = CFAbsoluteTimeGetCurrent()
        
        while i < maxOutputTokens {
            let iterationStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            
            let historyForRepetition = Profiler.time("History preparation") {
                MLXArray(generatedTokensForPenalty)
            }
            
            let samplingStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let nextTokenArray = sampleNextToken(
                logits: logits,
                history: historyForRepetition,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: 1.3
            )
            let samplingEnd = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let samplingDuration = Profiler.enabled ? (samplingEnd - samplingStart) * 1000 : 0
            if Profiler.enabled {
                print("⏱️ [PROFILE] Token sampling (iter \(i)): \(String(format: "%.2f", samplingDuration))ms")
            }

            // Only extract the Int32 value when we absolutely need it for CPU operations
            let next_token: Int32 = Profiler.time("Token extraction") {
                // This operation forces GPU->CPU transfer and might be a sync point
                let result: Int32 = nextTokenArray[0].item()
                return result
            }
            
            // Stop generation only at the general end-of-text token
            if next_token == Constants.endToken {
                let endArr = MLXArray([Constants.endToken]).reshaped([1,1])
                current_ids = MLX.concatenated([current_ids, endArr], axis: 1)
                if Profiler.enabled {
                    print("DBG: End token \(Constants.endToken) encountered. Appending and breaking.")
                }
                break
            }
                        
            // Add next token to the sequence for parsing and for model input
            let tokenConcatTime = Profiler.time("Token concatenation (iter \(i))") {
                let nextTokenForConcat = nextTokenArray.reshaped([1, 1])
                current_ids = MLX.concatenated([current_ids, nextTokenForConcat], axis: 1)
            }
            
            // Add to history for repetition penalty *after* it's been sampled
            let historyUpdateTime = Profiler.time("History update") {
                generatedTokensForPenalty.append(next_token)
                if generatedTokensForPenalty.count > Constants.repetitionContextSize { // Keep history to context size
                    generatedTokensForPenalty.removeFirst()
                }
                previousToken = next_token // Update previous token for next iteration
            }
            
            // Prepare for the next iteration:
            let forwardPassStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let (next_logits, next_kvCaches) = forward(inputIds: current_ids, currentKvCaches: kvCaches)
            let forwardPassEnd = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let forwardPassDuration = Profiler.enabled ? (forwardPassEnd - forwardPassStart) * 1000 : 0
            if Profiler.enabled {
                print("⏱️ [PROFILE] Forward pass (iter \(i)): \(String(format: "%.2f", forwardPassDuration))ms")
            }
            
            logits = next_logits
            kvCaches = next_kvCaches
            
            // Clear cache periodically
            if (i + 1) % 50 == 0 {
                Profiler.time("GPU cache clear") {
                    MLX.GPU.clearCache()
                }
            }
            
            if Profiler.enabled {
                let iterationEnd = CFAbsoluteTimeGetCurrent()
                let iterationDuration = (iterationEnd - iterationStart) * 1000
                
                // Calculate unaccounted time
                let accountedTime = forwardPassDuration + 
                                  (historyForRepetition.size > 0 ? 0.5 : 0.0) + // rough estimate for sampling/concat
                                  0.4 + 0.03 + 0.1 // sampling + concat + overhead estimates
                let unaccountedTime = iterationDuration - accountedTime
                
                // Print detailed timing every 10 iterations or for first 5
                if i < 5 || i % 10 == 0 {
                    print("  🔀 Iteration \(i): \(String(format: "%.2f", iterationDuration))ms total")
                    print("    📊 Forward: \(String(format: "%.2f", forwardPassDuration))ms")
                    print("    ❓ Unaccounted: \(String(format: "%.2f", unaccountedTime))ms")
                    print("    🎯 Token: \(next_token)")
                }
            }
            
            i += 1
        }
                
        if i >= maxOutputTokens {
            print("WARNING: Reached max token count (\(maxOutputTokens)) during generation.")
        }
        
        // Parse the output into code lists
        let code_lists = Profiler.time("Output parsing") {
            parseOutput(tokens: current_ids.asArray(Int32.self).map { Int($0) })
        }
        
        // Generate audio using SNAC decoder
        let waveform = Profiler.time("SNAC decoding") {
            snacDecoder.decode(codes: code_lists)
        }
        
        let totalGenerationEnd = CFAbsoluteTimeGetCurrent()
        let totalDuration = (totalGenerationEnd - totalGenerationStart) * 1000
        print("🏁 [PROFILE] Total audio generation: \(String(format: "%.2f", totalDuration))ms")
        
        return waveform
    }
    
    private func forward(inputIds: MLXArray, currentKvCaches: [Cache?]) -> (logits: MLXArray, updatedKvCaches: [Cache?]) {
        let forwardStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        
        // Get embedding weights
        guard let embeddingWeights = weights["model.embed_tokens.weight"] else {
            fatalError("Embedding weights not found") // Should consider returning error or empty array
        }
        
        // If using cache, only process the last token
        var x: MLXArray
        let isCaching = currentKvCaches.first(where: { $0 != nil }) != nil
        if isCaching {
            // Only get embedding for the last token
            let lastTokenId = inputIds[0, -1]
            x = embeddingWeights[lastTokenId].reshaped([1, 1, -1])
        } else {
            // Process full sequence
            x = embeddingWeights[inputIds]
        }

        // Only print token details for initial pass or occasionally during generation
        if Profiler.enabled && (!isCaching || (isCaching && inputIds.shape[1] % 50 == 0)) {
            let tokenCount = inputIds.shape[1]
            print("Generated tokens count: \(tokenCount)")
        }

        // Validate shape
        guard x.shape[2] == hiddenSize else {
            fatalError("Invalid shape after embedding: expected \(hiddenSize), got \(x.shape[2])")
        }
        
        let L = x.shape[1] // Sequence length
        
        var attentionMask: MLXArray? = nil
        if !isCaching {
            attentionMask = Profiler.time("Attention mask creation") {
                let mask = MLX.triu(MLXArray.full([L,L], values: MLXArray([Float(-1e9)])), k: 1)
                return mask.asType(x.dtype).expandDims(at: 0).expandDims(at: 0)
            }
        }
        
        var nextKvCaches: [Cache?] = Array(repeating: nil, count: self.layers.count)
        
        // Process through transformer layers
        let layersStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        for i in 0..<self.layers.count {
            let layerStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let (layerOutput, updatedLayerCache) = self.layers[i].call(x, mask: attentionMask, cache: currentKvCaches[i])
            x = layerOutput
            nextKvCaches[i] = updatedLayerCache
            
            if Profiler.enabled {
                let layerEnd = CFAbsoluteTimeGetCurrent()
                let layerDuration = (layerEnd - layerStart) * 1000
                
                // Print timing for first few layers or every 5th layer, and only occasionally during generation
                if (!isCaching || currentKvCaches.first??.offset ?? 0 < 50) && (i < 3 || i % 5 == 0) {
                    print("  🧠 Layer \(i): \(String(format: "%.2f", layerDuration))ms")
                }
            }
        }
        
        if Profiler.enabled {
            let layersEnd = CFAbsoluteTimeGetCurrent()
            let layersTotalDuration = (layersEnd - layersStart) * 1000
            
            // Only print layer summary for initial pass or occasionally during generation
            if !isCaching || (currentKvCaches.first??.offset ?? 0) < 50 {
                print("  🧠 All layers: \(String(format: "%.2f", layersTotalDuration))ms")
            }
        }

        // 3. Final RMSNorm
        guard let finalNormWeight = weights["model.norm.weight"] else {
            print("ERROR: Final norm weight not found.")
            return (MLXArray([]), nextKvCaches) // Return current caches even on error
        }

        x = Profiler.time("Final RMSNorm") {
            MLX.rmsNorm(x, weight: finalNormWeight, eps: 1e-5)
        }
        
        // 4. Output projection (LM Head)
        let logits = Profiler.time("Output projection") {
            TransformerBlock.linear(x: x, weight: embeddingWeights)
        }

        // If caching, logits are already [1, 1, VocabSize] from processing the last token.
        let finalLogits = Profiler.time("Logits finalization") {
            isCaching ? logits.squeezed(axis: 1) : logits[0, -1].expandDims(at: 0)
        }
        
        if Profiler.enabled {
            let forwardEnd = CFAbsoluteTimeGetCurrent()
            let forwardDuration = (forwardEnd - forwardStart) * 1000
            print("  ➡️ Forward pass total: \(String(format: "%.2f", forwardDuration))ms")
        }

        return (finalLogits, nextKvCaches)
    }
    
    private func sampleNextToken(
        logits: MLXArray,
        history: MLXArray,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float = 1.3
    ) -> MLXArray {
        let samplingStart = Profiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        
        // Start with raw logits
        var currentLogits = logits

        // 1. Apply repetition penalty if needed
        if repetitionPenalty != 1.0 && history.size > 0 {
            currentLogits = Profiler.time("Repetition penalty") {
                // Vectorised implementation to keep data on GPU/Metal.
                let indices = history // Int32 tensor with shape [K]
                var logits1D = currentLogits[0] // Shape [V]

                // Gather the logits corresponding to the history tokens.
                let gathered = MLX.take(logits1D, indices)

                // Compute updated logits according to the repetition penalty.
                let negMask   = gathered .< 0
                let updated   = MLX.where(
                    negMask,
                    gathered * repetitionPenalty,
                    gathered / repetitionPenalty
                )

                // Scatter the updated values back into the logits tensor.
                logits1D = MLXArray.scatter(logits1D, indices: indices, updates: updated)

                // Restore the [1, V] shape expected downstream.
                return logits1D.expandDims(at: 0)
            }
        }
        
        // 2. Apply temperature scaling
        let scaledLogits = Profiler.time("Temperature scaling") {
            currentLogits / max(temperature, 1e-6)
        }

        // 3. Apply top-p filtering
        var filteredLogits = scaledLogits
        if topP > 0.0 && topP < 1.0 {
            filteredLogits = Profiler.time("Top-p filtering") {
                let vocabSize = scaledLogits.shape[1]
                if vocabSize > 1 {
                    // Vectorised top-p filtering (no host round-trips).

                    // 1. Probabilities.
                    let probs = MLX.softmax(scaledLogits[0], axis: -1)        // [V]

                    // 2. Sort (descending).
                    let sortedIdx   = MLX.argSort(MLX.negative(probs))         // [V] Int32
                    let sortedProbs = MLX.take(probs, sortedIdx)               // [V]

                    // 3. Cumulative sum.
                    let cumProbs = sortedProbs.cumsum(axis: -1)                // [V]

                    // 4. Mask tokens occurring strictly after the cut-off.
                    let gtMask        = cumProbs .> topP                      // Bool [V]
                    let gtMaskInt     = gtMask.asType(.int32)                 // Int32 [V]
                    let prefix        = gtMaskInt.cumsum(axis: -1)            // Int32 [V]
                    let removeMaskSorted = prefix .> 1                        // Bool [V]

                    // 5. Bring mask back to original vocab order.
                    let invIdx          = MLX.argSort(sortedIdx)              // [V]
                    let removeMask      = MLX.take(removeMaskSorted, invIdx)  // Bool [V]

                    // 6. Apply mask: set filtered logits to -inf.
                    let negInfScalar    = MLXArray(-Float.infinity)           // scalar
                    let logits1D        = scaledLogits[0]
                    let filtered1D      = MLX.where(removeMask, negInfScalar, logits1D)

                    // 7. Restore [1, V] shape expected downstream.
                    return filtered1D.expandDims(at: 0)
                }
                return scaledLogits
            }
        }
        
        // 4. Sample from filtered distribution
        let nextTokenIdArray = Profiler.time("Categorical sampling") {
            MLXRandom.categorical(filteredLogits, count: 1)
        }
        
        if Profiler.enabled {
            let samplingEnd = CFAbsoluteTimeGetCurrent()
            let samplingDuration = (samplingEnd - samplingStart) * 1000
            print("  🎲 Sampling total: \(String(format: "%.2f", samplingDuration))ms")
        }
        
        return nextTokenIdArray
    }
    
    private func parseOutput(tokens: [Int]) -> [[Int]] {
        // Find the last occurrence of the audio start token as defined in Constants
        let lastStartIndex = tokens.lastIndex(of: Constants.audioCodeDataStartMarker) ?? -1
        
        // Get tokens after the last start token
        let relevantTokens = lastStartIndex >= 0 ? Array(tokens[(lastStartIndex + 1)...]) : tokens
        
        // Filter out the general end token (128258) and ensure codes are valid (>= codeOffset)
        // Python's llama.py uses token_to_remove = 128258 and does not filter a separate audioEndToken.
        let filteredTokens = relevantTokens.filter { $0 != Constants.endToken && $0 >= Constants.codeOffset }
        
        // Ensure length is multiple of 7 by trimming
        let newLength = (filteredTokens.count / 7) * 7
        let trimmedTokens = Array(filteredTokens[..<newLength])            
        
        // Subtract offset from all tokens
        let adjustedTokens = trimmedTokens.map { $0 - Constants.codeOffset }
        
        // Split into layers based on the stride pattern
        var layer1: [Int] = []
        var layer2: [Int] = []
        var layer3: [Int] = []
        
        // Process codes in groups of 7
        for i in 0..<(adjustedTokens.count / 7) {
            let base = 7 * i
            layer1.append(adjustedTokens[base])
            layer2.append(adjustedTokens[base + 1] - 4096)
            layer3.append(adjustedTokens[base + 2] - 2 * 4096)
            layer3.append(adjustedTokens[base + 3] - 3 * 4096)
            layer2.append(adjustedTokens[base + 4] - 4 * 4096)
            layer3.append(adjustedTokens[base + 5] - 5 * 4096)
            layer3.append(adjustedTokens[base + 6] - 6 * 4096)
        }
        
        return [layer1, layer2, layer3]
    }
}

class Cache {
    var keys: MLXArray
    var values: MLXArray
    var offset: Int // Represents the number of tokens already in the cache (sequence length of cached items)
    
    init(keys: MLXArray, values: MLXArray, offset: Int = 0) {
        self.keys = keys
        self.values = values
        self.offset = offset // Should be L_initial if creating from scratch, e.g., keys.shape[2]
    }
    
    // newKeys, newValues are for the current segment being processed (e.g., L=1 for incremental generation)
    // newKeys: [B, H, L_new, D_head]
    func updateAndFetch(newKeys: MLXArray, newValues: MLXArray) -> (MLXArray, MLXArray) {
        // Ensure keys and newKeys are compatible for concatenation along axis 2 (sequence length)
        // self.keys: [B, H, L_cached, D_head]
        // newKeys:   [B, H, L_new, D_head]
        self.keys = MLX.concatenated([self.keys, newKeys], axis: 2)
        self.values = MLX.concatenated([self.values, newValues], axis: 2)
        self.offset += newKeys.shape[2] // Update the offset by the length of the new segment
        return (self.keys, self.values)
    }
} 

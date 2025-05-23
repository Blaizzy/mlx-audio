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
        self.weights = OrpheusWeightLoader.loadWeightsOrpheus()

        self.snacDecoder = SNACDecoder(config: SNACDecoder.loadConfig()!)
        
        self.tokenizer = try OrpheusTokenizer()
        
        // Initialize transformer layers
        let numLayers = 28 // Based on config.json
        var tempLayers = [TransformerBlock]()
        for i in 0..<numLayers {
            tempLayers.append(TransformerBlock(weights: weights, layerIndex: i))
        }
        self.layers = tempLayers
    }
    
    public func generateAudio(voice: OrpheusVoice, text: String, temperature: Float = 0.6, topP: Float = 0.8) async throws -> MLXArray {
        // Prepare input with voice prefix
        let prompt = "\(voice.rawValue): \(text)"
        print("Orpheus prompt: \(prompt)")
        
        let input_ids_tuple = tokenizer.prepareInputIds(prompts: [prompt])
                
        // Convert the tokenizer output to a Swift [Int32]
        var current_ids = MLXArray(input_ids_tuple.0[0].asArray(Int32.self)) // Keep as MLXArray
        
        print("Input IDs: \(current_ids.shape) = \(current_ids.asArray(Int32.self))")
        
        // Ensure input_ids is 2D for concatenation
        if current_ids.ndim == 1 {
            current_ids = current_ids.reshaped([1, -1])
        }
        
        // Initialize KV Caches
        let numLayers = self.layers.count
        var kvCaches: [Cache?] = Array(repeating: nil, count: numLayers)
        
        // Process the initial prompt.
        // `logits` will be for the token immediately following the prompt.
        // `kvCaches` will be the state *after* processing the entire prompt.
        var (logits, updatedKvCachesAfterPrompt) = forward(inputIds: current_ids, currentKvCaches: kvCaches)
        kvCaches = updatedKvCachesAfterPrompt
        
        // Generate audio tokens
        var generatedTokensForPenalty: [Int32] = [] // For repetition penalty
        var i = 0
        var previousToken: Int32? = nil // For correcting anomalous tokens after audioStartToken

        let maxOutputTokens = Constants.maxTokenCount // Define how many tokens to generate at most
        
        while i < maxOutputTokens {
            let historyForRepetition = MLXArray(generatedTokensForPenalty)
            
            var next_token_int = sampleNextToken(
                logits: logits,
                history: historyForRepetition,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: 1.3
            )
            
            let next_token = Int32(next_token_int) // Ensure it's Int32 for MLXArray
                        
            // Stop generation only at the general end-of-text token
            if next_token == Constants.endToken {
                let endArr = MLXArray([Constants.endToken]).reshaped([1,1])
                current_ids = MLX.concatenated([current_ids, endArr], axis: 1)
                print("DBG: End token \(Constants.endToken) encountered. Appending and breaking.")
                break
            }
                        
            // Add next token to the sequence for parsing and for model input
            let nextTokenArray = MLXArray([next_token]).reshaped([1, 1])
            current_ids = MLX.concatenated([current_ids, nextTokenArray], axis: 1)
            
            // Add to history for repetition penalty *after* it's been sampled
            generatedTokensForPenalty.append(next_token)
            if generatedTokensForPenalty.count > Constants.repetitionContextSize { // Keep history to context size
                generatedTokensForPenalty.removeFirst()
            }
            
            previousToken = next_token // Update previous token for next iteration
            
            // Prepare for the next iteration:
            let (next_logits, next_kvCaches) = forward(inputIds: current_ids, currentKvCaches: kvCaches)
            logits = next_logits
            kvCaches = next_kvCaches
            
            // Clear cache periodically
            if (i + 1) % 50 == 0 {
                MLX.GPU.clearCache()
            }
            
            i += 1
        }
        
        if i >= maxOutputTokens {
            print("WARNING: Reached max token count (\(maxOutputTokens)) during generation.")
        }
        
        // Parse the output into code lists
        let code_lists = parseOutput(tokens: current_ids.asArray(Int32.self).map { Int($0) })
        
        // Generate audio using SNAC decoder
        let waveform = snacDecoder.decode(codes: code_lists)
        
        return waveform
    }
    
    private func forward(inputIds: MLXArray, currentKvCaches: [Cache?]) -> (logits: MLXArray, updatedKvCaches: [Cache?]) {
        // Get embedding weights
        guard let embeddingWeights = weights["model.embed_tokens.weight"] else {
            fatalError("Embedding weights not found") // Should consider returning error or empty array
        }
        
        // If using cache, only process the last token
        var x: MLXArray
        let isCaching = currentKvCaches.first(where: { $0 != nil }) != nil

        if isCaching {
            // Only get embedding for the last token
            // inputIds is [1, L], so inputIds[0, -1] gives the last token ID as a scalar MLXArray
            let lastTokenId = inputIds[0, -1]
            x = embeddingWeights[lastTokenId].reshaped([1, 1, -1])
        } else {
            // Process full sequence
            x = embeddingWeights[inputIds]
        }

        print("Generated tokens: \(inputIds.asArray(Int32.self)) (\(inputIds.asArray(Int32.self).count))")

        // Validate shape
        guard x.shape[2] == hiddenSize else {
            fatalError("Invalid shape after embedding: expected \(hiddenSize), got \(x.shape[2])")
        }
        
        let L = x.shape[1] // Sequence length
        
        var attentionMask: MLXArray? = nil
        if !isCaching {
            // Create causal attention mask [1, 1, L, L] for initial pass (no cache)
            // L is the sequence length of the input x (prompt length)
            attentionMask = MLX.triu(MLXArray.full([L,L], values: MLXArray([Float(-1e9)])), k: 1) // Use large negative float
                                     
            attentionMask = attentionMask!.asType(x.dtype) // Ensure same dtype as x for addition
            attentionMask = attentionMask!.expandDims(at: 0).expandDims(at: 0) // Shape becomes [1, 1, L, L]
        }
        
        var nextKvCaches: [Cache?] = Array(repeating: nil, count: self.layers.count)
        
        // Process through transformer layers
        for i in 0..<self.layers.count {
            let (layerOutput, updatedLayerCache) = self.layers[i].call(x, mask: attentionMask, cache: currentKvCaches[i])
            x = layerOutput
            nextKvCaches[i] = updatedLayerCache
        }

        // 3. Final RMSNorm
        guard let finalNormWeight = weights["model.norm.weight"] else {
            print("ERROR: Final norm weight not found.")
            return (MLXArray([]), nextKvCaches) // Return current caches even on error
        }

        x = MLX.rmsNorm(x, weight: finalNormWeight, eps: 1e-5)
        
        // 4. Output projection (LM Head)
        // Use embedding weights for output projection (weight tying)
        let logits = TransformerBlock.linear(x: x, weight: embeddingWeights)

        // If caching, logits are already [1, 1, VocabSize] from processing the last token.
        // We need to squeeze the middle dimension to get [1, VocabSize].
        // If not caching, logits are [1, L_prompt, VocabSize], and we take the last one.
        let finalLogits = isCaching ? logits.squeezed(axis: 1) : logits[0, -1].expandDims(at: 0)

        return (finalLogits, nextKvCaches)
    }
    
    private func sampleNextToken(
        logits: MLXArray,
        history: MLXArray,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float = 1.3
    ) -> Int {
        // Start with raw logits
        var currentLogits = logits

        // 1. Apply repetition penalty if needed
        if repetitionPenalty != 1.0 && history.size > 0 {
            // Vectorised implementation to keep data on GPU/Metal.
            // 1. Gather current logits for the history indices.
            let indices = history // Int32 tensor with shape [K]

            // Ensure logits is 2-D [1, V]. We will work on the 1-D slice.
            var logits1D = currentLogits[0]                     // Shape [V]

            // 2. Gather the logits corresponding to the history tokens.
            let gathered = MLX.take(logits1D, indices)

            // 3. Compute updated logits according to the repetition penalty.
            //    If the logit is < 0 multiply by penalty, else divide by penalty.
            let negMask   = gathered .< 0
            let updated   = MLX.where(
                negMask,
                gathered * repetitionPenalty,
                gathered / repetitionPenalty
            )

            // 4. Scatter the updated values back into the logits tensor.
            logits1D = MLXArray.scatter(logits1D, indices: indices, updates: updated)

            // 5. Restore the [1, V] shape expected downstream.
            currentLogits = logits1D.expandDims(at: 0)
        }
        
        // 2. Apply temperature scaling
        let scaledLogits = currentLogits / max(temperature, 1e-6)

        let startRep = Date.timeIntervalSinceReferenceDate

        // 3. Apply top-p filtering
        var filteredLogits = scaledLogits
        if topP > 0.0 && topP < 1.0 {
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
                //    A token is removed if: it appears after the FIRST time
                //    cumulative prob exceeds `topP`.
                //    Implementation: mark (cumProbs > topP), then remove all
                //    occurrences *after* the first such event using a prefix sum.
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
                filteredLogits = filtered1D.expandDims(at: 0)
            }
        }
        
        // 4. Sample from filtered distribution
        let nextTokenIdArray = MLXRandom.categorical(filteredLogits, count: 1)
        let nextTokenId: Int = nextTokenIdArray[0].item()

        // Validate token
        let vocabSizeOutput = filteredLogits.shape[1]
        if nextTokenId >= vocabSizeOutput {
            print("WARNING: Generated audio token \(nextTokenId) exceeds vocabulary size \(vocabSizeOutput)")
        }

        return nextTokenId
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

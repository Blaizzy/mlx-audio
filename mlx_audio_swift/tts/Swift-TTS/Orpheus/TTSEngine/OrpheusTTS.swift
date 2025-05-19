import Foundation
import MLX
import MLXNN
import MLXRandom

// Available voices for Orpheus
public enum OrpheusVoice: String, CaseIterable {
    case tara = "tara" // "default"
    // Add more voices as needed
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

    init() throws {
        // Load model weights
        weights = OrpheusWeightLoader.loadWeightsOrpheus()
        if weights.isEmpty {
            throw OrpheusTTSError.weightsNotAvailable
        }

        // Initialize SNAC decoder
        let snacConfig = SNACConfig()
        snacDecoder = SNACDecoder(weights: weights, config: snacConfig)
    }

    public func generateAudio(voice: OrpheusVoice, text: String, temperature: Float = 0.6, topP: Float = 0.8) async throws -> MLXArray {
        print("Orpheus voice: \(voice), text: \(text)")

        // Prepare input with voice prefix
        let prompt = "\(voice.rawValue): \(text)"
        print("Orpheus prompt: \(prompt)")

        // Tokenize input
        let inputIds = tokenize(prompt: prompt)

        print("Orpheus inputIDs: \(inputIds)")

        // Generate tokens using MLX
        let generatedTokens = generateTokens(inputIds: inputIds, temperature: temperature, topP: topP)

        print("Orpheus generatedTokens: \(generatedTokens)")

        // Convert tokens to audio using SNAC decoder
        let codeLists = parseOutput(tokens: generatedTokens)

        print("Code lists: \(codeLists)")

        // Clear cache after each chunk to avoid memory leaks
        MLX.clearCache()

        return MLXArray([])
//        let audio = snacDecoder.decode(codes: codeLists)
//        return audio
    }

    private func tokenize(prompt: String) -> MLXArray {
        // TODO: Implement proper tokenization
        // For now, just convert string to array of ASCII values
        let tokens = prompt.utf8.map { Int($0) }
        return MLXArray(tokens)
    }

    private func generateTokens(inputIds: MLXArray, temperature: Float, topP: Float) -> [Int] {
        var currentIds = inputIds
        var generatedTokens: [Int] = []

        for _ in 0..<Constants.maxTokenCount {
            // Get next token prediction
            let logits = forward(inputIds: currentIds)
            let nextToken = sampleNextToken(logits: logits, temperature: temperature, topP: topP)

            // Check for end token
            if nextToken == Constants.endToken {
                break
            }

            generatedTokens.append(nextToken)
            // Reshape arrays to 2D before concatenation
            let currentIds2D = currentIds.reshaped([1, -1])
            let nextToken2D = MLXArray([nextToken]).reshaped([1, 1])
            currentIds = MLX.concatenated([currentIds2D, nextToken2D], axis: 1)
        }

        return generatedTokens
    }

    private func rmsNorm(x: MLXArray, weight: MLXArray) -> MLXArray {
        let variance = MLX.mean(MLX.square(x), axis: -1, keepDims: true)
        let normalized = x / MLX.sqrt(variance + 1e-5)
        return normalized * weight
    }

    private func forward(inputIds: MLXArray) -> MLXArray {
        print("Input shape: \(inputIds.shape)")

        // Model configuration (from config.json)
        let hiddenSize = 3072
        let intermediateSize = 8192
        let numAttentionHeads = 24
        let vocabSize = 156940
        let numLayers = 32 // Assuming, verify if different

        // 1. Embed tokens
        guard let embeddings = weights["model.embed_tokens.weight"] else {
            print("ERROR: Embedding weights not found.")
            return MLXArray([])
        }
        print("Embeddings weight shape: \(embeddings.shape)") // Expect [vocabSize, embeddingDim]
        guard embeddings.shape[0] == vocabSize else {
            print("ERROR: Embedding vocab size mismatch. Expected \(vocabSize), got \(embeddings.shape[0])")
            return MLXArray([])
        }
        let embeddingDim = embeddings.shape[1]
        print("Detected Embedding Dim: \(embeddingDim)")

        var x = embeddings[inputIds]
        print("Embedded shape: \(x.shape)") // Expect [SeqLen, embeddingDim]

        // Check if projection is needed
        if embeddingDim != hiddenSize {
             print("ERROR: Embedding dimension (\(embeddingDim)) does not match hidden size (\(hiddenSize)).")
             print("This model structure requires an input projection layer, but no weight was found.")
             return MLXArray([])
             // If you find the projection weight (e.g., 'model.input_proj.weight'), add it here:
             // guard let projWeight = weights["model.input_proj.weight"] else { print("ERROR: Input projection weight not found"); return MLXArray([]) }
             // print("Applying input projection. Weight shape: \(projWeight.shape)") // Expect [hiddenSize, embeddingDim]
             // x = linear(x: x, weight: projWeight)
             // print("After input projection: \(x.shape)") // Expect [SeqLen, hiddenSize]
        }

        // 2. Process through transformer layers
        for i in 0..<numLayers {
            print("\nLayer \(i):")

            // Input RMSNorm
            guard let inputNormWeight = weights["model.layers.\(i).input_layernorm.weight"] else {
                print("ERROR: Layer \(i) input norm weight not found."); return MLXArray([])
            }
            guard inputNormWeight.shape == [hiddenSize] else {
                print("ERROR: Layer \(i) input norm weight shape mismatch. Expected [\(hiddenSize)], got \(inputNormWeight.shape)")
                return MLXArray([])
            }
            let normedX = rmsNorm(x: x, weight: inputNormWeight)
            print("After input norm: \(normedX.shape)")

            // Self attention
            guard let qWeight = weights["model.layers.\(i).self_attn.q_proj.weight"],
                  let kWeight = weights["model.layers.\(i).self_attn.k_proj.weight"],
                  let vWeight = weights["model.layers.\(i).self_attn.v_proj.weight"],
                  let oWeight = weights["model.layers.\(i).self_attn.o_proj.weight"] else {
                print("ERROR: Layer \(i) attention weights missing."); return MLXArray([])
            }

            // Weights loaded from safetensors for MLX linear layers are typically [output_features, input_features]
            let expectedAttnWeightShape = [hiddenSize, hiddenSize]
            guard qWeight.shape == expectedAttnWeightShape,
                  kWeight.shape == expectedAttnWeightShape,
                  vWeight.shape == expectedAttnWeightShape,
                  oWeight.shape == expectedAttnWeightShape else {
                 print("ERROR: Layer \(i) attention weight dimensions mismatch. Expected \(expectedAttnWeightShape)")
                 print("Q shape: \(qWeight.shape), K shape: \(kWeight.shape), V shape: \(vWeight.shape), O shape: \(oWeight.shape)")
                 return MLXArray([])
            }

            let q = linear(x: normedX, weight: qWeight)
            let k = linear(x: normedX, weight: kWeight)
            let v = linear(x: normedX, weight: vWeight)
            print("Q shape: \(q.shape), K shape: \(k.shape), V shape: \(v.shape)") // Expect [SeqLen, hiddenSize]

            let headDim = hiddenSize / numAttentionHeads
            guard headDim > 0 else { print("ERROR: Invalid head dimension"); return MLXArray([]) }
            let scale = sqrt(Float(headDim))

            // Note: Need to implement attention with multiple heads properly.
            // This is a simplified version and likely incorrect.
            // For multi-head attention, Q/K/V need reshaping before matmul.
            let scores = MLX.matmul(q, k.transposed(0, 1)) / scale
            let probs = MLX.softmax(scores, axis: -1)
            let attnOutput = MLX.matmul(probs, v)
            let attnProj = linear(x: attnOutput, weight: oWeight)
            print("Attention output shape: \(attnProj.shape)") // Expect [SeqLen, hiddenSize]

            // First residual connection
            let h = x + attnProj
            print("After attention: \(h.shape)")

            // Post attention RMSNorm
            guard let postNormWeight = weights["model.layers.\(i).post_attention_layernorm.weight"] else {
                print("ERROR: Layer \(i) post norm weight not found."); return MLXArray([])
            }
            guard postNormWeight.shape == [hiddenSize] else {
                print("ERROR: Layer \(i) post norm weight shape mismatch. Expected [\(hiddenSize)], got \(postNormWeight.shape)")
                return MLXArray([])
            }
            let normedH = rmsNorm(x: h, weight: postNormWeight)
            print("After post norm: \(normedH.shape)")

            // MLP (Llama style)
            guard let gateWeight = weights["model.layers.\(i).mlp.gate_proj.weight"],
                  let upWeight = weights["model.layers.\(i).mlp.up_proj.weight"],
                  let downWeight = weights["model.layers.\(i).mlp.down_proj.weight"] else {
                 print("ERROR: Layer \(i) MLP weights missing."); return MLXArray([])
            }

            let expectedGateUpShape = [intermediateSize, hiddenSize]
            let expectedDownShape = [hiddenSize, intermediateSize]

            guard gateWeight.shape == expectedGateUpShape,
                  upWeight.shape == expectedGateUpShape,
                  downWeight.shape == expectedDownShape else {
                  print("ERROR: Layer \(i) MLP weight dimensions mismatch.")
                  print("Gate Expected: \(expectedGateUpShape), Got: \(gateWeight.shape)")
                  print("Up Expected: \(expectedGateUpShape), Got: \(upWeight.shape)")
                  print("Down Expected: \(expectedDownShape), Got: \(downWeight.shape)")
                  return MLXArray([])
            }

            let gate = linear(x: normedH, weight: gateWeight)
            let up = linear(x: normedH, weight: upWeight)
            print("Gate shape: \(gate.shape), Up shape: \(up.shape)") // Expect [SeqLen, intermediateSize]

            // Apply SiLU to gate and multiply with up
            let gateUp = silu(gate) * up
            print("GateUp shape: \(gateUp.shape)") // Expect [SeqLen, intermediateSize]

            // Down projection
            let down = linear(x: gateUp, weight: downWeight)
            print("Down shape: \(down.shape)") // Expect [SeqLen, hiddenSize]

            // Second residual connection
            x = h + down
            print("After MLP: \(x.shape)") // Expect [SeqLen, hiddenSize]
        }

        // 3. Final RMSNorm
        guard let finalNormWeight = weights["model.norm.weight"] else {
            print("ERROR: Final norm weight not found."); return MLXArray([])
        }
        guard finalNormWeight.shape == [hiddenSize] else {
            print("ERROR: Final norm weight shape mismatch. Expected [\(hiddenSize)], got \(finalNormWeight.shape)")
            return MLXArray([])
        }
        x = rmsNorm(x: x, weight: finalNormWeight)
        print("Final shape: \(x.shape)")

        // 4. Output projection (LM Head)
        guard let lmHeadWeight = weights["lm_head.weight"] else {
            print("ERROR: LM head weight not found (and tie_word_embeddings is false)."); return MLXArray([])
        }
        print("LM Head weight shape: \(lmHeadWeight.shape)") // Expect [vocabSize, hiddenSize]

        let expectedLmHeadShape = [vocabSize, hiddenSize]
        guard lmHeadWeight.shape == expectedLmHeadShape else {
            print("ERROR: LM Head weight shape mismatch. Expected \(expectedLmHeadShape), Got \(lmHeadWeight.shape)")
            return MLXArray([])
        }

        let logits = linear(x: x, weight: lmHeadWeight)
        print("Logits shape: \(logits.shape)") // Expect [SeqLen, vocabSize]

        // Return only the logits for the last token for next token prediction
        return logits[-1, ..<vocabSize]
    }

    private func linear(x: MLXArray, weight: MLXArray) -> MLXArray {
        // For Llama, we need to handle the weight shapes correctly
        let weightShape = weight.shape
        if weightShape.count == 1 {
            // For layer norm weights
            return x * weight
        } else {
            // For linear projections, ensure correct dimensions
            let xShape = x.shape
            let xReshaped = x.reshaped([-1, xShape.last!])
            let result = MLX.matmul(xReshaped, weight.transposed(0, 1))
            return result.reshaped(xShape.dropLast() + [weightShape[0]])
        }
    }

    private func silu(_ x: MLXArray) -> MLXArray {
        return x * MLX.sigmoid(x)
    }

    private func sampleNextToken(logits: MLXArray, temperature: Float, topP: Float) -> Int {
        // Apply temperature
        let scaledLogits = logits / temperature

        // Convert to probabilities
        let probs = MLX.softmax(scaledLogits, axis: -1)

        // Sample from distribution
        let r = MLXRandom.uniform(0.0..<1.0)
        var cumulativeProb: Float = 0.0
        for i in 0..<probs.shape[0] {
            cumulativeProb += probs[i].item()
            if r.item() < cumulativeProb {
                return i
            }
        }

        return probs.shape[0] - 1 // Fallback to last token if no sample found
    }

    private func parseOutput(tokens: [Int]) -> [[Int]] {
        // Parse tokens into code lists for SNAC decoder
        var codeLists: [[Int]] = []
        var currentList: [Int] = []

        for token in tokens {
            if token == Constants.audioStartToken {
                currentList = []
            } else if token == Constants.audioEndToken {
                if !currentList.isEmpty {
                    codeLists.append(currentList)
                }
            } else {
                currentList.append(token)
            }
        }

        return codeLists
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
    }
}

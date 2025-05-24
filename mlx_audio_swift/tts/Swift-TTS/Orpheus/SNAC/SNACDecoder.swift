import Foundation
import MLX
import MLXNN

class SNACDecoder {
    private let weights: [String: MLXArray]
    private let config: SNACConfig
    
    // Decoder Layers
    let initialConv: (MLXArray) -> MLXArray
    let decoderBlocks: [DecoderBlock]
    let finalSnakeAlpha: MLXArray
    let finalConv: WNConv1d
    
    init(config: SNACConfig) {
        self.weights = SNACDecoder.loadWeights()
        self.config = config
        
        // 1. Initial Convolution Layer
        let dwWeightG = weights["decoder.model.layers.0.weight_g"]!
        let dwWeightV = weights["decoder.model.layers.0.weight_v"]!
        let dwBias = weights["decoder.model.layers.0.bias"]
        let depthwiseConv = WNConv1d(
            inChannels: config.latentDim,
            outChannels: config.latentDim,
            kernelSize: 7,
            padding: 3,
            groups: config.latentDim
        )
        depthwiseConv.weightG = dwWeightG
        depthwiseConv.weightV = dwWeightV
        depthwiseConv.bias = dwBias

        let pwWeightG = weights["decoder.model.layers.1.weight_g"]!
        let pwWeightV = weights["decoder.model.layers.1.weight_v"]!
        let pwBias = weights["decoder.model.layers.1.bias"]
        let pointwiseConv = WNConv1d(
            inChannels: config.latentDim,
            outChannels: config.decoderDim,
            kernelSize: 1,
            padding: 0
        )
        pointwiseConv.weightG = pwWeightG
        pointwiseConv.weightV = pwWeightV
        pointwiseConv.bias = pwBias

        self.initialConv = { (x: MLXArray) -> MLXArray in
            let y = depthwiseConv(x)
            return pointwiseConv(y)
        }

        // 2. Setup Decoder Blocks
        var blocks: [DecoderBlock] = []
        var currentDim = config.decoderDim // Starts with the output dim of the initial conv
        for i in 0..<config.decoderRates.count {
            let inputDim = currentDim
            let outputDim = config.decoderDim / Int(pow(2.0, Double(i + 1)))
            let stride = config.decoderRates[i]
            let groups = config.depthwise ? outputDim : 1 // Groups based on depthwise config
            let basePath = "decoder.model.layers.\(i + 2).block.layers"
            
            let block = DecoderBlock(
                inputDim: inputDim,
                outputDim: outputDim,
                stride: stride,
                noise: config.noise,
                groups: groups,
                weights: weights,
                basePath: basePath
            )
            blocks.append(block)
            currentDim = outputDim // Update dimension for the next block's input
        }
        self.decoderBlocks = blocks
        
        // 3. Final Layers
        let finalBlockIndex = 6 // for decoder.model.layers.6.alpha
        self.finalSnakeAlpha = weights["decoder.model.layers.\(finalBlockIndex).alpha"]!
        let finalConvWeightG = weights["decoder.model.layers.\(finalBlockIndex + 1).weight_g"]!
        let finalConvWeightV = weights["decoder.model.layers.\(finalBlockIndex + 1).weight_v"]!
        let finalConvBias = weights["decoder.model.layers.\(finalBlockIndex + 1).bias"]
        self.finalConv = WNConv1d(
            inChannels: currentDim,
            outChannels: 1,
            kernelSize: 7,
            padding: 3
        )
        self.finalConv.weightG = finalConvWeightG
        self.finalConv.weightV = finalConvWeightV
        self.finalConv.bias = finalConvBias
        
        // 4. Tanh is applied in the decode function
    }
    
    static func loadWeights() -> [String: MLXArray] {
        let filePath = Bundle.main.path(forResource: "snac_model", ofType: "safetensors")!
        
        if !FileManager.default.fileExists(atPath: filePath) {
            fatalError("SNAC: Weights not found at \(filePath)")
        }
        
        do {
            let weights = try MLX.loadArrays(url: URL(fileURLWithPath: filePath))
//            print("--- SNAC Weight Keys ---")
//            for key in weights.keys.sorted() {
//                let arr = weights[key]!
//                print("\(key): shape=\(arr.shape)")
//            }
//            print("------------------------")
            return weights
        } catch {
            print("SNAC: Error loading weights: \(error)")
            return [:]
        }
    }
    
    static func loadConfig() -> SNACConfig? {
        guard let filePath = Bundle.main.path(forResource: "snac_config", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("SNAC: Config not found or invalid")
            return nil
        }
        
        return SNACConfig(
            samplingRate: json["sampling_rate"] as? Int ?? 24000,
            encoderDim: json["encoder_dim"] as? Int ?? 48,
            encoderRates: json["encoder_rates"] as? [Int] ?? [2, 4, 8, 8],
            decoderDim: json["decoder_dim"] as? Int ?? 1024,
            decoderRates: json["decoder_rates"] as? [Int] ?? [8, 8, 4, 2],
            attnWindowSize: json["attn_window_size"] as? Int,
            codebookSize: json["codebook_size"] as? Int ?? 4096,
            codebookDim: json["codebook_dim"] as? Int ?? 8,
            vqStrides: json["vq_strides"] as? [Int] ?? [4, 2, 1],
            noise: json["noise"] as? Bool ?? true,
            depthwise: json["depthwise"] as? Bool ?? true,
            latentDim: json["latent_dim"] as? Int
        )
    }
    
    func decode(codes: [[Int]]) -> MLXArray {        
        // 1. Convert codes to embeddings
        var x = embedCodes(codes: codes)
        
        // 2. Apply Initial Convolution
        x = initialConv(x)
        
        // 3. Apply Decoder Blocks sequentially
        for block in decoderBlocks {
            x = block(x)
        }
        
        // 4. Apply Final Snake Activation
        if x.shape[1] != finalSnakeAlpha.shape[1] && x.shape[2] == finalSnakeAlpha.shape[1] {
            x = x.transposed(axes: [0, 2, 1])
        }
        x = SNACDecoder.snake(x, alpha: finalSnakeAlpha)
        
        // 5. Apply Final Convolution
        if x.shape[1] != finalConv.weightV.shape[2] && x.shape[2] == finalConv.weightV.shape[2] {
            x = x.transposed(axes: [0, 2, 1])
        }
        x = finalConv(x)
        
        // 6. Apply Final Tanh Activation
        x = MLX.tanh(x)
                
        // Normalize audio.  Python doesn't do this, but seems nice....
//        let xSqueezed = x.squeezed()
//        let floatArray: [Float] = xSqueezed.asArray(Float.self)
//        let maxAbs = floatArray.map { abs($0) }.max() ?? 1.0
//        let normFactor: Float = maxAbs > 0 ? 0.99 / maxAbs : 1.0
//        let normalizedArray = floatArray.map { $0 * normFactor }
        
        return x
    }
    
    private func embedCodes(codes: [[Int]]) -> MLXArray {
        // Determine D_input (e.g., 768) and max_expanded_length
        let D_input = config.latentDim
        var max_expanded_length = 0
        
        // Calculate max_expanded_length based on codes and strides
        for i in 0..<config.vqStrides.count {
            if i < codes.count && !codes[i].isEmpty {
                max_expanded_length = max(max_expanded_length, codes[i].count * config.vqStrides[i])
            }
        }
        
        if max_expanded_length == 0 && !codes.isEmpty {
            print("Warning: max_expanded_length is 0, but codes exist. This might happen if all code chunks are empty.")
        } else if max_expanded_length == 0 {
            print("Warning: max_expanded_length is 0 (no codes). Decoder might not produce output.")
        }
        
        // Initialize accumulator
        var z_q_sum = MLXArray.zeros([D_input, max_expanded_length])
        
        // Process each quantizer
        for i in 0..<config.vqStrides.count {
            guard i < codes.count else {
                print("Warning: Not enough code layers for VQ \(i). Skipping.")
                continue
            }
            
            let currentCodes = codes[i]
            if currentCodes.isEmpty {
                print("Warning: Empty codes for VQ \(i). Skipping.")
                continue
            }        
            
            // 1. Get codebook weights and decode codes
            guard let codeEmbeddings = weights["quantizer.quantizers.\(i).codebook.weight"] else {
                print("Error: quantizer.quantizers.\(i).codebook.weight not found")
                continue
            }
            
            // Convert codes to embeddings
            let codeIndices = MLXArray(currentCodes)
            let decoded_z_p_i = codeEmbeddings[codeIndices]  // Shape: [num_codes, codebook_dim]
            
            // 2. Apply output projection
            guard let outProjWeightG = weights["quantizer.quantizers.\(i).out_proj.weight_g"],
                  let outProjWeightV = weights["quantizer.quantizers.\(i).out_proj.weight_v"],
                  let outProjBias = weights["quantizer.quantizers.\(i).out_proj.bias"] else {
                print("Error: quantizer.quantizers.\(i) output projection weights not found")
                continue
            }
            
            // Compute normalized weight for output projection
            let weightG = outProjWeightG.squeezed()  // Shape: [D_input]
            let weightV = outProjWeightV.squeezed()  // Shape: [D_input, codebook_dim]
            
            // Compute norm of weightV along codebook_dim axis
            let normV = MLX.sqrt(MLX.sum(weightV * weightV, axis: 1, keepDims: true))  // Shape: [D_input, 1]
            
            // Compute effective weight: weightG * weightV / normV
            let effectiveWeight = weightG.reshaped([D_input, 1]) * weightV / (normV + 1e-12)  // Shape: [D_input, codebook_dim]
            
            // Apply projection: decoded_z_p_i @ effectiveWeight.T + bias
            let projected_z_q_i = MLX.matmul(decoded_z_p_i, effectiveWeight.transposed()) + outProjBias  // Shape: [num_codes, D_input]
            
            // Transpose to match expected shape [D_input, num_codes]
            let projected_z_q_i_t = projected_z_q_i.transposed()
            
            // 3. Handle stride expansion
            var expanded_z_q_i = projected_z_q_i_t
            let current_stride = config.vqStrides[i]
            
            if current_stride > 1 {
                let timesteps_before_stride = projected_z_q_i_t.shape[1]
                let expanded_len = timesteps_before_stride * current_stride
                
                // Create expanded tensor
                let expanded = MLXArray.zeros([D_input, expanded_len])
                
                // Fill expanded tensor with repeated values
                for t in 0..<timesteps_before_stride {
                    let val = projected_z_q_i_t[0..<D_input, t]
                    for s in 0..<current_stride {
                        expanded[0..<D_input, t * current_stride + s] = val
                    }
                }
                
                expanded_z_q_i = expanded
            }
            
            // 4. Add to running sum
            if expanded_z_q_i.shape == z_q_sum.shape {
                z_q_sum = z_q_sum + expanded_z_q_i
            } else {
                print("Warning: Shape mismatch for VQ \(i). Expected \(z_q_sum.shape), got \(expanded_z_q_i.shape)")
            }
        }
        
        return z_q_sum
    }
       
    static public func snake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
        // Snake requires x: [batch, channels, time]
        let x_permuted = x //.transposed(axes: [0, 2, 1])
        let alphaReshaped = alpha.reshaped([1, alpha.shape[1], 1]) // [1, C, 1]
        let sinSquared = MLX.pow(MLX.sin(alphaReshaped * x_permuted), 2)
        let term = (1.0 / (alphaReshaped + 1e-9)) * sinSquared
        let out = x_permuted + term
        
        // Permute back to [batch, time, channels]
        return out.transposed(axes: [0, 2, 1])
    }
}

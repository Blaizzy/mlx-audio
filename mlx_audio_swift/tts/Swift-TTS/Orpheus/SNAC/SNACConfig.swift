//
//  SNACConfig.swift
//  Swift-TTS
//
//  Created by Ben Harraway on 14/05/2025.
//
import Foundation

struct SNACConfig {
    let samplingRate: Int
    let encoderDim: Int
    let encoderRates: [Int]
    let decoderDim: Int
    let decoderRates: [Int]
    let attnWindowSize: Int?
    let codebookSize: Int
    let codebookDim: Int
    let vqStrides: [Int]
    let noise: Bool
    let depthwise: Bool
    let latentDim: Int
    
    init(
        samplingRate: Int = 24000,
        encoderDim: Int = 48,
        encoderRates: [Int] = [2, 4, 8, 8],
        decoderDim: Int = 1024,
        decoderRates: [Int] = [8, 8, 4, 2],
        attnWindowSize: Int? = nil,
        codebookSize: Int = 4096,
        codebookDim: Int = 8,
        vqStrides: [Int] = [4, 2, 1],
        noise: Bool = true,
        depthwise: Bool = true,
        latentDim: Int? = nil
    ) {
        self.samplingRate = samplingRate
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.attnWindowSize = attnWindowSize
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.vqStrides = vqStrides
        self.noise = noise
        self.depthwise = depthwise
        
        // Calculate latentDim if not provided
        if let latentDim = latentDim {
            self.latentDim = latentDim
        } else {
            self.latentDim = encoderDim * Int(pow(2.0, Double(encoderRates.count)))
        }
    }
}

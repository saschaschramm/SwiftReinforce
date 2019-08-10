//
//  Reinforce.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 10.08.19.
//  Copyright © 2019 Sascha Schramm. All rights reserved.
//

import Foundation
import TensorFlow

var generator = ARC4RandomNumberGenerator(seed: 0)

struct Model: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    var layer1: Dense<Float>
    var layer2: Dense<Float>
    init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        layer1 = Dense<Float>(inputSize: inputSize, outputSize: hiddenSize,  activation: relu, generator: &generator)
        layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: outputSize, activation: softmax, generator: &generator)
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: layer1, layer2)
    }
}

class Reinforce {
    let batchSize: Int
    let actionSpace: Int
    let observationSpace: Int
    var model: Model
    var optimizer: AnyObject!
    
    init(batchSize: Int, observationSpace: Int, actionSpace: Int, model: Model, optimizer: AnyObject) {
        self.batchSize = batchSize
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.optimizer = optimizer
        self.model = model
    }
    
    private func sample(_ probs: Tensor<Float>) -> Int32 {
        let randomUniform = Tensor<Float>(randomUniform: probs.shape, generator: &generator)
        let scaledRandomUniform = log(randomUniform) / probs
        return scaledRandomUniform.argmax(squeezingAxis: 1).scalarized()
    }
    
    func predict(_ observation: Int32) -> Int32  {
        Context.local.learningPhase = .inference
        let observationsTensor = Tensor<Float>(oneHotAtIndices: Tensor([observation]), depth: observationSpace)
        let probs = model(observationsTensor)
        return sample(probs)
    }
    
    func predict(_ observation: [Float]) -> Int32  {
        Context.local.learningPhase = .inference
        let observationTensor = Tensor<Float>(shape: [1, observationSpace], scalars:observation)
        let probs = model(observationTensor)
        return sample(probs)
    }
    
    private func compute(observationTensor: Tensor<Float>, rewards: [Float], actions: [Int32]) {
        Context.local.learningPhase = .training
        let gradients = model.gradient { model -> Tensor<Float> in
            let rewardsTensor = Tensor<Float>(shape: [self.batchSize, 1], scalars:rewards)
            let actionProbs = model(observationTensor)
            let tensorActions = Tensor<Int32>(shape: [self.batchSize], scalars: actions)
            let actionMask = Tensor<Float32>(oneHotAtIndices: tensorActions, depth: Int(self.actionSpace))
            let losses = rewardsTensor * (actionMask * log(actionProbs + 1e-13)).sum(alongAxes: 1)
            let loss = -losses.mean()
            return loss
        }
        
        if let optimizerSGD = optimizer as? SGD<Model> {
            optimizerSGD.update(&model.allDifferentiableVariables, along: gradients)
        } else if let optimizerRMSProb = optimizer as? RMSProp<Model> {
            optimizerRMSProb.update(&model.allDifferentiableVariables, along: gradients)
        } else {
            assertionFailure("Optimizer not implemented")
        }
    }
    
    func computeGradients(observations: [Int32], rewards: [Float], actions: [Int32]) {
        let observationTensor = Tensor<Float>(oneHotAtIndices: Tensor(observations), depth: observationSpace)
        compute(observationTensor: observationTensor, rewards: rewards, actions: actions)
    }
    
    func computeGradients(observations: [Float], rewards: [Float], actions: [Int32]) {
        let observationTensor = Tensor<Float>(shape: [self.batchSize, self.observationSpace], scalars:observations)
        compute(observationTensor: observationTensor, rewards: rewards, actions: actions)
    }
}

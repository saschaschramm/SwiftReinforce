//
//  Model.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 09.07.18.
//  Copyright © 2018 Sascha Schramm. All rights reserved.
//

import Foundation
import TensorFlow

enum Optimizer {
    case RMSProb
    case GradientDescent
}

func adjointSoftmax(_ x: Tensor<Float>, seed: Tensor<Float>) -> Tensor<Float> {
    let batchSize = Int(x.shape[0])
    let size = Int(x.shape[1])
    var jacobian = [Float](repeating: 0.0, count: size*size*batchSize)
    
    for k in 0 ..< batchSize {
        let scalars = x.array[k].scalars
        for i in 0 ..< size {
            for j in 0 ..< size {
                let index = i * size + j + k * size * size
                if i == j {
                    jacobian[index] = scalars[i] * (1-scalars[j])
                    
                } else {
                    jacobian[index] = -scalars[i] * scalars[j]
                }
            }
        }
    }
    let jacobianTensor = Tensor<Float>(shape: [Int32(batchSize), Int32(size), Int32(size)], scalars: jacobian)
    return (jacobianTensor * seed.expandingShape(at: 1)).sum(squeezingAxes: 2)
}

struct Model {
    var learningRate: Float
    var observationSpace: Int32
    var actionSpace: Int32
    var meanGradient1: Tensor<Float>
    var meanGradient2: Tensor<Float>
    var weights1: Tensor<Float>
    var weights2: Tensor<Float>
    let decay: Float = 0.90
    var optimizer: Optimizer
    
    init(observationSpace: Int32,
         actionSpace: Int32,
         learningRate: Float,
         hiddenUnits: Int32,
         optimizer: Optimizer
        ) {
        self.learningRate = learningRate
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        self.meanGradient1 = Tensor<Float>(zeros: [observationSpace, hiddenUnits])
        self.meanGradient2 = Tensor<Float>(zeros: [hiddenUnits, actionSpace])
        self.weights1 = Tensor<Float>(randomUniform: [observationSpace, hiddenUnits])/sqrtf(Float(observationSpace))
        self.weights2 = Tensor<Float>(randomUniform: [hiddenUnits, actionSpace])/sqrtf(Float(observationSpace))
        self.optimizer = optimizer
    }

    @inline(never)
    mutating func train(observations: [Int32], actions: [Int32], rewards: [Float], batchSize: Int) {
        let observationsTensor = Tensor<Float>(oneHotAtIndices: Tensor(observations), depth: observationSpace)
        train(observations: observationsTensor, actions: actions, rewards: rewards, batchSize: batchSize)
    }

    @inline(never)
    mutating func train(observations: [Float], actions: [Int32], rewards: [Float], batchSize: Int) {
        let observationsTensor = Tensor<Float>(
            shape: [Int32(batchSize), observationSpace],
            scalars: observations)
        
        train(observations: observationsTensor, actions: actions, rewards: rewards, batchSize: batchSize)
    }
    
    @inline(never)
    mutating func train(observations: Tensor<Float>,
                        actions: [Int32],
                        rewards: [Float],
                        batchSize: Int){

        let output1 = matmul(observations, weights1)
        let reluOutput = relu(output1)
        let output2 = matmul(reluOutput, weights2)
        
        let actionProbs = softmax(output2, alongAxis: 1)
        let logActionProbs = log(actionProbs)
        let actionMask = Tensor<Float>(oneHotAtIndices: Tensor(actions), depth: actionSpace)
        let selectedLogActionProbs = actionMask * logActionProbs
        let selectedLogActionProbsSum = selectedLogActionProbs.sum(squeezingAxes: 1)
        let rewardsTensor = Tensor<Float>(rewards)
        
        let losses = rewardsTensor * selectedLogActionProbsSum
        let loss = losses.mean(alongAxes: 0)
        
        let d6 = Tensor<Float>(Float(1)/Float(batchSize))
        
        let (_, d5) = #adjoint(Tensor.*)(
            rewardsTensor.expandingShape(at: 1), selectedLogActionProbsSum.expandingShape(at: 1), originalValue: losses, seed: d6
        )
        
        let d4 = Tensor<Float>(1) * d5
    
        let (_, d3) = #adjoint(Tensor.*)(
            actionMask, logActionProbs, originalValue: selectedLogActionProbs, seed: d4
        )
        
        let d2 = #adjoint(log)(
            actionProbs, originalValue: logActionProbs, seed: d3
        )
        
        let d1 = adjointSoftmax(actionProbs, seed: d2)
        
        // dOutput2/dRelu
        let (dRelu, dWeights2) = #adjoint(matmul)(
            reluOutput, weights2, originalValue: output2, seed: d1
        )
        
        let dOutput1 = #adjoint(relu)(
            output1, originalValue: reluOutput, seed: dRelu
        )
        
        // dOutput1/dWeights1
        let (_, dWeights1) = #adjoint(matmul)(
            observations, weights1, originalValue: output1, seed: dOutput1
        )
    
        if optimizer == Optimizer.GradientDescent {
            applyGradientDescent(dWeights1: dWeights1, dWeights2: dWeights2)
        } else if optimizer == Optimizer.RMSProb {
            applyRMSProb(dWeights1: dWeights1, dWeights2: dWeights2)
        }
    }
    
    @inline(never)
    mutating func applyGradientDescent(dWeights1: Tensor<Float>, dWeights2: Tensor<Float>) {
        weights1 += learningRate * dWeights1
        weights2 += learningRate * dWeights2
    }
    
    @inline(never)
    mutating func applyRMSProb(dWeights1: Tensor<Float>, dWeights2: Tensor<Float>) {
        meanGradient1 = decay * meanGradient1 + (1 - decay) * (pow(dWeights1,2))
        weights1 += learningRate * dWeights1 / (sqrt(meanGradient1) + 1e-10)
        meanGradient2 = decay * meanGradient2 + (1 - decay) * (pow(dWeights2,2))
        weights2 += learningRate * dWeights2 / (sqrt(meanGradient2) + 1e-10)
    }
    
    @inline(never)
    func predictAction(_ observation: [Float]) -> Int32 {
        let observationTensor = Tensor<Float>(
            shape: [1, observationSpace],
            scalars: observation)
        return predictAction(observationTensor)
    }
    
    @inline(never)
    func predictAction(_ observation: Int32) -> Int32 {
        let observationsTensor = Tensor<Float>(oneHotAtIndices: Tensor([observation]), depth: observationSpace)
        return predictAction(observationsTensor)
    }

    @inline(never)
    func predictAction(_ observation: Tensor<Float>) -> Int32 {
        let output1 = matmul(observation, weights1)
        let reluOutput = relu(output1)
        let output = matmul(reluOutput, weights2)
        let actionProbs = softmax(output, alongAxis: 1)
        let scaledRandomUniform = log(Tensor<Float>(randomUniform: actionProbs.shape))/actionProbs
        let action = scaledRandomUniform.argmax()
        return action
    }
}



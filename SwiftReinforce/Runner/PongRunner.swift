//
//  PongRunner.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 09.07.18.
//  Copyright © 2018 Sascha Schramm. All rights reserved.
//

import Python
import Foundation
import TensorFlow

class PongRunner {
    var env: PythonObject
    var batchSize: Int
    var discountRate: Float
    var observationSpace: Int32
    var actionSpace: Int32
    var learningRate: Float
    var timesteps: Int
    var summaryFrequency: Int
    var performanceNumEpisodes: Int
    
    func transform(_ column: UInt8) -> Float {
        switch column {
        case 144:
            return 0
        case 109:
            return 0
        case 0:
            return 0
        default:
            return 1
        }
    }
    
    init(env: PythonObject,
         observationSpace: Int32,
         actionSpace: Int32,
         timesteps: Int,
         learningRate: Float,
         discountRate: Float,
         summaryFrequency: Int,
         performanceNumEpisodes: Int,
         batchSize: Int) {
        self.env = env
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.timesteps = timesteps
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.summaryFrequency = summaryFrequency
        self.performanceNumEpisodes = performanceNumEpisodes
        self.batchSize = batchSize
    }
    
    func preprocess(_ step: PythonObject) -> [Float] {
        let numpyArray: PythonObject = step[35..<195]
        let observation: [UInt8] = Array<UInt8>(numpyArray: numpyArray)!
        
        var preprocessedImage = [Float](repeating: 0, count: 80 * 80)
        for i in stride(from: 0, to: 160, by: 2) {
            for j in stride(from: 0, to: 160, by: 2) {
                let index = (i * 160 + j)
                let oldIndex = index * 3
                let newIndex = i/2 * 80 + j/2
                preprocessedImage[newIndex] = transform(observation[oldIndex])
            }
        }
        return preprocessedImage
    }
    
    func run() {
        var observations: [[Float]] = []
        var rewards: [Float] = []
        var actions: [Int32] = []
        var dones: [Bool] = []
        
        var observation: PythonObject = env.reset()
        var statsRecorder = StatsRecorder(summaryFrequency: summaryFrequency,
                                      performanceNumEpisodes: performanceNumEpisodes)
        
        var network = Model(observationSpace:observationSpace,
                                  actionSpace: actionSpace,
                                  learningRate: learningRate,
                                  hiddenUnits: 256,
                                  optimizer: .RMSProb)
        
        for t in 0 ..< timesteps + 1 {            
            observations.append(preprocess(observation))
            let actionIndex = network.predictAction(observations.last!)
            var action: Int32 = 0
            
            if actionIndex == 0 {
                action = 2
            } else if actionIndex == 1 {
                action = 3
            }
            
            let step = env.step(action)
            observation = step[0]
            let done = Bool(step[2])!
            let reward = Float(step[1])!
            
            statsRecorder.afterStep(reward: Double(reward), done: done, t: t)
            rewards.append(reward)
            dones.append(done)
            actions.append(actionIndex)
    
            if t % batchSize == 0 && t > 0 {
                let discountedRewards = discount(rewards: rewards, dones: dones, discountRate: discountRate)
                                
                network.train(observations: observations.flatMap({$0}),
                              actions: actions,
                              rewards: discountedRewards,
                              batchSize: rewards.count
                            )
                rewards = []
                observations = []
                actions = []
                dones = []
            }
            
            if done {
                observation = env.reset()
            }
        }
    }
}

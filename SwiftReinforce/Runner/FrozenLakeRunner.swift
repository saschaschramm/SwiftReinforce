//
//  FrozenLakeRunner.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 08.07.18.
//  Copyright © 2018 Sascha Schramm. All rights reserved.
//

import Foundation
import Python
import TensorFlow

class FrozenLakeRunner {
    var timesteps: Int
    var network: Model
    var statsRecorder: StatsRecorder
    var discountRate: Float
    var env: PythonObject
    var batchSize: Int
    var observationSpace: Int32
    
    init(env: PythonObject,
         observationSpace: Int32,
         actionSpace: Int32,
         timesteps: Int,
         learningRate: Float,
         discountRate: Float,
         summaryFrequency: Int,
         performanceNumEpisodes: Int,
         batchSize: Int
        ) {
        
        self.discountRate = discountRate
        statsRecorder = StatsRecorder(summaryFrequency: summaryFrequency,
                                      performanceNumEpisodes: performanceNumEpisodes)
        
        self.timesteps = timesteps
        network = Model(observationSpace: Int32(observationSpace),
                                    actionSpace: Int32(actionSpace),
                                    learningRate: learningRate,
                                    hiddenUnits:128,
                                    optimizer: .GradientDescent)
        self.env = env
        self.batchSize = batchSize
        self.observationSpace = observationSpace
    }
    
    func run() {
        var observations: [Int32] = []
        var rewards: [Float] = []
        var actions: [Int32] = []
        var dones: [Bool] = []
        var observation = Int32(env.reset())!
        
        for t in 0 ..< timesteps+1 {
            observations.append(observation)
            let action = network.predictAction(observation)
           
            let next_step = env.step(action)
            observation = Int32(next_step[0])!
            let reward = Float(next_step[1])!
            let done = Bool(next_step[2])!
            
            statsRecorder.afterStep(reward: Double(reward), done: done, t: t)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            
            if t % batchSize == 0 && t > 0 {
                let discountedRewards = discount(rewards: rewards, dones: dones, discountRate: discountRate)
                                
                network.train(observations: observations,
                              actions: actions,
                              rewards: discountedRewards,
                              batchSize: rewards.count)
                rewards = []
                observations = []
                actions = []
                dones = []
            }
            
            if done {
                observation = Int32(env.reset())!
            }
        }
    }
}

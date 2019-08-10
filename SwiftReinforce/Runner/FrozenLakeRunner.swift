//
//  FrozenLakeRunner.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 10.08.19.
//  Copyright © 2019 Sascha Schramm. All rights reserved.
//

import Foundation
import Python
import TensorFlow

class FrozenLakeRunner {
    var timesteps: Int
    var statsRecorder: StatsRecorder
    var discountRate: Float
    var env: PythonObject
    var reinforce: Reinforce
    
    init(env: PythonObject,
         observationSpace: Int,
         actionSpace: Int,
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
        self.env = env
        
        let model = Model(inputSize: observationSpace,
                          hiddenSize: 128,
                          outputSize: actionSpace)
        
        self.reinforce = Reinforce(batchSize: batchSize,
                                   observationSpace: observationSpace,
                                   actionSpace: actionSpace,
                                   model: model,
                                   optimizer: SGD(for: model, learningRate: learningRate)
        )
    }
    
    func run() {
        var observations: [Int32] = []
        var rewards: [Float] = []
        var actions: [Int32] = []
        var terminals: [Bool] = []
        var observation = Int32(env.reset())!
        
        for t in 0 ..< timesteps+1 {
            observations.append(observation)
            
            let action = reinforce.predict(Int32(observation))
            let nextStep = env.step(action)
            observation = Int32(nextStep[0])!
            let reward = Float(nextStep[1])!
            let terminal = Bool(nextStep[2])!
            statsRecorder.afterStep(reward: Double(reward), terminal: terminal, t: t)
            rewards.append(reward)
            actions.append(action)
            terminals.append(terminal)
            
            if observations.count == reinforce.batchSize {
                let discountedRewards = discount(rewards: rewards,
                                                 terminals: terminals,
                                                 discountRate: discountRate)
                
                reinforce.computeGradients(observations: observations, rewards: discountedRewards, actions: actions)
                rewards = []
                observations = []
                actions = []
                terminals = []
            }
            
            if terminal {
                observation = Int32(env.reset())!
            }
        }
    }
}

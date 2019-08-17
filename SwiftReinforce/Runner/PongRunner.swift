//
//  PongRunner.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 10.08.19.
//  Copyright © 2019 Sascha Schramm. All rights reserved.
//

import Python
import Foundation
import TensorFlow

class PongRunner {
    var env: EnvWrapper
    var discountRate: Float
    var timesteps: Int
    var summaryFrequency: Int
    var performanceNumEpisodes: Int
    var reinforce: Reinforce<RMSProp<Model>>!
    
    init(env: EnvWrapper,
         observationSpace: Int,
         actionSpace: Int,
         timesteps: Int,
         learningRate: Float,
         discountRate: Float,
         summaryFrequency: Int,
         performanceNumEpisodes: Int,
         batchSize: Int) {
        self.env = env
        self.timesteps = timesteps
        self.discountRate = discountRate
        self.summaryFrequency = summaryFrequency
        self.performanceNumEpisodes = performanceNumEpisodes
        
        let model = Model(inputSize: observationSpace,
                          hiddenSize: 256,
                          outputSize: actionSpace)
        
        self.reinforce = Reinforce(batchSize: batchSize,
                                   observationSpace: observationSpace,
                                   actionSpace: actionSpace,
                                   model: model,
                                   optimizer: RMSProp(for: model, learningRate: learningRate))
    }
    
    func run() {
        var observations: [[Float]] = []
        var rewards: [Float] = []
        var actions: [Int32] = []
        var terminals: [Bool] = []
        var observation: [Float] = env.reset()
        var statsRecorder = StatsRecorder(summaryFrequency: summaryFrequency,
                                          performanceNumEpisodes: performanceNumEpisodes)
        
        for t in 0 ..< timesteps + 1 {
            observations.append(observation)
            let actionIndex = reinforce.predict(observation)
            
            let step = env.step(actionIndex)
            observation = step.observation
            let reward = step.reward
            let terminal = step.terminal
            
            statsRecorder.afterStep(reward: Double(reward), terminal: terminal, t: t)
            rewards.append(reward)
            terminals.append(terminal)
            actions.append(actionIndex)
            
            if observations.count == reinforce.batchSize {
                let discountedRewards = discount(rewards: rewards,
                                                 terminals: terminals,
                                                 discountRate: discountRate)
                reinforce.train(observations: observations.flatMap({$0}), rewards: discountedRewards, actions: actions)
                rewards = []
                observations = []
                actions = []
                terminals = []
            }
            
            if terminal {
                observation = env.reset()
            }
        }
    }
}

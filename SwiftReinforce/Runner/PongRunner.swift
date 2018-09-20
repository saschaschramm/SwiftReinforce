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
    var env: EnvWrapper
    var batchSize: Int
    var discountRate: Float
    var observationSpace: Int32
    var actionSpace: Int32
    var learningRate: Float
    var timesteps: Int
    var summaryFrequency: Int
    var performanceNumEpisodes: Int
    
    init(env: EnvWrapper,
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
    
    func run() {
        var observations: [[Float]] = []
        var rewards: [Float] = []
        var actions: [Int32] = []
        var terminals: [Bool] = []
        var observation: [Float] = env.reset()
        var statsRecorder = StatsRecorder(summaryFrequency: summaryFrequency,
                                          performanceNumEpisodes: performanceNumEpisodes)
        
        var model = Model(observationSpace:observationSpace,
                            actionSpace: actionSpace,
                            learningRate: learningRate,
                            hiddenUnits: 256,
                            optimizer: .RMSProb)
        
        for t in 0 ..< timesteps + 1 {
            observations.append(observation)
            let actionIndex = model.predictAction(observations.last!)
            let step = env.step(actionIndex)
            observation = step.observation
            let reward = step.reward
            let terminal = step.terminal

            statsRecorder.afterStep(reward: Double(reward), terminal: terminal, t: t)
            rewards.append(reward)
            terminals.append(terminal)
            actions.append(actionIndex)
            
            if (t % batchSize == 0) && (t > 0) {
                let discountedRewards = discount(rewards: rewards,
                                                 terminals: terminals,
                                                 discountRate: discountRate)
                
                model.train(observations: observations.flatMap({$0}),
                              actions: actions,
                              rewards: discountedRewards,
                              batchSize: rewards.count
                )
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

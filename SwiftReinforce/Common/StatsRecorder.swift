//
//  StatsRecorder.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 08.07.18.
//  Copyright © 2018 Sascha Schramm. All rights reserved.
//

import Foundation

struct StatsRecorder {
    var totalRewards: [Double] = []
    let summaryFrequency: Int
    let performanceNumEpisodes: Int
    var totalReward = 0.0
    var numEpisodes = 0
    var startTime: DispatchTime!

    init(summaryFrequency: Int, performanceNumEpisodes: Int) {
        self.summaryFrequency = summaryFrequency
        self.performanceNumEpisodes = performanceNumEpisodes
    }
    
    mutating func printScore(_ t: Int) {
        let endTime = DispatchTime.now()
        let elapsedTime = t == 0 ? 0.0 : Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds)/Double(1_000_000_000)
        let sum = totalRewards.suffix(performanceNumEpisodes).reduce(0, +)
        let score = sum/Double(performanceNumEpisodes)
        print(String(format: "%i %.2f %.1f", t, score, elapsedTime))
        startTime = endTime
    }
    
    mutating func afterStep(reward: Double, done: Bool, t: Int) {
        totalReward += reward
        
        if done {
            numEpisodes += 1
            totalRewards.append(totalReward)
            totalReward = 0
        }
        
        if t % summaryFrequency == 0 {
            printScore(t)
        }
    }
}

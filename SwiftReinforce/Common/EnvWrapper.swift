//
//  EnvWrapper.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 10.08.19.
//  Copyright © 2019 Sascha Schramm. All rights reserved.
//

import Foundation
import Python
import TensorFlow

enum Pixel: Int {
    case player = 92
    case enemy = 213
    case ball = 236
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
            let pixel = observation[oldIndex]
            
            if ((pixel == Pixel.player.rawValue) ||
                (pixel == Pixel.enemy.rawValue) ||
                (pixel == Pixel.ball.rawValue))
            {
                preprocessedImage[newIndex] = 1
            }
        }
    }
    return preprocessedImage
}

class EnvWrapper {
    var env: PythonObject
    let frameSkip = 4
    let inSize: Int = 80 * 80
    var observationBuffer: [[Float]]!
    
    init() {
        let gym = importGym()
        env = gym.make("PongNoFrameskip-v4")
        env.seed(0)
        
        observationBuffer = [
            [Float](repeating: 0, count: inSize),
            [Float](repeating: 0, count: inSize)
        ]
    }
    
    func reset() -> [Float] {
        env.reset()
        
        var step = env.step(1)
        var terminal = Bool(step[2])!
        
        if terminal {
            env.reset()
        }
        
        step = env.step(2)
        let observation = step[0]
        terminal = Bool(step[2])!
        
        if terminal {
            env.reset()
        }
        
        self.observationBuffer[0] = preprocess(observation)
        self.observationBuffer[1] = [Float](repeating: 0, count: inSize)
        return maxMerge()
    }
    
    private func maxMerge() -> [Float] {
        assert(frameSkip > 1)
        return zip(observationBuffer[0], observationBuffer[1]).map { max($0, $1) }
    }
    
    public func step(_ actionIndex: Int32) -> (observation: [Float], reward: Float, terminal: Bool) {
        var totalReward: Float = 0.0
        var terminal = false
        
        var action: Int32 = 0
        if actionIndex == 0 {
            action = 2
        } else if actionIndex == 1 {
            action = 3
        }
        
        for timeStep in 0 ..< frameSkip  {
            let step = env.step(action)
            let observation = step[0]
            let reward = Float(step[1])!
            terminal = Bool(step[2])!
            
            totalReward += reward
            
            if terminal {
                break
            } else if timeStep >= (frameSkip - 2) {
                let t = timeStep - (frameSkip - 2)
                self.observationBuffer[t] = preprocess(observation)
            }
        }
        let observation = maxMerge()
        return (observation, totalReward, terminal)
    }
}

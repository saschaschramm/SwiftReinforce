//
//  main.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 05.08.18.
//  Copyright © 2018 Sascha Schramm. All rights reserved.
//

import Foundation
import TensorFlow
import Python

RandomState.global.seed(with: 0)

func runFrozenLake() {
    let gym = Python.import("gym")
    gym.envs.registration.register(id:"FrozenLakeNotSlippery-v0",
                                   entry_point:"gym.envs.toy_text:FrozenLakeEnv",
                                   kwargs: ["is_slippery": false])
    
    let env = gym.make("FrozenLakeNotSlippery-v0")
    env.seed(0)
    
    let runner = FrozenLakeRunner(env: env,
                                  observationSpace: 16,
                                  actionSpace: 4,
                                  timesteps: 20000,
                                  learningRate: 0.1,
                                  discountRate: 0.99,
                                  summaryFrequency: 2000,
                                  performanceNumEpisodes: 100,
                                  batchSize: 16)
    
    let start = DispatchTime.now()
    runner.run()
    let end = DispatchTime.now()
    let time = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)/Double(1_000_000_000)
    print(time)
}

func runPong() {
    let gym = Python.import("gym")
    let sys = Python.import("sys")
    
    let path = "\(NSHomeDirectory())/gym/lib/python2.7/site-packages/"
    sys.path.append(path)
    
    let env = gym.make("Pong-v4")
    env.seed(0)
    let runner = PongRunner(env: env,
                            observationSpace: 80 * 80,
                            actionSpace: 2,
                            timesteps: Int(1e6),
                            learningRate: 0.0002,
                            discountRate: 0.99,
                            summaryFrequency: 40000,
                            performanceNumEpisodes: 10,
                            batchSize: 128)
    runner.run()
}

func main() {
    runFrozenLake()
    //runPong()
}

main()

//
//  Utilities.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 10.08.19.
//  Copyright © 2019 Sascha Schramm. All rights reserved.
//

import Foundation
import Python
import TensorFlow

func importGym() -> PythonObject {
    let sys = Python.import("sys")
    print(Python.version)
    let path = "\(NSHomeDirectory())/gym/lib/python3.6/site-packages/"
    sys.path.append(path)
    return Python.import("gym")
}

/*
func renderPixels(_ pixels: [UInt8], rows: Int, cols: Int) {
    let sys = Python.import("sys")
    let np = Python.import("numpy")
    let path = "\(NSHomeDirectory())/gym/lib/python3.6/site-packages/"
    sys.path.append(path)
    let image = Python.import("PIL.Image")
    let img = image.fromarray(np.uint8(np.array(pixels).reshape([rows,cols])))
    img.show()
}*/

func discount(rewards: [Float], terminals: [Bool], discountRate: Float) -> [Float] {
    var discounted: [Float] = []
    var totalReturn: Float = 0.0
    for (reward, terminal) in zip(rewards.reversed(), terminals.reversed()) {
        if terminal {
            totalReturn = reward
        } else {
            totalReturn = reward + discountRate * totalReturn
        }
        discounted.append(totalReturn)
    }
    return discounted.reversed()
}

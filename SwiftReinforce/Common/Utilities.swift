//
//  Utilities.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 09.07.18.
//  Copyright © 2018 Sascha Schramm. All rights reserved.
//

import Foundation
import TensorFlow
import Python

/*
@inline(never)
func adjointSoftmax2(tensor: Tensor<Float>,
                     seed: Tensor<Float>,
                     batchSize: Int,
                     observationSpace: Int32) -> Tensor<Float> {
    
    var jacobian = [Float]()
    
    for i in 0 ..< batchSize {
        let input = tensor.slice(lowerBounds: [Int32(i), 0],
                                 upperBounds: [Int32(i)+1, Int32(batchSize)])[0]
        
        let sExpanded = input.expandingShape(at: 1)
        let x1 = Raw.diag(diagonal: input)
        let x2 = matmul(sExpanded, sExpanded.transposed())
        let result = x1-x2
        jacobian += result.scalars
    }
    
    let jacobianTensor =
        Tensor<Float>(shape: [Int32(batchSize), Int32(observationSpace), Int32(observationSpace)],
                      scalars: jacobian)
    
    //return (jacobianTensor * seed.expandingShape(at: 1)).sum(squeezingAxes: 2)
    return jacobianTensor * seed.expandingShape(at: 1)
}*/

func discount(rewards: [Float], dones: [Bool], discountRate: Float) -> [Float] {
    var discounted: [Float] = []
    var totalReturn: Float = 0.0
    for (reward, done) in zip(rewards.reversed(), dones.reversed()) {
        if done {
            totalReturn = reward
        } else {
            totalReturn = reward + discountRate * totalReturn
        }
        discounted.append(totalReturn)
    }
    return discounted.reversed()
}

func renderPixels(_ pixels: [UInt8], rows: Int, cols: Int) {
    let sys = Python.import("sys")
    let np = Python.import("numpy")
    let path = "\(NSHomeDirectory())/gym/lib/python2.7/site-packages/"
    sys.path.append(path)
    let image = Python.import("PIL.Image")
    
    let foo = np.array(pixels).reshape([rows,cols])
    let img = image.fromarray(np.uint8(foo))
    img.show()
}

func printPixels(_ pixels:[UInt8], cols:Int, rows: Int) {
    var str = ""
    for i in 0 ..< rows { // row
        for j in 0 ..< cols { // column
            let index = i * cols + j
            str += "\(pixels[index]) "
        }
        str += String("\n")
    }
    print(str)
}

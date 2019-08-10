//
//  Array+NumPy.swift
//  SwiftReinforce
//
//  Created by Sascha Schramm on 09.07.18.
//  Copyright © 2018 Sascha Schramm. All rights reserved.
//

import Python

extension Array where Element == UInt8 {
    
    public init?(numpyArray: PythonObject) {
        let numpyArraySize = Int(numpyArray.size)!
        let numpyArrayPointerAddress = UInt(numpyArray.__array_interface__["data"].tuple2.0)!
        let pointer = UnsafePointer<Element>(bitPattern: numpyArrayPointerAddress)!
        let bufferPointer = UnsafeBufferPointer(start: pointer, count: numpyArraySize)
        self.init(bufferPointer)
    }
}

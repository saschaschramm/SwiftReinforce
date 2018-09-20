#  SwiftReinforce

## Introduction
Implementation of the Reinforce algorithm using Swift for Tensorflow.

## Install Swift for Tensorflow
This project is based on Swift for Tensorflow. To install Swift for Tensorflow visit [https://github.com/tensorflow/swift/](https://github.com/tensorflow/swift/blob/master/Installation.md).

The Tensorflow library for Swift is published in the [stdlib/public/TensorFlow](https://github.com/apple/swift/tree/tensorflow/stdlib/public/TensorFlow) directory.

Make sure to change the Xcode build system to legacy (File > Project Settings > Build System).

## Install OpenAI Gym
Take the following steps to install OpenAI Gym using Virtualenv:

1. Create a Virtualenv environment:
```
virtualenv --system-site-packages gym
```
2. Activate the Virtualenv environment:
```
cd gym
source ./bin/activate
```

3. Install OpenAI Gym into the active Virtualenv:
```
pip install gym[atari]
```

## Requirements
* Xcode 10.0 beta or later
* Swift for TensorFlow 2018-09-17

## Results
### FrozenLake
```
2000 0.00 1.1
4000 0.00 0.7
6000 0.05 0.7
8000 0.05 0.7
10000 0.09 0.7
12000 0.14 0.7
14000 0.31 0.7
16000 0.57 0.7
18000 0.76 0.7
20000 0.86 0.7
7.569058606
```
### Pong
```
40000 -19.90 73.0
80000 -19.60 72.6
120000 -19.40 72.5
160000 -18.80 72.6
200000 -19.30 72.7
240000 -19.60 72.6
280000 -18.30 72.7
320000 -18.00 72.6
360000 -18.90 74.1
400000 -19.10 72.7
440000 -18.00 72.6
480000 -15.60 72.5
520000 -14.90 72.6
560000 -15.60 72.5
600000 -16.30 72.7
640000 -14.80 72.8
680000 -16.10 79.8
720000 -16.20 72.7
760000 -13.50 72.6
800000 -7.90 72.5
840000 -9.20 72.6
880000 4.10 72.5
920000 8.30 72.6
960000 15.70 77.3
1000000 14.00 78.7
```

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
* Swift for TensorFlow 2018-09-05

## Results
### FrozenLake
```
0 0.00 0.0
2000 0.01 1.1
4000 0.05 0.7
6000 0.03 0.7
8000 0.05 0.7
10000 0.07 0.8
12000 0.20 0.8
14000 0.36 0.8
16000 0.62 0.8
18000 0.78 0.8
20000 0.86 0.8
8.073721319
```
### Pong
Performance after 1e6 steps:
```
0 0.00 0.0
40000 -20.40 65.4
80000 -20.20 64.4
120000 -20.30 64.4
160000 -20.10 66.8
200000 -19.30 68.0
240000 -19.10 72.4
280000 -18.70 71.1
320000 -18.50 72.4
360000 -19.30 71.9
400000 -18.30 67.1
440000 -18.30 68.6
480000 -18.30 74.1
520000 -18.70 68.2
560000 -17.80 66.1
600000 -17.50 67.4
640000 -18.00 65.4
680000 -18.40 66.7
720000 -17.40 68.2
760000 -16.60 65.7
800000 -16.10 66.4
840000 -18.00 66.4
880000 -17.30 67.8
920000 -16.60 68.1
960000 -18.30 68.6
1000000 -17.30 66.1
```

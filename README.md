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

## Results
### FrozenLake
```
0 0.00 0.0
2000 0.01 1.2
4000 0.05 1.1
6000 0.03 1.0
8000 0.05 1.0
10000 0.07 1.0
12000 0.20 1.0
14000 0.36 1.0
16000 0.62 1.0
18000 0.78 1.0
20000 0.86 1.0
10.41444578
```
### Pong
Performance after 1e6 steps:
```
0 0.00 0.0
40000 -20.40 69.4
80000 -20.20 66.1
120000 -20.30 67.4
160000 -20.10 73.3
200000 -19.30 69.4
240000 -19.10 69.6
280000 -18.70 66.7
320000 -18.50 66.8
360000 -19.30 68.4
400000 -18.30 66.1
440000 -18.30 69.2
480000 -18.30 69.2
520000 -18.70 69.6
560000 -17.80 66.0
600000 -17.50 65.4
640000 -18.00 65.3
680000 -18.40 65.5
720000 -17.40 65.7
760000 -16.60 65.3
800000 -16.10 65.4
840000 -18.00 65.3
880000 -17.30 65.3
920000 -16.60 65.7
960000 -18.30 66.4
1000000 -17.30 69.0
```

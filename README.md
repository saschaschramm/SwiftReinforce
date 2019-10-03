#  SwiftReinforce

## Introduction
Implementation of the Reinforce algorithm using Swift for Tensorflow.

## Install Swift for Tensorflow
This project is based on Swift for Tensorflow. To install Swift for Tensorflow visit [https://github.com/tensorflow/swift/](https://github.com/tensorflow/swift/blob/master/Installation.md).

The Tensorflow library for Swift is published in the [stdlib/public/TensorFlow](https://github.com/apple/swift/tree/tensorflow/stdlib/public/TensorFlow) directory.

Make sure to change the Xcode build system to legacy (File > Project Settings > Build System).

## Install OpenAI Gym
Take the following steps to install OpenAI Gym using Virtualenv:

1. System wide install
```
sudo pip3 install -U virtualenv
```

2. Create a Virtualenv environment:
```
virtualenv --system-site-packages gym
```

3. Activate the Virtualenv environment:
```
cd gym
source ./bin/activate
```

4. Install OpenAI Gym into the active Virtualenv:
```
pip3 install gym
pip3 install gym[atari]
```

## Requirements
* Xcode 11.0 or later
* Swift for TensorFlow 0.5

## Results
### FrozenLake
```
0 0.00 0.0
2000 0.01 0.8
4000 0.00 0.8
6000 0.02 0.8
8000 0.02 0.8
10000 0.02 0.8
12000 0.00 0.8
14000 0.04 0.8
16000 0.08 0.8
18000 0.14 0.8
20000 0.26 0.8
8.262566555
```

### Pong
```
0 0.00 0.0
40000 -19.70 73.3
80000 -18.00 73.3
120000 -19.30 73.5
160000 -17.70 73.4
200000 -17.40 73.3
240000 -17.40 73.3
280000 -16.90 73.2
320000 -16.90 73.2
360000 -16.40 73.3
400000 -17.30 73.2
440000 -16.30 73.3
480000 -13.60 73.2
520000 -15.20 73.2
560000 -12.60 73.2
600000 -12.30 73.2
640000 -12.60 73.3
680000 -10.00 73.5
720000 -12.10 71.8
760000 -9.70 71.8
800000 -7.50 71.9
840000 -8.40 71.9
880000 -2.50 72.1
920000 1.10 72.0
960000 2.70 71.9
1000000 -4.30 72.0
```

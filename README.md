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
* Swift for TensorFlow 2019-07-25

## Results
### FrozenLake
```
0 0.00 0.0
2000 0.02 0.6
4000 0.00 0.6
6000 0.05 0.6
8000 0.07 0.6
10000 0.18 0.6
12000 0.28 0.6
14000 0.64 0.6
16000 0.83 0.6
18000 0.92 0.6
20000 0.97 0.6
5.895307806
```

### Pong
```
0 0.00 0.0
40000 -19.20 64.9
80000 -20.60 64.9
120000 -19.20 65.1
160000 -19.00 65.0
200000 -17.80 65.1
240000 -19.30 64.9
280000 -19.30 64.8
320000 -18.60 64.8
360000 -17.80 64.8
400000 -18.00 66.2
440000 -17.70 65.1
480000 -16.00 65.0
520000 -14.60 65.6
560000 -15.10 65.0
600000 -14.70 65.0
640000 -15.50 64.9
680000 -14.70 64.9
720000 -13.00 64.9
760000 -11.90 65.0
800000 -10.80 65.0
840000 -8.80 66.0
880000 -7.50 65.2
920000 -5.70 65.3
960000 -6.60 65.0
1000000 -7.60 64.9
```

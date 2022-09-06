# Deep RL
## Overview
- This project aims at implementing and evaluating Q-Learning on a simple control task - the [CartPole](https://github.com/openai/gym/wiki/CartPole-v0) environment.
- The Q-learning algorithm and the nuts and bolts of a Deep Q-Network have been coded up and can be referred from [dqn.py](dqn.py).
- You do not require a GPU (and hence, AWS) for this task; the DQN can be trained easily within a few minutes on a decent CPU.
- While each of the runs won't take much time, depending on your implementation, you may have to play around with the parameters like learning rate, epsilon-annealing, batch size, hidden layer sizes, etc. which could take up the bulk of your time.
- The performance of the Q-network using a different set of hyperparameters is summarized in [report.pdf](report.pdf).

## Installation
- Install TensorFlow for your machine following the [documentation](https://www.tensorflow.org/install/pip#linux). We recommend using the Virtualenv installation since it provides a virtual and isolated Python environment, incapable of interfering with or being affected by other Python programs, versions, and libraries being used by your other projects on your machine.

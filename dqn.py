import gym
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from collections import deque
from keras.optimizers import adam
from keras.models import clone_model
import matplotlib.pyplot as plt
import random

print(tf.__version__)

# The below class is used to build a deep q network and learn from experiences to balance a pole on a moving cart. 
class DQN:

  def __init__(self, env, n_episodes):
    self.env = env
    self.input_size = self.env.observation_space.shape[0]
    self.output_size = self.env.action_space.n 

    # setting the hyper-parameters used in training and updating the network
    self.REPLAY_MEMORY_SIZE = 10000
    self.EPSILON = 0.6
    self.EPSILON_DECAY = 0.995
    self.EPSILON_MIN = 0.01
    self.EPISODES_NUM = n_episodes
    self.MAX_STEPS = 200
    self.MINIBATCH_SIZE = 64
    self.DISCOUNT_FACTOR = 0.99
    self.TARGET_UPDATE_FREQ = 150
    self.HIDDEN_LAYER1_SIZE = 128
    self.HIDDEN_LAYER2_SIZE = 64
    self.LEARNING_RATE = 0.0001
    self.q_net = self.build_net()
    self.target_net = clone_model(self.q_net)
    
  # The below function builds and returns a sequential model containing 2 hidden layers with tanh activation and a linear output layer   
  def build_net(self): 
    net = Sequential()
    net.add(Dense(self.HIDDEN_LAYER1_SIZE, input_shape=(self.input_size, ), activation='tanh'))
    net.add(Dense(self.HIDDEN_LAYER2_SIZE, activation='tanh'))
    net.add(Dense(self.output_size, activation='linear'))
    net.compile(loss='mse', optimizer=adam(lr=self.LEARNING_RATE))
    return net

  def update_net(self):
    
    # if buffer has minimum required amount of samples then continue to update network else return and wait till required samples are filled
    if len(self.replay_buffer) >= self.MINIBATCH_SIZE:
      
      # sample instances from the replay buffer of minibatch size  
      minibatch = random.sample(self.replay_buffer, self.MINIBATCH_SIZE) 

      # making separate arrays for current_states, actions, rewards, next_states, episode_dones from minibatch 
      curr_st = np.array([i[0] for i in minibatch]).squeeze() if self.MINIBATCH_SIZE > 1 else np.array([i[0] for i in minibatch]) 
      act = np.array([i[1] for i in minibatch])
      r = np.array([i[2] for i in minibatch])
      nxt_st = np.array([i[3] for i in minibatch]).squeeze() if self.MINIBATCH_SIZE > 1 else np.array([i[3] for i in minibatch])
      dn = np.array([i[4] for i in minibatch])

      # calculating targets based on episode done or not done
      _target = r + self.DISCOUNT_FACTOR * (np.amax(self.target_net.predict_on_batch(nxt_st), axis=1)) * (1 - dn)
      _target_all = self.q_net.predict_on_batch(curr_st)
      idx = np.array([i for i in range(self.MINIBATCH_SIZE)])
      _target_all[[idx], [act]] = _target

      self.q_net.fit(curr_st, _target_all, epochs=1, verbose=0) # fitting the model with training data -> (inputs, targets)
      
      # if epsilon is greater than minimum epsilon decrease it by a factor of epsilon decay
      if self.EPSILON > self.EPSILON_MIN:
        self.EPSILON *= self.EPSILON_DECAY
        
    else:
      return

  def train(self):
    num_episodes = self.EPISODES_NUM

    # initialize the replay buffer
    self.replay_buffer = deque(maxlen=self.REPLAY_MEMORY_SIZE)
    
    self.steps_taken = []
    self.reward_received = []
    self.last_100_avg = []
    scores = deque(maxlen=100)  # stores the rewards of last 100 episodes
    global_steps = 0

    for ep in range(num_episodes):
      ep_steps = 0
      ep_reward = 0
      curr_state = self.env.reset()
      done = False

      while not done and ep_steps < self.MAX_STEPS:

        Q = self.q_net.predict(curr_state.reshape((1, 4)))

        # choose action by eps-greedy policy
        if np.random.rand() < self.EPSILON:
          action = self.env.action_space.sample()
        else:
          action = np.argmax(Q[0])     

        next_state, reward, done, _ = self.env.step(action)
        ep_steps += 1
        global_steps += 1
        ep_reward += reward

        # fill the replay buffer with current sample
        self.replay_buffer.append((curr_state, action, reward, next_state, done))
        curr_state = next_state

        # update the q network
        self.update_net()
        
        # update the target network with given frequency
        if not global_steps % self.TARGET_UPDATE_FREQ:
          self.target_net.set_weights(self.q_net.get_weights()) 

      scores.append(ep_reward)
      avg_score = np.mean(scores)

      self.last_100_avg.append(avg_score)
      self.steps_taken.append(ep_steps)
      self.reward_received.append(ep_reward)

      if ep % 10 == 0:
        print(f"Episode: {ep}:- average reward(last 100 epsiodes) = {avg_score}")

if __name__ == '__main__':
  
  env = gym.make('CartPole-v0')
  env.seed(7)
  np.random.seed(7)
  random.seed(7)
  n_episodes = 500
  dqn = DQN(env, n_episodes)  # creating dqn class instance 
  dqn.train()  # Training starts

# Plotting the learning curve
plt.figure(figsize=(12, 7))
plt.plot(np.arange(100, n_episodes + 1), dqn.last_100_avg[99:])
plt.hlines(195, 0, n_episodes - 1)
plt.xlabel("No. of episodes", fontsize=20)
plt.ylabel("Avg Reward(last 100 episodes)", fontsize=20)
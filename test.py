#obtain distribution of observation space.

import gym
import matplotlib.pyplot as plt
import numpy as np

max_actions = 1000

bin_thetas = np.arange(-0.3,0.3,0.02)
bin_theta_nodes = np.arange(-2.5,2.5,0.2)

thetas = np.zeros(1000)
theta_nodes = np.zeros(1000)
numactions = 0

env = gym.make('CartPole-v0')
while numactions < max_actions:
    done = False
    observation = env.reset()
    while not done and numactions < max_actions:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        numactions = numactions + 1
        thetas[numactions - 1] = obs[2]
        theta_nodes[numactions - 1] = obs[3]


plt.hist(thetas, bins = bin_thetas)
plt.show()
plt.hist(theta_nodes, bins = bin_theta_nodes)
plt.show()

import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

#metadata of the environment
theta_high = 1.5
theta_max = 0.4
theta_range = theta_max*2
theta_node_max = env.observation_space.high[1]
theta_node_range = theta_node_max * 2

#discretization parameters
bin_t = float(500);
bin_extra = 10;
bin_n = float(500);

#learning parameters
max_episode = 2000
alpha = 0.5;
epsilon = 0.999
neg_reward = -10

#storing qa values
qtable = [[[0 for k in xrange(2)] for j in xrange(int(bin_n))] for i in xrange(int(bin_t+bin_extra))]

#list for storing rewards for every episode, analyze performance
rewards_list = np.zeros(max_episode)
x = np.arange(max_episode)

#compute which bin the state is in given its parameters
def compute_bin(obs):
    if(obs[0] > theta_max):
        bin1 = int(bin_t) + int((obs[0] - theta_max)/((theta_high-theta_max)/(float(bin_extra)/2))) + bin_extra/2
    elif(obs[0] < -theta_max):
        bin1 = int(bin_t) + int((-obs[0] - theta_max)/((theta_high-theta_max)/(float(bin_extra)/2)))
    else:
        bin1 = int((obs[0] + theta_max)/(theta_range/bin_t))

    bin2 = int((obs[1] + theta_node_max)/(theta_node_range/bin_n))
    return bin1,bin2

#select the policy with certain randomness
def policy_select(obs,rand_rate):
    x = random.random()
    if(x <= rand_rate):
        return env.action_space.sample()
    else:
        bin1,bin2 = compute_bin(obs)
        if(qtable[bin1][bin2][0] > qtable[bin1][bin2][1]):
            return 0
        else:
            return 1

#print(qtable)

for i_episode in range(max_episode):
    prev_obs = env.reset()
    done = False
    total_reward =  0
    times = 0

    #update learning rate and eploration rate for each episode, start exploiting more
    epsilon = epsilon / 1.0001
    alpha = alpha / 1.009

    while not done:
        env.render()
        times = times + 1
        action = policy_select(prev_obs,epsilon)
        obs, reward, done, info = env.step(action)

        if done:
            reward = neg_reward

        bin1,bin2 = compute_bin(obs)
        #the "nudge" from taking a sample:
        sample = float(reward) + max(qtable[bin1][bin2][0],qtable[bin1][bin2][1])

        bin1,bin2 = compute_bin(prev_obs)
        #update q-a value
        qtable[bin1][bin2][action] = (1-alpha)*qtable[bin1][bin2][action] + alpha * reward
        prev_obs = obs
        total_reward = total_reward + reward

        if done:
            print("Finished in {} timesteps, total reward is {}".format(times,total_reward))
            rewards_list[i_episode] = total_reward
            break

#the reward list visualized
print(epsilon)
plt.plot(x,rewards_list)
plt.show()

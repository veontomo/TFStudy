import gym
import numpy as np

env = gym.make('FrozenLake-v0')
# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
print("observation space size: " + str(env.observation_space.n))
print("action space size: " + str(env.action_space.n))
actions = env.action_space.n
space = env.observation_space.n
# Set learning parameters
lr = .8
y = .95
num_episodes = 100000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []
for i in range(num_episodes):
    s = env.reset()
    # Reset environment and get first new observation
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while not d:
        # Choose an action by greedily (with noise) picking from Q table
        qTmp = Q[s, :] + np.random.rand(1, actions) * (1.0 / (1 + i))
        #        print("qTmp: ", qTmp)
        a = np.argmax(qTmp)
        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        #        print("s1: " + str(s1) + ", reward: " + str(r) + (", is done" if d else ", not done"))
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
    rList.append(rAll)
    if i % 500 == 0:
        score = int(10000 * sum(rList) / (i + 1)) / 100
        print(str(i) + ": score = " + str(score) + "%")
print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
print(np.argmax(Q, 1))

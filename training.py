import dill
import numpy as np
import agent
import environment
import matplotlib.pyplot as plt

episodes = 2000         # number of training episodes
episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [0, 3]           # objective point
discount = 0.9          # exponential discount factor
softmax = False         # set to true to use Softmax policy
sarsa = False           # set to true to use the Sarsa algorithm

# TODO alpha and epsilon profile
reward_dict={}
reward_arr=[]
al= [0.0025,0.025,0.25,2.5,5]
ep=[0.5,0.6,0.7,0.8,0.9]
for i in range(0,len(ep)):
    alpha = np.ones(episodes)*al[i]
    epsilon = np.linspace(ep[i], 0.001,episodes)

# initialize the agent
    learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

# perform the training
    for index in range(0, episodes):
    # start from a random state
        initial = [np.random.randint(0, x), np.random.randint(0, y)]
    # initialize environment
        state = initial
        env = environment.Environment(x, y, state, goal)
        reward = 0
    # run episode
        for step in range(0, episode_length):
        # find state index
            state_index = state[0] * y + state[1]
        # choose an action
            action = learner.select_action(state_index, epsilon[index])
        # the agent moves in the environment
            result = env.move(action)
        # Q-learning update
            next_index = result[0][0] * y + result[0][1]
            learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
        # update state and reward
            reward += result[1]
            state = result[0]
            reward /= episode_length
            reward_arr.append(reward)


    # periodically save the agent
        if ((index + 1) % 10 == 0):
            with open('agent.obj', 'wb') as agent_file:
                dill.dump(agent, agent_file)
                print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial, 'with alpha:', al[i], 'with epsilon:', ep[i])
    reward_dict[i]=reward_arr
    for key in reward_dict.keys():
        fig= plt.figure(figsize=(6,8))
        plt.title('Reward')
        plt.hist(reward_dict[key], label='reward with epsilon:{} and alpha:{}'.format(ep[i], al[i]))
        plt.xlabel('index')
        plt.ylabel('Reward')
        plt.savefig('reward with epsilon:{} and alpha:{}.png'.format(ep[i], al[i]))

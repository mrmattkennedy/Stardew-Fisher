#!/usr/bin/env python3
import sys
import pdb
import gym
import time
import stardew_fisher
import numpy as np
from gym import error, spaces

sys.path.insert(1, 'models\\')
    
env = gym.make('StardewFisherEnv-v0')

eta = .628  #learning rate
gma = .9    #value placed on future rewards
epis = 10   #epochs
Q = np.zeros([env.observation_space[0].n, env.observation_space[1].n, env.action_space.n])

def train():
    for i in range(epis):
        # Reset environment
        obs = env.reset()
        done = False
        j = 0
        #The Q-Table learning algorithm
        while j < 2000: #give 2000 frames to catch a fish
            j+=1
            # Choose action from Q table
            act = np.argmax(Q[obs[0]][obs[1]] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            #Get new state & reward from environment
            pdb.set_trace()
            obs1,rew,done,_ = env.step(act)
            #Update Q-Table with new knowledge
            Q[obs[0]][obs[1]][act] = Q[obs[0]][obs[1]][act] + eta*(rew + gma*np.max(Q[s1[0]][s1[1]][act]) - Q[obs[0]][obs[1]][act])
            obs = obs1
            if done:
                break

#def fish():
train()

#!/usr/bin/env python3
import sys
import pdb
import gym
import time
import stardew_fisher
import numpy as np
from gym import error, spaces

#Necessary to find models
sys.path.insert(1, 'models\\')
    
eta = .628  #learning rate
gma = .9    #value placed on future rewards
epis = 1   #epochs
env = gym.make('StardewFisherEnv-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])

def train():
    for i in range(epis):
        # Reset environment
        diff = env.reset()
        done = False
        j = 0
        
        while j < 100: #give 2000 frames to catch a fish
            j+=1
            # Choose action from Q table. Less random in later iterations
            act = np.argmax(Q[diff] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            #Step with current action
            diff1,rew,done,_ = env.step(act)
            print(j, rew, act, diff1)
            #Update current obs,state in Q table.
            #np.max(Q[diff1]) - Q[diff][act] is the difference in states.
            Q[diff][act] = Q[diff][act] + eta*(rew + gma*np.max(Q[diff1]) - Q[diff][act])
            diff = diff1
            if done:
                break

#def fish():
train()
#pdb.set_trace()

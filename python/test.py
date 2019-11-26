import gym
import time
from gym import error, spaces
import stardew_fisher
import numpy as np

env = gym.make('StardewFisherEnv-v0')

eta = .628  #learning rate
gma = .9    #value placed on future rewards
epis = 5000 #epochs

Q = np.zeros([len(env.observation_space),env.action_space.n])
q = [[1,2,3,4],[5,6,7,8]]
#1, in a 2D array gets the second element in every sublist
print(Q[1,]) #gets the first list and nothing else

def train():
    for i in range(epis):
        # Reset environment
        s = env.reset()
        rAll = 0
        done = False
        j = 0
        #The Q-Table learning algorithm
        while j < 1000: #give 1000 frames to catch a fish=
            j+=1
            # Choose action from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            #Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if done:
                break
        rev_list.append(rAll)


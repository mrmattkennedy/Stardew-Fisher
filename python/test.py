#import gym
import time
#from gym import error, spaces
#import stardew_fisher
import numpy as np
import cv2

def test():
    return 3, 4
"""
Flow:


OpenCV aspect will continue to read where the positions of the fish and bar are.
Send in to step function, and do the option with the
"""
#observations = {
    #"A" : 100,
    #"B" : 100}
#spaces_total = spaces.Tuple(tuple(spaces.Discrete(observations[key]) for key in observations))
#print(spaces_total.sample())
#print(t)
#spaces.Discrete(i) for i in )
#t = None
#print(str(t is None))
#im = cv2.imread('fish.jpg')
#im_g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#print(type(im))
#print(im_g[0])
#cv2.imshow('', im_g)
#env = gym.make('StardewFisherEnv-v0')
temp = np.load('train_imgs.npy')
print(temp[0][450])

#!/usr/bin/env python3
import sys
import cv2
import pdb
import gym
import time
import atexit
import stardew_fisher
import numpy as np
from PIL import ImageGrab
from gym import error, spaces
from pynput.mouse import Button, Controller

#Necessary to find models
sys.path.insert(1, 'models\\')
    
eta = .628  #learning rate
gma = .9    #value placed on future rewards
epis = 1   #epochs
env = gym.make('StardewFisherEnv-v0')
Q = np.zeros([env.observation_space.n * 2, env.action_space.n])
offset = env.observation_space.n - 1
caught_something_space = (960, 480, 980, 550)
bar_space = (800, 305, 840, 855)

mouse = Controller()
def train():
    global mouse
    while True:
        #Cast fishing rod
        mouse.release(Button.left)
        
        time.sleep(3)
        mouse.click(Button.left)
        time.sleep(2.5)
        
        mouse.press(Button.left)
        time.sleep(0.5)
        mouse.release(Button.left)
        time.sleep(1.0)
        mouse.press(Button.left)
        time.sleep(0.5)
        mouse.release(Button.left)
        time.sleep(1.0)

        #pdb.set_trace()
        mouse.press(Button.left)
        time.sleep(0.5)
        mouse.release(Button.left)
        time.sleep(1.5)
        

        caught_area = np.array(ImageGrab.grab(bbox=caught_something_space))
        caught_area = cv2.cvtColor(caught_area, cv2.COLOR_BGR2RGB)
        last_caught = caught_area

        bar_area = np.array(ImageGrab.grab(bbox=bar_space))
        bar_area = cv2.cvtColor(bar_area, cv2.COLOR_BGR2RGB)
        last_bar = bar_area
        
        caught = False
        actual_fish = False
        
        #Wait until caught something
        while not caught:
            time.sleep(1/30)
            caught_area = np.array(ImageGrab.grab(bbox=caught_something_space))
            caught_area = cv2.cvtColor(caught_area, cv2.COLOR_BGR2RGB)
            temp = np.sum(np.where(caught_area == last_caught, 1, 0)) / caught_area.size
            if temp < 0.65:
                time.sleep(0.2)
                mouse.press(Button.left)
                time.sleep(0.1)
                mouse.release(Button.left)
                caught = True
            last_caught = caught_area

        #Get the bar area, see if any large change happens. If not, didn't catch fish.
        bar_area_change = list()
        for i in range(0, 20):
            bar_area = np.array(ImageGrab.grab(bbox=bar_space))
            bar_area = cv2.cvtColor(bar_area, cv2.COLOR_BGR2RGB)
            temp = np.sum(np.where(bar_area == last_bar, 1, 0)) / bar_area.size
            bar_area_change.append(temp)
            last_bar = bar_area
            time.sleep(1/30)
            
        if min(bar_area_change) > 0.5:
            actual_fish = False
        else:
            actual_fish = True

        time.sleep(0.2)
        if actual_fish:
            # Reset environment
            diff = env.reset()
            diff += 515
            done = False
            j = 0
            
            while j < 300: #give 2000 frames to catch a fish
                j+=1
                # Choose action from Q table. Less random in later iterations
                act = np.argmax(Q[diff] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
                #Step with current action
                diff1,rew,done,_ = env.step(act)
                diff1 += 515
                print(j, rew, act, diff1)
                #Update current obs,state in Q table.
                #np.max(Q[diff1]) - Q[diff][act] is the difference in states.
                Q[diff][act] = Q[diff][act] + eta*(rew + gma*np.max(Q[diff1]) - Q[diff][act])
                diff = diff1
                #print('Done!')
                if done:
                    env.second = True
                    break

def exit_func():
    pdb.set_trace()
"""
#def fish():
train()
#pdb.set_trace()
"""
#env._update_time()
train()
atexit.register(exit_func)


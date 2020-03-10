import os
import cv2
import sys
import pdb
import time
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
from pynput.mouse import Button, Controller
from PIL import ImageGrab

if __name__ == '__main__':
    sys.path.insert(1, '..\\..\\models')
import object_finder


#Action space
actions = {
    "START_HOLD" : 0,
    "NOTHING" : 1,
    "STOP_HOLD" : 2,
}

class StardewFisherEnv(gym.Env):
    def __init__(self):
        self.mouse = Controller()
        #Can be anywhere from fish at top, bar at bottom, to vice versa.
        self.top = 20
        self.bottom = 515
        self.observation_space = spaces.Discrete((self.bottom*2)+1)
        self.action_space = spaces.Discrete(len(actions))

        #Variables related to location and determining if fish being caught
        self.catching = True
        self.bar_location = (self.bottom - 158, self.bottom - (158 / 2))
        self.fish_location = self.bottom

        #Variables related to timer
        self.start_time = 0
        self.elapsed_time = 0

        self.max_time = 10 #max time is 8 seconds to catch the fish?
        self.max_diff = self.bottom - self.top

        #cv2/obj locater vars
        self.show_screen = True
        self.finder = object_finder.object_finder(load_model_path='models\\batch100_fish_id.h5')
        self.fish_start_col = 3
        self.fish_end_col = 38
        self.bar_start_col = 1
        self.bar_end_col = 40
        self.bar_height = 158
        self.catch_range = self.bar_height / 2
        
        if not os.path.isfile('models\\numpy_data\\screen_dims.npy'):
            self.get_screen_dims()
            
        self.screen_dims = np.load('models\\numpy_data\\screen_dims.npy').tolist()
        #(800, 305, 840, 855)

    def get_screen_dims(self):
        cv2.imshow('', cv2.imread('models\\image_data\\1.jpg'))
        x_diff = 40
        y_diff = 550
        done = False

        while not done:
            inp = input("Need to get dimensions for where to look for fish. Set zoom to 95%.\nHead to your fishing spot, get a fish to pop up, alt-tab out,\nand then try entering an x,y coordinate for the top left corner of the fishing area (see example).\nWhen done, hit enter in this window to see the capture area.\n")
            try:
                start_coord = list(map(int, inp.rstrip('\n').replace(' ', '').split(',')))
                dims = (start_coord[0], start_coord[1], start_coord[0]+x_diff, start_coord[1]+y_diff)
                screen = np.array(ImageGrab.grab(bbox=dims)) 
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                cv2.imshow('', screen)

                print("Left: {}, Top: {}, Right: {}, Bottom: {}".format(dims[0], dims[1], dims[2], dims[3]))
                inp = input("Is this a good capture? Should show only fishing bar.\n1 is yes.\n2 is no.\n3 is no, show example image again.\n")
                if inp == '1':
                    done = True
                if inp == '3':
                    cv2.imshow('', cv2.imread('models\\image_data\\1.jpg'))                    
            except:
                print('Input not correct. Try again')

        dims = np.array(dims)
        np.save('models\\numpy_data\\screen_dims.npy', dims)

                
    def step(self, action):
        assert self.action_space.contains(action)
        #Do action
        if action == 0:
            self.mouse.press(Button.left)
        elif action == 1:
            pass
        elif action == 2:
            self.mouse.release(Button.left)
            
        #Get current locations
        self._get_obs()
        """
        TODO:
        Fix caught condition
        """
        if self.fish_location > 550 or self.fish_location < 10: #fish caught
            print(self.fish_location)
            done = True
            reward = 50000
        else: #fish is somewhere
            done = False
            self._update_time()
                    
            #Get reward for not being done
            reward = self._get_reward()
            
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.bar_location = (self.bottom - self.bar_height, self.bottom - (self.bar_height / 2))
        self.fish_location = self.bottom
        self.catching = True
        self.start_time = time.time()
        self.elapsed_time = 0
        return self._get_difference()

    def _get_obs(self):
        #capture the window (wrote script to resize window)
        screen = np.array(ImageGrab.grab(bbox=self.screen_dims)) 
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        self.fish_location = self.finder.locate_fish(screen)
        self.bar_location = self.finder.locate_bar(screen)

        if self.show_screen:
            #Draw rect around fish
            cv2.rectangle(screen,
                          (self.fish_start_col, self.fish_location),
                          (self.fish_end_col, self.fish_location+27), [0, 255, 255], 1)
            #Draw rect around bar
            cv2.rectangle(screen,
                          (self.bar_start_col, self.bar_location[0]),
                          (self.bar_end_col, self.bar_location[1]), [255, 255, 0], 1)
            cv2.imshow('', screen)
        
        #Return roughly middle of bar
        return self._get_difference()

    def _update_time(self):
        #pdb.set_trace()
        if abs(self._get_difference()) > self.catch_range:
            #If was catching and now not, reset timers
            if self.catching:
                self.catching = False
                self.start_time = time.time()
                self.elapsed_time = 0
            else:
                self.elapsed_time = self._check_elapsed()
        else:
            #If still catching, update timer
            if self.catching:
                self.elapsed_time = self._check_elapsed()
            else:
                self.catching = True
                self.start_time = time.time()
                self.elapsed_time = 0
                
    #If fish is out of range, reset the timer
    def _check_elapsed(self):
        return time.time() - self.start_time

    #Get difference in two locations
    def _get_difference(self):
        return self.fish_location - int(self.bar_location[0] + (self.bar_height / 2))

    #Get current reward
    def _get_reward(self):
        reward_factor = 1 if self.catching == True else -1
        return self.elapsed_time * reward_factor

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
        self.fish_location = 497
        self.bar_location = (385,543)

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
        self.screen_dims = np.load('models\\numpy_data\\screen_dims.npy').tolist()
        #(800, 305, 840, 855)
        self.last_screen = None
        self.moving = 0

        self.second = False

        
    def step(self, action):
        assert self.action_space.contains(action)
        #Do action
        if action == 0:
            self.mouse.press(Button.left)
            self.moving = 1
        elif action == 1:
            pass
        elif action == 2:
            self.mouse.release(Button.left)
            self.moving = 0
            
        #Get current locations
        self._get_obs()
        screen = np.array(ImageGrab.grab(bbox=self.screen_dims)) 
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        temp = 1
        if self.last_screen is not None:
            temp = np.sum(np.where(screen == self.last_screen, 1, 0)) / screen.size
        self.last_screen = screen
        
        if temp < 0.1 or self.bar_location == (0, 0): #fish caught
            print('Temp is {}'.format(temp))
            done = True
            if self.elapsed_time < 0.3:
                self.catching = False
        else: #fish is somewhere
            done = False
            self._update_time()

        obs = self._get_obs()
        if self.bar_location == (0, 0):
            done = True
            if self.elapsed_time < 0.3:
                self.catching = False

        #Get reward for not being done
        reward = self._get_reward()
            
        return obs, reward, done, {}

    def reset(self):
        self.catching = True
        self.start_time = time.time()
        self.elapsed_time = 0
        self.fish_location = 497
        self.bar_location = (385,543)
        self.finder.last_bar_data = (385,543)
        self.last_screen = None
        self.moving = 0
        return self._get_difference() + (self.observation_space.n - 1) * self.moving

    def _get_obs(self):
        #capture the window (wrote script to resize window)
        screen = np.array(ImageGrab.grab(bbox=self.screen_dims)) 
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        #if self.second:
            #pdb.set_trace()
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
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        #Return roughly middle of bar
        return self._get_difference() + (self.observation_space.n - 1) * self.moving

    def _update_time(self):
       # pdb.set_trace()
        if abs(self._get_difference()) >= 75:
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
        return (self.fish_location+14) - int(self.bar_location[0] + (76))

    #Get current reward
    def _get_reward(self):
        self._update_time()
        reward_factor = 1 if self.catching == True else -1
        return reward_factor

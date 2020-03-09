import cv2
import sys
import pdb
import math
import time
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
from PIL import ImageGrab

if __name__ == '__main__':
    sys.path.insert(1, '..\\..\\models')
import object_finder


#Action space
actions = {
    "START_HOLD" : 0,
    "CONTINUE_HOLD" : 1,
    "STOP_HOLD" : 2,
    "NOTHING" : 3
}

class StardewFisherEnv(gym.Env):
    def __init__(self):
        #pdb.set_trace()
        #self.observation_space = spaces.Tuple(tuple(spaces.Discrete(observations[key]) for key in observations))
        self.observation_space = spaces.Tuple((
            spaces.Discrete(515+1),
            spaces.Discrete(515+1)))
        self.action_space = spaces.Discrete(len(actions))

        #Variables related to location and determining if fish being caught
        self.catching = True
        self.top = 20
        self.bottom = 515
        self.bar_location = self.bottom
        self.fish_location = self.bottom

        #Variables related to timer
        self.start_time = 0
        self.elapsed_time = 0

        self.max_time = 10 #max time is 8 seconds to catch the fish?
        self.max_diff = self.bottom - self.top

        #cv2/obj locater vars
        self.finder = object_finder.object_finder(load_model_path='models\\batch100_fish_id.h5')
        self.screen_dims = (800, 305, 840, 855)
        self.fish_start_col = 3
        self.fish_end_col = 38
        self.bar_start_col = 1
        self.bar_end_col = 40
        self.bar_height = 158
        self.catch_range = self.bar_height / 2

        
    def step(self, action):
        assert self.action_space.contains(action)
        #Get current locations
        self._get_obs()
        """
        TODO:
        Fix caught condition
        """
        if self.fish_location > 330 or self.fish_location < 40: #fish caught
            done = True
            reward = 50000
        else: #fish is somewhere
            done = False
            self._update_time()
                    
            #Get reward for not being done
            reward = self._get_reward()
            
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.bar_location = self.bottom - (self.bar_height / 2)
        self.fish_location = self.bottom
        self.catching = True
        self.start_time = time.time()
        self.elapsed_time = 0
        return (self.fish_location, int(self.bar_location))

    def _get_obs(self):
        #capture the window (wrote script to resize window)
        screen = np.array(ImageGrab.grab(bbox=self.screen_dims)) 
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        fish_row = self.finder.locate_fish(screen)
        bar_rows = self.finder.locate_bar(screen)

        #Draw rect around fish
        cv2.rectangle(screen,
                      (self.fish_start_col, fish_row),
                      (self.fish_end_col, fish_row+27), [0, 255, 255], 1)
        #Draw rect around bar
        cv2.rectangle(screen,
                      (self.bar_start_col, bar_rows[0]),
                      (self.bar_end_col, bar_rows[1]), [255, 255, 0], 1)

        #Return roughly middle of bar
        return (self.fish_location, int(self.bar_location + (self.bar_height / 2)))

    def _update_time(self):
        if math.abs(self._get_difference()) > self.catch_range:
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
        return time.time() - self.start_time()

    #Get difference in two locations
    def _get_difference(self):
        values = self._get_obs()
        return values[1] - values[0]

    #Get current reward
    def _get_reward(self):
        reward_factor = 1 if self.catching == True else -1
        return self.time_elapsed * reward_factor

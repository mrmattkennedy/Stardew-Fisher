import gym
import math
import time
import numpy as np
from gym import error, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete

#blackjack good example https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
observations = {
    "FISH_LOC" : 100,     #max: 100?
    "BAR_LOC" : 100,      #max 100?
    }

actions = {
    "START_HOLD" : 0,
    "CONTINUE_HOLD" : 1,
    "STOP_HOLD" : 2,
    "NOTHING" : 3
    }
class StardewFisherEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.observation_space = spaces.Tuple(tuple(spaces.Discrete(observations[key]) for key in observations))
        self.action_space = spaces.Discrete(len(actions))
        
        self.bar_location = None
        self.fish_location = None
        self.catching_range = 10
        self.catching = True
        
        self.last_time = None
        self.current_time = None
        self.time_elapsed = None
        self.max_time = 8 #max time is 8 seconds to catch the fish

    def step(self, action):
        """
        Make sure action exists
        Update timer
        Get reward based on difference and time
        """
        assert self.action_space.contains(action)
        self._update_time()

        done = False
        if self.fish_location > 100 or self.fish_location < 0: #fish caught
            done = True
            reward = 1
        else: #fish is somewhere
            reward = self._get_rew()
        
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.bar_location = 0
        self.fish_location = 0
        return self._get_obs()
            
    def _get_obs(self):
        return (self.fish_location, self.bar_location)

    def _start_timer(self):
        return time.time()

    #get time elapsed of catching fish
    def _update_time(self):
        self.current_time = self._start_timer()
        
        if self.last_time is not None and self.current_time is not None:            
            self.time_elapsed = self._check_elapsed() + (self.current_time - self.last_time)
            self.last_time = self.current_time
        else:
            self.last_time = time.time()

    def _check_elapsed(self):
        return ((lambda: 0, lambda: self.time_elapsed)[abs(self._get_difference()) <= self.catch_range]())

    def _get_difference(self)
        values = self._get_obs()
        return values[1] - values[0]
    
    def _get_rew(self):
        #diff max is 100?
        #greater difference = lesser reward
        #fix this
        return -(self.elapsed_time * abs(self.bar_location - self.fish_location))
    

"""
Need:
Determine if use models or actual game
If models: get equations for change in bar
If game: wait in step until next set of info comes in.

For reward:
If diff is within range, then diff * time
If not, reward slowly from 0 -> -1
"""

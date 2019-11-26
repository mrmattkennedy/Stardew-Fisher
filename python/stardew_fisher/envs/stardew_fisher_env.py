import math
import time
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete

#blackjack good example https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
observations = {
    "FISH_LOC" : 290,     #40 to 330
    "BAR_LOC" : 290,      #40 to 330
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

        #Variables related to location and determining if fish being caught
        self.bar_location = None
        self.fish_location = None
        self.catching_range = 10
        self.catching = True

        #Variables related to timer
        self.last_time = None
        self.current_time = None
        self.time_elapsed = None
        
        self.max_time = 8 #max time is 8 seconds to catch the fish?
        self.max_diff = 330-40

    def step(self, action):
        assert self.action_space.contains(action)
        self._update_time() #see if timer needs to change

        done = False
        if self.fish_location > 330 or self.fish_location < 40: #fish caught
            done = True
            reward = 500
        else: #fish is somewhere
            reward = self._get_reward()
        
        return self._get_obs(), reward, done, {}
    
    def reset(self):
        self.bar_location = 0
        self.fish_location = 0
        self.catching = True
        return self._get_obs()
            
    def _get_obs(self):
        return (self.fish_location, self.bar_location)
    
    def _start_timer(self):
        return time.time()

    #get time elapsed of catching fish
    def _update_time(self):
        self.current_time = self._start_timer()

        #If both timers are initiated, get diff
        if self.last_time is not None and self.current_time is not None:            
            self.time_elapsed = self._check_elapsed() + (self.current_time - self.last_time)
            self.last_time = self.current_time
        else: #start the last time, current time updated at start of method
            self.last_time = time.time()

    #If fish is out of range, reset the timer
    def _check_elapsed(self):
        if abs(self._get_difference()) <= self.catch_range: #fish in range
            if self.catching is False: #if wasn't in range prior, timer reset
                return 0
            self.catching = True
            return self.time_elapsed
        else:
            if self.catching is True: #if was in range prior, reset
                return 0
            self.catching = False
            return self.time_elapsed

    #get difference in two locations
    def _get_difference(self):
        values = self._get_obs()
        return values[1] - values[0]

    #get current reward
    def _get_reward(self):
        return ((lambda: self.time_elapsed * (self.max_diff-self.get_difference()),
                lambda: -(self.time_elapsed * self.get_difference()))
                [abs(self._get_difference()) <= self.catch_range]())
    

"""
Need:
Determine if use models or actual game
If models: get equations for change in bar
If game: wait in step until next set of info comes in.

For reward:
If diff is within range, then diff * time
If not, reward slowly from 0 -> -1
"""

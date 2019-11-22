import numpy as np
import gym
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
        self.action_space = spaces.Discrete(len(spaces))

    def step(self, action):
        assert self.action_space.contains(action)
        if action: #start hold
            #TODO: Update bar location
            
    def _get_obs(self):
        return (self.fish_location, self.bar_location)
    
    def reset(self):
        self.bar_location = 0
        self.fish_location = 0
        return self._get_obs()

"""
Need:
Determine if use models or actual game
If models: get equations for change in bar
If game: wait in step until next set of info comes in.
"""

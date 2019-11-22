import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym.envs.toy_text import discrete

observations = [
    "FISH_LOC",
    "FISH_TOP",
    "FISH_BOTTOM",
    "BAR_LOC",
    "BAR_TOP"
    "BAR_BOTTOM"
    "DIFF"
    ]
class StardewFisherEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.viewer = None
        self.num_states = 7
        self.num_actions = 4
        self.desc = np.asarray(observations,dtype='c')
        self.reward_range = (0, 1)

        isd = np.array(self.desc == b'BAR_LOC').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(self.num_actions)} for s in range(self.num_states)}
        super(StardewFisherEnv, self).__init__(self.num_states,
                                               self.num_actions,
                                               P,
                                               isd)

import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

class StardewFisher(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.viewer = None

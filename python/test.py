import gym
from gym import error, spaces
import stardew_fisher
import numpy as np

"""
nA = 4
nS = 7

P = {s : {a : [] for a in range(nA)} for s in range(nS)}
#for item in P:
    #print(item, P[item])
print(P[0][0])


MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

desc = MAPS["4x4"]
#print(desc)
desc = desc = np.asarray(desc,dtype='c')
#print(desc)
#isd: finds all the starting points.
#For fishing: starting point is 1.0 for the poisition of the rod
isd = np.array(desc == b'S').astype('float64').ravel()
#print(isd)
isd /= isd.sum()
#print(isd)
#print(desc.shape)
"""
observations = {
    "A" : 100,
    "B" : 100}
#spaces_total = spaces.Tuple(tuple(spaces.Discrete(observations[key]) for key in observations))
#print(spaces_total.sample())
#print(t)
#spaces.Discrete(i) for i in )
env = gym.make('StardewFisherEnv-v0')

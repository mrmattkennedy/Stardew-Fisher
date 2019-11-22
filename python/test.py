import numpy as np
nA = 4
nS = 7

P = {s : {a : [] for a in range(nA)} for s in range(nS)}
#print(type(P))


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
print(desc)
desc = desc = np.asarray(desc,dtype='c')
print(desc)
#isd: finds all the starting points.
#For fishing: starting point is 1.0 for the poisition of the rod
isd = np.array(desc == b'S').astype('float64').ravel()
print(isd)
isd /= isd.sum()
print(isd)
print(desc.shape)

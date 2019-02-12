import os
import sys
from pprint import pprint
import numpy as np

path_to_output = os.path.dirname(os.path.realpath(__file__)) + \
                 '/../qtables/2019-02-05T17:52:24.412393/'

for i in range(0,19):
    numpyfile1 = path_to_output + 'iter_' + str(i) + '/q_learner1.npy'
    numpyfile2 = path_to_output + 'iter_' + str(i) + '/q_learner2.npy'
    train1 = np.load(numpyfile1)
    train2 = np.load(numpyfile2)

    nonzero1 = train1[np.nonzero(train1)]
    nonzero2 = train2[np.nonzero(train2)]
    pprint("Number of nonzeroes in iteration " + str(i) + " for agent 1: " + str(len(nonzero1)))
    pprint("Nonzeroes: ")
    pprint(nonzero1)

    pprint("Number of nonzeroes in iteration " + str(i) + " for agent 2: " + str(len(nonzero2)))
    pprint("Nonzeroes: ")
    pprint(nonzero2)


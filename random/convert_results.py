import os
import pickle
from collections import defaultdict
import numpy as np
import shutil

def convert(file, barrier_range):

    results_dict = defaultdict(dict)
        
    for filename in os.listdir('{}/random/'.format(file)):

        params = filename.split('_')

        barrier, i = int(params[0]), int(params[1])

        data = pickle.load(open('{}/random/'.format(file) + filename, "rb"))

        results_dict[barrier][i] = data[1:] # Dropping the seed number 

    for barrier in barrier_range: # Trying a bunch of barriers to aim for

        filename = '{}/{}_dict.txt'.format(file, barrier)
        results = np.array(list(results_dict[barrier].values()))

        pickle.dump(results, open(filename, "wb" ))

    shutil.rmtree('{}/random'.format(file))

files = ['e2_lccsd', 'e2_mp2', 'michael', 'sn2_lccsd', 'sn2_mp2', 'vaska']
barrier_ranges = 2 * [np.arange(-20, 61, 20)] + [np.arange(10, 51, 10)] + 2 * [np.arange(-20, 61, 20)] + [np.arange(0, 26, 5)]

[convert(file, barrier_range) for file, barrier_range in zip(files, barrier_ranges)]
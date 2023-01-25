import sys
import os
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from finding_barriers import Barriers
from Heuristic_all import Search_agent
import numpy as np
import pickle
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def run(data, file, barrier_range):

    num_runs = 50

    for barrier in tqdm(barrier_range, position=0): # Trying a bunch of barriers to aim for

        barriers = Barriers(file=file, name=data, target_barrier=barrier)
        
        try: os.mkdir('{}/min_max_targ'.format(file))
        except: pass

        filename = '{}/min_max_targ/{}_dict.txt'.format(file, barrier)
        results = []

        for run in tqdm(range(num_runs), position=1):

            np.random.seed(run)

            barriers.reset()
            agent = Search_agent(barriers)            
            agent.find_barrier(barrier)
    
            results.append(agent.results)

        pickle.dump(results, open(filename, "wb" ))

datas = ['e2_barriers_groups_lccsd', 'e2_barriers_groups_mp2', 'barriers_groups', 'sn2_barriers_groups_lccsd', 'sn2_barriers_groups_mp2', 'vaska_barriers_groups']
files = ['e2_lccsd', 'e2_mp2', 'michael', 'sn2_lccsd', 'sn2_mp2', 'vaska']
barrier_ranges = 2 * [np.arange(-20, 61, 20)] + [np.arange(10, 51, 10)] + 2 * [np.arange(-20, 61, 20)] + [np.arange(0, 26, 5)]

[run(data, file, barrier_range) for data, file, barrier_range in zip(datas, files, barrier_ranges)]

    







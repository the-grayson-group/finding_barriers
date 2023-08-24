"""
Collects results for the guided search.
"""

import sys
import os
sys.path.insert(0, '../')
from finding_barriers import Barriers
from Heuristic import Search_agent
import numpy as np
import pickle
from multiprocessing import Process, Manager
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def run_single(barriers, barrier, results_dict):

    results = []

    for r in range(1, 26): # 25 runs

        np.random.seed(r)
        barriers.shuffle()
        agent = Search_agent(barriers)       
        agent.find_barrier(barrier, 'greedy')
        
        results.append(len(agent.results))

    results_dict[barrier] = np.array(results)
                    
def run(data, file):

    # Avoiding using GPU because the multiprocessing doesn't work.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    results_dict = Manager().dict()

    barriers = Barriers(file=file, name=data)
                
    num_p = 50 # Size of batch

    # Barriers to search
    target_indices = set()
    min_barrier = barriers.all_barriers_nosym.min()
    while min_barrier < barriers.barriers.max():
        differences = np.abs(min_barrier - barriers.all_barriers_nosym)
        target_indices.add(np.where(differences == np.min(differences))[0][0])
        min_barrier += 2.0
    target_indices.add(np.where(barriers.all_barriers_nosym == barriers.barriers.max())[0][0])
    target_indices = list(target_indices)
    targets = barriers.all_barriers_nosym[target_indices]
    targets = np.sort(targets)

    # Number of separate active processes
    for batch in tqdm(range(int(np.ceil(len(targets) / num_p)))):

        processes = [Process(target=run_single, args=(barriers, barrier, results_dict)) for barrier in targets[batch * num_p : (batch + 1) * num_p]]

        for p in processes: p.start()
        for p in processes: p.join()

    pickle.dump(dict(results_dict), open('{}/results_dict_greedy.txt'.format(file), "wb" ))

if __name__ == "__main__":

    datas = ['e2_barriers_groups_lccsd', 'e2_barriers_groups_mp2', 'barriers_groups', 'sn2_barriers_groups_lccsd', 'sn2_barriers_groups_mp2', 'vaska_barriers_groups']
    files = ['e2_lccsd', 'e2_mp2', 'michael', 'sn2_lccsd', 'sn2_mp2', 'vaska']

    [run(data, file) for data, file in zip(datas, files)]






import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from finding_barriers import Barriers
import numpy as np
import pickle
from multiprocessing import Process

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def run_single(barrier, r, i, data, file):

    np.random.seed(r)

    barriers = Barriers(file=file, name=data, target_barrier=barrier, actions=0)
    barriers.reset()

    mins = []
    maxs = []
    targets = []

    for ind in range(len(barriers.molecules)):

        Y_known = barriers.barriers[:ind + 1]

        mins.append(Y_known.min())
        maxs.append(Y_known.max())
        targets.append(Y_known[np.abs(Y_known - barrier).argmin()])

    results = [r, mins, maxs, targets]

    try: os.mkdir('{}/random'.format(file))
    except: pass

    pickle.dump(results, open('{}/random/{}_{}'.format(file, barrier, i), "wb" ))

def run(data, file, barrier_range):

    # Avoiding using GPU because the multiprocessing doesn't work.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    all_p = [] # The hyperparameters for all processes.

    for barrier in barrier_range: # Trying a bunch of barriers to aim for

        for i, r in enumerate(range(1, 51)): # 50 runs

            all_p.append((barrier, r, i))
                
    num_p = 40 # Size of batch

    # Number of separate active processes
    for r in range(int(np.ceil(len(all_p) / num_p))):

        processes = []

        for params in all_p[r * num_p : (r + 1) * num_p]:

            barrier, r, i = params[0], params[1], params[2]

            p = Process(target=run_single, args=(barrier, r, i, data, file))

            processes.append(p)

        for p in processes: p.start()
        for p in processes: p.join()


if __name__ == "__main__":

    datas = ['e2_barriers_groups_lccsd', 'e2_barriers_groups_mp2', 'barriers_groups', 'sn2_barriers_groups_lccsd', 'sn2_barriers_groups_mp2', 'vaska_barriers_groups']
    files = ['e2_lccsd', 'e2_mp2', 'michael', 'sn2_lccsd', 'sn2_mp2', 'vaska']
    barrier_ranges = 2 * [np.arange(-20, 61, 20)] + [np.arange(10, 51, 10)] + 2 * [np.arange(-20, 61, 20)] + [np.arange(0, 26, 5)]

    [run(data, file, barrier_range) for data, file, barrier_range in zip(datas, files, barrier_ranges)]
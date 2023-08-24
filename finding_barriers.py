import csv
import numpy as np
from collections import defaultdict
from itertools import product
import regex as re

class Barriers:
    """
    Deals with the barrier data, doubles as an environment
    """
    
    def __init__(self, file, name):
        
        self.read(file, name)
        self.point_change()
        self.transitions_point_change()
        
    def read(self, file, name):
        """
        Reads in data
        """

        # Read in the data
        self.file = file
        with open(file + '/' + name + '.csv', 'r') as file:

            data = np.array(list(csv.reader(file))[1:], dtype='O')

        # Separate into barriers, molecules and molecule symbols
        if name == 'barriers_groups':

            self.barriers = data[:, 4].astype(float) * 627.503 # Scaling barriers
            self.molecules_str = data[:, 5:].astype('O')

        else:

            self.barriers = data[:, -1].astype(float)
            self.molecules_str = data[:, :-1].astype('O')

        self.molecules_str = np.vectorize(lambda group_str : re.sub("\(?\[\*:\d\]\)?", "", group_str))(self.molecules_str)

        # Convert into numbers
        self.molecule_convert = defaultdict(lambda : len(self.molecule_convert))
        self.molecules = self.molecules_str.copy()
        for i, row in enumerate(self.molecules_str):
            for j, value in enumerate(row):
                
                self.molecules[i, j] = self.molecule_convert[value]

        self.molecules = self.molecules.astype(int)

        # Number of points in dataset, and size of molecules
        self.n_points = len(self.barriers)
        self.dim = len(self.molecules[0])

        # Available molecules in each index
        self.all = [np.unique(self.molecules[:, i]) for i in range(self.dim)]
        self.all_n = [len(i) for i in self.all]

        # Every possible molecule
        self.all_molecules = np.array(list(product(*self.all)))
        self.n_all_points = len(self.all_molecules)
        self.all_barriers = np.array([self.barriers[(self.molecules == molecule).all(axis=1)][0] if (self.molecules == molecule).all(axis=1).any() else np.inf for molecule in self.all_molecules])
        self.all_barriers = np.array([min(self.all_barriers[(self.all_molecules == molecule).all(axis=1)], 
                                          self.all_barriers[(self.all_molecules == mkswp(molecule, file)).all(axis=1)]) 
                                          for molecule in self.all_molecules])

        # Removing symmetry
        self.all_molecules_nosym = self.all_molecules[:1]
        self.all_barriers_nosym = self.all_barriers[:1]

        for molecule in self.all_molecules:

            if not (molecule == self.all_molecules_nosym).all(axis=1).any() and not (mkswp(molecule, file) == self.all_molecules_nosym).all(axis=1).any():

                self.all_molecules_nosym = np.append(self.all_molecules_nosym, [molecule], axis=0)
                self.all_barriers_nosym = np.append(self.all_barriers_nosym, self.all_barriers[(self.all_molecules == molecule).all(axis=1)])

        self.n_all_points_nosym = len(self.all_molecules_nosym)

    def shuffle(self):
        """
        Shuffles the molecules and targets
        """

        # Shuffle the data and targets
        index = np.random.choice(np.arange(self.n_points), self.n_points, replace=False)
        self.molecules = self.molecules[index]
        self.barriers = self.barriers[index]

        index = np.random.choice(np.arange(self.n_all_points), self.n_all_points, replace=False)
        self.all_molecules = self.all_molecules[index]
        self.all_barriers = self.all_barriers[index]

        index = np.random.choice(np.arange(self.n_all_points_nosym), self.n_all_points_nosym, replace=False)
        self.all_molecules_nosym = self.all_molecules_nosym[index]
        self.all_barriers_nosym = self.all_barriers_nosym[index]

    def point_change(self):
        """
        Makes the actions for method where an agent 
        selects an index and changes the molecule - point and change
        Action [a1, a2], a1 slot and a2 molecule.
        """

        self.action_dict = {}

        for i in range(self.dim):
            for j in self.all[i]:

                self.action_dict[len(self.action_dict)] = np.array([i, j])

    def transitions_point_change(self):
        """
        Transition for Action [a1, a2], a1 slot and a2 molecule.
        """

        self.trans = defaultdict(dict)

        for state in self.all_molecules:
            for action, action_vec in self.action_dict.items():

                # Take action, finding new state and reward
                new_state = state.copy()
                new_state[action_vec[0]] = action_vec[1]
                new_energy = self.all_barriers[(self.all_molecules == new_state).all(axis=1)][0]

                self.trans[tuple(state)][action] = tuple([list(new_state), new_energy])

def mkswp(s, file): 
    """
    Returns the symmetrically equivalent reaction.
    """
    if file == 'michael': return s
    elif file in ['e2_lccsd', 'e2_mp2', 'sn2_lccsd', 'sn2_mp2']: return s[[1, 0, 3, 2, 4, 5]]
    elif file == 'vaska': return s[[0, 1, 3, 2]]
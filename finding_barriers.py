import csv
import numpy as np
from collections import defaultdict

class Barriers:
    """
    Deals with the barrier data, doubles as an environment
    """
    
    def __init__(self, file, name, target_barrier, actions=0):

        self.target_barrier = target_barrier # This is what we are trying to optimise towards

        # Names of all the different datasets
        self.names = ['barriers_groups', 
                      'e2_barriers_groups_lccsd', 
                      'e2_barriers_groups_mp2', 
                      'sn2_barriers_groups_lccsd', 
                      'sn2_barriers_groups_mp2',
                      'vaska_barriers_groups']
        
        self.read(file, name)
        self.choose_actions(actions)

        self.target_state = self.molecules[np.abs(self.barriers - target_barrier).argmin()]
        
    def read(self, file, name):
        """
        Reads in data
        """

        # Read in the data
        with open(file + '/' + name + '.csv', 'r') as file:

            data = np.array(list(csv.reader(file))[1:], dtype='O')

        # Separate into barriers, molecules and molecule symbols
        if name == 'barriers_groups':

            self.barriers = data[:, 4].astype(float) * 627.503 # Scaling barriers
            self.molecules_str = data[:, 5:].astype('O')

        elif name in self.names:

            self.barriers = data[:, -1].astype(float)
            self.molecules_str = data[:, :-1].astype('O')

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

    def shuffle(self):
        """
        Shuffles the molecules and targets
        """

        # Shuffle the data and targets
        index = np.random.choice(np.arange(self.n_points), self.n_points, replace=False)
        self.molecules = self.molecules[index]
        self.barriers = self.barriers[index]

    def choose_actions(self, action_struc=0):
        """
        Changes the action structure that it currently being used.
        """

        if action_struc == 0:

            self.point_change()
            self.transitions_point_change()

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

        for state in self.molecules:
            for action, action_vec in self.action_dict.items():

                # Take action, finding new state and reward
                new_state = state.copy()
                new_state[action_vec[0]] = action_vec[1]

                if (self.molecules == new_state).all(axis=1).any(): # For missing data points

                    new_energy = self.barriers[(self.molecules == new_state).all(axis=1)][0]
                    self.trans[tuple(state)][action] = tuple([list(new_state), new_energy]) 
        
    def reset(self):
        """
        Resets the experiment.
        Shuffles data and creates a new split
        """
        
        self.shuffle()

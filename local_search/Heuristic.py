import numpy as np

class Search_agent:

    def __init__(self, barriers):

        self.barriers = barriers

    def check_barrier_local(self, state):
        """
        Checks if the current state is a local min.
        """

        best_state = state
        best_energy = abs(self.barrier - self.barriers.all_barriers_nosym[(self.barriers.all_molecules_nosym == state).all(axis=1)][0])

        # Check the energies of all surrounding states
        for new_state, new_energy in self.barriers.trans[tuple(state)].values():

            # Check energy if state has not been seen before
            if new_state in self.unseen_states: self.min_max_target(new_state)
            
            # Check if new neighbour is better
            if abs(self.barrier - new_energy) <= best_energy:

                best_energy = abs(self.barrier - new_energy)
                best_state = new_state

            if self.terminate: break # FOUND BARRIER

        return best_state

    def check_barrier_greedy(self, state):
        """
        Checks if the current state is a local min.
        """

        best_state = state
        best_energy = abs(self.barrier - self.barriers.all_barriers_nosym[(self.barriers.all_molecules_nosym == state).all(axis=1)][0])

        start = 0
        start_state = state
        for n_neighbours in self.barriers.all_n:
            for action in range(start, start + n_neighbours):

                new_state, new_energy = self.barriers.trans[tuple(start_state)][action]

                # Check energy if state has not been seen before
                if new_state in self.unseen_states: self.min_max_target(new_state)
                
                # Check if new neighbour is better
                if abs(self.barrier - new_energy) <= best_energy:

                    best_energy = abs(self.barrier - new_energy)
                    best_state = new_state

                if self.terminate: break # FOUND BARRIER

            start += n_neighbours
            start_state = best_state

        return best_state

    def find_barrier(self, barrier, version):
        """
        Uses a voting system to decide which local minimum is the true minimum.
        """

        self.barrier = barrier # The barrier that we are trying to find the closest to.
        self.seen_states = []
        self.seen_barriers = []
        self.unseen_states = [list(i) for i in self.barriers.all_molecules_nosym]
        self.results = []
        self.terminate = False

        best_state, state = (None, None)

        while True:

            if best_state == state:

                # Sample a state which has not been seen before
                state = self.unseen_states[np.random.randint(len(self.unseen_states))]
                self.min_max_target(state)
                if self.terminate: break # FOUND BARRIER

            else: state = best_state

            if version == 'local': best_state = self.check_barrier_local(state)
            elif version == 'greedy': best_state = self.check_barrier_greedy(state)

            if self.terminate: break # FOUND BARRIER

        self.results = np.array(self.results).T

    def min_max_target(self, state):
        """
        Finds the current best min, max and target
        """

        self.seen_states.append(state)
        self.seen_barriers.append(self.barriers.all_barriers_nosym[(self.barriers.all_molecules_nosym == state).all(axis=1)][0])
        self.unseen_states.remove(state)

        target = self.seen_barriers[np.abs(self.seen_barriers - self.barrier).argmin()]

        self.results.append(target)

        if np.abs(self.barrier - target) < 1: self.terminate = True

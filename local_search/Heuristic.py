import numpy as np

class Search_agent:

    def __init__(self, barriers):

        self.barriers = barriers

    def check_barrier(self, state):
        """
        Checks if the current state is a local min.
        """

        best_state = state
        best_energy = self.barriers.barriers[(self.barriers.molecules == state).all(axis=1)][0]

        # Check the energies of all surrounding states
        for new_state, new_energy in self.barriers.trans[tuple(state)].values():

            # Check energy if state has not been seen before
            if new_state in self.unseen_states: self.min_max_target(new_state)

            # Check if new neighbour is better
            if abs(self.barrier - new_energy) <= best_energy:

                best_energy = abs(self.barrier - new_energy)
                best_state = new_state

        return best_state

    def find_local_barrier(self, state):
        """
        From a starting state it finds a local minimum.
        """

        while True:
        
            best_state = self.check_barrier(state)
                        
            if best_state == state: break # No improvement
            else: state = best_state

    def find_barrier(self, barrier):
        """
        Uses a voting system to decide which local minimum is the true minimum.
        """

        self.barrier = barrier # The barrier that we are trying to find the closest to.
        self.seen_states = []
        self.seen_barriers = []
        self.unseen_states = [list(i) for i in self.barriers.molecules]
        self.results = []

        while len(self.unseen_states) > 0:

            # Sample a state which has not been seen before
            state = self.unseen_states[np.random.randint(len(self.unseen_states))]
            self.min_max_target(state)
            self.find_local_barrier(state) # Find the min from this start state

        self.results = np.array(self.results).T

    def min_max_target(self, state):
        """
        Finds the current best min, max and target
        """

        self.seen_states.append(state)
        self.seen_barriers.append(self.barriers.barriers[(self.barriers.molecules == state).all(axis=1)][0])
        self.unseen_states.remove(state)

        target = self.seen_barriers[np.abs(self.seen_barriers - self.barrier).argmin()]

        self.results.append([np.min(self.seen_barriers), np.max(self.seen_barriers), target])
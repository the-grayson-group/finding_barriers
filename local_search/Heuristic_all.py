import numpy as np

class Search_agent:

    def __init__(self, barriers):

        self.barriers = barriers

    def check_barrier(self, state, barrier):
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
            if abs(barrier - new_energy) <= best_energy:

                best_energy = abs(barrier - new_energy)
                best_state = new_state

        return best_state

    def find_barrier(self, barrier):
        """
        Uses a voting system to decide which local minimum is the true minimum.
        """

        self.barrier = barrier # The barrier that we are trying to find the closest to.
        self.seen_states = []
        self.seen_barriers = []
        self.unseen_states = [list(i) for i in self.barriers.molecules]
        self.results = []

        state_min, best_state_min, state_max, best_state_max, state_targ, best_state_targ = tuple(None for _ in range(6))

        while True:

            if best_state_min == state_min:

                if len(self.unseen_states) == 0: break
                state_min = self.unseen_states[np.random.randint(len(self.unseen_states))]
                self.min_max_target(state_min)

            else: state_min = best_state_min

            if best_state_max == state_max:

                if len(self.unseen_states) == 0: break
                state_max = self.unseen_states[np.random.randint(len(self.unseen_states))]
                self.min_max_target(state_max)

            else: state_max = best_state_max

            if best_state_targ == state_targ:

                if len(self.unseen_states) == 0: break
                state_targ = self.unseen_states[np.random.randint(len(self.unseen_states))]
                self.min_max_target(state_targ)

            else: state_targ = best_state_targ

            best_state_min = self.check_barrier(state_min, self.barriers.barriers.min())
            best_state_max = self.check_barrier(state_max, self.barriers.barriers.max())
            best_state_targ = self.check_barrier(state_targ, self.barrier)

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
import numpy as np

class MeasureProperties:
    def __init__(self) -> None:
        transitions = [] # store 1000 transitions
        states = [] # list of all states
        phis = [] # list of all representations of all states
        qs = [] # list of shape (n_s, n_a)
        n = len(self.transitions)




        pass

    def complexity_reduction(self):

        run_sum = 0
        n = self.n
        # iterate over all pairs of transitions
        for i in range(n):
            for j in range(n):
                if i < j:

                    d_qij = max(self.qs[i, :] - self.qs[j, :])
                    d_sij = np.sqrt(np.sum(np.square(self.phis[i] - self.phis[j])))
                    ratio = d_qij / d_sij
                    run_sum += ratio
        
        # TODO: calculate L_max

        return 1 - (run_sum * (2 / (n * (n + 1))))

    def orthogonality(self):
        orth_sum = 0
        n = self.n
        for i in range(n):
            for j in range(n):
                if i < j:
                    magn_phi_i = np.linalg.norm(self.phis[i, :])
                    magn_phi_j = np.linalg.norm(self.phis[j, :])
                    orth = np.dot(self.phis[i, :], self.phis[j, :]) / (magn_phi_j * magn_phi_j)
                    orth_sum += orth

        return 1 - (orth_sum * (2 / (n * (n + 1))))

    def sparsity(self):
        sparse_sum = 0
        n = self.n
        for i in range(n):
            d = self.phis[i, :].size
            for j in range(d):
                if (np.dot(self.phis[i, :], self.phis[j, :]) == 0):
                    sparse_sum += 1 / (d * n)

        return sparse_sum





class Transition:
    def __init__(self) -> None:
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

        pass
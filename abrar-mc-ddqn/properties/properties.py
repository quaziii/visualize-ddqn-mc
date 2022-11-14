import numpy as np


class MeasureProperties:
    def __init__(self) -> None:
        transitions = []  # store 1000 transitions
        states = []  # list of all states
        phis = []  # list of all representations of all states
        qs = []  # list of shape (n_s, n_a)
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

    def awareness(self):
        d_s_sum = 0
        sum_of_diff = 0
        n = self.n

        for i in range(n):
            _ = np.random.uniform(1, n, self.phis[j].size)
            for j in _:
                d_s = np.linalg.norm(self.phis[i] - self.phis[j])
                d_s_sum += d_s
            diff = np.linalg.norm(self.phis[i] - self.phis[i + 1])
            sum_of_diff += diff

        return (d_s_sum - sum_of_diff) / d_s_sum

    def diversity(self):
        diverse_sum = 0
        n = self.n
        arr_d_v = np.zeros((n, n))
        arr_d_s = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                d_v = abs(np.max(self.phis[i]) - np.max(self.phis[j]))
                d_s = np.linalg.norm(self.phis[i], self.phis[j])
                arr_d_v[i][j] = d_v
                arr_d_s[i][j] = d_s
        max_d_v = np.max(arr_d_v)
        max_d_s = np.max(arr_d_s)

        for i in range(n):
            for j in range(n):
                ratio = (d_v / max_d_v) / (d_s / (max_d_s + 0.01))
                diverse = ratio if ratio < 1 else 1
                diverse_sum += diverse

        return 1 - (1 / (n * n)) * diverse_sum

    def orthogonality(self):
        orth_sum = 0
        n = self.n

        for i in range(n):
            for j in range(n):
                if i < j:
                    magn_phi_i = np.linalg.norm(self.phis[i])
                    magn_phi_j = np.linalg.norm(self.phis[j])
                    orth = np.dot(self.phis[i], self.phis[j]) / (magn_phi_j * magn_phi_j)
                    orth_sum += orth

        return 1 - (orth_sum * (2 / (n * (n + 1))))

    def sparsity(self):
        sparse_sum = 0
        n = self.n

        for i in range(n):
            d = self.phis[i].size
            for j in range(d):
                if (np.dot(self.phis[i], self.phis[j]) == 0):
                    sparse_sum += 1 / (d * n)

        return sparse_sum


class Transition:
    def __init__(self) -> None:
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

        pass
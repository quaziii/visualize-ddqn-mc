import numpy as np

class MeasureProperties:
    def __init__(self) -> None:
        transitions = [] # store 1000 transitions
        states = [] # list of all states
        phis = [] # list of all representations of all states
        qs = [] # list of shape (n_s, n_a)




        pass

    def complexity_reduction(self):

        run_sum = 0
        n = len(self.transitions)
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



class Transition:
    def __init__(self) -> None:
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

        pass
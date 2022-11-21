import numpy as np

import gym

import random

import torch



class MeasureProperties:
    def __init__(self, n, representation_size, n_states, n_actions, env) -> None:
        self.transitions: list(Transition) = [] # store 1000 transitions
        # self.states = [] 
        self.states = np.zeros((n,)) # list of all states
        self.phis = np.zeros((n, representation_size)) # list of all representations of all states
        self.phis_next = np.zeros((n, representation_size))
        # self.phis = [] 
        self.qs = np.zeros((n, n_actions)) # list of shape (n_s, n_a)
        # self.qs = [] 
        self.n = n
        self.representation_size = representation_size
        self.n_actions = n_actions

        self.env = env

        pass

    def initialize_data(self, agent):
        # populate transitions list

        self.__init__(self.n, self.representation_size, self.n, self.n_actions, self.env)
        # env = gym.make('gym_mc:mc-v0')
        state = self.env.reset()
        # state is an np array

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.device = "cpu"

        


        # run random policy for n episodes
        for _ in range(self.n):
            done = False
            while not done:

                action = self.env.action_space.sample()

                next_state, reward, done, info = self.env.step(action)
                transition = Transition(state, action, next_state, reward)
                state = next_state
                self.transitions.append(transition)

        
        # randomly sample n transitions

        self.transitions = random.sample(self.transitions, self.n)




        for i, transition in enumerate(self.transitions):

            state = torch.FloatTensor(np.array(transition.state)).to(self.device)
            next_state = torch.FloatTensor(np.array(transition.next_state)).to(self.device)
            self.phis[i] = agent.target_model.get_representation(state).detach()
            self.phis_next[i] = agent.target_model.get_representation(next_state).detach()
            self.qs[i] = agent.target_model.forward(state).detach()
            # self.states[i] = transition.state

        # self.states = np.array(self.states)
        # self.phis = np.array(self.phis)
        # self.qs = np.array(self.qs)

        print('LENNN ', len(self.transitions))


        pass

    def get_L_rep(self):

        run_sum = 0
        n = self.n
        # iterate over all pairs of transitions
        for i in range(n):
            for j in range(n):
                if i < j: #  and not all(self.phis[i] == self.phis[j]):
                    d_qij = np.max(np.abs(self.qs[i, :] - self.qs[j, :]))

                    d_sij = np.sqrt(np.sum(np.square(self.phis[i] - self.phis[j])))
                    ratio = d_qij / (d_sij + 1e-9)

                    
                    # print('phiii fml ', self.phis[i] - self.phis[j])

                    if d_sij == 0:
                        # print('diff ', self.phis[i] - self.phis[j])
                        # print(d_sij)
                        # print('phiss i', self.phis[i])
                        # print('phiss j', self.phis[j])
                        # print('state i ', self.transitions[i].state)
                        # print('state j ', self.transitions[j].state)
                        pass
                    run_sum += ratio

        

        return (run_sum * (2 / (n * (n + 1))))


    def complexity_reduction(L):
        L_max = max(L)
        complexity_reduction_values = [1-(L_rep/L_max) for L_rep in L]
        return complexity_reduction_values


    def awareness(self):
        d_s_sum = 0
        sum_of_diff = 0
        n = self.n

        for i in range(n):
            # _ = np.random.uniform(1, n, self.phis[i].size).astype(int)
            _ = np.random.uniform(0, n, 1).astype(int)
            for j in _:
                d_s = np.linalg.norm(self.phis[i] - self.phis[j], ord=1)
                d_s_sum += d_s
            diff = np.linalg.norm(self.phis[i] - self.phis_next[i], ord=1)
            sum_of_diff += diff

        # print('sum of diff: ', sum_of_diff)
        # print('ds sum ', d_s_sum)

        return (d_s_sum - sum_of_diff) / d_s_sum

    def diversity(self):
        diverse_sum = 0
        n = self.n
        arr_d_v = np.zeros((n, n))
        arr_d_s = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                d_v = abs(np.max(self.phis[i]) - np.max(self.phis[j]))
                d_s = np.linalg.norm(self.phis[i] - self.phis[j])
                arr_d_v[i][j] = d_v
                arr_d_s[i][j] = d_s
        max_d_v = np.max(arr_d_v)
        max_d_s = np.max(arr_d_s)

        for i in range(n):
            for j in range(n):
                ratio = (arr_d_v[i][j] / max_d_v) / ((arr_d_s[i][j] / max_d_s) + 0.01)
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
                    orth = np.dot(self.phis[i], self.phis[j]) / ((magn_phi_i * magn_phi_j) + 1e-9)
                    orth_sum += orth

        return 1 - (orth_sum * (2 / (n * (n - 1))))

    def sparsity(self):
        sparse_sum = 0
        n = self.n

        for i in range(n):
            d = self.phis[i].size
            for j in range(d):
                if (self.phis[i][j] == 0):
                    sparse_sum += 1 / (d * n)

        return sparse_sum


class Transition:
    def __init__(self, state, action, next_state, reward) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

        pass

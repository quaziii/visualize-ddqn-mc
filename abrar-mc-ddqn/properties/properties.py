import numpy as np

import gym

import random

import torch

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from ddqn.my_ddqn import DQNAgent



class MeasureProperties:
    def __init__(self, n, representation_size, env) -> None:
        self.transitions: list(Transition) = [] # store 1000 transitions
        # self.states = [] 
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.states = np.zeros((n,)) # list of all states
        self.phis = np.zeros((n, representation_size)) # list of all representations of all states
        self.phis_next = np.zeros((n, representation_size))
        # self.phis = [] 
        self.qs = np.zeros((n, self.n_actions)) # list of shape (n_s, n_a)
        # self.qs = [] 
        self.n = n
        self.representation_size = representation_size
        

        

        pass

    def initialize_data(self, agent):
        # populate transitions list

        self.__init__(self.n, self.representation_size, self.env)
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




class VisualizeRepresentation:
    def __init__(self, agent, env) -> None:
        self.agent = agent
        self.env = env
        pass
    def visualize_env(self, layer='output'):
        # print('XXX ', xx)
        # print('YY', yy)

        xx, yy, zz = self.return_visualization(layer)

        print('ZZZZ', np.flip(zz, axis=0))

        cs = plt.contourf(xx, yy, zz, cmap='Paired')
        plt.clabel(cs, inline=1, fontsize=10)
        plt.show()
        # plt.legend()

        
    # def return_visualization(self, layer='output'):
    def return_decision_boundary(self):
        # show image of NN layer outputs of all states in env

        # if layer == 'output':
        #     network = self.agent.model.forward
        # elif layer == 'hidden':
        #     network = self.agent.model.get_representation

        output_network = self.agent.model.forward
        
        min_position, max_position = self.env.min_position, self.env.max_position

        min_speed, max_speed = -self.env.max_speed, self.env.max_speed


        position_grid = np.arange(min_position, max_position, 0.1)
        speed_grid = np.arange(min_speed, max_speed, 0.01)

        xx, yy = np.meshgrid(position_grid, speed_grid)

        

        r1, r2 = xx.flatten(), yy.flatten()

        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        grid = torch.tensor(np.hstack((r1,r2))).float()

        grid_output = torch.argmax(output_network(grid), dim=1).detach().cpu().numpy()
        # print('OUTPUTSSS ', grid_output)

        zz = grid_output.reshape(xx.shape)

        return xx, yy, zz

    def return_tsne_class_clusters(self):
        min_position, max_position = self.env.min_position, self.env.max_position

        min_speed, max_speed = -self.env.max_speed, self.env.max_speed

        output_network = self.agent.model.forward
        representation_network = self.agent.model.get_representation

        position_grid = np.arange(min_position, max_position, 0.1)
        speed_grid = np.arange(min_speed, max_speed, 0.01)

        xx, yy = np.meshgrid(position_grid, speed_grid)

        r1, r2 = xx.flatten(), yy.flatten()

        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        grid = torch.tensor(np.hstack((r1,r2))).float()

        grid_output = torch.argmax(output_network(grid), dim=1).detach().cpu().numpy()


        grid_representations = representation_network(grid).detach().cpu().numpy()

        tsne = TSNE(n_components=2).fit_transform(grid_representations)

        tsne_0_range = (np.max(tsne[:, 0]) - np.min(tsne[:, 0]))
        tsne_1_range = (np.max(tsne[:, 1]) - np.min(tsne[:, 1]))

        tsne_0 = tsne[:, 0] - np.min(tsne[:, 0])
        tsne_1 = tsne[:, 1] - np.min(tsne[:, 1])

        tsne_0 = tsne_0 / tsne_0_range
        tsne_1 = tsne_1 / tsne_1_range
        tsne[:, 0] = tsne_0
        tsne[:, 1] = tsne_1

        # plot scatter plot of grid_output vs tsne

        # print(' --- TSNEEEEE ---')
        # print(tsne[:,0])
        # print(tsne[:,1])
        # print(grid_output.flatten())

        return tsne[:, 0], tsne[:, 1], grid_output.flatten()
        scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=grid_output.flatten())
        classes = ['Left', 'Coast', 'Right']
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        plt.show()

    
class AgentPropertiesWrapper:
    '''
    Agent wrapper class is just to compute all the properties and representation stuff for each epoch. 
    '''
    def __init__(self, env, gamma, tau, learning_rate, n) -> None:
        '''
        Need agent parameters to create a new agent from each model state dictionary in return_properties.
        '''
        
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.n = n

        pass
    def return_properties(self, agent_model_state_dicts):
        '''
        Calculate properties and representation stuff and return everything for each model state dictionary in agent_model_state_dicts
        '''


        # episode properties and rewards keep track of properties and rewards for each of the milestone episodes
        milestone_properties = {
                'l_rep': [],
                'awareness': [],
                'orthogonality': [],
                'sparsity': [],
                'diversity': [],
                'complexity_reduction': []
            }

        milestone_rewards = []

        n_milestones = len(agent_model_state_dicts)
        # use dummy agent to get shapes of heatmap, decision boundary and tsne visualization
        dummy_agent = DQNAgent(self.env, gamma=self.gamma, tau=self.tau, learning_rate=self.learning_rate)


        # used to plot heatmaps of hidden layer weights for each milestone
        milestone_heatmaps = np.empty((n_milestones, dummy_agent.model.representation_layer[2].weight.shape[0], dummy_agent.model.representation_layer[2].weight.shape[1]))


        

        
        properties = MeasureProperties(self.n, dummy_agent.model.representation_size, self.env)
        visualize_rep = VisualizeRepresentation(dummy_agent, self.env)

        decision_boundary_classes_shape = visualize_rep.return_decision_boundary()[2].shape
        tsne_classes_shape = visualize_rep.return_tsne_class_clusters()[2].shape


        # used to plot decision boundaries of model for each milestone
        milestone_decision_boundaries = {
            'state_x': np.empty((n_milestones, *decision_boundary_classes_shape)),
            'state_y': np.empty((n_milestones, *decision_boundary_classes_shape)),
            'class':  np.empty((n_milestones, *decision_boundary_classes_shape)),
        }

        # used to plot tsne plots of hidden layer vs output class of each milestone
        milestone_tsne_class_clusters = {
            'tsne_x': np.empty((n_milestones, *tsne_classes_shape)),
            'tsne_y': np.empty((n_milestones, *tsne_classes_shape)),
            'class': np.empty((n_milestones, *tsne_classes_shape)),
        }

        for i, agent_model_state_dict in enumerate(agent_model_state_dicts):
            agent = DQNAgent(self.env, gamma=self.gamma, tau=self.tau, learning_rate=self.learning_rate)
            agent.model.load_state_dict(agent_model_state_dict)

            # initialize classes for this particular agent config
            properties = MeasureProperties(self.n, agent.model.representation_size, self.env)
            visualize_rep = VisualizeRepresentation(agent, self.env)

            # store properties
            properties.initialize_data(agent)
            milestone_properties['l_rep'].append(properties.get_L_rep())
            milestone_properties['awareness'].append(properties.awareness())
            milestone_properties['orthogonality'].append(properties.orthogonality())
            milestone_properties['sparsity'].append(properties.sparsity())
            milestone_properties['diversity'].append(properties.diversity())

            # store heatmap
            milestone_heatmaps[i] = agent.model.representation_layer[2].weight.detach().numpy()

            # store decision boundaries
            milestone_decision_boundaries['state_x'][i], milestone_decision_boundaries['state_y'][i], milestone_decision_boundaries['class'][i] = visualize_rep.return_decision_boundary()

            # store tsne plots of hidden layers 
            milestone_tsne_class_clusters['tsne_x'], milestone_tsne_class_clusters['tsne_y'],milestone_tsne_class_clusters['class'] = visualize_rep.return_tsne_class_clusters()

            # store reward
            # LATER IF NEEDED
        
        # compute complexity reductions
        milestone_properties['complexity_reduction'] = MeasureProperties.complexity_reduction(milestone_properties['l_rep'])

        # convert milestone lists to np arrays

        for item in milestone_properties:
            milestone_properties[item] = np.array(milestone_properties[item])
        
        # return milestone_properties, milestone_heatmaps, milestone_decision_boundaries, milestone_tsne_class_clusters

        return {
            'milestone_properties': milestone_properties,
            'milestone_heatmaps': milestone_heatmaps,
            'milestone_decision_boundaries': milestone_decision_boundaries,
            'milestone_tsne_class_clusters': milestone_tsne_class_clusters
        }



class Transition:
    def __init__(self, state, action, next_state, reward) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

        pass

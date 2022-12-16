import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# torch.manual_seed(100)

import numpy as np
import gym

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.representation_size = 16

        # print('self.input dim ', self.input_dim[0])
        
        # self.fc = nn.Sequential(
        #     nn.Linear(self.input_dim[0], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(self.representation_size, self.output_dim)
        # )

        self.representation_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.representation_size),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(self.representation_size, self.output_dim)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0.01)


    def forward(self, state):

        y = self.get_representation(state)


        qvals = self.output_layer(y)
        return qvals

    def get_representation(self, state):
        
        
        # y = fc(state)
        # print('shapeee ' , y.shape)
        return self.representation_layer(state)



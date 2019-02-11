import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, fc1=256, fc2=128, leak=0.01, seed=42):
        """ Initialize parameters and build model.

        Params
        ======
            state_size (int): Size of state tensor
            action_size (int): Size of action tensor
            seed (int): Random seed
            fc1 (int): Number of hidden units in FC layer 1
            fc2 (int): Number of hidden units in FC layer 2           
            leak (float): the leak rate for leaky ReLU, i.e. the alpha in (x < 0) * alpha * x + (x >= 0) * x
        """
        super(Actor, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

        self.bn = nn.BatchNorm1d(state_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize FC layers followed by leaky ReLU using Kaiming He's (2015) approach.
        Source: https://arxiv.org/pdf/1502.01852v1.pdf
        For more info see here:
            https://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/
        """
        torch.nn.init.kaiming_normal_(
            self.fc1.weight.data,
            a=self.leak, 
            nonlinearity='leaky_relu',
            mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data,
                                      a=self.leak,
                                      nonlinearity='leaky_relu',
                                      mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data,
                               -3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = self.bn(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x =  torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, fc1=256, fc2=128, fc3=64, leak=0.01, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize FC layers followed by leaky ReLU using Kaiming He's (2015) approach.
        Source: https://arxiv.org/pdf/1502.01852v1.pdf
        For more info see here:
            https://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/
        """
        torch.nn.init.kaiming_normal_(self.fcs1.weight.data, a=self.leak,
                                      nonlinearity='leaky_relu', mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak,
                                      nonlinearity='leaky_relu', mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self.bn(state)
        x = F.leaky_relu(self.fcs1(state), negative_slope=self.leak)
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leak)
        x =  self.fc4(x)
        return x

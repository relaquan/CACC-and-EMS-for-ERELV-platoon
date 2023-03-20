"""
Actor (Policy) Model:
state_size -> fc1 -> 256 -> fc2 -> 128 -> fc3 -> action_size

Critic (Value) Model:
state_size -> fc1 -> 128 -> 128+action_size -> fc2 -> 64 -> fc3 -> 32 -> fc4 -> Q-values
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)   # 状态 -> 256
        self.bn1 = nn.BatchNorm1d(fc1_units)       # 归一化
        self.fc2 = nn.Linear(fc1_units, fc2_units)   # 256 -> 128
        self.fc3 = nn.Linear(fc2_units, action_size)  # 128 -> 动作
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))# 状态 -> 256
        x = self.bn1(x)
        x = F.relu(self.fc2(x))# 256 -> 128
        return torch.tanh(self.fc3(x))# 128 -> 动作

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, full_state_size, actions_size, seed=0, fcs1_units=128, fc2_units=64, fc3_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)# 状态 -> 128
        self.bn1 = nn.BatchNorm1d(fcs1_units)# 归一化
        self.fc2 = nn.Linear(fcs1_units+actions_size, fc2_units)# 128+动作 -> 64
        self.fc3 = nn.Linear(fc2_units, fc3_units)# 64 -> 32
        self.bn2 = nn.BatchNorm1d(fc3_units)# 归一化
        self.fc4 = nn.Linear(fc3_units, 1)# 32 -> 1
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state)) # 状态 -> 128
        xs = self.bn1(xs)
        x = torch.cat((xs, action), dim=1)# 128+动作
        x = F.relu(self.fc2(x))# 128+动作 -> 64
        x = F.relu(self.fc3(x)) # 64 -> 32
        return self.fc4(x)# 32 -> 1

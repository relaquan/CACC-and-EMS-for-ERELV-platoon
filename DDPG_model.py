"""
Actor（策略）网络：
state_size -> fc1 -> 256 -> fc2 -> 128 -> fc3 -> action_size

Critic（价值）网络：
state_size -> fc1 -> 128 -> 128+action_size -> fc2 -> 64 -> fc3 -> 32 -> fc4 -> Q-values
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 隐藏层定义
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Actor（策略）网络
class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """
        初始化参数和建立网络模型
            state_size (int): 每个状态维度
            action_size (int):每个动作维度
            seed (int):     随机种子
            fc1_units (int):  第一隐藏层节点数
            fc2_units (int):  第二隐藏层节点数
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)   # 状态 -> 256
        self.bn1 = nn.BatchNorm1d(fc1_units)      # 归一化
        self.fc2 = nn.Linear(fc1_units, fc2_units)    # 256 -> 128
        self.fc3 = nn.Linear(fc2_units, action_size) # 128 -> 动作
        self.reset_parameters()

    def reset_parameters(self):
        """网络参数初始化"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """构建一个Actor(策略)网络，状态 -> 动作"""
        x = F.relu(self.bn1(self.fc1(state))) # 状态 -> 256
        x = F.relu(self.fc2(x))   # 256 -> 128
        return F.tanh(self.fc3(x))   # 128 -> 动作

# Critic（价值）网络
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64, fc3_units=32):
        """
        初始化参数和建立网络模型
            state_size (int):   每个状态维度
            action_size (int): 每个动作维度
            seed (int):           随机种子
            fc1_units (int):    第一隐藏层节点数
            fc2_units (int):    第二隐藏层节点数
            fc3_units (int):    第三隐藏层节点数
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units) # 状态 -> 128
        self.bn1 = nn.BatchNorm1d(fc1_units) # 归一化
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units) # 128+动作 -> 64
        self.fc3 = nn.Linear(fc2_units, fc3_units)# 64 -> 32
        self.bn2 = nn.BatchNorm1d(fc3_units)   # 归一化
        self.fc4 = nn.Linear(fc3_units, 1)# 32 -> 1
        self.reset_parameters()

    def reset_parameters(self):
        """网络参数初始化"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """构建Critic(价值)网络，（状态，动作）-> Q值"""
        xs = F.relu(self.bn1(self.fc1(state)))  # 状态 -> 128
        x = torch.cat((xs, action), dim=1) # 128+动作
        x = F.relu(self.fc2(x)) # 128+动作 -> 64
        x = F.relu(self.fc3(x)) # 64 -> 32
        return self.fc4(x) # 32 -> 1
        

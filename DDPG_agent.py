"""
Agent类：
        __init__ ：state_size, action_size, random_seed
        step ：state, action, reward, next_state, done, t
        act：state, add_noise=True
        learn：experiences, gamma 
        soft_update：local_model, target_model, tau 
OUNoise类：
        __init__：size, seed, mu=0., theta=0.15, sigma=0.1
ReplayBuffer类：
        __init__：action_size, buffer_size, batch_size, seed 
        add：state, action, reward, next_state, done 
"""

import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from DDPG_model import Actor, Critic

BUFFER_SIZE = int(1e6)  # 重放缓冲区大小
BATCH_SIZE = 512        # 训练样本大小
GAMMA = 0.99            # 折扣因子
TAU = 1e-3                  # 用于软更新的目标参数
LR_ACTOR = 1e-4         # Actor 学习率
LR_CRITIC = 1e-3        # Critic学习率
WEIGHT_DECAY = 0.0      # 权重衰减项，防止过拟合
EPISODES_BEFORE_TRAINING = 3 # 多少回合后开始训练

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 智能体
class Agent():

    def __init__(self, state_size, action_size, random_seed):
        """
        初始化智能体
            state_size (int):   每个状态维度
            action_size (int): 每个动作维度
            random_seed (int): 随机种子
        """
        self.state_size = state_size # 状态维度
        self.action_size = action_size # 动作维度
        self.seed = random.seed(random_seed) 

        # Actor 网络
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)   # Actor当前网络
        self.actor_target = Actor(state_size, action_size, random_seed).to(device) # Actor目标网络
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) 

        # Critic网络 
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)  # Critic当前网络
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)# Critic目标网络
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # 噪音
        self.noise = OUNoise(action_size, random_seed)

        # 经验池
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done, t):
        """将经验保存在重放缓冲区中，并使用随机样本从缓冲区学习"""
        # 保存每个Agent的经验/奖励
        for state, action, reward, next_state, done in zip(state, action, reward, next_state, done):
            self.memory.add(state, action, reward, next_state, done)

        # 有足够的样本后开始学习
        if len(self.memory) > BATCH_SIZE and t > EPISODES_BEFORE_TRAINING:  # 三个循环后开始训练
            for i in range(3):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """根据当前策略返回相应状态的动作"""
        state = torch.from_numpy(state).float().to(device)# 转为tensor
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise: # 添加噪声
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        使用给定的经验元组更新策略网络和价值网络的参数 
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        式中：actor_target(state) -> action
                  critic_target(state, action) -> Q-value
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- Critic网络更新---------------------------- #
        # 从目标网络中获得下一状态的动作和Q值
        actions_next = self.actor_target(next_states) # Actor目标网络：状态->动作
        Q_targets_next = self.critic_target(next_states, actions_next)# Critic目标网络：（状态，动作）-> Q值
        # 计算当前状态的Q值
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # 计算Critic当前网络损失
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # 最小化损失
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ----------------------------Actor网络更新---------------------------- #
        # 计算Actor当前网络损失
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # 最小化损失
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----------------------更新目标网络----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """
        软更新网络模型参数
        θ_target = τ*θ_local + (1 - τ)*θ_target
            local_model: 当前网络（复制）
            target_model: 目标网络（粘贴）
            tau (float): 用于目标参数的软更新
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Ornstein-Uhlenbeck过程 
class OUNoise:

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """初始化参数"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """恢复初始状态"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """更新内部状态并将其作为噪声样本返回"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

# 重放缓冲区
class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """初始化重放缓冲区
            buffer_size (int): 最大容量
            batch_size (int):  batch大小
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """在记忆中添加新经验"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """从记忆中随机抽取batch大小的经验"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """返回当前内存大小"""
        return len(self.memory)

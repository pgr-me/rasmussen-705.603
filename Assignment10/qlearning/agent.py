#!/usr/bin/env python3

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlearning.experience import ReplayBuffer
from qlearning.qnet import QNetwork

# Adapted from Johns Hopkins Masters in AI course EN.705.603.82.SP23, module 10


class Agent():
    """
    --------
    Deep Q-Learning Agent
    --------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'lr' -> learning rate
        'gamma' -> discount factor
        'tau' -> For soft update of target parameters
        'buffer_size' -> Replay buffer size
        'batch_size' -> Number of observations trained per batch
        'update_every' -> update frequency
        'seed' -> used for random module
        'device' -> CPU or GPU
    --------
    """
    def __init__(
            self,
            state_size: int,
            action_size: int,
            lr: float,
            gamma: float,
            tau: float,
            buffer_size: int,
            batch_size: int,
            update_every: int,
            seed: int,
            device: torch.device
        ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.seed = random.seed(seed)
        self.device = device

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """
        --------------
        Take an action given the current state (S(i))
        --------------
        [Params]
            'state' -> current state
            'eps' -> current epsilon value
        --------------
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).cpu().data.numpy()
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values), np.max(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """
        --------
        Update our target network with the weights from the local network
        --------
        Formula for each param (w): w_target = τ*w_local + (1 - τ)*w_target
        See https://arxiv.org/pdf/1509.02971.pdf
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


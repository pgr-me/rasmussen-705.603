#!/usr/bin/env python3

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch

# Adapted from Johns Hopkins Masters in AI course EN.705.603.82.SP23, module 10

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    ------------
    Used to store agent experiences
    ------------
    [Params]
        'action_size' -> length of the action space
        'buffer_size' -> Max size of our memory buffer
        'batch_size' -> how many memories to randomly sample
        'seed' -> seed for random module
        'device' -> CPU or GPU
    ------------
    """
    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int, device: torch.device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    

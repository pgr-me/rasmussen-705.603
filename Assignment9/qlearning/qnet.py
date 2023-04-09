import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from Johns Hopkins Masters in AI course EN.705.603.82.SP23, module 10

class QNetwork(nn.Module):
    """
    -------
    Neural Network Used for Agent to Approximate Q-Values
    -------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    """
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
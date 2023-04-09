#!/usr/bin/env python3

# Adapted from Johns Hopkins Masters in AI course EN.705.603.82.SP23, module 10

import random
from typing import Tuple

from gym.wrappers.order_enforcing import OrderEnforcing
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class JupyterDisplay(object):
    def __init__(self, env: OrderEnforcing, figsize: Tuple):
        self.figsize = figsize
        self.mode = "rgb_array"
        self.env = env
    
    def show(self, env=None):
        if env is None:
            env = self.env
        plt.figure(figsize=self.figsize)
        plt.imshow(env.render())               # Removed render mode for compatibility
        plt.axis('off')
        display.clear_output(wait=True)
        display.display(plt.gcf())


def get_action(env: OrderEnforcing, qtable: NDArray, state: Tuple, epsilon: float):
    """
    Function that obatins the action to be taken as either exploration or exploitation

    Arguments:
        env: The blackjack gym environment
        qtable: numpy array with dimension for each of three state description lus a dimension for each action
        state: tuple of shape (idx1, idx2, idx3) where idx 1 and 2 are integers and idx 3 is a bool. idx 1 2 and 3 are described above.
        epsilon: float representing likelihood of action being random exploration.

    Returns: int for action
    """
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        idx1, idx2, idx3 = get_state_idxs(state)
        action = np.argmax(qtable[idx1][idx2][idx3])
    return action
        

def get_state_idxs(state: Tuple):
    """
    Function that plays blackjack with provided qtable as policy.  This function required some modification as that original was for an older version of gym.  Mods are noted below.

    Arguments:
        state: tuple of shape (idx1, idx2, idx3) where idx 1 and 2 are integers and idx 3 is a bool

    Returns: Tuple of state indexes
    """
    idx1, idx2, idx3 = state
    idx3 = int(idx3)
    return idx1, idx2, idx3


def print_policy(qtable):
    """
    Print policy given qtable.
    
    Arguments:
        qtable: numpy array with dimension for each of the three state descriptions plus a dimension for each action.

    """
    print('PC DC Soft Pol')
    dim1, dim2, dim3, dim4 = qtable.shape
    for player_count in range(10,21):
        for dealer_card in range(dim2):
            for soft in range(dim3):
                q_stay = qtable[player_count, dealer_card, soft, 0]
                q_hit  = qtable[player_count, dealer_card, soft, 1]
                pol = "Stay" if q_stay>=q_hit else "Hit"
                print(player_count+1, dealer_card+1, soft, pol)


def train_agent(env: OrderEnforcing,
                qtable: np.ndarray,
                num_episodes: int,
                alpha: float, 
                gamma: float, 
                epsilon: float, 
                epsilon_decay: float) -> np.ndarray:
    """
    Function that develops q table by playing game with passed parameters.  This function required some modification as that original was for an older version of gym.  Mods are noted below.

    Arguments:
        env: The blackjack gym environment
        qtable: numpy array with dimension for each of three state description plus a dimension for each action
        num_episodes: int for number of games to play
        alpha: float learning rate
        gamma: float discount factor
        epsilon: float representing likelihood of action being random exploration.
        epsilon_decay: float rate of decline of epsilon

    Returns: qtable for trained policy
    """

    for episode in range(num_episodes):
        state, _ = env.reset()                                     # Added blank for extra returned argument
        done = False
        while True:
            action = get_action(env, qtable, state, epsilon)
            new_state, reward, done, _, info = env.step(action)    # Added blank for extra returned argument
            qtable = update_qtable(qtable, state, action, reward, new_state, alpha, gamma)
            state = new_state
            if done:
                break
        epsilon = np.exp(-epsilon_decay*episode)
    return qtable


def update_qtable(
        qtable: NDArray,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        alpha: float,
        gamma: float
):
    """
    Function that uses the Bellman equation to update a qtable given a learning rate and a discount faactor. 

    Arguments:
        qtable: numpy array with dimension for each of three state description lus a dimension for each action
        state: tuple of shape (idx1, idx2, idx3) where idx 1 and 2 are integers and idx 3 is a bool. idx 1 2 and 3 are described above.
        action: int represeting Hit(1) or Stick(0)
        reward: float representing reward for a given step
        next_state: the next state resulting from action.
        alpha: learning rate
        gamma: discount factor

    Returns: Updated qtable
    """
    curr_idx1, curr_idx2, curr_idx3 = get_state_idxs(state)
    next_idx1, next_idx2, next_idx3 = get_state_idxs(next_state)
    curr_state_q = qtable[curr_idx1][curr_idx2][curr_idx3]
    next_state_q = qtable[next_idx1][next_idx2][next_idx3]
    qtable[curr_idx1][curr_idx2][curr_idx3][action] += \
            alpha * (reward + gamma * np.max(next_state_q) - curr_state_q[action])
    return qtable


def watch_trained_agent(
        env: OrderEnforcing,
        qtable: NDArray,
        num_rounds: int,
        epsilon: float,
        figsize: Tuple[int, int],
        showfig: bool=True,
):
    """
    Function that plays blackjack with provided qtable as policy.  This function required some modification as that original was for an older version of gym.  Mods are noted below.

    Arguments:
        env: The blackjack gym environment
        qtable: numpy array with dimension for each of three state description plus a dimension for each action
        num_rounds: int for number of games to play
        epsilon: likelihood of action being random exploration
        figsize: Figure size
        showfig: True to display figure
    
    Returns: List of rewards
    """
    if showfig:
        envdisplay = JupyterDisplay(env, figsize=figsize)
    rewards = []
    for s in range(1, num_rounds+1):
        state, _ = env.reset()
        done = False
        round_rewards = 0
        while True:
            action = get_action(env, qtable, state, epsilon)          
            new_state, reward, done, _, info = env.step(action)  # Added blank for extra returned argument
            if showfig:
                envdisplay.show(env)

            round_rewards += reward
            state = new_state
            if done == True:
                break
        rewards.append(round_rewards)
    return rewards


def watch_trained_agent_no_exploration(env: OrderEnforcing, qtable: NDArray, num_rounds: int, epsilon: float, figsize: Tuple[int, int], showfig: bool=True):
    """
    Function that plays blackjack with provided qtable as policy. This function required soem modification as that original was for an older version of gym. Mods are noted below. This function has epsilon set to zero.

    Arguments:
        env: The blackjack gym environment
        qtable: NumPy array with dimension for each of three state description plus a dimension for each action
        num_rounds: Int for number of games to play
        epsilon: likelihood of action being random exploration
        figsize: Figure size
        showfig: True to display figure
    """
    if showfig:
        envdisplay = JupyterDisplay(env, figsize=figsize)
    rewards = []
    for s in range(1, num_rounds+1):
        state, _ = env.reset()
        done = False
        round_rewards = 0
        while True:
            action = get_action(env, qtable, state, 0)                # epsilon set to 0
            new_state, reward, done, _, info = env.step(action)  # Added blank for extra returned argument
            if showfig:
                envdisplay.show(env)

            round_rewards += reward
            state = new_state
            if done == True:
                break
                
        rewards.append(round_rewards)
    return rewards

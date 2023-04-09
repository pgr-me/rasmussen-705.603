#!/usr/bin/env python3
from collections import deque
from pathlib import Path

import numpy as np
import torch

from qlearning.lookup_rl import get_state_idxs

# Adapted from Johns Hopkins Masters in AI course EN.705.603.82.SP23, module 10

def dqn(agent, model_path: Path, env, n_episodes: int=2000, eps_start: float=0.99, eps_end: float=0.02, eps_decay: float=0.995, verbose: bool=False):
    """
    -------------
    Train a Deep Q-Learning Agent
    -------------
    [Params]
        'agent' -> agent to train
        'model_path' -> path to model
        'env' -> environment
        'n_episodes' -> number of episodes to train for
        'eps_start' -> epsilon starting value
        'eps_end' -> epsilon minimum value
        'eps_decay' -> how much to decrease epsilon every iteration
        'verbose' -> True to print state, epsilon, and action every loop
    -------------
    """

    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start                   
    
    for episode in range(1, n_episodes+1):
        done = False
        episode_score = 0
        
        state, _ = env.reset()                                 # Added _ for new version of gym
        state = np.array(get_state_idxs(state), dtype=float)
        state[0] = state[0]/32
        state[1] = state[1]/10
        
        while not done:
            if verbose:
                print(state, eps)
                print(type(state), type(eps))
            action = agent.act(state, eps)
            if verbose:
                print(action, type(action))
            if isinstance(action, tuple):
                action, value = action
            else:
                value = 1.
            next_state, reward, done, _, _ = env.step(action)   # Added second _ for new version of gym
            reward *= value
            next_state = np.array(get_state_idxs(next_state), dtype=float)
            next_state[0] = next_state[0]/32
            next_state[1] = next_state[1]/10
        
            agent.step(state, action, reward, next_state, done)   
            state = next_state
            episode_score += reward
        
        scores_window.append(episode_score)
        scores.append(episode_score)
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 5000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_path)
            
    return scores

#!/usr/bin/env python
# coding: utf-8

# Assignment 10 - 705.603 - Local Blackjack Optimization 
# Adapted from Johns Hopkins Masters in AI course EN.705.603.82.SP23, module 10
# ## Blackjack
# 
# ### Environment Details
# 
#     ### Action Space
#     There are two actions: stick (0), and hit (1).
#     
#     ### Observation Space
#     Tuple(Discrete(32), Discrete(11), Discrete(2))
#     The observation consists of a 3-tuple containing: 
#         1. the player's current sum
#         2. the value of the dealer's one showing card (1-10 where 1 is ace)
#         3. whether the player holds a usable ace (0 or 1).
#         
#     ### Rewards
#     - win game: +1
#     - lose game: -1
#     - draw game: 0
#     - win game with natural blackjack:
#         +1.5 (if natural is True)
#         +1 (if natural is False)


import argparse
from pathlib import Path

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlearning import Agent, dqn
from qlearning.lookup_rl import (
    JupyterDisplay,
    get_state_idxs,
    train_agent,
    watch_trained_agent,
    watch_trained_agent_no_exploration,
)


if __name__ == "__main__":
    print("")
    print(80 * "~")
    print("Define inputs")
    FIGSIZE = (8, 4)
    RANDOM_SEED = 777
    rl_dir = Path("/rl")
    data_dir = rl_dir / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(exist_ok=True, parents=True)
    interim_dir.mkdir(exist_ok=True, parents=True)
    processed_dir.mkdir(exist_ok=True, parents=True)
    model_path = processed_dir / "local_notebook_model_checkpoint.pth"

    print("")
    print(80 * "~")
    print("Commence tabular / lookup reinforcement learning routine.")
    print("Create Blackjack Environment and test methods.")
    env = gym.make("Blackjack-v1")
    env.reset(seed=RANDOM_SEED)

    # get initial state
    state = env.reset()

    state_size = [x.n for x in env.observation_space]
    action_size = env.action_space.n

    qtable = np.zeros(state_size + [action_size]) #init with zeros

    alpha = 0.3 # learning rate
    gamma = 0.1 # discount rate
    epsilon = 0.9     # probability that our agent will explore
    decay_rate = 0.005

    # training variables
    num_hands = 500_000
    print("Train look-based Q-learning agent.")
    qtable = train_agent(env,
                        qtable,
                        num_hands,
                        alpha,
                        gamma,
                        epsilon,
                        decay_rate)

    print(f"\tQtable Max: {np.max(qtable)}")
    print(f"\tQtable Mean: {np.mean(qtable)}")
    print(f"\tQtable Num Unique Vals: {len(np.unique(qtable))}")

    print("Watch trained agent.")
    env = gym.make("Blackjack-v1", render_mode='rgb_array')    # Added render mode for compatibility
    rewards = watch_trained_agent(env, qtable, num_rounds=100, epsilon=epsilon, figsize=FIGSIZE, showfig=False)
    env.close()

    print("Output reward over hands played")
    dst = processed_dir / "local_py_lookup_total_rewards_over_time.png"
    plt.figure(figsize=(12,8))
    plt.plot(np.cumsum(rewards))
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title("Total Rewards Over Time")
    plt.savefig(dst)
    print(f"Save test with exploration fig at {dst}")


    print("Test without exploration")
    # Watch trained agent
    env = gym.make("Blackjack-v1", render_mode='rgb_array')    # Added render mode for compatibility
    #env.action_space.seed(42)
    rewards = watch_trained_agent_no_exploration(env, qtable, num_rounds=100, epsilon=epsilon, figsize=FIGSIZE, showfig=False)
    env.close()

    dst = processed_dir / "local_py_lookup_total_rewards_over_time_no_exploration.png"
    plt.figure(figsize=(12,8))
    plt.plot(np.cumsum(rewards))
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title("Total Rewards Over Time")
    plt.savefig(dst)
    print(f"Save test without exploration fig at {dst}")

    print("")
    print(80 * "~")
    print("Commence deep reinforcement learning routine.")
    print("Deep reinforcement learning classes and functions are provided in the `qlearning` module.")
    
    print("Set deep reinforcement learning params.")
    BUFFER_SIZE = int(1e3)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 3e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    N_EPISODES = 70_000


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    print("Instantiate env and agent.")
    env = gym.make("Blackjack-v1")
    env.action_space.seed(42)
    agent = Agent(
        state_size=3,
        action_size=2,
        lr=LR,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        update_every=UPDATE_EVERY,
        seed=0,
        device=device,
    )
    print("Train the deep learning reinforcement model.")
    scores = dqn(agent, model_path, env, n_episodes=N_EPISODES)
    env = gym.make("Blackjack-v1", render_mode='rgb_array')    # Added render mode for compatibility
    num_hands = 100
    
    agent.qnetwork_local.load_state_dict(torch.load(model_path))

    results = []
    for hand in range(num_hands):
        state, _ = env.reset()                                   # Added _ for new version of gym
        state = np.array(get_state_idxs(state), dtype=float)
        state[0] = state[0]/32
        state[1] = state[1]/10

        done = False
        while not done:
            frame = env.render()        
            action = agent.act(state)
            if isinstance(action, tuple):
                action, value = action
            else:
                value = 1.

            state, reward, done, _, _ = env.step(action)         # Added second _ for new version of gym
            reward *= value
            state = np.array(get_state_idxs(state), dtype=float)
            state[0] = state[0]/32
            state[1] = state[1]/10

        # envdisplay.show(env)
        results.append(reward)

    env.close()

    batting_avg = np.argwhere(np.array(results) > 0).size / len(results)
    print(f"Batting Average: {batting_avg*100:.2f}%")

    dst = processed_dir / "local_py_dl_rewards_over_time.png"
    plt.figure(figsize=(10,6))
    plt.plot(np.cumsum(results))
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title("Total Rewards Over Time")
    plt.savefig(dst)

    dst = processed_dir / "local_py_dl_total_cash_over_time.png"
    start_cash = 100.
    pct_gain = ((start_cash + np.sum(results)*1000) - 100) / 100
    print(f"Percent Gain: {pct_gain*100:.2f}%")
    plt.figure(figsize=(12,8))
    plt.plot(start_cash + np.cumsum(results)*1000, c="g")
    plt.ylabel("Cash ($)")
    plt.xlabel("Dealt Hands")
    plt.title(f"Total Cash Over Time | Starting Cash: ${int(start_cash)} | Win Pct: {batting_avg:.2f}%", c="darkgreen")
    plt.savefig(dst)

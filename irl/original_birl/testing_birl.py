# They just parameterize their rewards as a function of the state.
# There is a separate element of the reward vector for each state.
# Already not sustainable I think especially in high dimensional spaces.

# This is exactly what the AVRIl paper does.

# How do you generate a posterior with that? There's no way to, because the reward is specific to each state.
# You can't even visualize the posterior in the original BIRL paper.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from irl.original_birl.birl.gridworld import GridWorld
from irl.original_birl.birl.mdp import MDP
import numpy as np
from irl.original_birl.birl.birl import birl
from irl.original_birl.birl.main import initialize_gridworld, initialize_rewards, initialize_true_rewards, convert_state_coord_to_state_idx, convert_state_idx_to_state_coord
from irl.original_birl.birl.prior import *
import sys


## Create gridworld
grid_dimension = 2
iterations = 900
reward_vec = [0.0, 0.0, 1.0, 0.0]
r_max = 1
transitions = initialize_gridworld(grid_dimension, grid_dimension)
n_unique_states, n_unique_actions, _ = transitions.shape

## Create state coordinates
x1s = np.arange(0, grid_dimension)
x2s = np.arange(0, grid_dimension)
X1, X2 = np.meshgrid(x1s, x2s)
state_coords = np.vstack([X1.ravel(), X2.ravel()]).T

## Initialize MDP
mdp = MDP(transitions, initialize_rewards(5, n_unique_states, state_coords), 0.99)
mdp.true_reward = initialize_true_rewards(5, n_unique_states, state_coords, reward_vec)
thing = GridWorld(mdp)
# demos = thing.record(1)


## Create trajectories for training
n_trajectories = 200

LEFT = [0, -1]
RIGHT = [0, 1]
UP = [-1, 0]
DOWN = [1, 0]
ACTIONS = [UP, DOWN, LEFT, RIGHT]

## This basically goes through the grid toward the goal state repeatedly
trajectories = [(0, np.random.choice([1, 3]))]
curr_state_coord = np.array([0, 0])
curr_action_idx = trajectories[0][1]
for ii in range(1, n_trajectories):
    # if ii % 20 == 0:
    # 	trajectories.append((0, np.random.choice([1, 3])))
    # 	curr_state_coord = np.array([0, 0])
    # 	continue


    last_state, last_action = trajectories[ii - 1]
    curr_state_coord += ACTIONS[last_action]
    curr_state_idx = convert_state_coord_to_state_idx(curr_state_coord.reshape(1, -1), grid_dimension)[0]

    ## Reached goal state
    if curr_state_idx == n_unique_states - 1:
        trajectories.append((n_unique_states - 1, np.random.choice([1, 3])))
        curr_state_coord = np.array([0, 0])
        continue

    hypothetical_next_states = curr_state_coord + np.vstack(ACTIONS)
    valid_idx = np.where(((hypothetical_next_states < 0) | (hypothetical_next_states > grid_dimension - 1)).sum(1) == 0)[0]
    valid_actions = np.vstack(ACTIONS)[valid_idx]
    hypothetical_next_states = hypothetical_next_states[valid_idx]
    hypothetical_next_states_idx = convert_state_coord_to_state_idx(hypothetical_next_states, grid_dimension)
    hypothetical_next_rewards = mdp.true_reward[hypothetical_next_states_idx]

    # Add random noise to break ties
    hypothetical_next_rewards += np.random.uniform(low=-1e-6, high=1e-6, size=len(hypothetical_next_rewards))

    max_idx = np.argmax(hypothetical_next_rewards)
    next_state_idx = hypothetical_next_states_idx[max_idx]
    best_action_idx = valid_idx[max_idx]

    curr_tuple = (curr_state_idx, best_action_idx)
    trajectories.append(curr_tuple)




# ## Fit BIRL (this takes a long time even when the gridsize is 10)
# for iteration in range(10):
#     for it in [20, 50, 100, 500, 1000, 5000, 10000]:
#         policy, reward_samples = birl(mdp=mdp,step_size=0.05,r_max=r_max,demos=demos,iterations=it,burn_in=0,sample_freq=1,
#                                       prior=PriorDistribution.UNIFORM)
#         cols = [str(i) for i in range(grid_dimension*grid_dimension)]
#         sample_df = pd.DataFrame(np.vstack(reward_samples), columns=cols)
#         sample_df.to_csv("birl_reward_samples_grid=" + str(grid_dimension) + "_iterations=" + str(it) + "_allneighbors_reward_vec=" + str(
#             reward_vec) +
#                          "_rmax=" + str(r_max) + "_rmin=0_iteration=" + str(iteration) + ".csv")

# hyperparams = [1000]
# for h in hyperparams:
#     print("H: ", h)
demos = [(1, trajectories, 400)]
policy, reward_samples = birl(mdp=mdp,step_size=0.05,r_max=r_max,demos=demos,iterations=iterations,burn_in=0,sample_freq=1,
                                      prior=PriorDistribution.UNIFORM)
cols = [str(i) for i in range(grid_dimension*grid_dimension)]
sample_df = pd.DataFrame(np.vstack(reward_samples), columns=cols)
sample_df.to_csv("additional_experiments/reward_vec=" + str(reward_vec) + ".csv")






#!/usr/bin/env python
from irl.original_birl.birl.gridworld import GridWorld
from irl.original_birl.birl.mdp import MDP
import numpy as np
from irl.original_birl.birl.birl import birl
from irl.original_birl.birl.prior import *


def initialize_gridworld(width, height):
	# where 24 is a goal state that always transitions to a
	# special zero-reward terminal state (25) with no available actions
	num_states = width * height
	trans_mat = np.zeros((num_states, 4, num_states))

	# NOTE: the following iterations only happen for states 0-23.
	# This means terminal state 25 has zero probability to transition to any state,
	# even itself, making it terminal, and state 24 is handled specially below.

	# Action 1 = down
	for s in range(num_states):
		if s < num_states - width:
			trans_mat[s, 1, s + width] = 1
		else:
			trans_mat[s, 1, s] = 1

		# Action 0 = up
	for s in range(num_states):
		if s >= width:
			trans_mat[s, 0, s - width] = 1
		else:
			trans_mat[s, 0, s] = 1

	# Action 2 = left
	for s in range(num_states):
		if s % width > 0:
			trans_mat[s, 2, s - 1] = 1
		else:
			trans_mat[s, 2, s] = 1

	# Action 3 = right
	for s in range(num_states):
		if s % width < width - 1:
			trans_mat[s, 3, s + 1] = 1
		else:
			trans_mat[s, 3, s] = 1

	# Finally, goal state always goes to zero reward terminal state
	for a in range(4):
		for s in range(num_states):
			trans_mat[num_states - 1, a, s] = 0
		trans_mat[num_states - 1, a, num_states - 1] = 1

	return trans_mat


def initialize_rewards(dims, num_states, state_coords):
	# weights = np.random.normal(0, 0.25, dims)
	# rewards = dict()
	# for i in range(num_states):
	# 	rewards[i] = np.dot(weights, np.random.normal(-3, 1, dims))
	# 	#Give goal state higher value
	# rewards[num_states - 1] = 10

	# rewards = np.random.normal(loc=-3, scale=1, size=num_states)
	rewards = np.zeros(num_states)
	# rewards[-1] = 10.
	return rewards

def initialize_true_rewards(dims, num_states, state_coords, reward_vec=None):
	# weights = np.random.normal(0, 0.25, dims)
	# rewards = dict()
	# for i in range(num_states):
	# 	rewards[i] = np.dot(weights, np.random.normal(-3, 1, dims))
	# 	#Give goal state higher value
	# rewards[num_states - 1] = 10
	weights = np.array([0.1, 0.1])
	rewards = state_coords @ weights - 2
	if reward_vec is not None:
		rewards = np.asarray(reward_vec) - 2
		print(str(rewards))
		# rewards = np.random.normal(loc=-3, scale=1, size=num_states)
	else:
		rewards[-1] = 10.

	return rewards

def convert_state_coord_to_state_idx(state_coord, grid_dimension):
	return grid_dimension * state_coord[:, 0] + state_coord[:, 1]

def convert_state_idx_to_state_coord(state_idx, grid_dimension):
	return np.array([state_idx // grid_dimension, state_idx % grid_dimension])

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd

	## Create gridworld
	grid_dimension = 10
	transitions = initialize_gridworld(grid_dimension, grid_dimension)
	n_unique_states, n_unique_actions, _ = transitions.shape

	## Create state coordinates
	x1s = np.arange(0, grid_dimension)
	x2s = np.arange(0, grid_dimension)
	X1, X2 = np.meshgrid(x1s, x2s)
	state_coords = np.vstack([X1.ravel(), X2.ravel()]).T

	## Initialize MDP
	mdp = MDP(transitions, initialize_rewards(5, n_unique_states, state_coords), 0.99)
	mdp.true_reward = initialize_true_rewards(5, n_unique_states, state_coords)
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


	# demonstration_density_map = np.zeros((grid_dimension, grid_dimension))
	# for traj in trajectories:
	# 	curr_coord = convert_state_idx_to_state_coord(traj[0], grid_dimension=grid_dimension)
	# 	demonstration_density_map[curr_coord[0], curr_coord[1]] += 1


	demos = [(1, trajectories, 400.0)]

	## Fit BIRL
	policy, reward_samples = birl(mdp=mdp,step_size=0.05,r_max=1.0,demos=demos,iterations=600,burn_in=50,sample_freq=1,prior=PriorDistribution.UNIFORM)

	sums = np.sum(reward_samples, axis=0)
	np.save("birl_posterior_sum.npy", sums)

	import ipdb; ipdb.set_trace()

	## Plot posteriors
	sample_df = pd.DataFrame(np.vstack(reward_samples), columns=["Top left", "Top right", "Bottom left", "Bottom right (target)"])
	sns.kdeplot(data=sample_df) #, palette=["gray", "gray", "gray", "red"])
	plt.xlabel("Reward")
	plt.ylabel("p(R | data)")
	plt.show()
	import ipdb

	ipdb.set_trace()
	print("Finished BIRL")
	print("Agent Playing")
	reward, playout = thing.play(policy)
	print("Reward is " + str(reward))
	print("Playout is " + str(playout))

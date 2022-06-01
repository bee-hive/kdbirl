'''
All the IRL functions for the inventory class.
'''
import os, sys
import pandas as pd
import numpy as np
from irl.inventory.inventory_env import InventoryEnv
from irl.clinical.baselines.baselines.deepq import deepq
from irl.clinical.sepsis_env import SepsisEnv as ModifiedSepsisEnv

def callback(lcl, _glb):
	# stop training if reward exceeds 199
	is_solved = lcl['t'] > 1000  # and sum(lcl['episode_rewards'][:100]) / 100 >= 199
	return is_solved

def runFQI(reward_vec, it=100):
	env = InventoryEnv(n=reward_vec[0], k=reward_vec[1], p=reward_vec[2], c=reward_vec[3], h=reward_vec[4])
	act = deepq.learn(
		env,
		network='mlp',
		lr=1e-3,
		total_timesteps=10,  # Change back to 1000
		buffer_size=500,
		exploration_fraction=0.1,
		exploration_final_eps=0.02,
		print_freq=10,
		callback=callback
	)
	# print("Saving model to sepsis_qmodel.pkl")
	# act.save("sepsis_qmodel_reward=" + str(reward_vec) + ".pkl")

	# Create trajectories
	points = []
	rewards = []
	actions = []
	for traj in range(it):
		dem_points, dem_rewards, dems_actions = generate_trajectories(reward_vec, 20, act=act)
		points.extend(dem_points)
		rewards.extend(dem_rewards)
		actions.extend(dems_actions)
	return points, actions, rewards

def generate_trajectories(reward_vec, iterations, act):
	env = InventoryEnv(n=reward_vec[0], k=reward_vec[1], p=reward_vec[2], c=reward_vec[3], h=reward_vec[4])
	_ = env.reset()
	dems_points = []
	dems_actions = []
	dems_rewards = []
	for it in range(iterations):
		obs = np.expand_dims(env.state, axis=0)
		action = act.step(obs)
		action = action[0].numpy()[0]
		env.step(action)
		# state, action, next state,
		dems_points.append(env.state)
		dems_actions.append(action)
		dems_rewards.append(reward_vec)
	return dems_points, dems_rewards, dems_actions

# reward_vec = [100, 5, 2, 3, 2]
# env = InventoryEnv(n=reward_vec[0], k=reward_vec[1], p=reward_vec[2], c=reward_vec[3], h=reward_vec[4])
# states = []
# for i in range(10):
# 	states.append(env.state)
# 	env.step(1)
# import ipdb; ipdb.set_trace()

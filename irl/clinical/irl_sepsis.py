from gym_sepsis.envs.sepsis_env import SepsisEnv
from irl.clinical.sepsis_env import SepsisEnv as ModifiedSepsisEnv
import tqdm
import pystan
from hashlib import md5
from os.path import join as pjoin
import os
import pickle
import numpy as np
import tqdm
import pandas as pd
import scipy
from irl.clinical.baselines.baselines.deepq import deepq


def callback(lcl, _glb):
	# stop training if reward exceeds 199
	is_solved = lcl['t'] > 1000  # and sum(lcl['episode_rewards'][:100]) / 100 >= 199
	return is_solved


def runFQI(reward_vec, dim, it=100, full=False):
	env = ModifiedSepsisEnv(reward_params=reward_vec, dim=dim)
	act = deepq.learn(
		env,
		network='mlp',
		lr=1e-3,
		total_timesteps=100,  # Change back to 1000
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
		if full:
			dem_points, dem_rewards, dem_actions = generate_trajectories(reward_vec, 20, dim=dim, act=act, full=full)
			points.extend(dem_points)
			rewards.extend(dem_rewards)
			actions.extend(dem_actions)
		else:
			dem_points, dem_rewards = generate_trajectories(reward_vec, 20, act=act, dim=dim, full=full)
			points.extend(dem_points)
			rewards.extend(dem_rewards)
	if full:
		return points, actions, rewards
	else:
		return points, rewards

def runFQI_evd(reward_vec, dim):
	env = ModifiedSepsisEnv(reward_params=reward_vec, dim=dim)
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
	features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
				'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
				'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
				'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
				'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
				'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
				'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
				'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
				'blood_culture_positive', 'action', 'state_idx']
	if dim == 3:
		states = ['sofa', 'qsofa', 'qsofa_sysbp_score']
	else:
		states = ['sofa', 'qsofa', 'qsofa_sysbp_score', 'qsofa_gcs_score', 'qsofa_resprate_score', 'LACTATE']
	state_idx = [features.index(s) for s in states]
	env = ModifiedSepsisEnv(reward_params=reward_vec, dim=dim)
	_ = env.reset()
	done = False
	while not done:
		obs = np.expand_dims(env.s, axis=0)
		action = act.step(obs)
		action = action[0].numpy()[0]
		_, r, done, _ = env.step(action)
		# state, action, next state,
		points.append(env.s.flatten()[state_idx])

	return points


def value_function(true_reward_fn, eval_fn, dim):
	points = runFQI_evd(eval_fn, dim=3)
	reward = 0
	for i in range(len(points) - 1):
		diff = np.subtract(points[i], points[i+1])
		reward += np.dot(true_reward_fn, diff)
	return reward


def generate_trajectories(reward_vec, iterations, dim, act=None, full=False):
	features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
				'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
				'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
				'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
				'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
				'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
				'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
				'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
				'blood_culture_positive', 'action', 'state_idx']

	if dim == 3:
		states = ['sofa', 'qsofa', 'LACTATE']#['sofa', 'qsofa', 'qsofa_sysbp_score']
	else:
		states = ['sofa', 'LACTATE']#['sofa', 'qsofa',
	# 'qsofa_sysbp_score', 'qsofa_gcs_score', 'qsofa_resprate_score', 'LACTATE']
	state_idx = [features.index(s) for s in states]
	env = ModifiedSepsisEnv(reward_params=reward_vec, dim=dim)
	_ = env.reset()
	dems_points = []
	dems_actions = []
	dems_rewards = []
	for it in range(iterations):
		if act == None:
			env.step(1)
		else:
			obs = np.expand_dims(env.s, axis=0)
			action = act.step(obs)
			action = action[0].numpy()[0]
			env.step(action)
		# state, action, next state,
		dems_points.append(env.s.flatten()[state_idx])
		dems_actions.append(action)
		dems_rewards.append(reward_vec)
	if full:
		return dems_points, dems_rewards, dems_actions
	else:
		return dems_points, dems_rewards


# Find optimal bandwidth
def distance_points(p1, p2, h):
	dist = scipy.spatial.distance.euclidean(p1, p2)  # + scipy.spatial.distance.euclidean(p1[1], p2[1])
	return np.exp(-np.power(dist, 2) / (2 * h))


def distance_r(r, r_prime, h_prime):
	dist = scipy.spatial.distance.euclidean(r, r_prime)
	return np.exp(-(np.power(dist, 2) / (2 * h_prime)))


def find_bandwidth(observations_points, observations_rewards):
	# Distance between rewards, h_prime
	all_distances_r = []
	all_distances_p = []
	for ii, o in enumerate(tqdm.tqdm(observations_points)):
		for jj, o_prime in enumerate(observations_points):
			all_distances_p.append(distance_points(o, o_prime, h=0.2))
			all_distances_r.append(distance_r(observations_rewards[ii], observations_rewards[jj], h_prime=0.2))
	h_prime = np.std(all_distances_r) * np.std(all_distances_r)
	h = np.std(all_distances_p) * np.std(all_distances_p)

	return h, h_prime

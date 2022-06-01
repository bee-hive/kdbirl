import os, sys
import numpy as np
import pandas as pd
from irl.irl_gridworld import runNonparametricFQI
import scipy
import tqdm
import pickle


ns = [10, 100, 500, 1000, 10000]
m = 600
true_reward = [0.1, 0.2, 0.3, 0.4]
eps = 3.5
gridsize = 2
num_rl = 15

# Generate the training dataset
observations_points = []
observations_rewards = []
observations = []

reward_vecs = []
for i in range(num_rl):
	reward_vecs.append(np.random.uniform(low=0.0, high=1.0, size=gridsize*gridsize))

idx = [i for i in range(len(reward_vecs))]
size_rl = min(num_rl, len(reward_vecs))
new_idx = np.random.choice(idx, size=size_rl, replace=False)
for i, r in enumerate(reward_vecs):
	if i in new_idx:
		reward_vec = reward_vecs[i]
		policy_rollouts = runNonparametricFQI(reward_vec, gridsize, num_rollouts=1000)
		for sample in policy_rollouts:
			observations_points.append(sample[0])
			observations_rewards.append(reward_vec)
			observations.append((sample[0], reward_vec))


idx = [i for i in range(len(observations_points))]
size_obs = min(m, len(observations_points))
new_idx = np.random.choice(idx, size=size_obs, replace=False)
observations_points = np.asarray(observations_points)[new_idx]
observations_rewards = np.asarray(observations_rewards)[new_idx]
pickle.dump(observations_points, open("training_points.pkl", 'wb'))
pickle.dump(observations_rewards, open("training_rewards.pkl", 'wb'))
observations_points = pickle.load(open("training_points.pkl", 'rb'))
observations_rewards = pickle.load(open("training_rewards.pkl", 'rb'))
print("len observations: ", observations_points.shape[0])

ratios = []

def distance_r_euclidean(r, r_prime):
	h_prime = 0.2
	dist = scipy.spatial.distance.euclidean(r, r_prime)
	# Euclidean distance
	return np.exp(-(np.power(dist, 2) / (2 * h_prime)))

def distance_rewards(r_k, observations):
	sum_diff = 0
	for sample in observations:
		sum_diff += distance_r_euclidean(r_k, sample)
	return sum_diff

def distance_points(p1, p2):
	h = 0.05
	dist = scipy.spatial.distance.euclidean(p1, p2)
	return np.exp(-np.power(dist, 2) / (2 * h))

def conditional_dist(sa_i, dataset_points, dataset_rewards, reward):
	sum = 0
	dist_rewards = distance_rewards(reward, dataset_rewards)
	for sa_j, r_j in zip(dataset_points, dataset_rewards):
		weight = distance_r_euclidean(reward, r_j) / dist_rewards
		dist = distance_points(sa_i, sa_j)
		est = dist * weight
		sum += est
	return sum

def l1_distance(reward_sample, true_reward, observations_points, observations_rewards):
	l1 = 0
	for sa_i in observations_points:
		likelihood_sample = conditional_dist(sa_i, observations_points, observations_rewards, reward_sample)
		likelihood_true = conditional_dist(sa_i, observations_points, observations_rewards, true_reward)
		l1 += np.abs(likelihood_true-likelihood_sample)
	return l1
m = 1000
mean_distances = []
for n in ns:
	count_under = 0
	total = 0
	dists = []
	for i in range(2):
		samples = pd.read_csv("/home/amandyam/contrastive-rl/contrastive-rl/irl/experiments/kdbirl_concentration/iterations/fig1/kdbirl_samples_rmax=1"
							  ".0_linear=False_m=" + str(m) + "_n=" + str(n) + "_iteration=" + str(i) + "_rmin=0_fn=[0.1, 0.2, 0.3, "
																			   "0.4]h=0.01_h_prime=0.2_numrl_1000.csv")

		for kk, i in enumerate(tqdm.tqdm(range(samples.shape[0]))):
			if i%20 != 0:
				continue
			sample = samples.iloc[i][1:]
			# Calculate L1 distance between the likelihood of that reward sample and the true reward
			dist = l1_distance(sample, true_reward, observations_points, observations_rewards)
			dists.append(dist)
			if dist <= eps:
				count_under += 1
			total += 1
	mean_distances.append(np.mean(dists))
	ratios.append(count_under/total)
	print(str(ratios))

print("Ns: ", ns)
print("ratios: ", ratios)
print(str(mean_distances))
pickle.dump(zip(ns, ratios), open("contraction/eps=" + str(eps) + ".pkl", 'wb'))


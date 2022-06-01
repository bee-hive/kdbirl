'''
Metropolis-Hastings algorithm for sampling from the KD-BIRL posterior.
'''
import os, sys
from irl.irl_gridworld import runNonlinearFQI, targetstate_to_rewardvec, runLinearFQI, linearreward_to_rewardvec, \
	estimate_expert_posterior_nonparametric, runNonparametricFQI
import numpy as np
import tqdm
import pandas as pd

np.random.seed(123)
linear = True
gridsize=2
reward_fn = [1, 1]
target_state = [1, 1]
iterations = 100
m = 1000
n = 100

if linear:
	observations_points = []
	observations_rewards = []
	observations = []
	init_experience = 200
	for aa, i in enumerate(tqdm.tqdm(range(1, 11, 5))):
		for j in range(1, 11, 5):
			r = [i / 10, j / 10]
			if np.sum(np.asarray(r)) == 0:
				continue
			behavior_opt, opt_agent = runLinearFQI(dataset='bg', init_experience=init_experience, num_rollouts=1000, behavior=True,
												   reward_weights_shared=r, gridsize=gridsize)
			r_vec = linearreward_to_rewardvec(r, gridsize)
			r_vec /= np.max(np.abs(r_vec))
			for sample in behavior_opt:
				observations_points.append(sample[0])
				observations_rewards.append(r_vec)
				s = (sample[0], sample[1], r_vec)
				observations.append(s)

	vecs = []
	for i in range(1, 11, 5):
		for j in range(1, 11, 5):
			for k in range(1, 11, 5):
				for l in range(1, 11, 5):
					vecs.append([i/10, j/10, k/10, l/10])
	for aa, ll in enumerate(tqdm.tqdm(range(len(vecs)))):
		reward_vec = vecs[aa]
		policy_rollouts = runNonparametricFQI(reward_vec, gridsize, num_rollouts=100)
		for sample in policy_rollouts:
			observations_points.append(sample[0])
			observations_rewards.append(reward_vec)
			s = (sample[0], sample[1], reward_vec)
			observations.append(s)

	behavior_opt, _ = runLinearFQI(init_experience=200, behavior=True, reward_weights_shared=reward_fn, gridsize=gridsize, num_rollouts=1000)
	behavior_points = []
	for b in behavior_opt:
		behavior_points.append(b[0])

else:
	observations_points = []
	observations_rewards = []
	observations = []
	init_experience = 200
	target_states = []
	for i in range(0, gridsize):
		for j in range(0, gridsize):
			target_states.append([i, j])

	for kk, r in enumerate(tqdm.tqdm(target_states)):
		behavior_opt = runNonlinearFQI(init_experience=init_experience, behavior=True, target_state=r, n=gridsize, num_rollouts=1000)
		for sample in behavior_opt:
			observations.append((sample[0], targetstate_to_rewardvec(r, gridsize)))
			observations_points.append(sample[0])
			observations_rewards.append(targetstate_to_rewardvec(r, gridsize))

	behavior_opt = runNonlinearFQI(init_experience=200, behavior=True, target_state=target_state, n=2, num_rollouts=100)

	behavior_points = []
	for b in behavior_opt:
		behavior_points.append(b[0])

idx = [i for i in range(len(behavior_points))]
size_obs = min(n, len(behavior_points))
new_idx = np.random.choice(idx, size=size_obs, replace=False)
behavior_opt = np.asarray(behavior_opt)[new_idx]

idx = [i for i in range(len(observations_points))]
size_obs = min(m, len(observations_points))
new_idx = np.random.choice(idx, size=size_obs, replace=False)
observations = np.asarray(observations)[new_idx]

# x is a reward vector
# probability function is a posterior calculation
def p_x(x):
	prob = estimate_expert_posterior_nonparametric(x, behavior_opt, observations)
	return prob

# Metropolis hastings
x_0 = np.random.rand(4)
thresh = 1
reward_samples = []
for ii, it in enumerate(tqdm.tqdm(range(iterations))):
	prob_x_0 = p_x(x_0)

	x_1 = x_0 + np.random.normal(scale=0.5)

	prob_x_1 = p_x(x_1)

	if prob_x_0 / prob_x_1 > thresh:
		x_0 = x_1
	else:
		x_0 = x_0
	reward_samples.append(x_0)

cols = [str(i) for i in range(gridsize*gridsize)]
sample_df = pd.DataFrame(np.vstack(reward_samples), columns=cols)
sample_df.to_csv("kdbirl_concentration/kdbirl_mh_samples_linear=" + str(linear) + "_m=" + str(m) + "_n=" + str(n) +
					 "fn=" + str(reward_fn) + ".csv")
print(str(x_0))
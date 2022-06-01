from irl.irl_gridworld import runNonlinearFQI, targetstate_to_rewardvec, runLinearFQI, linearreward_to_rewardvec, find_optimal_bandwidth, \
	runNonparametricFQI
import pystan
from hashlib import md5
from os.path import join as pjoin
import os
import pickle
import numpy as np
import tqdm
import pandas as pd

with open("../kdbirl.stan", "r") as file:
	model_code = file.read()
code_hash = md5(model_code.encode("ascii")).hexdigest()
cache_fn = pjoin("cached_models", "cached-model-{}.pkl".format(code_hash))

if os.path.isfile(cache_fn):
	print("Loading cached model...")
	sm = pickle.load(open(cache_fn, "rb"))
else:
	print("Saving model to cache...")
	sm = pystan.StanModel(model_code=model_code)
	with open(cache_fn, "wb") as f:
		pickle.dump(sm, f)


def sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n, m, h, h_prime, linear, step, iteration, reward_fn=None, save=True,
				  num_rl=1000):
	if linear:
		observations_points = []
		observations_rewards = []
		observations = []
		init_experience = 200
		for aa, i in enumerate(tqdm.tqdm(range(-110, 110, 5))):
			for j in range(-110, 110, 5):
				r = [i / 100, j / 100]
				behavior_opt, opt_agent = runLinearFQI(dataset='bg', init_experience=init_experience, num_rollouts=1000, behavior=True,
													   reward_weights_shared=r, gridsize=gridsize)
				r_vec = linearreward_to_rewardvec(r, gridsize)
				r_vec /= np.max(np.abs(r_vec))
				for sample in behavior_opt:
					observations_points.append(sample[0])
					observations_rewards.append(r_vec)
					observations.append((sample[0], r_vec))

		behavior_opt, _ = runLinearFQI(init_experience=200, behavior=True, reward_weights_shared=reward_fn, gridsize=gridsize, num_rollouts=1000)
		behavior_points = []
		for b in behavior_opt:
			behavior_points.append(b[0])

	elif step:
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

		behavior_opt = runNonlinearFQI(init_experience=init_experience, behavior=True, target_state=target_state, n=gridsize, num_rollouts=1000)

		behavior_points = []
		for b in behavior_opt:
			behavior_points.append(b[0])

	else:  # Nonzero reward
		observations_points = []
		observations_rewards = []
		observations = []

		reward_vecs = []
		# for i in range(0, 10, 3):
		# 		# 	for j in range(0, 10, 3):
		# 		# 		for k in range(0, 10, 3):
		# 		# 			for l in range(0, 10, 3):
		# 		# 				reward_vec = [i/10, j/10, k/10, l/10]
		# 		# 				reward_vecs.append(reward_vec)
		for i in range(40):
			reward_vecs.append(np.random.uniform(low=0.0, high=1.0, size=gridsize * gridsize))

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

		behavior_opt = runNonparametricFQI(reward_fn, gridsize, num_rollouts=10000)
		behavior_points = []
		for b in behavior_opt:
			behavior_points.append(b[0])

	idx = [i for i in range(len(behavior_points))]
	size_obs = min(n, len(behavior_points))
	new_idx = np.random.choice(idx, size=size_obs, replace=False)
	behavior_points = np.asarray(behavior_points)[new_idx]
	print("Len behavior: ", behavior_points.shape[0])

	idx = [i for i in range(len(observations_points))]
	size_obs = min(m, len(observations_points))
	new_idx = np.random.choice(idx, size=size_obs, replace=False)
	observations_points = np.asarray(observations_points)[new_idx]
	observations_rewards = np.asarray(observations_rewards)[new_idx]
	print("len observations: ", observations_points.shape[0])

	# print(str(find_optimal_bandwidth(observations, gridsize, metric_r='euclidean')))

	# Fit and check if chains mixed
	rhat_failed = True
	if linear:
		kdbirl_data = {"J": gridsize * gridsize, "n": n, "m": m, "training_points":
			observations_points, "training_rewards": observations_rewards, "behavior_points": behavior_points, "h": h, "h_prime": h_prime}
	else:
		kdbirl_data = {"J": gridsize * gridsize, "n": n, "m": m, "training_points":
			observations_points, "training_rewards": observations_rewards, "behavior_points": behavior_points, "h": h, "h_prime": h_prime}

	while rhat_failed:
		fit = sm.sampling(data=kdbirl_data, iter=n_iter, warmup=n_iter - n_posterior_samples, chains=1, control={"adapt_delta": 0.85}
						  )
		rhat_vals = fit.summary()["summary"][:, -1]
		print("RHAT: ", rhat_vals)
		rhat_failed = np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1)

	# Get samples, it's an 800 by 4 numpy array
	sample_reward = np.squeeze(fit.extract()["sample_reward"])
	cols = [str(i) for i in range(gridsize * gridsize)]
	sample_df = pd.DataFrame(np.vstack(sample_reward), columns=cols)
	print(str(fit))
	if save:
		sample_df.to_csv("kdbirl_concentration/additional_experiments/kdbirl_samples__m=" + str(m) + "_n=" + str(n) +
						 "_fn=" + str(reward_fn) + "h=" + str(h) + "_h_prime=" + str(h_prime) + ".csv")


# Fit the step function model
# n_iter = 1000
# n_posterior_samples = 400
# gridsize = 2
# target_state = [1, 1]
# # h = 1.0
# # h_prime = 1.0
# linear = False
# step = True
# m = 1000
# n = 1000
# reward_fn = [0, 0, 0, 1]
#
# hs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# hprimes = [0, 0.2, 0.4, 0.6, 0.6, 1.0]
# for h in hs:
# 	for h_prime in hprimes:
# 		sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n, m, h, h_prime, linear, step, reward_fn=reward_fn, save=True)

# Fit the nonzero function model
n_iter = 1000
n_posterior_samples = 400
gridsize = 2
target_state = [1, 1]
linear = False
step = False
m = 100
n = 100
# reward_fn = [0.2, 0.6, 0.4, 0.8]
hs = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
hprimes = [0.01, 0.2, 0.4, 0.6, 0.6, 1.0]
reward_fn = [0, 0, 0, 1]

for h in hs:
	for h_prime in hprimes:
		sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n, m, h, h_prime, linear, step, reward_fn=reward_fn, save=True,
					  iteration=0)
# h = 0.01
# h_prime = 0.2
# print("H: " + str(h) + " Hprime: " + str(h_prime))
# # sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n, m, h, h_prime, linear, step, reward_fn=reward_fn, save=True,
# # 			  iteration=0)
#
# # reward_fn = [0, 0.5, 0.55, 0.9]
# # for h in hs:
# # 	for h_prime in hprimes:
# print("H: " + str(h) + " Hprime: " + str(h_prime))
# # sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n, m, h, h_prime, linear, step, reward_fn=reward_fn, save=True,
# # 			  iteration=0)
#
# reward_fn = [0, 0, 1, 0]
# step = True
# target_state = [1, 0]
# # for h in hs:
# # 	for h_prime in hprimes:
# print("H: " + str(h) + " Hprime: " + str(h_prime))
# sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n, m, h, h_prime, linear, step, reward_fn=reward_fn, save=True,
# 			  iteration=0)
# sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n=1000, m=1000, h=h, h_prime=h_prime, linear=False, step=False, iteration=0,
# 			  reward_fn=reward_fn, save=True)

# for m in [500, 1000, 10000]:#[10, 100, 500, 1000, 10000]:
# 	for n in [10, 100, 500, 1000, 10000]:
# 		for it in range(5):
# 			num_rl = 1000
# 			if os.path.exists("kdbirl_concentration/iterations/fig1/kdbirl_samples_rmax=1.0_linear=" + str(linear) +
# 			"_m=" + str(m) + "_n=" + str(n) + "_iteration=" + str(it) +
# 			"_rmin=0_fn=" + str(reward_fn) + "h=" + str(h) + "_h_prime=" + str(h_prime) + "_numrl_" + str(num_rl) + ".csv"):
# 				print("continuing")
# 			else:
# 				break

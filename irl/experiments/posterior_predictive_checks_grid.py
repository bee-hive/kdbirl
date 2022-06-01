from irl.irl_gridworld import heatmap_dataset_conditional_density_posterior, most_common_reward, find_target_state, runNonlinearFQI, \
    generate_dataset_plot, conditional_dist, estimate_expert_posterior, find_optimal_bandwidth, \
    heatmap_dataset_conditional_density_posterior_nonlinear_nonparametric, heatmap_dataset_conditional_density_posterior_nonparametric, runLinearFQI
import numpy as np
import os
import pickle
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# reward = [-1, 1]
# target_state = [0, 3]
#
# num_samples_per_rollout = 10000
# observations = []

# for file in os.listdir("/tigress/BEE/bayesian_irl/datasets/grid_datasets/"):
#     fname = "/tigress/BEE/bayesian_irl/datasets/grid_datasets/" + file
#     if "target" in fname:
#         continue
#     filehandler = open(fname, "rb")
#     behav = pickle.load(filehandler)
#     idx = [i for i in range(len(behav))]
#     size = min(num_samples_per_rollout, len(behav))
#     if size > 0:
#         new_behav_idx = np.random.choice(idx, size=size, replace=False)
#         new_behav = np.asarray(behav)[new_behav_idx]
#         observations.extend(new_behav)
#
# # n = 50
# # for m in [1, 2, 5, 10, 15]:
# # 	print("m=" + str(m) + " n=" + str(n))
# # 	heatmap_posterior = heatmap_dataset_conditional_density_posterior(reward, observations, gridsize=10, plot_loc="plots/grid/", m=m, n=n)


# target_state = [9, 9]
# num_samples_per_rollout = 10000
# observations = []
#
# datadir = "/tigress/BEE/bayesian_irl/datasets/grid_datasets/"
# for i in range(0, 10, 2):
# 	for j in range(0, 10, 2):
# 		fname = datadir + "rollouts_target=" + str([i, j]) + ".pkl"
# 		filehandler = open(fname, "rb")
# 		behav = pickle.load(filehandler)
# 		idx = [k for k in range(len(behav))]
# 		size = min(num_samples_per_rollout, len(behav))
# 		if size > 0:
# 			new_behav_idx = np.random.choice(idx, size=size, replace=False)
# 			new_behav = np.asarray(behav)[new_behav_idx]
# 			observations.extend(new_behav)

# target_states = []
# for i in range(2):
# 	for j in range(2):
# 		target_states.append([i, j])
#
# observations = []
# for kk, r in enumerate(tqdm.tqdm(target_states)):
# 	obs = []
# 	behavior_opt = runNonlinearFQI(init_experience=200, behavior=True, target_state=r, n=2, num_rollouts=10)
# 	for sample in behavior_opt:
# 		s = (sample[0], sample[1], r)
# 		obs.append(s)
# 	observations.extend(obs)

observations = []
init_experience = 200
gridsize = 2
for aa, i in enumerate(tqdm.tqdm(range(-10, 11, 3))):
	for j in range(-10, 11, 3):
		r = [i / 10, j / 10]
		behavior_opt, opt_agent = runLinearFQI(dataset='bg', init_experience=init_experience, num_rollouts=10, behavior=True, reward_weights_shared=r, gridsize=gridsize)
		for sample in behavior_opt:
			s = (sample[0], sample[1], r)
			observations.append(s)

plot_loc = 'plots/grid/'
reward = [1, 1]
heatmap_dataset_conditional_density_posterior_nonparametric(reward, observations, gridsize=2, plot_loc=plot_loc, m=1000, n=1000)
# m = 2000
# for n in [100, 500, 1000, 2000, 3000, 6000, 10000]:
# 	plot_loc = 'plots/grid/'
# 	heatmap_dataset_conditional_density_posterior_nonlinear(target_state, observations, gridsize=10, plot_loc=plot_loc, m=m, n=n)
#
#
# n = 2500
# for m in [100, 500, 1000, 2000, 3000, 6000, 10000]:
# 	plot_loc = 'plots/grid/'
# 	heatmap_dataset_conditional_density_posterior_nonlinear(target_state, observations, gridsize=10, plot_loc=plot_loc, m=m, n=n)
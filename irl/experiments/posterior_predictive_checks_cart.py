import random
import pickle
import numpy as np
from irl.irl_cart import heatmap_dataset_conditional_density_posterior, most_common_reward, fqi, test_reward_parameters
import os
random.seed(42)
import matplotlib.pyplot as plt

#Unpackage the observations dataset
reward_pos = [i/10 for i in range(5, 6, 2)]
reward_ang = [i/10 for i in range(1, 4, 2)]

# Generate a lot of data points for each reward function combination. We can arbitrarily use as many samples as we need.
num_samples_per_rollout = 100
observations = []
#
for file in os.listdir("/tigress/BEE/bayesian_irl/datasets/successful_rollouts/"):
	fname = "/tigress/BEE/bayesian_irl/datasets/successful_rollouts/" + file
	filehandler = open(fname, "rb")
	b = pickle.load(filehandler)
	idx = [i for i in range(len(b))]
	size = min(num_samples_per_rollout, len(b))
	if size > 0:
		new_behav_idx = np.random.choice(idx, size=size, replace=False)
		new_behav = np.asarray(b)[new_behav_idx]
		observations.extend(new_behav)

# Varying n (number of expert trajectories)
# Take 20 samples from each of the posteriors
# m = 5000
# reward = [1.7, 0.7]
# original_perf = test_reward_parameters(reward[0], reward[1], rollouts=10, nns=1, verbose=True)
# reward_pos = [i / 10 for i in range(5, 25, 2)]
# reward_ang = [i / 10 for i in range(1, 10, 2)]
# n_s = [10, 100, 1000, 10000, 150, 200, 2000, 250, 2500, 3000, 4000, 50, 500, 5000, 6000, 8000]
# evds = []
# all_rl = []
# all_ra = []
# for n in n_s:
# 	fname = "plots/cart/" + "r=" + str(reward) + "_m=" + str(m) + "_n=" + str(n) + ".npy"
# 	posterior = np.load(fname)
# 	# TODO: sample 20 elements from the posterior here
# 	rewards = most_common_reward(posterior, reward_pos, reward_ang)
# 	mode_reward = rewards[0][0]
# 	perf = test_reward_parameters(mode_reward[0], mode_reward[1], rollouts=10, nns=1, verbose=True)
# 	evds.append(abs(np.mean(original_perf) - np.mean(perf)))
#
# plt.plot(n_s, evds)
# plt.xlabel("n (number of expert demonstrations)")
# plt.ylabel("Difference in reward generated")
# plt.title("Cartpole reward comparison with mode of each posterior estimate at n (m=5000)")
# plt.savefig("plots/cart/predictive_checks_varying_n.png")

#Run the posterior calculations, saving some of the plots
reward = [1.7, 0.7]
m = 5000
for n in [6000, 8000, 10000]:
	heatmap_posterior, cart_pos, cart_angles = heatmap_dataset_conditional_density_posterior(reward, observations, plot_loc="plots/cart/", n=n, m=m)


# Do the top reward functions correspond to the same performance?
# rewards = most_common_reward(heatmap_posterior, cart_pos, cart_angles, pct=0.05)
# original_reward_perf = fqi(reward[0], reward[1], )
# perfs = []
# print("Number of reward functions to test: " + str(len(rewards)))
# for r in rewards:
# 	perf = fqi(r[0][0], r[0][1])
# 	perfs.extend(perf)
#
# fig, axs = plt.subplots(figsize=(14, 6))
# sns.distplot(original_reward_perf, label='Original Reward Function Performance', ax=axs)
# sns.distplot(perfs, label='Top 5% Reward Functions Performance', ax=axs)
# plt.xlabel("# Steps survived in evaluation")
# plt.legend()
# plt.savefig("plots/posterior_predictive_checks_" + str(reward) + ".png")

# # Do the lowest reward functions correspond to worse performance?
# rewards = most_common_reward(heatmap_posterior, cart_pos, cart_angles, top=False, pct=0.1)
# perfs = []
# for r in rewards:
# 	perf = fqi(r[0][0], r[0][1])
# 	perfs.extend(perf)
# print("Number of reward functions to test: " + str(len(rewards)))
# fig, axs = plt.subplots(figsize=(14, 6))
# sns.distplot(original_reward_perf, label='Original Reward Function Performance', ax=axs)
# sns.distplot(perfs, label='Bottom 10% Reward Functions Performance', ax=axs)
# plt.xlabel("# Steps survived in evaluation")
# plt.legend()

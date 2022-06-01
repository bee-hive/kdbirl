# Show that we're converging faster and to the correct posterior
# The posterior results in comparisons between the most dense rewards and least dense rewards
# To make this a reasonable pace, 3000 observations and 50 behavior opt samples
import random
from irl.irl_cart import generate_policy_rollouts, test_reward_parameters
import pickle
import os
random.seed(42)


reward_pos = [i/10 for i in range(5, 25, 2)]
reward_ang = [i/10 for i in range(1, 15, 2)]

# Generate a lot of data points for each reward function combination. We can arbitrarily use as many samples as we need.
for i, pos in enumerate(reward_pos):
	for j, ang in enumerate(reward_ang):
		fname = "/tigress/BEE/bayesian_irl/datasets/successful_rollouts/rollouts_" + str(pos) + "_" + str(ang) + ".pkl"
		rollouts, behav = generate_policy_rollouts(pos, ang, epoch=1000, init_experience=200, rollout_length=10, nns=4)
		if len(behav) > 0:
			filehandler = open(fname, "wb")
			pickle.dump(behav, filehandler)
			filehandler.close()



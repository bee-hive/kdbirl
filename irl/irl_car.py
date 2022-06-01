import configargparse
import torch
import torch.optim as optim

from simulated_fqi.environments.continuous_mountaincar import Continuous_MountainCarEnv
from simulated_fqi.models.agents import NFQAgent
from simulated_fqi.models.networks import NFQNetwork, ContrastiveNFQNetwork
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import tqdm


def generate_data(init_experience=400, goal=0.4):
	env_bg = Continuous_MountainCarEnv(group=0, goal_position=goal)
	all_rollouts = []
	if init_experience > 0:
		for _ in range(init_experience):
			rollout_bg, episode_cost = env_bg.generate_rollout(
				None, render=False, group=0
			)
			all_rollouts.extend(rollout_bg)
	return all_rollouts, env_bg


def runFQI(epoch=400, evaluations=10, big=False, num_rollouts=10, goal=0.45, verbose=False):
	print("Generating Data")
	train_rollouts, train_env = generate_data(goal=goal)
	test_rollouts, eval_env = generate_data(goal=goal)

	nfq_net = ContrastiveNFQNetwork(
		state_dim=train_env.state_dim, is_contrastive=False, big=big
	)
	optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)

	nfq_agent = NFQAgent(nfq_net, optimizer)

	# NFQ Main loop
	success_queue = [0] * 3
	for k, epoch in enumerate(tqdm.tqdm(range(epoch + 1))):
		state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set_car(
			train_rollouts
		)

		if not nfq_net.freeze_shared:
			loss = nfq_agent.train((state_action_b, target_q_values, groups))

		(eval_episode_length, eval_success, eval_episode_cost) = nfq_agent.evaluate_car(eval_env, render=False)
		success_queue = success_queue[1:]
		success_queue.append(1 if eval_success else 0)

		if sum(success_queue) == 3 and not nfq_net.freeze_shared == True:
			nfq_net.freeze_shared = True
			if verbose:
				print("FREEZING SHARED")
			break

	eval_env.step_number = 0
	eval_env.max_steps = 1000

	performance = []
	total = 0
	for it in range(evaluations):
		(eval_episode_length,eval_success,eval_episode_cost) = nfq_agent.evaluate_car(eval_env, False)
		performance.append(eval_episode_length)
		total += 1
		train_env.close()
		eval_env.close()

	# generate rollouts
	rollouts = []
	for it in range(num_rollouts):
		rollout_bg, episode_cost = train_env.generate_rollout(
			nfq_agent, render=False, group=0
		)
		rollouts.extend(rollout_bg)

	return performance, rollouts

def demonstration_density(rollouts, reward=""):
    car_pos = [i / 10 for i in range(-12, 8)]
    demonstration_density = np.zeros((len(car_pos), 1))
    for r in rollouts:
        state = r[0]
        pos, vel = state
        x = np.round(pos, 1)
        x_ind = car_pos.index(x)
        demonstration_density[x_ind] += 1

    plt.figure(figsize=(18, 2))
    ax = sns.heatmap(demonstration_density.T, xticklabels=car_pos)
    plt.xlabel("Cart Position")
    ax.invert_yaxis()
    plt.title("Demonstration Density with " + str(len(rollouts)) + " samples for reward " + str(reward))


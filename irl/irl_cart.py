import seaborn as sns
import torch.optim as optim
import scipy
from simulated_fqi.models.agents import NFQAgent
from simulated_fqi.models.networks import ContrastiveNFQNetwork
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import math
from simulated_fqi.environments.cartpole import CartPoleRegulatorEnv
import os
import pickle
import random
from multiprocessing import shared_memory, Process, Lock
plt.rcParams.update({'font.size': 10})



def plot_cart_reward(x_success_range, theta_success_range):
	x_threshold = 2.4
	theta_threshold_radians = math.pi / 2
	cart_pos = [i / 10 for i in range(-30, 30)]
	angles = [i / 10 for i in range(-20, 25, 2)]
	reward_matrix = np.zeros((len(angles), len(cart_pos)))
	for i, pos in enumerate(cart_pos):
		for j, ang in enumerate(angles):
			# In a forbidden state
			if (pos < -x_threshold
				or pos > x_threshold
				or ang < -theta_threshold_radians
				or ang > theta_threshold_radians):
				reward = 1
			# In success range
			elif (-x_success_range < pos < x_success_range
				  and -theta_success_range < ang < theta_success_range):
				reward = 0
			# Accumulating cost
			else:
				reward = 0.2
			reward_matrix[j, i] = reward
	plt.figure(figsize=(18, 7))
	ax = sns.heatmap(reward_matrix, xticklabels=cart_pos, yticklabels=angles)
	plt.xlabel("Cart Position")
	plt.ylabel("Pole Angle (radians)")
	ax.invert_yaxis()
	plt.title("Cartpole Reward: x_success=" + str(x_success_range) + " theta_success:" + str(theta_success_range))


# Reverse all rollouts
def demonstration_density(rollouts, reward="", vmax=150):
	cart_pos = [i / 10 for i in range(-30, 30)]
	angles = [i / 10 for i in range(-20, 25)]
	demonstration_density = np.zeros((len(angles), len(cart_pos)))
	for r in rollouts:
		state = r[0]
		x = np.round(state[0], 1)
		theta = np.round(state[2], 1)

		x_ind = cart_pos.index(x)
		theta_ind = angles.index(theta)

		demonstration_density[theta_ind, x_ind] += 1
	plt.figure(figsize=(18, 7))
	ax = sns.heatmap(demonstration_density, xticklabels=cart_pos, yticklabels=angles, vmax=vmax)
	plt.xlabel("Cart Position")
	plt.ylabel("Pole Angle (radians)")
	ax.invert_yaxis()
	plt.title("Demonstration Density with " + str(len(rollouts)) + " samples for reward " + str(reward))


def fqi(x_success, theta_success, init_experience=200, epoch=2000, verbose=False):
	evaluations = 15
	train_env = CartPoleRegulatorEnv(group=0, masscart=1.0, mode="train", x_success_range=x_success,
									 theta_success_range=theta_success)
	eval_env = CartPoleRegulatorEnv(group=0, masscart=1.0, mode='eval', x_success_range=x_success,
									theta_success_range=theta_success)

	# Setup agent
	nfq_net = ContrastiveNFQNetwork(
		state_dim=train_env.state_dim, is_contrastive=False
	)
	optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)
	nfq_agent = NFQAgent(nfq_net, optimizer)

	# NFQ Main loop
	# A set of transition samples denoted as D
	bg_rollouts = []
	total_cost = 0
	if init_experience > 0:
		for _ in range(init_experience):
			rollout_bg, episode_cost, _ = train_env.generate_rollout(
				None, render=False, group=0
			)
			bg_rollouts.extend(rollout_bg)
			total_cost += episode_cost
	all_rollouts = bg_rollouts.copy()

	bg_success_queue = [0] * 3
	for kk, ep in enumerate(tqdm.tqdm(range(epoch + 1))):
		state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
			all_rollouts
		)
		nfq_agent.train((state_action_b, target_q_values, groups))
		(eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg) = nfq_agent.evaluate(eval_env, render=False)
		bg_success_queue = bg_success_queue[1:]
		bg_success_queue.append(1 if eval_success_bg else 0)

		if sum(bg_success_queue) == 3 and not nfq_net.freeze_shared == True:
			nfq_net.freeze_shared = True
			if verbose:
				print("FREEZING SHARED")
			break

	eval_env.step_number = 0
	eval_env.max_steps = 1000
	performance_bg = []
	num_steps_bg = []
	for it in range(evaluations):
		(
			eval_episode_length_bg,
			eval_success_bg,
			eval_episode_cost_bg,
		) = nfq_agent.evaluate(eval_env, False)
		if verbose:
			print(eval_episode_length_bg, eval_success_bg)
		num_steps_bg.append(eval_episode_length_bg)
		performance_bg.append(eval_episode_length_bg)
		train_env.close()
		eval_env.close()

	return num_steps_bg


def generate_policy_rollouts(x_success, theta_success, init_experience=200, rollout_length=100, epoch=2000, verbose=False, nns=10):
	is_contrastive = False
	if verbose:
		evaluations = 5
	else:
		evaluations = 0
	rollouts = []
	costs = []
	for n in range(nns):
		train_env = CartPoleRegulatorEnv(group=0, masscart=1.0, mode="train", x_success_range=x_success,
										 theta_success_range=theta_success)
		eval_env = CartPoleRegulatorEnv(group=0, masscart=1.0, mode='eval', x_success_range=x_success,
										theta_success_range=theta_success)
		# Setup agent
		nfq_net = ContrastiveNFQNetwork(
			state_dim=train_env.state_dim, is_contrastive=is_contrastive
		)
		optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)
		nfq_agent = NFQAgent(nfq_net, optimizer)

		# NFQ Main loop
		# A set of transition samples denoted as D
		bg_rollouts = []
		total_cost = 0
		if init_experience > 0:
			for _ in range(init_experience):
				rollout_bg, episode_cost, _ = train_env.generate_rollout(
					None, render=False, group=0
				)
				bg_rollouts.extend(rollout_bg)
				total_cost += episode_cost
		all_rollouts = bg_rollouts.copy()

		bg_success_queue = [0] * 3
		for kk, ep in enumerate(tqdm.tqdm(range(epoch + 1))):
			state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
				all_rollouts
			)

			if not nfq_net.freeze_shared:
				loss = nfq_agent.train((state_action_b, target_q_values, groups))

			(eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg) = nfq_agent.evaluate(eval_env, render=False)
			bg_success_queue = bg_success_queue[1:]
			bg_success_queue.append(1 if eval_success_bg else 0)

			if sum(bg_success_queue) == 3 and not nfq_net.freeze_shared == True:
				nfq_net.freeze_shared = True
				if verbose:
					print("FREEZING SHARED")
				break

		if verbose:
			eval_env.step_number = 0
			eval_env.max_steps = 1000
			performance_bg = []
			num_steps_bg = []
			for it in range(evaluations):
				(
					eval_episode_length_bg,
					eval_success_bg,
					eval_episode_cost_bg,
				) = nfq_agent.evaluate(eval_env, False)
				if verbose:
					print(eval_episode_length_bg, eval_success_bg)
				num_steps_bg.append(eval_episode_length_bg)
				performance_bg.append(eval_episode_length_bg)
				train_env.close()
				eval_env.close()
			print("Stayed up for steps: ", num_steps_bg)
		count = 0
		for r in range(rollout_length):
			rollout, _, episode_cost = eval_env.generate_rollout(nfq_agent, render=False, group=0)
			if episode_cost > 600:
				rollouts.extend(rollout)
				costs.append(episode_cost)
				count += 1

	print("Steps stayed up: ", costs)
	observations = []
	for r in rollouts:
		observations.append((r[0], r[1], [x_success, theta_success]))

	new_behav = []
	for sample in observations:
		point = [sample[0][0], sample[0][2]]
		reward = sample[-1]
		new_sample = [point, reward]
		new_behav.append(new_sample)

	return rollouts, new_behav


def distance_r(r, r_prime, h_prime):
	dist_pos = np.absolute(r[0] - r_prime[0])
	dist_ang = np.absolute(r[1] - r_prime[1])
	dist = dist_pos + dist_ang
	return np.exp(-(np.power(dist, 2) / (2 * h_prime)))


def distance_points(p1, p2, h):
	dist = scipy.spatial.distance.euclidean(p1, p2)  # + scipy.spatial.distance.euclidean(p1[1], p2[1])
	return np.exp(-np.power(dist, 2) / (2 * h))


def distance_rewards(r_k, observations, h_prime):
	sum_diff = 0
	for sample in observations:
		r = sample[1]
		sum_diff += distance_r(r_k, r, h_prime)
	return sum_diff


def conditional_dist(s_i, dataset, reward, h_prime=0.3, h=0.08):
	sum = 0
	dist_rewards = distance_rewards(reward, dataset, h_prime)
	for s_j in dataset:
		weight = distance_r(reward, s_j[1], h_prime=h_prime) / dist_rewards
		dist = distance_points(s_i, s_j[0], h)
		est = dist * weight
		sum += est
	return sum


def estimate_expert_posterior(r_k, behavior_opt, observations, h, h_prime):
	post = 0
	dist_rewards = distance_rewards(r_k, observations, h_prime=h_prime)
	for s_i in behavior_opt:
		sum_si = 0
		for s_j in observations:
			weight = distance_r(r_k, s_j[1], h_prime=h_prime) / dist_rewards
			likelihood = distance_points(s_i[0], s_j[0], h=h) * weight
			sum_si += likelihood
		if sum_si == 0:
			post += np.log(0.000000000001)
		else:
			post += np.log(sum_si)
	return post


def generate_demonstration_density(rollouts):
	cart_pos = [i / 10 for i in range(-25, 25)]
	angles = [i / 10 for i in range(-16, 26)]
	demonstration_density = np.zeros((len(angles), len(cart_pos)))

	for r in rollouts:
		pos = r[0][0]
		theta = r[0][1]
		pos = np.round(pos, 1)
		theta = np.round(theta, 1)
		x_ind = cart_pos.index(pos)
		theta_ind = angles.index(theta)

		demonstration_density[theta_ind, x_ind] += 1
	return demonstration_density, cart_pos, angles


def generate_training(samples=100):
	reward_pos = [i / 10 for i in range(5, 25, 5)]  # use a step of 2
	reward_ang = [i / 10 for i in range(1, 15, 3)]  # use a step of 2
	behavior = []
	for i, pos in enumerate(reward_pos):
		for j, ang in enumerate(reward_ang):
			rollouts, b = generate_policy_rollouts(pos, ang, epoch=1000, init_experience=200, rollout_length=10, nns=2)
			idx = [i for i in range(len(b))]
			size = min(samples, len(b))
			if size > 0:
				new_behav_idx = np.random.choice(idx, size=size, replace=False)
				new_behav = np.asarray(b)[new_behav_idx]
				behavior.extend(new_behav)
			print(str(len(behavior)))

	return behavior

def find_optimal_bandwidth(observations):
	# Distance between rewards, h_prime
	all_distances_r = []
	all_distances_p = []
	for ii, o in enumerate(tqdm.tqdm(observations)):
		for o_prime in observations:
			all_distances_r.append(distance_r(o[1], o_prime[1], h_prime=0.2))
			all_distances_p.append(distance_points(o[0], o_prime[0], h=0.2))
	h_prime = np.std(all_distances_r) * np.std(all_distances_r)
	h = np.std(all_distances_p) * np.std(all_distances_p)

	return h, h_prime

# Why isn't gridsize used.
def heatmap_dataset_conditional_density_posterior(reward, observations, plot_loc=".", m=300, n=40, verbose=False):
	lock = Lock()

	def create_shared_block(pos, angles):
		a = np.zeros((len(angles), len(pos)), dtype=np.float32)
		shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
		np_array = np.ndarray(a.shape, dtype=np.float32, buffer=shm.buf)
		np_array[:] = a[:]
		return shm, np_array

	def conditional_density_add(states, observations, reward, h, h_prime, shr_name, i_s, j_s):
		cart_pos = [i / 10 for i in range(-25, 25)]
		cart_angles = [i / 10 for i in range(-16, 26)]
		for state, i, j in zip(states, i_s, j_s):
			c_est = conditional_dist(state, observations, reward, h=h, h_prime=h_prime)
			existing_shm = shared_memory.SharedMemory(name=shr_name)
			hmap_conditional = np.ndarray((len(cart_angles), len(cart_pos)), dtype=np.float32, buffer=existing_shm.buf)
			lock.acquire()
			hmap_conditional[i, j] = c_est
			lock.release()
			existing_shm.close()

	def posterior_add(r_k, behavior_opt, observations, h, h_prime, shr_name, i, j):
		reward_pos = [i / 10 for i in range(5, 25, 2)]
		reward_ang = [i / 10 for i in range(1, 10, 2)]
		post = estimate_expert_posterior(r_k, behavior_opt, observations, h=h, h_prime=h_prime)
		existing_shm = shared_memory.SharedMemory(name=shr_name)
		hmap_post = np.ndarray((len(reward_ang), len(reward_pos)), dtype=np.float32, buffer=existing_shm.buf)
		lock.acquire()
		hmap_post[i, j] = post
		lock.release()
		existing_shm.close()

	# Generate Data
	fname = "/tigress/BEE/bayesian_irl/datasets/successful_rollouts/rollouts_" + str(reward[0]) + "_" + str(reward[1]) + ".pkl"
	if os.path.exists(fname):
		if verbose:
			print("Found behavior, using saved file")
		filehandler = open(fname, "rb")
		behavior_opt = pickle.load(filehandler)
		filehandler.close()
	else:
		if verbose:
			print("Generating new behavior")
		rollouts, behavior_opt = generate_policy_rollouts(reward[0], reward[1], epoch=1000, init_experience=200, rollout_length=10, nns=2)
		fname = open(fname, "wb")
		pickle.dump(behavior_opt, fname)

	h, h_prime = find_optimal_bandwidth(random.choices(observations, k=1000))
	if verbose:
		print("Found hyperparameters: h=" + str(h) + " h_prime=" + str(h_prime))
	idx = [i for i in range(len(behavior_opt))]
	size = min(n, len(behavior_opt))
	if size > 0:
		new_behav_idx = np.random.choice(idx, size=size, replace=False)
		behavior_opt = np.asarray(behavior_opt)[new_behav_idx]

	idx = [i for i in range(len(observations))]
	size = min(m, len(observations))
	if size > 0:
		new_idx = np.random.choice(idx, size=size, replace=False)
		observations = np.asarray(observations)[new_idx]

	# Demonstration Density
	if verbose:
		print("Drawing demonstration density")
	demonstration_density, cart_pos, angles = generate_demonstration_density(behavior_opt)
	fig, axs = plt.subplots(3, 1, figsize=(18, 14), constrained_layout=True)
	sns.heatmap(demonstration_density, xticklabels=cart_pos, yticklabels=angles, ax=axs[0], vmax=10)
	axs[0].set_xlabel("Cart Position")
	axs[0].set_ylabel("Pole Angle (radians)")
	axs[0].set_title("Demonstration density for reward: " + str(reward))
	axs[0].invert_yaxis()

	# Conditional Density
	if verbose:
		print("Calculating conditional density")
	cart_pos = [i / 10 for i in range(-25, 25)]
	cart_angles = [i / 10 for i in range(-16, 26)]
	shr, heatmap_conditional = create_shared_block(pos=cart_pos, angles=cart_angles)
	processes = []
	for i, ang in enumerate(tqdm.tqdm(cart_angles)):
		states = []
		i_s = []
		j_s = []
		for j, pos in enumerate(cart_pos):
			state = [pos, ang]
			states.append(state)
			i_s.append(i)
			j_s.append(j)
		_process = Process(target=conditional_density_add, args=(states, observations, reward, h, h_prime, shr.name, i_s, j_s))
		processes.append(_process)
		_process.start()
	for _process in processes:
		_process.join()

	sns.heatmap(heatmap_conditional, xticklabels=cart_pos, yticklabels=cart_angles, ax=axs[1])
	axs[1].invert_yaxis()
	axs[1].set_xlabel("Cart Position")
	axs[1].set_ylabel("Pole Angle (radians)")
	axs[1].set_title("Conditional density wrt reward: " + str(reward))

	shr.close()
	shr.unlink()

	# Posterior Distribution
	if verbose:
		print("Calculating posterior distribution")
	reward_pos = [i / 10 for i in range(5, 25, 2)]
	reward_ang = [i / 10 for i in range(1, 10, 2)]
	shr, heatmap_posterior = create_shared_block(pos=reward_pos, angles=reward_ang)
	processes = []
	for i, ang in enumerate(tqdm.tqdm(reward_ang)):
		for j, pos in enumerate(reward_pos):
			r_k = [pos, ang]
			_process = Process(target=posterior_add, args=(r_k, behavior_opt, observations, h, h_prime, shr.name, i, j))
			processes.append(_process)
			_process.start()
	for _process in processes:
		_process.join()

	sns.heatmap(heatmap_posterior, xticklabels=reward_pos, yticklabels=reward_ang, ax=axs[2])
	axs[2].invert_yaxis()
	axs[2].set_xlabel("X Success Range")
	axs[2].set_ylabel("Angle Success Range")
	axs[2].set_title("Expert posterior, true reward=" + str(reward))

	file = plot_loc + "r=" + str(reward) + "_m=" + str(m) + "_n=" + str(n)
	plot_file = file + ".png"
	posterior_file = file + ".npy"

	np.save(posterior_file, heatmap_posterior)
	shr.close()
	shr.unlink()

	plt.savefig(plot_file)
	plt.close()
	return heatmap_posterior, reward_pos, reward_ang


def most_common_reward(hmap_posterior, reward_pos, reward_ang, top=True, pct=0.1):
	post = hmap_posterior / np.max(np.abs(hmap_posterior))
	post = np.exp(post)
	post /= np.sum(post)

	rewards = np.random.multinomial(1000000, post.flatten())
	rewards = rewards.reshape((len(reward_ang), len(reward_pos)))

	evaluations = rewards // 100
	x = reward_pos
	y = reward_ang
	# xx is the position, yy is the angle
	xx, yy = np.meshgrid(x, y, sparse=True)

	reward_fns = []
	for kk, i in enumerate(range(len(reward_pos))):
		for ll, j in enumerate(range(len(reward_ang))):
			r_k = [xx[0][i], yy[j][0]]
			count = evaluations[ll, kk]
			for aa in range(count):
				reward_fns.append(r_k)
	fn_to_count = {}
	for fn in reward_fns:
		if tuple(fn) not in fn_to_count:
			fn_to_count[tuple(fn)] = 0
		fn_to_count[tuple(fn)] += 1

	fns = sorted(fn_to_count.items(), key=lambda item: item[1], reverse=True)
	if top:
		most_common_fns = fns[:int(len(fns) * pct)]
	else:
		most_common_fns = fns[int(len(fns) * pct):]
	return most_common_fns

# TODO: parallelize this
def test_reward_parameters(parameter_pos, parameter_ang, rollouts=10, nns=5, verbose=False):
	steps_stayed_up = []
	epochs=2000
	init_experience=1000 # TODO: Use init_experience when getting rollouts from saved

	train_env = CartPoleRegulatorEnv(group=0, masscart=1.0, mode="train", x_success_range=parameter_pos,
									 theta_success_range=parameter_ang)
	eval_env = CartPoleRegulatorEnv(group=0, masscart=1.0, mode='eval', x_success_range=parameter_pos,
									theta_success_range=parameter_ang)
	# Setup agent
	nfq_net = ContrastiveNFQNetwork(state_dim=train_env.state_dim, is_contrastive=False)
	optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)
	nfq_agent = NFQAgent(nfq_net, optimizer)

	# Gather data from existing dataset if applicable
	fname = "/tigress/BEE/bayesian_irl/datasets/cart_eval/rollouts_" + str(parameter_pos) + "_" + str(parameter_ang) + ".pkl"
	if os.path.exists(fname):
		if verbose:
			print("Found behavior, using saved file")
		filehandler = open(fname, "rb")
		bg_rollouts = pickle.load(filehandler)
		filehandler.close()
	else:
		if verbose:
			print("Generating new training data")
		bg_rollouts = []
		total_cost = 0
		if init_experience > 0:
			for _ in range(init_experience):
				rollout_bg, episode_cost, _ = train_env.generate_rollout(None, render=False, group=0)
				bg_rollouts.extend(rollout_bg)
				total_cost += episode_cost
		filehandler = open(fname, 'wb')
		pickle.dump(bg_rollouts, filehandler)
	all_rollouts = bg_rollouts.copy()

	for n in range(nns):
		bg_success_queue = [0] * 3
		for kk, ep in enumerate(tqdm.tqdm(range(epochs + 1))):
			state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
				all_rollouts
			)
			nfq_agent.train((state_action_b, target_q_values, groups))

			(eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg) = nfq_agent.evaluate(eval_env, render=False)
			bg_success_queue = bg_success_queue[1:]
			bg_success_queue.append(1 if eval_success_bg else 0)

			if sum(bg_success_queue) == 3 and not nfq_net.freeze_shared == True:
				nfq_net.freeze_shared = True
				if verbose:
					print("FREEZING SHARED")
				break

		eval_env.step_number = 0
		eval_env.max_steps = 1000
		for it in range(rollouts):
			(eval_episode_length_bg,eval_success_bg,eval_episode_cost_bg) = nfq_agent.evaluate(eval_env, False)
			steps_stayed_up.append(eval_episode_length_bg)
			train_env.close()
			eval_env.close()
		if verbose:
			print("Stayed up for steps: ", steps_stayed_up)

	return steps_stayed_up

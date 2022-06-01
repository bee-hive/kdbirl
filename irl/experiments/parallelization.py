import os
import numpy as np
from multiprocessing import shared_memory, Process, Lock
import time
import pickle
import tqdm
import random
from irl.irl_cart import generate_policy_rollouts, find_optimal_bandwidth, conditional_dist, estimate_expert_posterior

lock = Lock()

def create_shared_block(pos, angles):
	a = np.zeros((len(angles), len(pos)), dtype=np.float16)
	shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
	np_array = np.ndarray(a.shape, dtype=np.float16, buffer=shm.buf)
	np_array[:] = a[:]
	return shm, np_array


def conditional_density_add(state, observations, reward, h, h_prime, shr_name, i, j):
	cart_pos = [i / 10 for i in range(-25, 25)]
	cart_angles = [i / 10 for i in range(-16, 26)]
	c_est = conditional_dist(state, observations, reward, h=h, h_prime=h_prime)
	existing_shm = shared_memory.SharedMemory(name=shr_name)
	hmap_conditional = np.ndarray((len(cart_angles), len(cart_pos)), dtype=np.float16, buffer=existing_shm.buf)
	lock.acquire()
	hmap_conditional[i, j] = c_est
	lock.release()
	existing_shm.close()

def posterior_add(r_k, behavior_opt, observations, h, h_prime, shr_name, i, j):
	reward_pos = [i / 10 for i in range(5, 25, 2)]
	reward_ang = [i / 10 for i in range(1, 10, 2)]
	post = estimate_expert_posterior(r_k, behavior_opt, observations, h=h, h_prime=h_prime)
	existing_shm = shared_memory.SharedMemory(name=shr_name)
	hmap_post = np.ndarray((len(reward_ang), len(reward_pos)), dtype=np.float16, buffer=existing_shm.buf)
	lock.acquire()
	hmap_post[i, j] = post
	lock.release()
	existing_shm.close()


def parallel_heatmap_dataset_conditional_density_posterior(reward, observations, sample_size=1000, verbose=False):
	# Generate Data
	fname = "datasets/successful_rollouts/rollouts_" + str(reward[0]) + "_" + str(reward[1]) + ".pkl"
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
	size = min(sample_size, len(behavior_opt))
	if size > 0:
		new_behav_idx = np.random.choice(idx, size=size, replace=False)
		behavior_opt = np.asarray(behavior_opt)[new_behav_idx]

	# Conditional Density
	if verbose:
		print("Calculating conditional density")

	cart_pos = [i / 10 for i in range(-25, 25)]
	cart_angles = [i / 10 for i in range(-16, 26)]
	shr, heatmap_conditional = create_shared_block(pos=cart_pos, angles=cart_angles)
	processes = []
	for i, ang in enumerate(tqdm.tqdm(cart_angles)):
		for j, pos in enumerate(cart_pos):
			state = [pos, ang]
			_process = Process(target=conditional_density_add, args=(state, observations, reward, h, h_prime, shr.name, i, j))
			processes.append(_process)
			_process.start()
	for _process in processes:
		_process.join()
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
	shr.close()
	shr.unlink()

	return heatmap_posterior, reward_pos, reward_ang


# Unpackage the observations dataset
reward_pos = [i / 10 for i in range(5, 6, 2)]
reward_ang = [i / 10 for i in range(1, 4, 2)]

# Generate a lot of data points for each reward function combination. We can arbitrarily use as many samples as we need.
num_samples_per_rollout = 1000
observations = []
for i, pos in enumerate(reward_pos):
	for j, ang in enumerate(reward_ang):
		fname = "datasets/successful_rollouts/rollouts_" + str(pos) + "_" + str(ang) + ".pkl"
		filehandler = open(fname, "rb")
		b = pickle.load(filehandler)
		idx = [i for i in range(len(b))]
		size = min(num_samples_per_rollout, len(b))
		if size > 0:
			new_behav_idx = np.random.choice(idx, size=size, replace=False)
			new_behav = np.asarray(b)[new_behav_idx]
			observations.extend(new_behav)

# Run the posterior calculations, saving some of the plots
reward = [0.5, 0.1]
heatmap_posterior, cart_pos, cart_angles = parallel_heatmap_dataset_conditional_density_posterior(reward, observations, sample_size=200)


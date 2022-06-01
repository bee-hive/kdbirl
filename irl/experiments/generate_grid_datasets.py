from irl.irl_gridworld import runLinearFQI, runNonlinearFQI
import pickle
import tqdm
import numpy as np

gridsize=10
init_experience=200

datadir = "/tigress/BEE/bayesian_irl/datasets/grid_datasets/"
# for aa, i in enumerate(tqdm.tqdm(range(-10, 11, 3))):
# 	for j in range(-10, 11, 3):
# 		r = [i / 10, j / 10]
# 		behavior_opt, opt_agent = runLinearFQI(dataset='bg', init_experience=init_experience, num_rollouts=100, behavior=True, reward_weights_shared=r, gridsize=gridsize)
# 		observations = []
# 		for sample in behavior_opt:
# 			s = (sample[0], sample[1], r)
# 			observations.append(s)
# 		fname = datadir + "rollouts_r=" + str(r) + ".pkl"
# 		filehandler = open(fname, "wb")
# 		pickle.dump(observations, filehandler)
# 		filehandler.close()

target_states = []
for i in range(0, 10, 2):
	for j in range(0, 10, 2):
		target_states.append([i, j])
target_states.append([9, 9])

for kk, r in enumerate(tqdm.tqdm(target_states)):
	obs = []
	behavior_opt = runNonlinearFQI(init_experience=init_experience, behavior=True, target_state=r, n=gridsize, num_rollouts=1000)
	for sample in behavior_opt:
		s = (sample[0], sample[1], r)
		obs.append(s)
	fname = datadir + "rollouts_target=" + str(r) + ".pkl"
	filehandler = open(fname, "wb")
	pickle.dump(obs, filehandler)
	filehandler.close()


